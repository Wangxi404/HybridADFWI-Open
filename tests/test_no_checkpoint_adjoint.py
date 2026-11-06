"""True full-history propagation: neither PyTorch checkpoint nor replay."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybridadfwi.adjoint import (
    COMPONENTS, _full_history_step, disable_native_checkpointing,
    hybrid_forward, preserve_padding_dtype,
)
from hybridadfwi.experiment import build, verify_upstream


def config(tmp_path, free_surface):
    nz, nx = 18, 24
    z, x = np.indices((nz, nx), dtype=np.float64)
    values = {"vp": 2500+30*z+5*x, "vs": 1350+10*z+2*x, "rho": 2100+2*z+x}
    path = tmp_path/"model.npz"
    np.savez(path, **{p+"_"+k: v for p, v in values.items() for k in ("true", "initial")},
             dx=10., dz=10., ox=0., oz=0.)
    return dict(model=str(path), nt=36, dt=.0005, f0=200., depth_m=40., shots=2,
                receivers=5, parameters=["vp", "vs", "rho"], free_surface=free_surface,
                nabc=5, device="cpu")


def graph_node_names(tensor):
    names, pending, seen = [], [tensor.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        pending.extend(next_node for next_node, _ in node.next_functions)
    return names


@pytest.mark.parametrize("free_surface", [False, True])
def test_true_no_checkpoint_forward_and_all_parameter_gradients(tmp_path, free_surface):
    torch.set_num_threads(1)
    cfg = config(tmp_path, free_surface)
    model_ad, prop_ad = build(cfg)
    prop_ad.rcv_x[1] = prop_ad.rcv_x[0]
    prop_ad.rcv_z[1] = prop_ad.rcv_z[0]
    with preserve_padding_dtype(), disable_native_checkpointing():
        reference = prop_ad.forward(fd_order=8, checkpoint_segments=1)
    names = graph_node_names(reference["vz"])
    assert not any("Checkpoint" in name for name in names)
    assert reference["vz"].requires_grad
    generator = torch.Generator().manual_seed(842)
    weights = {key: torch.randn(reference[key].shape, dtype=torch.float64, generator=generator)
               / reference[key].detach().abs().max().clamp_min(1e-100) / reference[key].numel()
               for key in COMPONENTS}
    loss = sum((reference[key]*weights[key]).sum() for key in COMPONENTS)
    loss.backward()
    expected = {key: getattr(model_ad, key).grad.detach().clone() for key in cfg["parameters"]}

    _, prop = build(cfg)
    prop.rcv_x[1] = prop.rcv_x[0]
    prop.rcv_z[1] = prop.rcv_z[0]
    output, tape = hybrid_forward(prop, checkpointing=False)
    for key in COMPONENTS:
        torch.testing.assert_close(output[key], reference[key], rtol=0, atol=0)
        assert output[key].grad_fn is None
        assert not output[key].requires_grad
    assert tape.stats["checkpoint_segments"] == 0
    assert tape.stats["checkpoint_state_tensors"] == 0
    assert len(tape.full_history) == cfg["nt"]
    assert all(value is None for value in tape.segments[0][1][28:43])
    storages = set()
    for old_vx, old_vz, stresses in tape.full_history:
        assert all(t.grad_fn is None and not t.requires_grad for t in (old_vx, old_vz, *stresses))
        for field in (old_vx, old_vz, *stresses):
            storages.add(field.untyped_storage().data_ptr())
            assert field.untyped_storage().nbytes() == cfg["nt"]*field.numel()*field.element_size()
    # Five preallocated contiguous device buffers, not 5*nt clone allocations.
    assert len(storages) == 5
    # A replay attempt would fail, so a passing backward proves direct use of
    # retained original-forward history rather than reconstruction.
    def forbidden_replay(*args, **kwargs):
        raise AssertionError("No forward replay is allowed")
    tape.step = forbidden_replay
    leaves = {key: output[key].detach().requires_grad_() for key in COMPONENTS}
    loss = sum((leaves[key]*weights[key]).sum() for key in COMPONENTS)
    vjps = torch.autograd.grad(loss, tuple(leaves.values()))
    gradients = tape.backward(dict(zip(COMPONENTS, vjps)))
    for key in cfg["parameters"]:
        relative = (torch.linalg.vector_norm(gradients[key]-expected[key]) /
                    torch.linalg.vector_norm(expected[key])).item()
        assert relative < 1e-10, (key, relative)
    assert tape.stats["replay_seconds"] == 0
    assert tape.stats["peak_history_steps"] == cfg["nt"]
    assert tape.full_history == []
    verify_upstream()


def test_no_checkpoint_requires_single_native_loop(tmp_path):
    _, prop = build(config(tmp_path, True))
    with pytest.raises(ValueError, match="checkpoint_segments=1"):
        hybrid_forward(prop, checkpoint_segments=2, checkpointing=False)


def test_history_instrumentation_keeps_compiled_native_loop():
    function, digest = _full_history_step()
    assert "prim::Loop" in str(function.graph)
    assert len(digest) == 64
