"""Numerical checks against the *same* unmodified upstream FD8 simulator."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybridadfwi.adjoint import COMPONENTS, hybrid_forward, preserve_padding_dtype
from hybridadfwi.experiment import build


def _config(tmp_path, free_surface, segments):
    nz, nx = 18, 24
    z, x = np.indices((nz, nx), dtype=np.float64)
    vp = 2500 + 30 * z + 5 * x
    vs = 1350 + 10 * z + 2 * x
    rho = 2100 + 2 * z + x
    arrays = {name + "_" + kind: value for name, value in
              (("vp", vp), ("vs", vs), ("rho", rho))
              for kind in ("true", "initial")}
    path = tmp_path / "model.npz"
    np.savez(path, **arrays, dx=10., dz=10., ox=0., oz=0.)
    return dict(model=str(path), nt=36, dt=.0005, f0=200., depth_m=40.,
                shots=2, receivers=5, parameters=["vp", "vs", "rho"],
                free_surface=free_surface, nabc=5, device="cpu",
                checkpoint_segments=segments)


def _relative_l2(actual, expected):
    return (torch.linalg.vector_norm(actual - expected) /
            torch.linalg.vector_norm(expected).clamp_min(torch.finfo(expected.dtype).tiny)).item()


@pytest.mark.parametrize("free_surface", [False, True])
@pytest.mark.parametrize("segments", [1, 3])
def test_all_material_gradients_match_upstream_autograd(tmp_path, free_surface, segments):
    """All five outputs, duplicate receivers, all three physical parameters."""
    torch.set_num_threads(1)
    config = _config(tmp_path, free_surface, segments)
    ad_model, ad_prop = build(config)
    # Exercise the sampling transpose for coincident receivers as well.
    ad_prop.rcv_x[1] = ad_prop.rcv_x[0]
    ad_prop.rcv_z[1] = ad_prop.rcv_z[0]
    with preserve_padding_dtype():
        expected = ad_prop.forward(fd_order=8, checkpoint_segments=segments)
    generator = torch.Generator().manual_seed(390)
    weights = {name: torch.randn(expected[name].shape, dtype=torch.float64, generator=generator)
               / expected[name].detach().abs().max().clamp_min(1e-100)
               / expected[name].numel() for name in COMPONENTS}
    loss = sum((expected[name] * weights[name]).sum() for name in COMPONENTS)
    loss.backward()
    reference = {name: getattr(ad_model, name).grad.detach().clone()
                 for name in config["parameters"]}

    hybrid_model, hybrid_prop = build(config)
    hybrid_prop.rcv_x[1] = hybrid_prop.rcv_x[0]
    hybrid_prop.rcv_z[1] = hybrid_prop.rcv_z[0]
    actual, tape = hybrid_forward(hybrid_prop, fd_order=8, checkpoint_segments=segments)
    for name in COMPONENTS:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        assert not actual[name].requires_grad
        assert actual[name].grad_fn is None
    # AD computes a source only in receiver-data space.
    leaves = {name: actual[name].detach().requires_grad_() for name in COMPONENTS}
    loss = sum((leaves[name] * weights[name]).sum() for name in COMPONENTS)
    data_sources = torch.autograd.grad(loss, tuple(leaves.values()))
    gradients = tape.backward(dict(zip(COMPONENTS, data_sources)))
    for name in config["parameters"]:
        error = _relative_l2(gradients[name], reference[name])
        assert error < 1e-10, (name, error)
        assert getattr(hybrid_model, name).grad is None
    assert not tape.stats["temporal_autograd"]
    assert tape.stats["static_material_autograd"]


def test_one_output_and_directional_derivative(tmp_path):
    """Independent central differences check the reported model-space gradient."""
    torch.set_num_threads(1)
    config = _config(tmp_path, True, 3)
    config["parameters"] = ["vs"]
    model, prop = build(config)
    output, tape = hybrid_forward(prop, checkpoint_segments=3)
    scale = output["vz"].abs().max().clamp_min(1e-100)
    leaf = output["vz"].detach().requires_grad_()
    loss = .5 * (leaf / scale).square().mean()
    source, = torch.autograd.grad(loss, leaf)
    gradient = tape.backward({"vz": source})["vs"]
    direction = torch.randn(model.vs.shape, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(917))
    direction /= torch.linalg.vector_norm(direction)
    predicted = (gradient * direction).sum().item()
    eps = .05
    values = []
    initial = model.vs.detach().clone()
    for sign in (-1, 1):
        with torch.no_grad():
            model.vs.copy_(initial + sign * eps * direction)
        with preserve_padding_dtype(), torch.no_grad():
            out = prop.forward(fd_order=8, checkpoint_segments=3)
        values.append((.5 * (out["vz"] / scale).square().mean()).item())
    actual = (values[1] - values[0]) / (2 * eps)
    assert abs(actual - predicted) / max(abs(predicted), 1e-30) < 1e-5

