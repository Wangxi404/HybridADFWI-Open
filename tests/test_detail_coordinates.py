"""A data-free numerical check of the demonstrator's actual parameter map."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import gradient_detail as detail


def test_coordinate_gradient_matches_fd_when_reusing_native_propagator(tmp_path):
    torch.set_num_threads(1)
    nz, nx = 18, 24
    z, x = np.indices((nz, nx), dtype=float)
    water = z < 4
    fields = {"vp": np.where(water, 1500., 2450.+10*z+2*x),
              "vs": np.where(water, 0., 1300.+5*z+x),
              "rho": np.full((nz, nx), 2100.)}
    arrays = {}
    for name, value in fields.items():
        arrays[name+"_true"] = value
        arrays[name+"_initial"] = np.where(water, value, value*(.98 if name != "rho" else 1.))
    path = tmp_path/"synthetic.npz"
    np.savez(path, **arrays, dx=10., dz=10., ox=0., oz=0.)
    cfg = dict(model=str(path), device="cpu", nt=320, dt=.0005, f0=40., shots=3,
               selected_shots=[1], receivers=6, depth_m=20., nabc=5, free_surface=True,
               segments=8, parameters=["vp", "vs", "rho"],
               fixed_pml_reference_vmax_m_s=4700., constrained_coordinates=True,
               fixed_buffer_m=20., parameter_taper_m=10., minimum_offset_m=0.,
               offset_taper_m=10., mute_velocity_m_s=1500., direct_mute_after_s=0.,
               time_taper_s=.001, end_taper_s=.005)
    with torch.no_grad():
        _, truth_prop, _, _ = detail.build_model(cfg, truth=True, trainable=False)
        observed = detail.pressure(detail.native(truth_prop, cfg, 1))
        window = detail.data_window(truth_prop, cfg, 1, torch)
        scale = (observed*window).square().mean().sqrt().item()
    model, prop, coordinates, mask = detail.build_model(cfg)
    output = detail.native(prop, cfg, 1)
    loss = detail.loss_function(detail.pressure(output), observed, window, scale)
    loss.backward()
    physical = model.vs.grad.detach().clone()
    gradient = coordinates["vs"].grad.detach().clone()
    torch.testing.assert_close(gradient, mask*physical, rtol=0, atol=0)
    assert gradient.norm() > 0
    frozen = mask == 0
    # Also perturb frozen coordinates: their physical field must remain fixed.
    direction = gradient/gradient.norm() + frozen.to(torch.float64)/frozen.sum().sqrt()
    direction /= direction.norm()
    expected = (gradient*direction).sum().item()
    initial = coordinates["vs"].detach().clone()
    values = []
    eps = .05
    for sign in (-1, 1):
        with torch.no_grad():
            coordinates["vs"].copy_(initial+sign*eps*direction)
            prediction = detail.pressure(detail.native(prop, cfg, 1))
            values.append(detail.loss_function(prediction, observed, window, scale).item())
            torch.testing.assert_close(model.vs[frozen], initial[frozen], rtol=0, atol=0)
    finite_difference = (values[1]-values[0])/(2*eps)
    assert abs(finite_difference-expected)/abs(expected) < 1e-5
