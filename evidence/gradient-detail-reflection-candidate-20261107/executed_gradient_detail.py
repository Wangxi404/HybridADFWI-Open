#!/usr/bin/env python3
"""A separate marine-pressure Marmousi-II gradient-detail experiment.

This does not replace the original performance benchmarks. Raw model-space
gradients are saved, without smoothing, illumination division or preconditioning.
Geometry/time/offset weights define a fixed, explicitly documented data objective.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    for name in ("model", "observations"):
        if name in cfg and not Path(cfg[name]).is_absolute():
            cfg[name] = str(ROOT / cfg[name])
    return cfg


def initialize(cfg):
    import torch
    from hybridadfwi.experiment import verify_upstream
    verify_upstream()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Keep upstream default dtype float32 for FD constants; fields are explicit float64")
    if cfg["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no implicit fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def sync(torch, cfg):
    if cfg["device"] == "cuda":
        torch.cuda.synchronize()


def pressure(output):
    """Isotropic pressure at fluid receivers from the native normal stresses."""
    return -.5 * (output["txx"] + output["tzz"])


def build_model(cfg, truth=False, trainable=True):
    """External frozen-PML and explicit fixed-water/buffer parameterization."""
    import torch
    from hybridadfwi.experiment import build
    model, prop = build(cfg, truth=truth, trainable=trainable)
    from ADFWI.propagator.boundary_condition import bc_pml_xz
    vmax = cfg.get("fixed_pml_reference_vmax_m_s")
    if vmax is not None:
        bcx, bcz = bc_pml_xz(prop.nx, prop.nz, prop.dx, prop.dz, pml=prop.nabc,
                             vmax=vmax, free_surface=prop.free_surface)
        prop.bcx = torch.as_tensor(bcx, dtype=prop.dtype, device=prop.device)
        prop.bcz = torch.as_tensor(bcz, dtype=prop.dtype, device=prop.device)
    if truth or not cfg.get("constrained_coordinates", False):
        return model, prop, None, None
    with np.load(cfg["model"]) as stored:
        reference = {name: torch.as_tensor(stored[name+"_initial"], dtype=torch.float64,
                                          device=prop.device) for name in cfg["parameters"]}
        solid = torch.as_tensor(stored["vs_true"] > 0, device=prop.device)
    x = torch.arange(prop.nx, device=prop.device, dtype=torch.float64)*prop.dx
    z = torch.arange(prop.nz, device=prop.device, dtype=torch.float64)*prop.dz
    distance = torch.minimum(x[None, :].expand(prop.nz, -1),
                             (x[-1]-x)[None, :].expand(prop.nz, -1))
    distance = torch.minimum(distance, (z[-1]-z)[:, None].expand(-1, prop.nx))
    u = ((distance-cfg["fixed_buffer_m"])/cfg["parameter_taper_m"]).clamp(0, 1)
    mask = (.5-.5*torch.cos(math.pi*u))*solid
    coordinates = {}
    for name in cfg["parameters"]:
        coordinate = torch.nn.Parameter(getattr(model, name).detach().clone(), requires_grad=trainable)
        # Keep the vendor model class, methods and kernels intact. Its physical
        # field becomes a differentiable static realization of new coordinates.
        del model._parameters[name]
        model.register_parameter("coordinate_"+name, coordinate)
        coordinates[name] = coordinate
    def realize_fields():
        for name, coordinate in coordinates.items():
            field = reference[name] + mask*(coordinate-reference[name])
            setattr(model, name, field)
            if field.requires_grad:
                field.retain_grad()  # retain the unconstrained physical VJP too
    realize_fields()
    original_model_forward = model.forward
    def forward_with_coordinates():
        # Recreate the static map for every solve, including finite-difference
        # perturbations/iterations; do not reuse a stale eager realization.
        realize_fields()
        return original_model_forward()
    model.forward = forward_with_coordinates
    return model, prop, coordinates, mask


def native(prop, cfg, shot):
    from hybridadfwi.adjoint import preserve_padding_dtype
    with preserve_padding_dtype():
        return prop.forward(fd_order=8, checkpoint_segments=cfg["segments"], shot_index=[shot])


def data_window(prop, cfg, shot, torch):
    offset = torch.abs(prop.rcv_x - prop.src_x[shot]).to(torch.float64) * prop.dx
    lo, width = cfg["minimum_offset_m"], cfg["offset_taper_m"]
    position = ((offset - lo) / width).clamp(0, 1)
    offset_weight = .5 - .5 * torch.cos(math.pi * position)
    time_axis = torch.arange(prop.nt, dtype=torch.float64, device=prop.device) * prop.dt
    delay = 1.5 / cfg["f0"]
    if cfg.get("mute_reference", "water_direct") == "known_seafloor_primary":
        # Use only supplied known water geometry; no unknown target reflectors.
        source_z = prop.src_z[shot].to(torch.float64)*prop.dz + prop.oz
        receiver_z = prop.rcv_z.to(torch.float64)*prop.dz + prop.oz
        vertical = 2*cfg["known_water_depth_m"]-source_z-receiver_z
        path = torch.sqrt(offset.square()+vertical.square())
        start = delay + path/cfg["mute_velocity_m_s"] + cfg["seafloor_mute_after_s"]
    else:
        start = delay + offset / cfg["mute_velocity_m_s"] + cfg["direct_mute_after_s"]
    position = ((time_axis[:, None] - start[None, :]) / cfg["time_taper_s"]).clamp(0, 1)
    weight = (.5 - .5 * torch.cos(math.pi * position)) * offset_weight[None, :]
    end_taper = ((time_axis[-1] - time_axis) / cfg["end_taper_s"]).clamp(0, 1)
    return weight[None] * (.5 - .5 * torch.cos(math.pi * end_taper))[None, :, None]


def loss_function(prediction, observation, window, scale):
    return .5 * (((prediction - observation) * window) / scale).square().mean()


def dispersion_curve(cfg, fields):
    """Uniform-medium FD8 + leapfrog plane-wave analysis, not an interface test."""
    from hybridadfwi.experiment import load_upstream
    load_upstream()
    from ADFWI.propagator.elastic_kernels import DiffCoef
    coefficients = DiffCoef(4, "s").numpy().astype(np.float64)
    h, dt = cfg["grid"]["dx"], cfg["dt"]
    positive = fields["vs"][fields["vs"] > 0]
    speeds = {"minimum_positive_vs": float(positive.min()), "vs_p05_positive": float(np.percentile(positive, 5)),
              "water_vp": 1500., "maximum_vp": float(fields["vp"].max())}
    rows = []
    for name, speed in speeds.items():
        for multiple in (1., 2., 2.5):
            f = cfg["f0"] * multiple
            kh = 2 * np.pi * f * h / speed
            courant = speed * dt / h
            for angle in (0., 15., 30., 45.):
                theta = np.deg2rad(angle)
                symbol = []
                for component in (np.cos(theta), np.sin(theta)):
                    symbol.append(sum(c * np.sin((j + .5) * kh * component)
                                      for j, c in enumerate(coefficients)))
                argument = courant * np.linalg.norm(symbol)
                ratio = 2 * np.arcsin(argument) / (courant * kh) if argument <= 1 else None
                rows.append(dict(speed_label=name, speed_m_s=speed, frequency_hz=f,
                                 frequency_multiple=multiple, angle_degrees=angle,
                                 points_per_wavelength=speed/f/h,
                                 ricker_amplitude_relative_to_spectral_peak=multiple**2 * math.exp(1-multiple**2),
                                 numerical_phase_velocity_over_physical=ratio,
                                 relative_phase_velocity_error=None if ratio is None else ratio-1))
    return {"definition": "sin(omega_num*dt/2)=c*dt/h*sqrt(S(kx*h)^2+S(kz*h)^2), S(q)=sum_j a_j sin((j+1/2)q)",
            "fd_coefficients_upstream_float32": coefficients.tolist(), "dt_s": dt, "spacing_m": h,
            "cfl_max_vp_dt_over_dx": speeds["maximum_vp"]*dt/h,
            "scope": "Homogeneous elastic eigenmodes on axis/diagonal/intermediate angles; excludes material staggering, sharp interfaces, PML and accumulated propagation error. Pressure data are P-dominated; Vs kernels still arise through conversions/coupling.",
            "rows": rows}


def prepare_model(cfg, raw_dir):
    from hybridadfwi.models import download_sources, _read_resample, _smooth_solid
    path = Path(cfg["model"])
    if path.exists():
        raise RuntimeError("Prepared model exists; choose a new model filename")
    sources = download_sources(raw_dir)
    grid = cfg["grid"]
    x = grid["ox"] + np.arange(grid["nx"], dtype=np.float64) * grid["dx"]
    z = grid["oz"] + np.arange(grid["nz"], dtype=np.float64) * grid["dz"]
    fields, shapes = {}, {}
    for name, prefix in (("vp", "vp"), ("vs", "vs"), ("rho", "density")):
        fields[name], shapes[name] = _read_resample(Path(raw_dir)/(prefix+"_marmousi-ii.segy.gz"), z, x)
    fields["rho"] = np.full_like(fields["rho"], 2450.)
    water = fields["vs"] == 0
    sigma = (cfg["initial_smoothing_sigma_m"]/grid["dz"], cfg["initial_smoothing_sigma_m"]/grid["dx"])
    arrays = {}
    for name, value in fields.items():
        arrays[name+"_true"] = value
        arrays[name+"_initial"] = _smooth_solid(value, water, sigma)
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise FloatingPointError("Nonfinite prepared model")
    analysis = dispersion_curve(cfg, fields)
    metadata = {"name": "Marmousi-II horizontal ROI; not the full original model", "experiment": cfg["name"],
                "grid": grid, "x_extent_m": [float(x[0]), float(x[-1])],
                "z_extent_m": [float(z[0]), float(z[-1])], "sources": sources, "source_shapes_xz": shapes,
                "resampling": "bilinear from source SEG-Y; SI x1000, arrays z,x", "dtype": "float64",
                "initial_smoothing_sigma_m": cfg["initial_smoothing_sigma_m"],
                "initial_model": "normalized Gaussian within solid; true water restored; a closer starting model than old sigma150m benchmark",
                "density": "constant2450kg/m3, same control convention as old benchmark",
                "water_cells": int(water.sum()), "dispersion": analysis,
                "script_sha256": digest(__file__)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, **grid, x=x, z=z, water_mask=water,
                        metadata_json=np.array(json.dumps(metadata)))
    metadata["prepared_sha256"] = digest(path)
    save(path.with_suffix(".json"), metadata)
    print(json.dumps({"model": str(path), "shape_zx": [grid["nz"], grid["nx"]],
                      "cfl": analysis["cfl_max_vp_dt_over_dx"],
                      "minimum_positive_vs": min(fields["vs"][fields["vs"]>0])}), flush=True)


def observations(cfg):
    torch = initialize(cfg)
    _, prop, _, _ = build_model(cfg, truth=True, trainable=False)
    arrays = {"pml_profile_json": np.array(json.dumps({
        "reference_vmax_m_s": cfg["fixed_pml_reference_vmax_m_s"],
        "bcx_sha256": hashlib.sha256(prop.bcx.cpu().numpy().tobytes()).hexdigest(),
        "bcz_sha256": hashlib.sha256(prop.bcz.cpu().numpy().tobytes()).hexdigest()}))}
    with torch.no_grad():
        for shot in cfg["selected_shots"]:
            out = native(prop, cfg, shot)
            p = pressure(out)
            window = data_window(prop, cfg, shot, torch)
            scale = float((p*window).square().mean().sqrt())
            if scale <= 1e-30 or not bool(torch.isfinite(p).all()):
                raise FloatingPointError("Empty/nonfinite weighted observation")
            arrays["pressure_"+str(shot)] = p.cpu().numpy()
            arrays["window_"+str(shot)] = window.cpu().numpy()
            arrays["scale_"+str(shot)] = np.array(scale)
            arrays["source_"+str(shot)] = np.array([int(prop.src_x[shot]), int(prop.src_z[shot])])
            arrays["receiver_x"] = prop.rcv_x.cpu().numpy()
            arrays["receiver_z"] = prop.rcv_z.cpu().numpy()
            del out, p, window
            print("OBSERVATION", shot, "scale", scale, flush=True)
    np.savez_compressed(cfg["observations"], **arrays)


def worker(cfg, method, shot, result_path):
    from hybridadfwi.experiment import UPSTREAM_COMMIT
    from hybridadfwi.adjoint import hybrid_forward
    import psutil
    torch = initialize(cfg)
    model, prop, coordinates, parameter_mask = build_model(cfg)
    with np.load(cfg["observations"]) as f:
        observation = torch.as_tensor(f["pressure_"+str(shot)], dtype=torch.float64, device=cfg["device"])
        window = torch.as_tensor(f["window_"+str(shot)], dtype=torch.float64, device=cfg["device"])
        scale = float(f["scale_"+str(shot)])
    model.zero_grad(set_to_none=True)
    gc.collect()
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sync(torch, cfg)
    rss_before = psutil.Process().memory_info().rss
    begin = time.perf_counter()
    print("FORWARD_START", method, shot, flush=True)
    if method == "hybrid":
        out, tape = hybrid_forward(prop, fd_order=8, checkpoint_segments=cfg["segments"], shot_index=[shot])
    else:
        out = native(prop, cfg, shot)
    sync(torch, cfg)
    forward_end = time.perf_counter()
    print("FORWARD_DONE", method, shot, forward_end-begin, flush=True)
    if method == "hybrid":
        txx, tzz = (out[name].detach().requires_grad_(True) for name in ("txx", "tzz"))
        p = -.5*(txx+tzz)
    else:
        p = pressure(out)
    loss = loss_function(p, observation, window, scale)
    if method == "hybrid":
        source_p, source_x, source_z = torch.autograd.grad(loss, (p, txx, tzz))
    sync(torch, cfg)
    objective_end = time.perf_counter()
    if method == "hybrid":
        gradients = tape.backward({"txx": source_x, "tzz": source_z})
        coordinate_gradients = {name: parameter_mask*value for name, value in gradients.items()} if coordinates else gradients
    else:
        loss.backward()
        gradients = {name: getattr(model, name).grad for name in cfg["parameters"]}
        coordinate_gradients = {name: value.grad for name, value in coordinates.items()} if coordinates else gradients
    sync(torch, cfg)
    end = time.perf_counter()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss")
    arrays = {"predicted_pressure": p.detach().cpu().numpy()}
    arrays.update(x_m=model.ox+np.arange(model.nx)*model.dx,
                  z_m=model.oz+np.arange(model.nz)*model.dz,
                  source_x_m=np.array(model.ox+int(prop.src_x[shot])*model.dx),
                  source_z_m=np.array(model.oz+int(prop.src_z[shot])*model.dz),
                  receiver_x_m=model.ox+prop.rcv_x.cpu().numpy()*model.dx,
                  receiver_z_m=model.oz+prop.rcv_z.cpu().numpy()*model.dz)
    if parameter_mask is not None:
        arrays["parameter_mask"] = parameter_mask.cpu().numpy()
        arrays["active_mask"] = parameter_mask.cpu().numpy()
    norms = {}
    for name, gradient in gradients.items():
        if gradient is None or gradient.dtype != torch.float64 or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Missing/nonfinite gradient " + name)
        norms[name] = float(torch.linalg.vector_norm(gradient))
        if norms[name] == 0:
            raise FloatingPointError("Exactly zero gradient " + name + "; the pilot did not reach a model-dependent signal")
        arrays["gradient_"+name] = gradient.detach().cpu().numpy()
        coordinate_gradient = coordinate_gradients[name]
        if coordinate_gradient is None or not bool(torch.isfinite(coordinate_gradient).all()):
            raise FloatingPointError("Nonfinite constrained-coordinate derivative " + name)
        arrays["gradient_coordinate_"+name] = coordinate_gradient.detach().cpu().numpy()
        arrays["active_gradient_"+name] = coordinate_gradient.detach().cpu().numpy()
        arrays["evaluation_model_"+name] = getattr(model, name).detach().cpu().numpy()
    if method == "hybrid":
        if not bool(torch.isfinite(source_p).all()):
            raise FloatingPointError("Nonfinite data VJP")
        arrays["adjoint_source_pressure"] = source_p.detach().cpu().numpy()
    np.savez_compressed(Path(result_path).with_suffix(".npz"), **arrays)
    row = {"status": "ok", "method": method, "shot_id": shot, "loss": float(loss.detach()), "kernel_l2": norms,
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "timing": {"forward_s": forward_end-begin, "objective_and_source_s": objective_end-forward_end,
                      "backward_s": end-objective_end, "total_s": end-begin},
           "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cfg["device"]=="cuda" else None,
           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if cfg["device"]=="cuda" else None,
           "rss_before_bytes": rss_before, "config": cfg, "python": sys.version, "pytorch": torch.__version__,
           "upstream_commit": UPSTREAM_COMMIT, "model_sha256": digest(cfg["model"]),
           "observations_sha256": digest(cfg["observations"]), "script_sha256": digest(__file__),
           "gradient_processing": "none; raw derivative of explicit fixed-window pressure objective",
           "parameterization": "m(theta)=m_initial+P*(theta-m_initial), diagonal P=0 in known water and fixed lateral/bottom buffer, raised-cosine transition; fixed buffer values come only from initial model. True observations use original true MII without this map. AD differentiates P map; Hybrid applies its exact P^T VJP. Full physical and constrained-coordinate derivatives saved separately.",
           "pml_profile": {"reference_vmax_m_s": cfg["fixed_pml_reference_vmax_m_s"],
                           "bcx_sha256": hashlib.sha256(prop.bcx.detach().cpu().numpy().tobytes()).hexdigest(),
                           "bcz_sha256": hashlib.sha256(prop.bcz.detach().cpu().numpy().tobytes()).hexdigest(),
                           "fixed_for_truth_initial_and_model_derivatives": True},
           "window": {"formula": "offset raised-cosine * time raised-cosine * end raised-cosine, all ramps clamped [0,1]; time starts at t0+sqrt(offset^2+(2*known_water_depth-z_s-z_r)^2)/water_velocity+seafloor_delay for known_seafloor_primary, or t0+abs(offset)/water_velocity+direct_delay for water_direct",
                      "reference": cfg.get("mute_reference", "water_direct"),
                      "scope": "Fixed arrival-data exclusion only; later data may still be sensitive to shallow model parameters",
                      "sha256": hashlib.sha256(window.detach().cpu().numpy().tobytes()).hexdigest(),
                      "positive_fraction": float((window>0).to(torch.float64).mean()),
                      "scale_windowed_observed_rms": scale,
                      "parameters_are_fixed_before_derivative": True},
           "data_objective": "0.5*mean((fixed_window*(pressure-p_observed)/fixed_windowed_observation_RMS)^2)",
           "performance_scope": "Cold functional detail experiment; not a warmed repeated benchmark; timing not comparable to old benchmark"}
    if method == "hybrid":
        row["adapter_details"] = dict(tape.stats)
    save(result_path, row)
    print(json.dumps({"method": method, "shot": shot, "loss": row["loss"], "timing": row["timing"], "kernel_l2": norms}), flush=True)


def compare(cfg, output):
    comparisons, mean_gradients, losses = [], {}, {}
    passed = True
    for method in ("autograd", "hybrid"):
        mean_gradients[method] = {}
        losses[method] = []
        for shot in cfg["selected_shots"]:
            path = output/(method+"-shot"+str(shot))
            row = json.loads(path.with_suffix(".json").read_text())
            losses[method].append(row["loss"])
            with np.load(path.with_suffix(".npz")) as f:
                for prefix in ("", "coordinate_"):
                    for name in cfg["parameters"]:
                        key = prefix+name
                        mean_gradients[method].setdefault(key, np.zeros_like(f["gradient_"+key]))
                        mean_gradients[method][key] += f["gradient_"+key]/len(cfg["selected_shots"])
    for shot in cfg["selected_shots"]:
        with np.load(output/("autograd-shot"+str(shot)+".npz")) as a, np.load(output/("hybrid-shot"+str(shot)+".npz")) as b:
            exact = bool(np.array_equal(a["predicted_pressure"], b["predicted_pressure"]))
            row = {"shot_id": shot, "pressure_bitwise_equal": exact, "kernels": {}}
            passed &= exact
            for name in cfg["parameters"]:
                x, y = a["gradient_"+name].ravel(), b["gradient_"+name].ravel()
                error = float(np.linalg.norm(y-x)/np.linalg.norm(x))
                row["kernels"][name] = {"relative_l2": error, "cosine": float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))}
                passed &= error <= cfg["tolerance"]
            comparisons.append(row)
    arrays, aggregate = {}, {}
    for prefix in ("", "coordinate_"):
        for name in cfg["parameters"]:
            key = prefix+name
            x, y = mean_gradients["autograd"][key], mean_gradients["hybrid"][key]
            error = float(np.linalg.norm(y-x)/np.linalg.norm(x))
            passed &= error <= cfg["tolerance"]
            aggregate[key] = {"relative_l2": error, "cosine": float(np.vdot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))}
            arrays["gradient_autograd_"+key] = x
            arrays["gradient_hybrid_"+key] = y
            if prefix == "coordinate_":
                arrays["autograd_active_gradient_"+name] = x
                arrays["hybrid_active_gradient_"+name] = y
            else:
                arrays["autograd_gradient_"+name] = x
                arrays["hybrid_gradient_"+name] = y
    with np.load(output/("hybrid-shot"+str(cfg["selected_shots"][0])+".npz")) as last:
        for key in last.files:
            if key in ("parameter_mask", "active_mask", "x_m", "z_m", "source_x_m", "source_z_m", "receiver_x_m", "receiver_z_m"):
                arrays[key] = last[key]
    source_x, source_z = [], []
    for shot in cfg["selected_shots"]:
        with np.load(output/("hybrid-shot"+str(shot)+".npz")) as values:
            source_x.append(float(values["source_x_m"]))
            source_z.append(float(values["source_z_m"]))
    arrays["source_x_m"] = np.asarray(source_x)
    arrays["source_z_m"] = np.asarray(source_z)
    arrays["selected_shot_ids"] = np.asarray(cfg["selected_shots"], dtype=np.int64)
    np.savez_compressed(output/"raw_kernels.npz", **arrays)
    summary = {"passed": bool(passed), "config": cfg, "shots": comparisons, "mean_kernel_agreement": aggregate,
               "mean_loss": {name: statistics.mean(values) for name, values in losses.items()},
               "shot_aggregation": "Arithmetic mean of each selected shot's separately fixed-RMS normalized scalar objective and its raw derivative, identically for AD and Hybrid",
               "gradient_processing": "none; fixed data weights belong to objective; constrained-coordinate derivatives are actual P^T chain-rule VJPs, saved separately from full physical derivatives; no model-gradient preconditioning",
               "limitation": "Changed grid, marine pressure acquisition, frequency, loss and sigma60m initial model form a separate illustrative experiment, not a continuation or improvement score for the old performance benchmark."}
    save(output/"summary.json", summary)
    print(json.dumps(summary), flush=True)
    if not passed:
        raise RuntimeError("AD and Hybrid did not satisfy the recorded raw-kernel tolerance")


def run(cfg, output, methods=None):
    from hybridadfwi.resources import run_checked
    output.mkdir(parents=True, exist_ok=False)
    cfg["observations"] = str(output/"observations.npz")
    protocol = output/"protocol.json"
    save(protocol, cfg)
    base = [sys.executable, str(Path(__file__).resolve())]
    jobs = [("observations", base+["observations", "--config", str(protocol)])]
    for shot in cfg["selected_shots"]:
        for method in (methods or ("autograd", "hybrid")):
            label = method+"-shot"+str(shot)
            jobs.append((label, base+["worker", "--config", str(protocol), "--method", method,
                                     "--shot", str(shot), "--result", str(output/(label+".json"))]))
    for label, command in jobs:
        with (output/(label+".log")).open("w") as log:
            audit = run_checked(command, stdout=log, device=cfg["device"], min_gpu_gib=10., min_ram_gib=8.)
        save(output/(label+".resources.json"), audit)
        if audit["status"] != "complete":
            raise RuntimeError("Detail worker failed/blocked/contaminated: " + label)
        print("DONE", label, flush=True)
    complete = all((output/(method+"-shot"+str(shot)+".json")).is_file()
                   for method in ("autograd", "hybrid") for shot in cfg["selected_shots"])
    if complete:
        compare(cfg, output)
    else:
        save(output/"stage_summary.json", {"status": "reference_comparison_pending",
             "methods_run": methods, "numeric_agreement_passed": None,
             "note": "Hybrid-only diagnostic stage; no AD agreement claim until matching reference worker completes"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-model", "run", "observations", "worker", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "prepare-model":
            p.add_argument("--raw-dir", type=Path, required=True)
        if name in ("run", "compare"):
            p.add_argument("--output", type=Path, required=True)
        if name == "run":
            p.add_argument("--device", choices=("cpu", "cuda"))
            p.add_argument("--shot-ids", nargs="+", type=int)
            p.add_argument("--methods", nargs="+", choices=("autograd", "hybrid"),
                           help="Run one diagnostic method first; final comparison waits for both methods")
        if name == "worker":
            p.add_argument("--method", choices=("autograd", "hybrid"), required=True)
            p.add_argument("--shot", type=int, required=True)
            p.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "prepare-model":
        initialize({**cfg, "device": "cpu"})
        prepare_model(cfg, args.raw_dir)
    elif args.command == "run":
        if args.device:
            cfg["device"] = args.device
        if args.shot_ids is not None:
            cfg["selected_shots"] = args.shot_ids
        if any(shot < 0 or shot >= cfg["shots"] for shot in cfg["selected_shots"]):
            raise ValueError("Selected shot outside configured source geometry")
        run(cfg, args.output.resolve(), args.methods)
    elif args.command == "observations":
        observations(cfg)
    elif args.command == "worker":
        worker(cfg, args.method, args.shot, args.result)
    else:
        compare(cfg, args.output)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
