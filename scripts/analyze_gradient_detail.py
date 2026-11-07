#!/usr/bin/env python3
"""Collect model-free detail evidence and audit depth sensitivity, on CPU only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def energy_by_depth(a, z, water):
    energy = a*a
    total = float(energy.sum())
    end = float(z[-1])+1e-6
    boundaries = sorted({value for value in (0., 450., 750., 1000., 1500., 2000., 2500., 3000.)
                         if value < end} | {end})
    rows = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi <= lo:
            continue
        selected = (z >= lo) & (z < hi)
        value = float(energy[selected].sum())
        rows.append({"depth_interval_m": [lo, min(hi, float(z[-1]))],
                     "squared_norm": value, "fraction_of_total": value/total if total else None})
    return {"l2": float(np.sqrt(total)), "max_absolute": float(np.abs(a).max()),
            "water_fraction": float(energy[water].sum()/total) if total else None,
            "deeper_than_1000m_fraction": float(energy[z>=1000].sum()/total) if total else None,
            "deeper_than_2000m_fraction": float(energy[z>=2000].sum()/total) if total else None,
            "depth_bins": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, help="Override prepared-model path when analyzing on another host")
    parser.add_argument("--canonical-arrays", action="store_true", help="Omit redundant alias copies of the same physical/coordinate gradients")
    parser.add_argument("--include-failed-agreement", action="store_true", help="Explicit diagnostic collection only; preserve summary.passed=false")
    args = parser.parse_args()
    cfg = json.loads((args.run/"protocol.json").read_text())
    summary = json.loads((args.run/"summary.json").read_text())
    if not summary["passed"] and not args.include_failed_agreement:
        raise RuntimeError("Only validated complete AD/Hybrid cases can be collected")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.model or cfg["model"]) as f:
        water = f["vs_true"] == 0
        initial = {name: f[name+"_initial"].copy() for name in cfg["parameters"]}
    with np.load(args.run/"raw_kernels.npz") as f:
        canonical = {"gradient_"+method+"_"+prefix+name for method in ("autograd", "hybrid")
                     for prefix in ("", "coordinate_") for name in cfg["parameters"]}
        public_arrays = {key: f[key].copy() for key in f.files
                         if "gradient" in key or key in ("x_m", "z_m", "source_x_m", "source_z_m",
                             "receiver_x_m", "receiver_z_m", "active_mask", "parameter_mask", "selected_shot_ids")}
        if args.canonical_arrays:
            public_arrays = {key: value for key, value in public_arrays.items()
                             if key in canonical or ("gradient" not in key and key != "parameter_mask")}
    report = {"source_run": str(args.run), "numeric_agreement_passed": bool(summary["passed"]),
              "collection_role": "validated_evidence" if summary["passed"] else "diagnostic_failed_agreement; not a successful validation",
              "gradient_processing": "none; full physical and actual constrained-coordinate derivatives reported separately",
              "config": cfg, "depth_energy": {}, "data_checks": [], "model_arrays_in_public_bundle": False}
    z = public_arrays["z_m"]
    for name in cfg["parameters"]:
        report["depth_energy"][name] = {
            "full_physical": energy_by_depth(public_arrays["gradient_autograd_"+name], z, water),
            "active_coordinate": energy_by_depth(public_arrays["gradient_autograd_coordinate_"+name], z, water)}
    pressure_arrays = {}
    with np.load(cfg["observations"]) as observations:
        for shot in cfg["selected_shots"]:
            with np.load(args.run/("hybrid-shot"+str(shot)+".npz")) as hybrid:
                pred = hybrid["predicted_pressure"]
                obs = observations["pressure_"+str(shot)]
                window = observations["window_"+str(shot)]
                scale = float(observations["scale_"+str(shot)])
                source = hybrid["adjoint_source_pressure"]
                analytic = (pred-obs)*window*window/(scale*scale*pred.size)
                delta = source-analytic
                check = {"shot_id": shot,
                    "pressure_cotangent_relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(analytic)),
                    "pressure_cotangent_max_abs": float(np.abs(delta).max()),
                    "objective_recomputed_numpy": float(.5*np.mean(((pred-obs)*window/scale)**2)),
                    "evaluation_model_equal_prepared_initial": {
                        name: bool(np.array_equal(hybrid["evaluation_model_"+name], initial[name])) for name in cfg["parameters"]}}
                report["data_checks"].append(check)
                for label, array in (("predicted_pressure", pred), ("observed_pressure", obs),
                                     ("window", window), ("adjoint_source_pressure", source), ("scale", np.array(scale))):
                    pressure_arrays[label+"_shot"+str(shot)] = array
        if "pml_profile_json" in observations:
            report["observation_pml_profile"] = json.loads(str(observations["pml_profile_json"]))
    np.savez_compressed(args.output/"raw_kernels.npz", **public_arrays)
    np.savez_compressed(args.output/"pressure_data.npz", **pressure_arrays)
    metadata_whitelist = {"protocol.json", "summary.json", "observations.log", "observations.resources.json"}
    for method in ("autograd", "hybrid"):
        for shot in cfg["selected_shots"]:
            base = method+"-shot"+str(shot)
            metadata_whitelist.update({base+".json", base+".log", base+".resources.json"})
    for path in args.run.iterdir():
        if path.is_file() and path.suffix in (".json", ".log"):
            if args.canonical_arrays and path.name not in metadata_whitelist:
                continue
            shutil.copy2(path, args.output/path.name)
    (args.output/"depth_and_data_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.output/"array_conventions.json").write_text(json.dumps({
        "physical_derivative": "gradient_{autograd|hybrid}_{vp|vs|rho}",
        "constrained_coordinate_derivative": "gradient_{autograd|hybrid}_coordinate_{vp|vs|rho}",
        "static_parameter_map": "m=m_initial+active_mask*(theta-m_initial); coordinate gradient=P^T physical gradient",
        "canonical_array_filter": bool(args.canonical_arrays),
        "values_changed_by_collection": False,
        "excluded": ["input SEG-Y", "prepared model arrays", "evaluation_model_*", "kernel_preview"],
        "array_keys": list(public_arrays)}, indent=2)+"\n")
    manifest = {path.name: {"bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in args.output.iterdir() if path.is_file() and path.name != "bundle_manifest.json"}
    (args.output/"bundle_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "depth_energy": report["depth_energy"]}), flush=True)


if __name__ == "__main__":
    main()
