#!/usr/bin/env python3
"""Package explicitly subsampled, model-free evidence without rerunning a solver.

The full pressure archive stays at the run host; this tool records its exact
hash and byte count. A sampled cotangent always retains the FULL objective
element count in its normalization. --tar-stdout emits only a small whitelist.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime, timezone

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bytes_of(array):
    return np.ascontiguousarray(array).tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--time-stride", type=int, default=8)
    parser.add_argument("--receiver-stride", type=int, default=16)
    parser.add_argument("--tar-stdout", action="store_true")
    args = parser.parse_args()
    if args.time_stride < 1 or args.receiver_stride < 1:
        raise ValueError("Strides must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((args.run/"protocol.json").read_text())
    shot = int(cfg["selected_shots"][0])
    if len(cfg["selected_shots"]) != 1:
        raise ValueError("This small proof packages one selected shot")
    hp, ap = args.run/f"hybrid-shot{shot}.npz", args.run/f"autograd-shot{shot}.npz"
    op = Path(cfg["observations"])
    with np.load(hp) as h, np.load(ap) as a, np.load(op) as o:
        full = {"predicted_pressure": h["predicted_pressure"],
                "observed_pressure": o[f"pressure_{shot}"],
                "window": o[f"window_{shot}"],
                "adjoint_source_pressure": h["adjoint_source_pressure"]}
        scale = float(o[f"scale_{shot}"])
        ad_pressure = a["predicted_pressure"]
    n = int(full["predicted_pressure"].size)
    analytic = ((full["predicted_pressure"]-full["observed_pressure"])
                *full["window"]**2/(scale**2*n))
    actual = full["adjoint_source_pressure"]
    ti = np.arange(full["window"].shape[1])[::args.time_stride]
    ri = np.arange(full["window"].shape[2])[::args.receiver_stride]
    sampled = {key: value[:, ti][:, :, ri] for key, value in full.items()}
    sampled.update(time_indices=ti, receiver_indices=ri, selected_shot_ids=np.array([shot]),
                   dt_s=np.array(cfg["dt"]), fixed_observed_rms=np.array(scale),
                   full_objective_element_count=np.array(n, dtype=np.int64))
    full_path = args.output/"pressure_data_full.npz"
    np.savez_compressed(full_path, **full, fixed_observed_rms=np.array(scale),
                        full_objective_element_count=np.array(n, dtype=np.int64))
    sample_path = args.output/"pressure_data_sampled.npz"
    np.savez_compressed(sample_path, **sampled)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Model-free synthetic pressure and cotangent evidence; NOT full pressure data",
        "sampling": {"operation": "Exact node selection; no interpolation, averaging, or rescaling",
                     "time_stride": args.time_stride, "receiver_stride": args.receiver_stride,
                     "original_shape": list(full["window"].shape),
                     "sample_shape": list(sampled["window"].shape),
                     "full_objective_element_count": n,
                     "warning": "Recomputing mean on the sample is a DIFFERENT objective; use full count for cotangent checks"},
        "dt_s": cfg["dt"], "shot_id": shot, "fixed_observed_rms": scale,
        "full_objective_loss": float(.5*np.mean((full["window"]*(full["predicted_pressure"]-full["observed_pressure"])/scale)**2)),
        "cotangent_formula": "(p-p_observed)*fixed_window**2/(fixed_observed_RMS**2*full_objective_element_count)",
        "full_cotangent_formula_relative_l2": float(np.linalg.norm(analytic-actual)/np.linalg.norm(actual)),
        "full_pressure_ad_hybrid_uint64_equal": bool(np.array_equal(ad_pressure.view(np.uint64), full["predicted_pressure"].view(np.uint64))),
        "full_ad_pressure_bytes_sha256": sha(bytes_of(ad_pressure)),
        "full_array_byte_hashes": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                         "sha256_c_contiguous_bytes": sha(bytes_of(value))}
                                   for key, value in full.items()},
        "source_files": {str(path): {"sha256": sha(path.read_bytes()), "bytes": path.stat().st_size}
                         for path in (hp, ap, op)},
        "full_pressure_archive": {"path_at_run_host": str(full_path.resolve()),
                                  "sha256": sha(full_path.read_bytes()), "bytes": full_path.stat().st_size,
                                  "downloaded_in_this_small_proof": False,
                                  "contents": "Only four full pressure/window/cotangent arrays plus scale and full count; no model arrays"},
        "sample_file": {"name": sample_path.name, "sha256": sha(sample_path.read_bytes()), "bytes": sample_path.stat().st_size},
        "packaging_code_sha256": sha(Path(__file__).read_bytes()) if __file__ != "<stdin>" else None,
    }
    if args.control:
        control = args.control/"pressure_traces.npz"
        with np.load(control) as c:
            previous = c[f"pressure_initial_shot{shot}"]
            truth = c[f"pressure_truth_shot{shot}"]
        delta = full["predicted_pressure"]-previous
        report["standalone_no_gradient_trace_control"] = {
            "source": str(control), "sha256": sha(control.read_bytes()),
            "prediction_uint64_equal": bool(np.array_equal(previous.view(np.uint64), full["predicted_pressure"].view(np.uint64))),
            "prediction_difference_absolute_linf": float(np.max(np.abs(delta))),
            "prediction_difference_relative_l2": float(np.linalg.norm(delta)/np.linalg.norm(previous)),
            "observed_truth_uint64_equal": bool(np.array_equal(truth.view(np.uint64), full["observed_pressure"].view(np.uint64))),
            "interpretation": "Quantifies different execution-path arithmetic only; the cause of the small delta is not established by this comparison"}
    report_path = args.output/"pressure_proof.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    sources = [sample_path, report_path]
    names = ["physical_gradient_discrepancy_audit.json", "observations.log", "observations.resources.json"]
    for method in ("autograd", "hybrid"):
        names.extend(f"{method}-shot{shot}"+suffix for suffix in (".json", ".log", ".resources.json"))
    for name in names:
        path = args.run/name
        if path.is_file():
            sources.append(path)
    if args.tar_stdout:
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for source in sources:
                archive.add(source, arcname=source.name)
    else:
        print(json.dumps({"files": [str(p) for p in sources], "full_pressure_bytes": full_path.stat().st_size}))


if __name__ == "__main__":
    main()
