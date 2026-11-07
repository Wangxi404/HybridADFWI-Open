#!/usr/bin/env python3
"""Single-factor diagnostic: freeze initial PML to the old true-model profile.

Read original observations/model/RMS without modifying them. All old acquisition,
grid, frequency, loss, active parameters and checkpoint settings are retained.
This runs one Hybrid derivative only: it is an objective-contamination diagnostic,
not an additional AD-versus-Hybrid correctness/performance benchmark.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import gradient_detail as detail


def energy_report(gradient, true_vs, dx, dz):
    nz, nx = gradient.shape
    x, z = np.arange(nx)*dx, np.arange(nz)*dz
    solid = true_vs > 0
    edge = ((x[None, :] <= 200) | (x[None, :] >= x[-1]-200)) & solid
    top = (z[:, None] >= 900) & (z[:, None] <= 3200)
    middle = (x[None, :] >= 500) & (x[None, :] <= x[-1]-500) & top & solid
    energy = gradient**2
    total, solid_total = energy.sum(), energy[solid].sum()
    count = max(1, int(np.ceil(gradient.size*.001)))
    return {"total_squared_norm": float(total), "water_fraction_total": float(energy[~solid].sum()/total),
            "solid_fraction_total": float(solid_total/total),
            "lateral_200m_fraction_of_solid": float(energy[edge].sum()/solid_total),
            "central_deep_fraction_of_solid": float(energy[middle].sum()/solid_total),
            "top_0p1_percent_cells_fraction_total": float(np.partition(energy.ravel(), -count)[-count:].sum()/total),
            "absolute_max_index_zx": [int(v) for v in np.unravel_index(np.abs(gradient).argmax(), gradient.shape)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--reference-kernels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = detail.load_config(args.reference_protocol)
    with np.load(cfg["model"]) as stored:
        exact_true_vmax = float(stored["vp_true"].max())
    cfg.update(fixed_pml_reference_vmax_m_s=exact_true_vmax, constrained_coordinates=False)
    detail.save(args.output/"protocol.json", cfg)
    torch = detail.initialize(cfg)
    from hybridadfwi.experiment import objective
    from hybridadfwi.adjoint import hybrid_forward
    model, prop, _, _ = detail.build_model(cfg)
    with np.load(cfg["observations"]) as data:
        observed = torch.as_tensor(data["vz"], dtype=torch.float64, device=cfg["device"])
        scale = float(data["scale"])
    if cfg["device"] == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    detail.sync(torch, cfg)
    begin = time.perf_counter()
    output, tape = hybrid_forward(prop, fd_order=8, checkpoint_segments=cfg["segments"])
    data = output["vz"].detach().requires_grad_(True)
    loss = objective(data, observed, scale, cfg["dt"], cfg["objective"])
    source, = torch.autograd.grad(loss, data)
    gradients = tape.backward({"vz": source})
    detail.sync(torch, cfg)
    elapsed = time.perf_counter()-begin
    with np.load(cfg["model"]) as model_file:
        true_vs = model_file["vs_true"]
        dx, dz = float(model_file["dx"]), float(model_file["dz"])
    report = {"status": "ok", "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "single-factor initial PML profile control; same old true-model observations/RMS/grid/frequency/acquisition/loss/parameters; no AD rerun",
              "reference_protocol_sha256": detail.digest(args.reference_protocol),
              "model_sha256": detail.digest(cfg["model"]), "observations_sha256": detail.digest(cfg["observations"]),
              "reference_kernels_sha256": detail.digest(args.reference_kernels),
              "script_sha256": detail.digest(__file__), "loss": float(loss.detach()),
              "cold_functional_elapsed_s": elapsed, "adapter_details": tape.stats,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if cfg["device"] == "cuda" else None,
              "kernels": {}}
    arrays = {}
    with np.load(args.reference_kernels) as old:
        for name, value in gradients.items():
            new = value.cpu().numpy()
            if not np.isfinite(new).all():
                raise FloatingPointError("Nonfinite " + name)
            reference = old["gradient_"+name]
            arrays["gradient_"+name] = new
            report["kernels"][name] = {
                "old_variable_profile": energy_report(reference, true_vs, dx, dz),
                "matched_fixed_profile": energy_report(new, true_vs, dx, dz),
                "relative_l2_change_from_old": float(np.linalg.norm(new-reference)/np.linalg.norm(reference)),
            }
    np.savez_compressed(args.output/"raw_kernels.npz", **arrays)
    detail.save(args.output/"summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
