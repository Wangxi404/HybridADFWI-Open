#!/usr/bin/env python3
"""No-gradient native boundary/stability control, separate from kernel comparisons."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import gradient_detail as detail
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--free-surface", choices=("true", "false"), required=True)
    parser.add_argument("--nabc", type=int)
    args = parser.parse_args()
    cfg = detail.load_config(args.config)
    reference = dict(cfg)
    cfg["free_surface"] = args.free_surface == "true"
    if args.nabc is not None:
        cfg["nabc"] = args.nabc
    args.output.mkdir(parents=True, exist_ok=False)
    detail.save(args.output/"protocol.json", cfg)
    torch = detail.initialize(cfg)
    report = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_config": reference, "control_config": cfg,
              "scope": "native no-gradient truth/initial pressure traces; no model-gradient or AD agreement claim",
              "script_sha256": detail.digest(__file__),
              "model_sha256": detail.digest(cfg["model"]), "traces": {}}
    arrays = {}
    for truth in (True, False):
        name = "truth" if truth else "initial"
        for shot in cfg["selected_shots"]:
            print("START", name, shot, flush=True)
            with torch.no_grad():
                model, prop, _, _ = detail.build_model(cfg, truth=truth, trainable=False)
                detail.sync(torch, cfg)
                begin = time.perf_counter()
                out = detail.native(prop, cfg, shot)
                p = detail.pressure(out)
                detail.sync(torch, cfg)
                elapsed = time.perf_counter()-begin
                if not bool(torch.isfinite(p).all()):
                    raise FloatingPointError("Nonfinite native pressure " + name)
                p = p.cpu().numpy()
                t = np.arange(cfg["nt"])*cfg["dt"]
                rows = []
                for lo, hi in ((0, 1), (1, 2), (2, 3), (3, cfg["nt"]*cfg["dt"]+1e-6)):
                    selected = (t>=lo) & (t<hi)
                    if not selected.any():
                        continue
                    data = p[:, selected]
                    rows.append({"time_interval_s": [lo, hi], "max_abs": float(np.abs(data).max()),
                                 "rms": float(np.sqrt(np.mean(data*data)))})
                key = name+"_shot"+str(shot)
                arrays["pressure_"+key] = p
                energy = out["forward_wavefield_txx"].cpu().numpy()+out["forward_wavefield_tzz"].cpu().numpy()
                index = np.unravel_index(energy.argmax(), energy.shape)
                report["traces"][key] = {"shape": list(p.shape), "all_finite": True,
                    "cold_forward_seconds": elapsed, "max_abs": float(np.abs(p).max()),
                    "rms": float(np.sqrt(np.mean(p*p))), "time_bands": rows,
                    "cumulative_normal_stress_energy_max_index_zx": [int(v) for v in index]}
                arrays["time_s"] = t
                arrays["receiver_x_m"] = prop.ox+prop.rcv_x.cpu().numpy()*prop.dx
                arrays["receiver_z_m"] = prop.oz+prop.rcv_z.cpu().numpy()*prop.dz
                del model, prop, out, p
            if cfg["device"] == "cuda":
                torch.cuda.empty_cache()
            detail.save(args.output/"summary.json", report)
            print("DONE", key, elapsed, flush=True)
    np.savez_compressed(args.output/"pressure_traces.npz", **arrays)
    report["status"] = "ok"
    detail.save(args.output/"summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
