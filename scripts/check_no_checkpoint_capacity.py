#!/usr/bin/env python3
"""Attempt the unchanged full case; record CUDA OOM separately from runtime."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from benchmark import save_json, sha256
from hybridadfwi.resources import run_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("Select a new output directory to preserve existing results")
    reference = json.loads(args.reference_protocol.read_text())
    output.mkdir(parents=True)
    # Reuse the exact prior common observations, without another simulator.
    shutil.copy2(reference["observations"], output/"observations.npz")
    config = {**reference, "checkpointing":False, "segments":1, "repeats":1,
              "observations":str(output/"observations.npz"),
              "reference_protocol":str(args.reference_protocol.resolve()),
              "reference_protocol_sha256":sha256(args.reference_protocol),
              "observation_source_sha256":sha256(reference["observations"]),
              "timing_scope":"original forward + objective/data VJP + reverse/static model VJP; no checkpoint and no replay"}
    save_json(output/"protocol.json", config)
    rows, audits = {}, {}
    for method in ("autograd", "hybrid"):
        result = output/(method+"-0.json")
        command = [sys.executable, str(ROOT/"scripts/benchmark.py"), "worker", "--config",
                   str(output/"protocol.json"), "--method", method, "--result", str(result)]
        with result.with_suffix(".log").open("w") as log:
            audit = run_checked(command, stdout=log, device=config["device"], min_gpu_gib=18, min_ram_gib=12)
        save_json(output/(method+"-0.resources.json"), audit)
        if audit["contaminated"] or audit["status"] in ("blocked", "monitor_error"):
            raise SystemExit(f"Resource audit {audit['status']}; inspect {method} audit before continuing")
        if not result.is_file():
            raise SystemExit(f"{method} failed without a result; inspect its log")
        row = json.loads(result.read_text())
        expected = row["status"] == "cuda_oom" and audit["returncode"] == 86
        if not expected and (row["status"] != "ok" or audit["returncode"] != 0):
            raise SystemExit(f"Unexpected {method} failure")
        rows[method], audits[method] = row, {"status":audit["status"], "returncode":audit["returncode"], "contaminated":audit["contaminated"]}
        print(json.dumps({"method":method,"status":row["status"],"output":str(result)}), flush=True)
    paired = None
    if all(r["status"] == "ok" for r in rows.values()):
        import numpy as np
        with np.load(output/"autograd-0.npz") as a, np.load(output/"hybrid-0.npz") as b:
            paired = {"forward_bitwise_equal":bool(np.array_equal(a["predicted_vz"], b["predicted_vz"])),
                      "kernel_relative_l2":{p:float(np.linalg.norm(a["gradient_"+p]-b["gradient_"+p])/np.linalg.norm(a["gradient_"+p])) for p in config["parameters"]}}
    summary = {"status":"capacity_recorded", "config":config, "methods":{}, "paired_comparison":paired,
               "resource_audit":audits, "note":"CUDA OOM is a capacity outcome, not a completed runtime or a measured gradient error."}
    lines = ["# Unchanged full-size no-checkpoint capacity test", "",
             "Both arms bypass temporal checkpointing. Common inputs and reference observations are unchanged.", "",
             "| Method | Status | Completed gradient seconds | Peak CUDA allocated (GiB) |",
             "|---|---|---:|---:|"]
    for method, row in rows.items():
        ok = row["status"] == "ok"
        peak = row["cuda_memory"]["peak_allocated_bytes"] if ok else row["peak_allocated_bytes"]
        seconds = row["timing"]["total_s"] if ok else None
        summary["methods"][method] = {"status":row["status"],"completed_gradient_seconds":seconds,"peak_allocated_bytes":peak}
        lines.append(f"| {method} | {row['status']} | {seconds if ok else 'N/A (OOM)'} | {peak/2**30:.6f} |")
    lines += ["", "An OOM attempt has no completed gradient time or kernel; its failure elapsed time is only in the individual diagnostic JSON.",
              "Use a matched smaller shot count in both arms for a complete no-checkpoint comparison if needed."]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    save_json(output/"capacity_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
