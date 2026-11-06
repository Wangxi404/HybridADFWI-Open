#!/usr/bin/env python3
"""Make a small auditable bundle; keep full traces in the original run directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capacity", action="store_true", help="Collect a capacity attempt including explicitly recorded CUDA OOM outcomes")
    args = parser.parse_args()
    summary = json.loads((args.run/("capacity_summary.json" if args.capacity else "summary.json")).read_text())
    if args.capacity and summary.get("status") != "capacity_recorded":
        raise SystemExit("Capacity attempt is incomplete")
    if not args.capacity and not summary["passed"]:
        raise SystemExit("Only validated runs can be collected as successful evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"run_directory": str(args.run.resolve()), "files": {}}
    for path in sorted(args.run.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            if path.suffix in (".json", ".md", ".log"):
                shutil.copy2(path, args.output/path.name)
            elif path.suffix == ".npz" and path.name != "observations.npz":
                with np.load(path) as data:
                    kernels = {k: data[k] for k in data.files if k.startswith("gradient_")}
                np.savez_compressed(args.output/(path.stem+".kernels.npz"), **kernels)
    (args.output/"raw_files_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output/"BUNDLE.md").write_text(
        "# Evidence bundle\n\n" + ("Raw kernels for successful workers are included; CUDA OOM workers have no completed kernel. " if args.capacity else "Raw Vp/Vs/rho kernels for every measurement are included. ") +
        "Full traces and observations remain in the original run directory; their SHA256 hashes and sizes are recorded. "
        "Protocol, runtime versions, source/data hashes, raw timing/memory measurements and process-audit samples are preserved.\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
