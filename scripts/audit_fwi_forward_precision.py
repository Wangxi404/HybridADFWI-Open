#!/usr/bin/env python3
"""Read saved forward buffers and compare same-point Hybrid/loss caches.

No solver, CUDA context, optimizer, or numerical-source mutation is used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import zipfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.namelist():
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            result[item.removesuffix(".npy")] = {"shape": list(shape), "dtype": str(dtype)}
    return result


def difference(left, right):
    delta = left - right
    return {"shape": list(left.shape), "left_dtype": str(left.dtype), "right_dtype": str(right.dtype),
            "actual_uint64_bits_equal": bool(np.array_equal(left.view(np.uint64), right.view(np.uint64))),
            "different_float64_bitpatterns": int(np.count_nonzero(left.view(np.uint64) != right.view(np.uint64))),
            "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(right), 1e-300)),
            "absolute_linf": float(np.max(np.abs(delta)))}


def audit(run):
    run = run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    groups = defaultdict(list)
    records = []
    for path in sorted(run.glob("evaluations/*/shots_*.json")):
        if path.name.endswith(".producer.json"):
            continue
        row = json.loads(path.read_text())
        if row.get("signature") != protocol["signature"]:
            raise ValueError("Unexpected operator signature in cache")
        data = path.with_suffix(".npz")
        if digest(data) != row["data_sha256"]:
            raise ValueError("Saved forward cache checksum mismatch")
        record = {"metadata": str(path.relative_to(run)), "method": row["method"],
                  "shot_ids": row["shot_ids"], "q_sha256": row["q_array_sha256"],
                  "physical_fields_sha256": row["physical_fields_sha256"], "archive": headers(data),
                  "data_sha256": row["data_sha256"]}
        records.append(record)
        groups[(row["q_array_sha256"], row["objective_id"], json.dumps(row["physical_fields_sha256"], sort_keys=True))].append((row, data))
    comparisons = []
    for point, entries in groups.items():
        loss = [(row, path) for row, path in entries if row["method"] == "loss"]
        hybrid = [(row, path) for row, path in entries if row["method"] == "hybrid"]
        for loss_row, loss_path in loss:
            with np.load(loss_path) as data:
                loss_pressure = data["predicted_pressure"]
            for hybrid_row, hybrid_path in hybrid:
                common = set(loss_row["shot_ids"]) & set(hybrid_row["shot_ids"])
                if not common:
                    continue
                with np.load(hybrid_path) as data:
                    hybrid_pressure = data["predicted_pressure"]
                for shot in sorted(common):
                    left = hybrid_pressure[hybrid_row["shot_ids"].index(shot)]
                    right = loss_pressure[loss_row["shot_ids"].index(shot)]
                    hybrid_loss = hybrid_row["per_shot_loss"][hybrid_row["shot_ids"].index(shot)]
                    loss_only = loss_row["per_shot_loss"][loss_row["shot_ids"].index(shot)]
                    comparisons.append({"shot_id": shot, "q_sha256": point[0],
                        "hybrid_batch_ids": hybrid_row["shot_ids"], "loss_batch_ids": loss_row["shot_ids"],
                        "hybrid_archive": str(hybrid_path.relative_to(run)),
                        "loss_archive": str(loss_path.relative_to(run)),
                        "hybrid_per_shot_loss": hybrid_loss, "loss_only_per_shot_loss": loss_only,
                        "stored_per_shot_loss_absolute_difference": abs(hybrid_loss-loss_only),
                        **difference(left, right)})
    return {"scope": "Read-only actual saved GPU forward arrays; no solver or CUDA context used by auditor",
            "source_sha256": digest(__file__), "protocol_sha256": digest(run / "protocol.json"),
            "operator_signature": protocol["signature"], "observed_archive": headers(run / "observations.npz"),
            "training_archive": headers(run / "training_input.npz"), "saved_cache_records": records,
            "same_q_same_material_same_objective_pressure_comparisons": comparisons,
            "same_point_comparison_available": bool(comparisons),
            "limitation": "Archived pressure/gradient dtypes are direct measurements. Unarchived wavefield/stress/cotensor dtypes require separate source or runtime evidence; no inference is presented as an archived runtime measurement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"records": len(report["saved_cache_records"]), "same_point_comparisons": len(report["same_q_same_material_same_objective_pressure_comparisons"]), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
