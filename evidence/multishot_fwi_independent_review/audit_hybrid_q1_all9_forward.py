"""Compare all nine archived q1 Hybrid and loss-only forward pressure arrays.

NumPy and the standard library only: no solver, Torch, CUDA or optimizer call.
Private arrays stay on the machine holding the run. The report contains hashes
and scalar differences, not model or pressure samples.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def ah(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def bits(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype == np.float64
            and np.array_equal(left.view(np.uint64), right.view(np.uint64)))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def audit(root, run):
    protocol = read(run / "protocol.json")
    cfg = protocol["config"]
    assert cfg["shots"] == 9 and cfg["selected_shots"] == list(range(9))
    assert cfg["dtype"] == "float64" and cfg["parameters"] == ["vp"]
    assert protocol["training_has_true_vp"] is False
    assert protocol["source_encoding"] is False
    assert protocol["model_error_used_by_optimizer"] is False
    source_hashes = protocol["source_hashes"]
    for name, expected in source_hashes.items():
        assert sha(root / name) == expected
        assert sha(run / "executed_sources" / name.replace("/", "__")) == expected
    assert hashed({"sources": source_hashes, "config": cfg,
                   "training_sha256": protocol["training_input_sha256"],
                   "observations_sha256": protocol["observations_sha256"],
                   "torch": protocol["torch_version"]}) == protocol["signature"]
    for name, key in (("training_input.npz", "training_input_sha256"),
                      ("observations.npz", "observations_sha256"),
                      ("geometry.npz", "geometry_sha256"),
                      ("models/map.npz", "map_sha256")):
        assert sha(run / name) == protocol[key]
    training = load(run / "training_input.npz")
    assert "vp_true" not in training
    assert {key: ah(training[key]) for key in protocol["training_array_hashes"]} == protocol["training_array_hashes"]
    mapping = load(run / "models/map.npz")
    P = mapping["P"]
    observed = load(run / "observations.npz")
    geometry = load(run / "geometry.npz")
    assert observed["shot_ids"].tolist() == list(range(9))
    assert observed["pressure"].shape == observed["window"].shape == (9, cfg["nt"], cfg["receivers"])
    for name, value in geometry.items():
        assert bits(value, observed[name])
    objective = {"stage": 0, "cutoff_hz": cfg["stages"][0]["cutoff_hz"],
                 "filter_transition_hz": cfg["filter_transition_hz"],
                 "filter_order": "zero-padded FFT residual lowpass then fixed window",
                 "window_sha256": ah(observed["window"]),
                 "scales_sha256": ah(observed["scales"][0]),
                 "target_sha256": ah(observed["pressure"]), "global_normalizer": 9,
                 "regularizer": None}
    assert hashed(objective) == protocol["objective_ids"][0]

    commits = [path for path in (run / "commits").glob("*.json")
               if read(path)["state_after"]["accepted_total"] == 1]
    assert len(commits) == 1
    commit_path = commits[0]
    commit = read(commit_path)
    assert commit["sources"] == source_hashes and commit["model_true_vp_used"] is False
    state = commit["state_after"]
    q = np.load(run / state["current_q"], allow_pickle=False)
    assert ah(q) == state["current_q_sha256"]
    reference = training["vp_initial"]
    realized = reference + P * ((reference + cfg["velocity_scale_m_s"] * q) - reference)
    vp = load(run / "models/vp_0001.npz")["vp"]
    assert bits(realized, vp)
    fields = {"vp": ah(vp), "vs": ah(training["vs_fixed"]), "rho": ah(training["rho_fixed"])}
    assert hashed(fields) == state["model_sha256"] == commit["candidate_proof"]["model_sha256"]
    assert commit["candidate_proof"]["normalizer"] == 9
    assert len(commit["candidate_proof"]["proof"]) == 1
    item = commit["candidate_proof"]["proof"][0]
    loss_path = run / item["metadata"]
    loss = read(loss_path)
    assert loss["method"] == "loss" and loss["shot_ids"] == list(range(9))
    assert item["data"] == str(loss_path.with_suffix(".npz").relative_to(run))
    assert item["reused_exact_point"] is False

    expected_batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8]]
    batches = []
    for ids in expected_batches:
        filename = "shots_" + "-".join(map(str, ids)) + ".json"
        matches = []
        for path in run.glob("evaluations/*_hybrid/" + filename):
            row = read(path)
            if row["q_array_sha256"] == ah(q) and row["signature"] == protocol["signature"]:
                matches.append((path, row))
        assert len(matches) == 1
        batches.extend(matches)

    records = []
    for path, row in [(loss_path, loss)] + batches:
        ids = row["shot_ids"]
        assert row["signature"] == protocol["signature"]
        assert row["q_array_sha256"] == ah(q) and row["physical_fields_sha256"] == fields
        assert row["P_sha256"] == ah(P)
        assert row["objective_id"] == hashed(objective)
        assert row["stage"] == 0 and row["cutoff_hz"] == cfg["stages"][0]["cutoff_hz"]
        assert row["per_shot_scale"] == observed["scales"][0, ids].tolist()
        assert row["window_sha256"] == ah(observed["window"][ids])
        assert row["requires_grad"]["vs"] is False and row["requires_grad"]["rho"] is False
        assert "import_provenance" not in row
        data_path = path.with_suffix(".npz")
        assert sha(data_path) == row["data_sha256"]
        assert abs(sum(row["per_shot_loss"]) - row["loss_sum"]) <= 1e-14 * row["loss_sum"]
        records.append({"metadata": str(path.relative_to(run)), "metadata_sha256": sha(path),
                        "data_sha256": row["data_sha256"], "shot_ids": ids,
                        "method": row["method"], "imported_execution": False})

    with np.load(loss_path.with_suffix(".npz"), allow_pickle=False) as archive:
        right = archive["predicted_pressure"]
    assert right.dtype == np.float64 and right.shape == (9, cfg["nt"], cfg["receivers"])
    assert np.isfinite(right).all() and ah(right) == loss["prediction_sha256"]
    comparisons = []
    for (path, row), ids in zip(batches, expected_batches):
        assert row["method"] == "hybrid" and row["shot_ids"] == ids
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            left = archive["predicted_pressure"]
        assert left.dtype == np.float64 and left.shape == (len(ids), cfg["nt"], cfg["receivers"])
        assert np.isfinite(left).all() and ah(left) == row["prediction_sha256"]
        for k, shot in enumerate(ids):
            a, b = left[k], right[shot]
            assert bits(a, b)
            delta = a - b
            comparisons.append({"shot_id": shot, "hybrid_batch_ids": ids,
                "shape": list(a.shape), "dtype": str(a.dtype), "pressure_sha256": ah(a),
                "actual_uint64_equal": True, "different_bitpatterns": int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))),
                "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(b)),
                "absolute_linf": float(np.max(np.abs(delta))),
                "hybrid_per_shot_GPU_loss": row["per_shot_loss"][k],
                "loss_only_per_shot_GPU_loss": loss["per_shot_loss"][shot],
                "GPU_loss_absolute_difference": abs(row["per_shot_loss"][k] - loss["per_shot_loss"][shot])})
    assert [row["shot_id"] for row in comparisons] == list(range(9))
    return {"passed": True, "scope": "Actual archived q1 forward pressure: all nine shots, Hybrid batches 2+2+2+2+1 versus loss-only batch9. Read-only NumPy; no solver, GPU or optimizer execution by reviewer.",
            "run": str(run.relative_to(root)), "protocol_sha256": sha(run / "protocol.json"),
            "operator_signature": protocol["signature"], "source_hashes": source_hashes,
            "current_and_executed_source_hashes_verified": True,
            "protocol_signature_recomputed": True, "stage0_objective_id_recomputed": hashed(objective),
            "observations_sha256": protocol["observations_sha256"],
            "geometry_sha256": protocol["geometry_sha256"], "map_archive_sha256": protocol["map_sha256"],
            "P_sha256": ah(P), "q1_sha256": ah(q), "physical_fields_sha256": fields,
            "actual_q1_Vp_matches_static_P_map_uint64": True,
            "fixed_Vs_rho_match_training_array_bytes": True, "training_has_true_vp": False,
            "first_accepted_commit_sha256": sha(commit_path), "first_accepted_iteration_id": commit["iteration_id"],
            "same_q_same_material_same_objective_same_window_and_scale_verified": True,
            "cache_records": records, "all_nine_pressure_comparisons": comparisons,
            "maximum_GPU_loss_absolute_difference": max(row["GPU_loss_absolute_difference"] for row in comparisons),
            "all_nine_GPU_losses_bitwise_equal_claimed": False,
            "GPU_wavefield_or_cotensor_dtype_claimed": False,
            "initial_point_FD_nonmonotonicity_explained": False,
            "general_all_models_forward_equivalence_claimed": False,
            "full_inversion_completion_or_geological_recovery_claimed": False,
            "limitation": "This finite saved q1 comparison certifies these pressure arrays and their provenance only. It does not establish an FD error mechanism, all future models, unarchived runtime tensor dtypes, convergence or recovery."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run", type=Path, default=Path("runs/fwi-20261108-fullshot-v3"))
    parser.add_argument("--output", type=Path, help="New output directory; default is a fresh system temporary directory")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    run = args.run.resolve() if args.run.is_absolute() else (root / args.run).resolve()
    report = audit(root, run)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="hybrid-q1-all9-independent-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    report["output_directory"] = str(output)
    (output / "independent_hybrid_q1_all9_forward_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
