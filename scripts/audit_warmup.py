#!/usr/bin/env python3
"""Check first-use overhead when a benchmark's native checkpoint length changes.

Run each method in a separate process, after other GPU experiments finish:
  python scripts/audit_warmup.py --config results/protocol.json \
      --method autograd --output results/warmup-autograd.json

This diagnostic reproduces the original 16-step warmup, then evaluates three
representative native chunks on the same grid/shots/parameters.  It does not
change the benchmark protocol or existing result files.  A first-versus-steady
gap suggests first-use overhead; it is not by itself proof of JIT compilation
or a correction that can simply be subtracted from the full experiment.
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

# Use the exact benchmark initialization, precision hook and synchronization.
_spec = importlib.util.spec_from_file_location("_hybridadfwi_benchmark", HERE / "benchmark.py")
benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=("autograd", "hybrid"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Output exists; choose a new file to preserve the audit")
    config = json.loads(args.config.read_text())
    if min(config["nt"], config["segments"]) < 1:
        raise SystemExit("nt and segments must be positive")

    import numpy as np
    from hybridadfwi.experiment import build, objective
    from hybridadfwi.adjoint import hybrid_forward
    torch = benchmark.initialize(config)
    # torch.chunk may produce fewer than the requested number of chunks.
    # This gives the largest actual checkpoint chunk in the full experiment.
    chunk_nt = (config["nt"] + config["segments"] - 1) // config["segments"]
    warm_nt = min(config["nt"], 16)
    warm_segments = min(config["segments"], warm_nt)
    started = time.perf_counter()
    report = {
        "status": "running", "method": args.method,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config, "protocol_path": str(args.config.resolve()),
        "protocol_sha256": benchmark.sha256(args.config),
        "script_sha256": benchmark.sha256(__file__),
        "benchmark_sha256": benchmark.sha256(HERE / "benchmark.py"),
        "adjoint_sha256": benchmark.sha256(ROOT / "src/hybridadfwi/adjoint.py"),
        "python": sys.version, "pytorch": torch.__version__,
        "hostname": platform.node(), "platform": platform.platform(),
        "device_name": torch.cuda.get_device_name() if config["device"] == "cuda" else platform.processor(),
        "old_warmup": {"nt": warm_nt, "segments": warm_segments, "evaluations": 1,
                       "observations": "zeros", "scale": 1.0},
        "representative": {"nt": chunk_nt, "segments": 1, "evaluations": 3,
                           "observations": "actual full-experiment observed-data prefix",
                           "scale": "unchanged full-experiment scale"},
        "scope": "forward + objective/data VJP + reverse/static VJP; setup, synchronization before calls, cleanup and finite checks excluded",
        "caution": "Short-chunk first-use sensitivity audit, not a full-run speed or memory benchmark. Interpret gaps alongside full-run variation; do not subtract them as known JIT costs.",
        "measurements": [],
    }
    benchmark.save_json(args.output, report)

    def evaluate(model, prop, observed, scale, segments):
        benchmark.sync(torch, config)
        start = time.perf_counter()
        tape = None
        source = None
        if args.method == "hybrid":
            output, tape = hybrid_forward(prop, fd_order=8, checkpoint_segments=segments)
        else:
            output = benchmark.native_forward(prop, segments)
        benchmark.sync(torch, config)
        after_forward = time.perf_counter()
        data = output["vz"].detach().requires_grad_(True) if tape else output["vz"]
        loss = objective(data, observed, scale, config["dt"], config["objective"])
        if tape:
            source, = torch.autograd.grad(loss, data)
        benchmark.sync(torch, config)
        after_loss = time.perf_counter()
        if tape:
            gradients = tape.backward({"vz": source})
        else:
            loss.backward()
            gradients = {name: getattr(model, name).grad for name in config["parameters"]}
        benchmark.sync(torch, config)
        finish = time.perf_counter()
        timings = {"forward_s": after_forward - start,
                   "objective_and_source_s": after_loss - after_forward,
                   "backward_s": finish - after_loss, "total_s": finish - start}
        # Finite checks are deliberately outside measured phase boundaries.
        if source is None:
            # Inspect the same data-space VJP without adding a retain_grad hook
            # or an extra backward pass to the measured upstream computation.
            check_leaf = output["vz"].detach().requires_grad_(True)
            check_loss = objective(check_leaf, observed, scale, config["dt"], config["objective"])
            source, = torch.autograd.grad(check_loss, check_leaf)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(output["vz"]).all()):
            raise FloatingPointError("Nonfinite loss or native receiver data")
        if source.dtype != torch.float64 or not bool(torch.isfinite(source).all()):
            raise FloatingPointError("Nonfinite or non-float64 data-space VJP")
        norms = {}
        for name in config["parameters"]:
            value = gradients[name]
            if value is None or value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Nonfinite, missing or non-float64 " + name + " kernel")
            norms[name] = float(torch.linalg.vector_norm(value))
        row = {"timing": timings, "loss": float(loss.detach()), "kernel_l2": norms,
               "data_vjp_l2": float(torch.linalg.vector_norm(source)),
               "finite_checks": "passed; receiver data, loss, data VJP and all requested kernels"}
        if tape:
            row["adapter_details"] = dict(tape.stats)
        return row

    def cleanup():
        gc.collect()
        if config["device"] == "cuda":
            torch.cuda.empty_cache()
        benchmark.sync(torch, config)

    try:
        # The original worker uses zero observations and the same full geometry.
        model, prop = build(config, nt=warm_nt)
        warm_obs = torch.zeros((config["shots"], warm_nt, prop.rcv_n),
                               dtype=torch.float64, device=config["device"])
        report["old_warmup"]["result"] = evaluate(model, prop, warm_obs, 1., warm_segments)
        del model, prop, warm_obs
        cleanup()
        with np.load(config["observations"]) as saved:
            prefix = np.ascontiguousarray(saved["vz"][:, :chunk_nt], dtype=np.float64)
            scale = float(saved["scale"])
        observed = torch.as_tensor(prefix, dtype=torch.float64, device=config["device"])
        report["observations_sha256"] = benchmark.sha256(config["observations"])
        for trial in range(3):
            model, prop = build(config, nt=chunk_nt)
            model.zero_grad(set_to_none=True)
            cleanup()
            row = evaluate(model, prop, observed, scale, 1)
            row["trial"] = trial
            report["measurements"].append(row)
            benchmark.save_json(args.output, report)
            print(json.dumps({"stage": "representative_chunk", "method": args.method,
                              "trial": trial, **row["timing"]}), flush=True)
            del model, prop
            cleanup()
        comparison = {}
        rows = report["measurements"]
        for key in ("forward_s", "objective_and_source_s", "backward_s", "total_s"):
            first = rows[0]["timing"][key]
            steady = statistics.mean(row["timing"][key] for row in rows[1:])
            comparison[key] = {"first": first, "mean_last_two": steady,
                               "first_minus_steady": first - steady,
                               "first_over_steady": first / steady if steady else None}
        report.update(status="ok", comparison=comparison,
                      elapsed_wall_s=time.perf_counter() - started)
    except BaseException as error:
        report.update(status="error", error=repr(error), traceback=traceback.format_exc(),
                      elapsed_wall_s=time.perf_counter() - started)
        benchmark.save_json(args.output, report)
        raise
    benchmark.save_json(args.output, report)
    print(json.dumps({"status": report["status"], "method": args.method,
                      "output": str(args.output), "elapsed_wall_s": report["elapsed_wall_s"],
                      "comparison": report["comparison"]}), flush=True)


if __name__ == "__main__":
    main()
