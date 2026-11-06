#!/usr/bin/env python3
"""Each measurement runs in a fresh process; compare raw kernels on identical inputs."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(config):
    import torch
    from hybridadfwi.experiment import verify_upstream
    verify_upstream()
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no implicit CPU fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Native FD constant construction requires float32 default; fields are explicit float64")
    return torch


def sync(torch, config):
    if config["device"] == "cuda":
        torch.cuda.synchronize()


def native_forward(prop, segments, checkpointing=True):
    from hybridadfwi.adjoint import preserve_padding_dtype, disable_native_checkpointing
    with preserve_padding_dtype():
        if not checkpointing:
            if segments != 1:
                raise ValueError("No-checkpoint mode requires segments=1")
            with disable_native_checkpointing():
                return prop.forward(fd_order=8, checkpoint_segments=1)
        return prop.forward(fd_order=8, checkpoint_segments=segments)


def prepare(config):
    import numpy as np
    from hybridadfwi.experiment import build
    torch = initialize(config)
    _, prop = build(config, truth=True, trainable=False)
    with torch.no_grad():
        observed = native_forward(prop, config["segments"], config.get("checkpointing", True))["vz"]
    sync(torch, config)
    if not torch.isfinite(observed).all():
        raise FloatingPointError("Non-finite true-model data")
    scale = max(float(observed.square().mean().sqrt()), 1e-30)
    np.savez(config["observations"], vz=observed.cpu().numpy(), scale=np.array(scale))
    print(json.dumps({"stage": "observations", "shape": list(observed.shape), "scale": scale}), flush=True)


def worker(config, method, result, progress=None):
    import numpy as np
    import psutil
    from hybridadfwi.experiment import build, objective, UPSTREAM_COMMIT
    from hybridadfwi.adjoint import hybrid_forward
    torch = initialize(config)
    progress = {} if progress is None else progress
    progress.update(stage="warmup", phase="setup", worker_started=time.perf_counter())

    def evaluate(model, prop, obs, scale, segments):
        start = time.perf_counter()
        progress.update(phase="forward", evaluation_started=start)
        progress.pop("forward_s", None)
        progress.pop("objective_and_source_s", None)
        if method == "hybrid":
            out, tape = hybrid_forward(prop, fd_order=8, checkpoint_segments=segments,
                                       checkpointing=config.get("checkpointing", True))
        else:
            out = native_forward(prop, segments, config.get("checkpointing", True))
        sync(torch, config)
        after_forward = time.perf_counter()
        progress.update(phase="objective_and_source", forward_s=after_forward-start)
        data = out["vz"].detach().requires_grad_(True) if method == "hybrid" else out["vz"]
        loss = objective(data, obs, scale, config["dt"], config["objective"])
        if method == "hybrid":
            (source,) = torch.autograd.grad(loss, data)
        sync(torch, config)
        after_loss = time.perf_counter()
        progress.update(phase="backward", objective_and_source_s=after_loss-after_forward)
        if method == "hybrid":
            gradients = tape.backward({"vz": source})
        else:
            loss.backward()
            gradients = {name: getattr(model, name).grad for name in config["parameters"]}
        sync(torch, config)
        finish = time.perf_counter()
        timings = {
            "forward_s": after_forward-start,
            "objective_and_source_s": after_loss-after_forward,
            "backward_s": finish-after_loss,
            "total_s": finish-start,
        }
        if method == "hybrid":
            timings["adapter_details"] = dict(tape.stats)
        return out, gradients, loss.detach(), timings

    # Same grid, source geometry and active parameters; warmup excluded from measurement.
    warm_nt = min(config["nt"], 16)
    model, prop = build(config, nt=warm_nt)
    warm_obs = torch.zeros((config["shots"], warm_nt, prop.rcv_n), dtype=torch.float64, device=config["device"])
    for _ in range(3 if not config.get("checkpointing", True) else 1):
        model.zero_grad(set_to_none=True)
        warm_result = evaluate(model, prop, warm_obs, 1., min(config["segments"], warm_nt))
        del warm_result
    del model, prop, warm_obs
    gc.collect()
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    model, prop = build(config)
    with np.load(config["observations"]) as f:
        obs = torch.as_tensor(f["vz"], dtype=torch.float64, device=config["device"])
        scale = float(f["scale"])
    model.zero_grad(set_to_none=True)
    gc.collect()
    sync(torch, config)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    rss = [baseline_rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(.005):
            rss.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    baseline_allocated = baseline_reserved = 0
    if config["device"] == "cuda":
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    progress.update(stage="measurement", baseline_allocated_bytes=baseline_allocated,
                    baseline_reserved_bytes=baseline_reserved, baseline_rss_bytes=baseline_rss)
    monitor.start()
    try:
        out, gradients, loss, times = evaluate(model, prop, obs, scale, config["segments"])
    finally:
        rss.append(process.memory_info().rss)
        progress["sampled_peak_rss_bytes"] = max(rss)
        stop.set()
        monitor.join()
    gpu_memory = {}
    if config["device"] == "cuda":
        gpu_memory = {
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_allocated_increment_bytes": torch.cuda.max_memory_allocated()-baseline_allocated,
        }
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss")
    saved = {"predicted_vz": out["vz"].detach().cpu().numpy()}
    for name in config["parameters"]:
        gradient = gradients[name]
        if gradient is None or gradient.dtype != torch.float64 or not torch.isfinite(gradient).all():
            raise FloatingPointError("Missing, wrong-dtype or non-finite " + name + " kernel")
        saved["gradient_"+name] = gradient.detach().cpu().numpy()
    np.savez(Path(result).with_suffix(".npz"), **saved)
    row = {
        "status": "ok", "method": method, "timing": times, "loss": float(loss),
        "cpu_memory": {"baseline_rss_bytes": baseline_rss, "sampled_peak_rss_bytes": max(rss),
                       "sampled_peak_increment_bytes": max(rss)-baseline_rss, "sample_interval_s": .005},
        "cuda_memory": gpu_memory, "upstream_commit": UPSTREAM_COMMIT,
        "python": sys.version, "pytorch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "hostname": platform.node(), "cpu_load": list(os.getloadavg()),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "device_name": torch.cuda.get_device_name() if config["device"] == "cuda" else platform.processor(),
        "config": config, "script_sha256": sha256(__file__), "model_sha256": sha256(config["model"]),
        "upstream_manifest_sha256": sha256(ROOT/"upstream.lock.json"),
        "observations_sha256": sha256(config["observations"]),
        "precision": "float64 models, fields, receiver data and kernels; native FD constants computed in float32",
        "padding": "identical dtype-preserving replicate-padding adapter in both arms; vendor source unchanged",
        "checkpointing": config.get("checkpointing", True),
        "warmup_evaluations": 3 if not config.get("checkpointing", True) else 1,
        "implementation_sha256": {str(p.relative_to(ROOT)):sha256(p) for p in
                                  (ROOT/"src/hybridadfwi/adjoint.py", ROOT/"src/hybridadfwi/experiment.py", ROOT/"src/hybridadfwi/resources.py")},
        "scope": "forward + objective/data VJP + backward/static model VJP; setup, warmup and result storage excluded",
    }
    save_json(result, row)
    print(json.dumps({"method": method, "total_s": times["total_s"], "loss": row["loss"]}), flush=True)


def compare(directory, config):
    import numpy as np
    rows = []
    comparisons = []
    passed = True
    audits = []
    if config.get("resource_audit"):
        for method in ("prepare", *[f"{m}-{r}" for r in range(config["repeats"]) for m in ("autograd", "hybrid")]):
            audit = json.loads((directory/(method+".resources.json")).read_text())
            audits.append({"worker": method, "status": audit["status"], "contaminated": audit["contaminated"]})
            passed &= audit["status"] == "complete"
    for repeat in range(config["repeats"]):
        paths = {m: directory / (m+"-"+str(repeat)+".json") for m in ("autograd", "hybrid")}
        rows.extend(json.loads(p.read_text()) for p in paths.values())
        with np.load(paths["autograd"].with_suffix(".npz")) as a, np.load(paths["hybrid"].with_suffix(".npz")) as b:
            exact = bool(np.array_equal(a["predicted_vz"], b["predicted_vz"]))
            report = {"repeat": repeat, "forward_bitwise_equal": exact, "kernels": {}}
            passed &= exact
            for name in config["parameters"]:
                x, y = a["gradient_"+name].ravel(), b["gradient_"+name].ravel()
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                relative = float(np.linalg.norm(x-y)/max(nx, 1e-300))
                valid = bool(nx > 0 and ny > 0 and np.isfinite(relative) and relative <= config["tolerance"])
                report["kernels"][name] = {
                    "relative_l2": relative, "relative_linf": float(np.max(np.abs(x-y))/max(np.max(np.abs(x)), 1e-300)),
                    "cosine": float(np.dot(x,y)/(nx*ny)) if nx*ny else None,
                    "autograd_l2": float(nx), "hybrid_l2": float(ny), "passed": valid,
                }
                passed &= valid
            comparisons.append(report)
    stats = {}
    for method in ("autograd", "hybrid"):
        selected = [r for r in rows if r["method"] == method]
        times = [r["timing"]["total_s"] for r in selected]
        stats[method] = {"total_s": times, "mean_total_s": float(np.mean(times)),
                         "sample_std_s": float(np.std(times, ddof=1)) if len(times)>1 else None,
                         "peak_rss_bytes": [r["cpu_memory"]["sampled_peak_rss_bytes"] for r in selected],
                         "peak_cuda_allocated_bytes": [r["cuda_memory"].get("peak_allocated_bytes") for r in selected]}
    summary = {"passed": bool(passed), "config": config, "comparison": comparisons, "statistics": stats,
               "resource_audit": audits,
               "autograd_over_hybrid_time_ratio": stats["autograd"]["mean_total_s"]/stats["hybrid"]["mean_total_s"],
               "caveat": "Small runs establish correctness only; timings include any enabled checkpoint replay and host contention."}
    save_json(directory/"summary.json", summary)
    lines = ["# ADFWI vs HybridADFWI", "", "Result: " + ("PASS" if passed else "FAIL"), "",
             "Same Marmousi-II model, native FD8 forward, float64 states and raw unprocessed kernels.", "",
             "Checkpointing: " + (f"enabled ({config['segments']} segments)" if config.get("checkpointing", True)
                                   else "disabled in both arms; full AD tape versus original-forward history; zero replay"), "",
             "| Method | Mean seconds | Peak process RSS (MiB) | Peak CUDA allocated (MiB) |",
             "|---|---:|---:|---:|"]
    for method, values in stats.items():
        gpu = values["peak_cuda_allocated_bytes"]
        gpu_text = f"{max(gpu)/2**20:.2f}" if gpu[0] is not None else "N/A"
        lines.append(f"| {method} | {values['mean_total_s']:.6f} | {max(values['peak_rss_bytes'])/2**20:.2f} | {gpu_text} |")
    lines += ["", "| Repeat | Forward bitwise equal | Parameter | Relative L2 kernel error | Cosine |",
              "|---:|---|---|---:|---:|"]
    for c in comparisons:
        for name, v in c["kernels"].items():
            lines.append(f"| {c['repeat']} | {c['forward_bitwise_equal']} | {name} | {v['relative_l2']:.4e} | {v['cosine']} |")
    lines += ["", "Full protocol, individual phase times, hashes and limitations are in the JSON files.",
              "CPU RSS includes the Python/PyTorch runtime; CUDA allocated/reserved are recorded separately.",
              "The upstream float32-generated FD constants are shared. Both paths use the same external dtype-preserving padding adapter.",
              "Warmup, observations and output storage are excluded. No optimizer, gradient clipping, normalization or illumination preconditioning."]
    (directory/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"passed": bool(passed), "summary": str(directory/"summary.json"), "comparisons": comparisons}), flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    run.add_argument("--model", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--parameters", nargs="+", choices=["vp", "vs", "rho"], default=["vp", "vs", "rho"])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--nt", type=int)
    run.add_argument("--shots", type=int)
    run.add_argument("--segments", type=int)
    run.add_argument("--no-checkpoint", action="store_true", help="Bypass checkpoint callbacks; retain the full AD tape or hybrid field history, with no replay")
    run.add_argument("--objective", choices=["l2", "l2_time_difference"], default="l2")
    run.add_argument("--tolerance", type=float, default=1e-8)
    for name in ("prepare", "worker", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "worker":
            p.add_argument("--method", choices=["autograd", "hybrid"], required=True)
            p.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "run":
        config = json.loads(args.config.read_text())
        if args.command == "prepare":
            prepare(config)
        elif args.command == "worker":
            progress = {}
            try:
                worker(config, args.method, args.result, progress)
            except Exception as error:
                import torch
                message = str(error).lower()
                cuda_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    config["device"] == "cuda" and
                    ("cuda out of memory" in message or "cuda error: out of memory" in message))
                if not cuda_oom:
                    raise
                failure = {"status": "cuda_oom", "method": args.method, "config": config,
                           "error": str(error), "progress": {k:v for k,v in progress.items()
                                                                if k not in ("worker_started", "evaluation_started")},
                           "attempt_elapsed_s": time.perf_counter()-progress.get("worker_started", time.perf_counter()),
                           "evaluation_elapsed_to_failure_s": time.perf_counter()-progress.get("evaluation_started", time.perf_counter()),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                           "script_sha256": sha256(__file__), "model_sha256": sha256(config["model"]),
                           "note": "Failed attempt time is not a completed gradient runtime; no kernel comparison is available."}
                save_json(args.result, failure)
                print(json.dumps({"status":"cuda_oom", "method":args.method, "result":str(args.result)}), flush=True)
                raise SystemExit(86)
        else:
            compare(args.config.parent, config)
        return
    output = args.output.resolve()
    if (output/"protocol.json").exists():
        raise SystemExit("Output already contains a protocol; select a new directory to preserve results")
    output.mkdir(parents=True, exist_ok=True)
    nt = args.nt or (192 if args.preset == "smoke" else 2500)
    if args.no_checkpoint and args.segments not in (None, 1):
        raise SystemExit("--no-checkpoint requires --segments 1 (or omit --segments)")
    config = dict(model=str((args.model or ROOT/"data"/"prepared"/f"marmousi2_{args.preset}.npz").resolve()),
                  observations=str(output/"observations.npz"), preset=args.preset, device=args.device,
                  nt=nt, dt=.004 if args.preset == "smoke" else .003, f0=5.,
                  shots=args.shots or (1 if args.preset == "smoke" else 9),
                  receivers=25 if args.preset == "smoke" else 100,
                  depth_m=540., nabc=10 if args.preset == "smoke" else 20, free_surface=True,
                  fd_order=8, dtype="float64", parameters=args.parameters, repeats=args.repeats,
                  segments=1 if args.no_checkpoint else args.segments or math.ceil(math.sqrt(nt)),
                  checkpointing=not args.no_checkpoint, threads=args.threads,
                  objective=args.objective, tolerance=args.tolerance,
                  resource_audit="per-worker NVIDIA process audit, sampled every 3 seconds" if args.device == "cuda" else None,
                  order="alternating AB/BA across repetitions; fresh process per arm",
                  timing_scope="forward + data objective/VJP + reverse; includes replay only when checkpointing is enabled; no optimizer")
    if min(config["nt"], config["shots"], config["segments"], config["repeats"], config["threads"]) < 1:
        raise SystemExit("Counts must be positive")
    if not Path(config["model"]).is_file():
        raise SystemExit("Prepare the real Marmousi-II model first: scripts/prepare_marmousi.py")
    save_json(output/"protocol.json", config)
    base = [sys.executable, str(Path(__file__).resolve())]
    from hybridadfwi.resources import run_checked

    def checked(command, name, log=None):
        evidence = run_checked(command, stdout=log, device=config["device"], min_gpu_gib=18, min_ram_gib=12)
        save_json(output/(name+".resources.json"), evidence)
        if evidence["status"] != "complete":
            raise SystemExit(f"{name}: {evidence['status']}; inspect its resources JSON and log; remaining measurements stopped")

    checked(base+["prepare", "--config", str(output/"protocol.json")], "prepare")
    for repeat in range(args.repeats):
        for method in (("autograd", "hybrid") if repeat % 2 == 0 else ("hybrid", "autograd")):
            result = output/(method+"-"+str(repeat)+".json")
            with result.with_suffix(".log").open("w") as log:
                checked(base+["worker", "--config", str(output/"protocol.json"), "--method", method,
                              "--result", str(result)], method+"-"+str(repeat), log)
            print(f"Completed {method} repeat {repeat}", flush=True)
    compare(output, config)


if __name__ == "__main__":
    main()
