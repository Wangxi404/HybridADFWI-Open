"""Check GPU availability for each worker and audit concurrent GPU processes.

Monitoring stays in the parent, creates no CUDA context and never terminates
another job. Sampling cannot exclude workloads shorter than its interval.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import subprocess
import time

import psutil


@lru_cache(maxsize=1)
def _gate():
    # Reuse the standalone launcher's exact desktop-process and resource policy.
    # Loading this script has no effects: its main entry point is guarded.
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_when_idle.py"
    spec = importlib.util.spec_from_file_location("_hybridadfwi_resource_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_snapshot(gpu_index="0", min_gpu_gib=8., min_ram_gib=8.):
    return _gate().snapshot(gpu_index, min_gpu_gib, min_ram_gib, 5.)


def _external_processes(snapshot, worker_pids):
    return [p for p in snapshot["compute_processes"] if p["pid"] not in worker_pids]


def run_checked(command, stdout=None, device="cuda", gpu_index="0", *, sample_interval=3.,
                idle_timeout=120., stable_checks=3, min_gpu_gib=8., min_ram_gib=8.):
    """Return resource evidence and the child return code.

    Caller must save the evidence and stop future workers unless status is
    ``complete``. GPU contamination does not terminate the current worker, so
    its failure or output remains inspectable; it invalidates its benchmark.
    ``blocked`` means the child never launched. CPU runs need no NVIDIA tools.
    stdout accepts an open log file as subprocess.Popen does.
    """
    if sample_interval <= 0 or idle_timeout <= 0 or stable_checks < 1:
        raise ValueError("Intervals, idle timeout and stable-check count must be positive")
    evidence = {
        "command": [str(x) for x in command], "device": device,
        "status": "waiting" if device == "cuda" else "starting", "returncode": None,
        "contaminated": False, "snapshots": [], "sample_interval_s": sample_interval,
        "monitor_scope": "entire child lifetime including setup, warmup and result writes",
        "limitation": "Sampled process audit cannot rule out sub-interval competing workloads; lock is cooperative.",
        "parent_pid": os.getpid(), "cpu_load_before": list(os.getloadavg()),
    }
    start = time.monotonic()

    def capture(phase, allowed_pids=()):
        try:
            snap = gpu_snapshot(gpu_index, min_gpu_gib, min_ram_gib)
        except Exception as error:
            snap = {"phase": phase, "monitor_error": str(error), "eligible": False}
            evidence["snapshots"].append(snap)
            evidence["contaminated"] = True
            return snap
        snap["phase"] = phase
        snap["seconds_since_gate_start"] = time.monotonic() - start
        if phase != "before":
            external = _external_processes(snap, set(allowed_pids))
            snap["allowed_worker_pids"] = sorted(allowed_pids)
            snap["external_compute_processes"] = external
            # GPU use and lower free memory are expected during our own worker.
            # Unknown compute processes, rather than utilization, contaminate it.
            evidence["contaminated"] |= bool(external)
        evidence["snapshots"].append(snap)
        return snap

    if device == "cuda":
        deadline = time.monotonic() + idle_timeout
        stable = 0
        while True:
            snap = capture("before")
            if "monitor_error" in snap:
                evidence["status"] = "monitor_error"
                return evidence
            stable = stable + 1 if snap["eligible"] else 0
            if stable >= stable_checks:
                break
            if time.monotonic() >= deadline:
                evidence["status"] = "blocked"
                evidence["block_reason"] = "GPU did not remain idle before the worker timeout"
                return evidence
            time.sleep(min(sample_interval, max(0., deadline - time.monotonic())))

    process = subprocess.Popen(command, stdout=stdout,
                               stderr=subprocess.STDOUT if stdout is not None else None)
    evidence["worker_pid"] = process.pid
    evidence["status"] = "running"
    known_worker_pids = {process.pid}
    # A separate worker should use a single process, but allow its descendants
    # in case future backends create their own explicitly related subprocesses.
    while True:
        if device == "cuda":
            try:
                known_worker_pids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
            capture("during", known_worker_pids)
        try:
            returncode = process.wait(timeout=sample_interval)
            break
        except subprocess.TimeoutExpired:
            continue
    if device == "cuda":
        capture("after", known_worker_pids)
    evidence["returncode"] = returncode
    evidence["cpu_load_after"] = list(os.getloadavg())
    evidence["elapsed_including_resource_checks_s"] = time.monotonic() - start
    evidence["status"] = "contaminated" if evidence["contaminated"] else "complete" if returncode == 0 else "failed"
    return evidence
