#!/usr/bin/env python3
"""Launch one experiment only after the selected NVIDIA GPU is idle.

This cooperates through a file lock; it never terminates or reprioritizes other
users' work. Unknown compute processes always block launch. CPU-only jobs are
left alone. Record concurrent CPU load when interpreting wall-clock timings.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

DESKTOP_EXECUTABLE = "/snap/snapd-desktop-integration/391/usr/bin/snapd-desktop-integration"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def query_csv(arguments):
    result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def snapshot(index, minimum_gpu_gib, minimum_ram_gib, maximum_utilization):
    rows = query_csv([f"--id={index}", "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu"])
    if len(rows) != 1:
        raise RuntimeError(f"Expected one GPU for index {index}, found {len(rows)}")
    gpu_index, gpu_uuid, name, total_mib, free_mib, utilization = rows[0]
    processes = query_csv([f"--id={index}", "--query-compute-apps=pid,process_name,used_gpu_memory"])
    compute, desktop = [], []
    for row in processes:
        if not row or len(row) != 3:
            continue
        pid, executable, memory = row
        process = {"pid": int(pid), "executable": executable, "used_memory_mib": memory}
        # Exact known desktop binary, one instance, small memory. Do not ignore
        # arbitrary C+G processes or unknown applications with a similar name.
        if executable == DESKTOP_EXECUTABLE and float(memory) <= 64:
            desktop.append(process)
        else:
            compute.append(process)
    if len(desktop) > 1:
        compute.extend(desktop)
        desktop = []
    available_ram = psutil.virtual_memory().available
    blockers = []
    if compute:
        blockers.append("compute_process_present")
    if float(utilization) > maximum_utilization:
        blockers.append("gpu_utilization")
    if float(free_mib) < minimum_gpu_gib * 1024:
        blockers.append("gpu_memory")
    if available_ram < minimum_ram_gib * 1024 ** 3:
        blockers.append("host_memory")
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu": {"index": int(gpu_index), "uuid": gpu_uuid, "name": name, "total_mib": float(total_mib), "free_mib": float(free_mib), "utilization_percent": float(utilization)},
        "compute_processes": compute,
        "ignored_known_desktop_processes": desktop,
        "host_available_ram_bytes": available_ram,
        "load_average": list(os.getloadavg()),
        "blockers": blockers,
        "eligible": not blockers,
    }


def cpu_policy():
    process = psutil.Process()
    inherited = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
    percent = psutil.cpu_percent(interval=1.0, percpu=True)
    groups = {}
    for cpu in inherited or range(len(percent)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (path / "physical_package_id").read_text().strip()
            core = (path / "core_id").read_text().strip()
            key = f"{package}:{core}"
        except OSError:
            key = str(cpu)
        groups.setdefault(key, []).append(cpu)
    ranked = sorted(groups.items(), key=lambda item: sum(percent[cpu] for cpu in item[1]) / len(item[1]))
    selected = None
    reason = "No available physical core measured below 90%; retain inherited affinity."
    if ranked:
        _, siblings = ranked[0]
        if max(percent[cpu] for cpu in siblings) < 90:
            selected = [min(siblings, key=lambda cpu: percent[cpu])]
            reason = "One logical CPU on the least busy available physical core."
    return {"inherited_affinity": inherited, "per_cpu_percent": percent, "selected_affinity": selected, "reason": reason, "threads": 1, "nice": 10}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--wait", type=float, default=3600, help="Maximum seconds to await GPU availability, including stable checks.")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--min-gpu-gib", type=float, default=8)
    parser.add_argument("--min-ram-gib", type=float, default=8)
    parser.add_argument("--max-utilization", type=float, default=5)
    parser.add_argument("--lock", type=Path, default=Path.home() / ".cache/hybridadfwi/gpu0.lock")
    parser.add_argument("--evidence", type=Path, default=Path("results/resource_gate.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command and not args.dry_run:
        parser.error("Supply a command after --, or use --dry-run")
    if args.interval <= 0 or args.stable_checks < 1 or args.wait <= 0:
        parser.error("interval, stable-checks and wait must be positive")
    if not sys.platform.startswith("linux"):
        parser.error("This NVIDIA GPU launcher requires Linux flock and CPU scheduling support")
    import fcntl
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait
    evidence = {"command": command, "lock_path": str(args.lock), "known_desktop_executable": DESKTOP_EXECUTABLE, "snapshots": [], "status": "waiting"}
    with args.lock.open("a+") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    evidence["status"] = "lock_timeout"
                    atomic_json(args.evidence, evidence)
                    return 75
                time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        stable = 0
        while True:
            current = snapshot(args.gpu_index, args.min_gpu_gib, args.min_ram_gib, args.max_utilization)
            evidence["snapshots"].append(current)
            stable = stable + 1 if current["eligible"] else 0
            evidence["stable_checks"] = stable
            atomic_json(args.evidence, evidence)
            print(json.dumps({"resource_gate": current, "stable_checks": stable}), flush=True)
            if stable >= args.stable_checks:
                break
            if time.monotonic() >= deadline:
                evidence["status"] = "resource_timeout"
                atomic_json(args.evidence, evidence)
                return 75
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        policy = cpu_policy()
        evidence["cpu_policy"] = policy
        evidence["status"] = "eligible" if args.dry_run else "running"
        atomic_json(args.evidence, evidence)
        if args.dry_run:
            return 0
        # Environment variables apply to this experiment and its children only.
        environment = dict(os.environ)
        environment.update({name: "1" for name in THREAD_ENV})
        environment["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        if policy["selected_affinity"] is not None:
            psutil.Process().cpu_affinity(policy["selected_affinity"])
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.setpriority(os.PRIO_PROCESS, 0, 10)
        started = time.monotonic()
        result = subprocess.run(command, env=environment)
        evidence["status"] = "complete" if result.returncode == 0 else "failed"
        evidence["return_code"] = result.returncode
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["final_snapshot"] = snapshot(args.gpu_index, args.min_gpu_gib, args.min_ram_gib, args.max_utilization)
        atomic_json(args.evidence, evidence)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
