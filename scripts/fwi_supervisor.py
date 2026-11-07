#!/usr/bin/env python3
"""Watch only our worker; yield the GPU if another compute job appears.

No foreign PID is signalled. Long-run workers must checkpoint at batch/update
boundaries and resume interrupted evaluations under the same immutable anchor.
"""
import argparse
from datetime import datetime,timezone
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import psutil
from hybridadfwi.resources import gpu_snapshot


def atomic(path,value):
    temp=path.with_suffix(".tmp");temp.write_text(json.dumps(value,indent=2)+"\n");temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--idle-wait-s",type=float,default=600.)
    parser.add_argument("--maximum-walltime-s",type=float,default=86400.)
    parser.add_argument("--reserve-gib",type=float,default=2.)
    parser.add_argument("--min-free-gib",type=float,default=22.)
    parser.add_argument("--resume-on-release",action="store_true")
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:parser.error("Worker command required")
    args.evidence.parent.mkdir(parents=True,exist_ok=True)
    events=args.evidence.with_suffix(".jsonl")
    state={"command":command,"status":"waiting","supervisor_pid":os.getpid(),"attempts":[],
        "policy":"Signal only this supervisor's own worker on foreign compute; never signal other jobs",
        "sample_interval_s":3.,"reserve_gib":args.reserve_gib}
    started=time.monotonic();attempt=0
    def record(event):
        event["utc"]=datetime.now(timezone.utc).isoformat()
        with events.open("a") as f:f.write(json.dumps(event)+"\n")
        state["last_event"]=event;state["elapsed_s"]=time.monotonic()-started;atomic(args.evidence,state)
    import fcntl
    lockpath=Path.home()/".cache/hybridadfwi/gpu0.lock";lockpath.parent.mkdir(parents=True,exist_ok=True)
    while time.monotonic()-started<args.maximum_walltime_s:
        deadline=time.monotonic()+args.idle_wait_s;stable=0
        with lockpath.open("a+") as lock:
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
                except BlockingIOError:locked=False
                snap=gpu_snapshot(min_gpu_gib=args.min_free_gib,min_ram_gib=8.)
                stable=stable+1 if locked and snap["eligible"] else 0
                record({"event":"await_idle","stable_checks":stable,"snapshot":snap})
                if stable>=3:break
                if time.monotonic()>=deadline:
                    state["status"]="paused_waiting_for_resources";record({"event":"idle_window_elapsed"})
                    if args.resume_on_release and time.monotonic()-started<args.maximum_walltime_s:
                        deadline=min(time.monotonic()+args.idle_wait_s,started+args.maximum_walltime_s)
                    else:
                        if time.monotonic()-started>=args.maximum_walltime_s:
                            state["status"]="paused_walltime_cap";record({"event":"walltime_cap_while_waiting"})
                        return 75
                time.sleep(3.)
            attempt+=1;state["status"]="running"
            env=dict(os.environ);env.update({k:"1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")})
            # Inherit one quiet logical CPU and low priority; never change other processes.
            allowed=psutil.Process().cpu_affinity();load=psutil.cpu_percent(interval=1,percpu=True)
            chosen=min(allowed,key=lambda cpu:(load[cpu],cpu));psutil.Process().cpu_affinity([chosen])
            if os.getpriority(os.PRIO_PROCESS,0)<10:os.setpriority(os.PRIO_PROCESS,0,10)
            worker=subprocess.Popen(command,env=env,start_new_session=True)
            state["attempts"].append({"attempt":attempt,"worker_pid":worker.pid,"cpu_affinity":[chosen]})
            interrupted=None
            while worker.poll() is None:
                allowed_pids={worker.pid}
                try:allowed_pids.update(p.pid for p in psutil.Process(worker.pid).children(recursive=True))
                except psutil.NoSuchProcess:pass
                snap=gpu_snapshot(min_gpu_gib=args.reserve_gib,min_ram_gib=8.)
                foreign=[p for p in snap["compute_processes"] if p["pid"] not in allowed_pids]
                record({"event":"worker_resource_sample","worker_pid":worker.pid,"foreign_compute":foreign,"snapshot":snap})
                if foreign:interrupted="foreign_compute"
                elif snap["gpu"]["free_mib"]<args.reserve_gib*1024:interrupted="gpu_reserve_violated"
                elif time.monotonic()-started>=args.maximum_walltime_s:interrupted="walltime_cap"
                if interrupted:
                    # ONLY the new session/process group created for our own
                    # worker (and its descendants), never any NVIDIA foreign PID.
                    try:os.killpg(worker.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:worker.wait(timeout=20.)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(worker.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        worker.wait()
                    record({"event":"own_worker_paused","reason":interrupted,"worker_pid":worker.pid})
                    break
                time.sleep(3.)
            state["attempts"][-1].update(returncode=worker.returncode,interruption=interrupted)
            if interrupted is None:
                state["status"]="complete" if worker.returncode==0 else "worker_failed"
                record({"event":"worker_exit","returncode":worker.returncode});return worker.returncode
            state["status"]="paused_"+interrupted;record({"event":"checkpoint_retained","reason":interrupted})
        if interrupted!="foreign_compute" or not args.resume_on_release:return 75
        # Released our cooperative lock and CUDA worker before waiting again.
        time.sleep(3.)
    state["status"]="paused_walltime_cap";record({"event":"walltime_cap"});return 75


if __name__=="__main__":raise SystemExit(main())
