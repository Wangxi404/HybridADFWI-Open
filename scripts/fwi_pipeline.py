#!/usr/bin/env python3
"""A bounded real job: supervised inversion, then independent offline evidence.

This is a normal resumable computation, not a scheduled app automation.
Unknown Vp is passed only to the post-hoc plotting process after optimization.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--truth-posthoc",type=Path,required=True)
    parser.add_argument("--plot-python",type=Path,default=Path(sys.executable),
                        help="Independent NumPy/matplotlib environment; numeric worker retains this pipeline's Python")
    parser.add_argument("--maximum-walltime-s",type=float,default=172800.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    python=sys.executable
    command=[python,str(ROOT/"scripts/fwi_supervisor.py"),"--evidence",str(args.output/"resource_summary.json"),
        "--resume-on-release","--idle-wait-s","600","--maximum-walltime-s",str(args.maximum_walltime_s),
        "--min-free-gib","22","--reserve-gib","2","--",python,"-u",str(ROOT/"scripts/run_multishot_fwi.py"),
        "--config",str(args.config.resolve()),"--observations",str(args.observations.resolve()),"--output",str(args.output.resolve())]
    result=subprocess.run(command)
    # No optimizer/solver is invoked after this point. Close only a genuinely
    # interrupted metadata prefix before auditing the actual accepted commits.
    import run_multishot_fwi as controller
    state_file=args.output/"state.json"
    state=json.loads(state_file.read_text()) if state_file.exists() else {}
    resource=json.loads((args.output/"resource_summary.json").read_text()) if (args.output/"resource_summary.json").exists() else {}
    for commit in (args.output/"commits").glob("*.json") if (args.output/"commits").exists() else []:
        after=json.loads(commit.read_text())["state_after"]
        if after["accepted_total"]>state.get("accepted_total",0):state=after
    if result.returncode and state and state.get("status")=="running":
        state["status"]="failed_worker" if resource.get("status")=="worker_failed" else "paused_"+str(resource.get("status","resource_or_walltime"))
        controller.save(state_file,state)
    dummy=types.SimpleNamespace(output=args.output,ledger=args.output/"iterations.jsonl")
    dummy.event=types.MethodType(controller.Engine.event,dummy)
    if dummy.ledger.exists():controller.repair_prefix(dummy,state)
    ledger_file=args.output/"ledger_audit.json"
    ledger=subprocess.run([python,str(ROOT/"scripts/fwi_ledger_check.py"),str(dummy.ledger)],text=True,capture_output=True)
    try:ledger_report=json.loads(ledger.stdout)
    except ValueError:ledger_report={"passed":False,"error":ledger.stdout+ledger.stderr}
    ledger_file.write_text(json.dumps(ledger_report,indent=2)+"\n")
    count=int(state.get("accepted_total",0));plot_result=None
    if count:
        plot_command=[str(args.plot_python),str(ROOT/"scripts/plot_multishot_fwi.py"),"--format","hybrid","--run",str(args.output.resolve()),
            "--truth-posthoc",str(args.truth_posthoc.resolve()),"--output",str((args.output/"offline_figures").resolve())]
        with (args.output/"offline_plot.log").open("w") as stream:
            plot_result=subprocess.run(plot_command,stdout=stream,stderr=subprocess.STDOUT).returncode
    summary={"optimizer_returncode":result.returncode,"resource_status":resource.get("status"),
        "inversion_status":state.get("status","not_started"),"actual_accepted_updates":count,
        "converged":state.get("status")=="converged","declared_cap_completed":state.get("status")=="complete_iteration_cap",
        "ledger_audit":ledger_report,"offline_plot_returncode":plot_result,
        "numeric_python":python,"plot_python":str(args.plot_python),
        "true_vp_used_only_by_separate_observation_preparation_and_posthoc_evaluation":True,
        "limitation":"A launch, resource pause or iteration cap is not an inversion-convergence certificate"}
    (args.output/"pipeline_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary),flush=True)
    return result.returncode if result.returncode else (0 if ledger_report.get("passed") and (plot_result in (None,0)) else 1)


if __name__=="__main__":raise SystemExit(main())
