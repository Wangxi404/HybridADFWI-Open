#!/usr/bin/env python3
"""Plot actual accepted full-shot iterates; true Vp is post-hoc evidence only.

This offline reader never calls the optimizer or a wave solver. It exports
physical models, raw globally accumulated derivatives, search directions and
actual accepted changes as distinct quantities. No gradient filtering/gain.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/seismic-fullshot-figures")
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/figure_qa"))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .65,
    "pdf.fonttype": 42, "svg.fonttype": "none", "image.composite_image": False,
    "savefig.facecolor": "white"})


def read(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_digest(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def arrays(path):
    with np.load(path, allow_pickle=False) as values: return {key: values[key].copy() for key in values.files}


def load_hybrid(run, truth_path, training_path=None):
    protocol = read(run / "protocol.json"); cfg = protocol["config"]
    training_path = Path(training_path or cfg["model"])
    if digest(training_path) != protocol["training_input_sha256"]:
        raise ValueError("Training container differs from the executed protocol")
    training = arrays(training_path); truth = arrays(truth_path)
    if "vp_true" in training: raise ValueError("Training file contains unknown true Vp")
    for field in ("vs", "rho"):
        if not np.array_equal(training[field + "_fixed"], truth[field + "_true"]):
            raise ValueError("Post-hoc truth has different known priors")
    mapping = arrays(run / "models/map.npz")
    mask = mapping["P"]
    for key, map_key in (("x","x_m"),("z","z_m")):
        if not np.array_equal(training[key],truth[key]) or not np.array_equal(training[key],mapping[map_key]):
            raise ValueError("Post-hoc truth/training/map coordinates differ")
    initial = arrays(run / "models/vp_initial.npz")["vp"]
    if not np.array_equal(initial, training["vp_initial"]): raise ValueError("Initial checkpoint differs from training input")
    commits = sorted((read(path) for path in (run / "commits").glob("*.json")), key=lambda row: row["state_after"]["accepted_total"])
    if not commits: raise ValueError("No accepted full-shot iteration yet")
    trajectory = [initial]; history = []; paths = []
    for expected, commit in enumerate(commits, 1):
        state = commit["state_after"]
        if state["accepted_total"] != expected or commit["accepted_count_this_iteration"] != 1:
            raise ValueError("Accepted model sequence is incomplete")
        folder = run / "iterations" / commit["iteration_id"]
        vp_file = run / "models" / f"vp_{expected:04d}.npz"
        vp = arrays(vp_file)["vp"]
        q = np.load(run/state["current_q"],allow_pickle=False)
        if array_digest(q) != state["current_q_sha256"]: raise ValueError("Accepted q checkpoint changed")
        proof = commit["candidate_proof"]
        if proof["model_sha256"] != state["model_sha256"]:
            raise ValueError("Accepted model differs from its full-shot candidate")
        for batch in proof["proof"]:
            meta = read(run/batch["metadata"])
            if (meta["q_array_sha256"] != array_digest(q) or meta["P_sha256"] != array_digest(mask)
                or meta["physical_fields_sha256"]["vp"] != array_digest(vp)):
                raise ValueError("Accepted fields, q or map differ from the actually evaluated model")
        trajectory.append(vp)
        history.append(dict(accepted=expected, stage=state["stage"],
            loss_before=commit["loss_before"], loss_after=commit["loss_after"],
            max_delta_vp_m_s=commit["maximum_actual_delta_vp_m_s"]))
        paths.append(folder)
    first_g = arrays(paths[0] / "raw_gradient_sums.npz")["gradient_coordinate_sum"]
    last_g = arrays(paths[-1] / "raw_gradient_sums.npz")["gradient_coordinate_sum"]
    direction_q = arrays(paths[-1] / "optimizer_direction.npz")["direction_q"]
    # Linear static map: this is the search direction, before projection/cap/alpha.
    direction = cfg["velocity_scale_m_s"] * mask * direction_q
    update = arrays(paths[-1] / "actual_update.npz")["delta_vp"]
    if not np.array_equal(update, trajectory[-1] - trajectory[-2]): raise ValueError("Recorded update differs from accepted checkpoint change")
    stage_labels = {i: (f"{s['cutoff_hz']:g} Hz low-pass" if s["cutoff_hz"] else f"Full {cfg['f0']:g} Hz source") for i, s in enumerate(cfg["stages"])}
    return dict(method="HybridADFWI", truth=truth["vp_true"], initial=initial,
        trajectory=trajectory, x=training["x"], z=training["z"], mask=mask,
        gradient_initial=first_g, gradient_last=last_g, direction=direction, update=update,
        history=history, stage_labels=stage_labels, shots=cfg["shots"],
        state=read(run / "state.json"), gradient_label="Σ shots dJₛ/dθVp (objective / [m/s])",
        geometry=f"{cfg['shots']} independent sources · known true Vs and density · Vp only",
        sources={"protocol_sha256": digest(run / "protocol.json"), "truth_sha256_posthoc_only": digest(truth_path),
            "training_input_sha256":digest(training_path),"ledger_sha256": digest(run / "iterations.jsonl"),
            "first_gradient_container_sha256":digest(paths[0]/"raw_gradient_sums.npz"),
            "last_gradient_container_sha256":digest(paths[-1]/"raw_gradient_sums.npz"),
            "accepted_direction_container_sha256":digest(paths[-1]/"optimizer_direction.npz")})


def load_native(run, prepared, truth_path):
    protocol = read(run / "protocol_copy.json"); training = arrays(prepared / "training_input.npz")
    truth = arrays(truth_path)
    if "vp_true" in training or "vp" in training: raise ValueError("Unknown true Vp is present in training input")
    if digest(prepared / "training_input.npz") != protocol["training_input_sha256"]:
        raise ValueError("Training file differs from the executed protocol")
    if digest(truth_path) != protocol["truth_file_for_observations_only_sha256"]:
        raise ValueError("Post-hoc truth is not the observation model")
    if not all(np.array_equal(training[field+"_fixed"],truth[field]) for field in ("vs","rho")):
        raise ValueError("Known priors differ between training and post-hoc reference")
    if not all(np.array_equal(training[key],truth[key]) for key in ("x_native","z_native")):
        raise ValueError("Native truth/training grids differ")
    # Native z points upward; reverse the actual parameter-grid rows for depth.
    flip = lambda value: value[::-1].copy()
    initial = training["vp_initial"]
    commits = sorted((read(path) for path in (run/"epochs").glob("*/accepted.json")),key=lambda row:row["next_state"]["updates"])
    if not commits: raise ValueError("No accepted full-shot native iteration yet")
    trajectory = [flip(initial)]; history=[]; folders=[]
    stages = {stage["id"]:stage for stage in protocol["stages"]}
    for expected, commit in enumerate(commits,1):
        state = commit["next_state"]
        if state["updates"] != expected: raise ValueError("Accepted native model sequence is incomplete")
        statefile = run/state["array_path"]
        if digest(statefile) != state["array_sha256"]: raise ValueError("Accepted native checkpoint changed")
        vp = arrays(statefile)["vp"]; trajectory.append(flip(vp))
        folder = run/"epochs"/commit["iteration_id"]; folders.append(folder)
        stage = commit["iteration_id"].rsplit("_e",1)[0]
        history.append(dict(accepted=expected,stage=stage,loss_before=commit["loss_before"],
            loss_after=commit["loss_after"],max_delta_vp_m_s=commit["decision"]["step_linf_m_s"]))
    if (run/"visual_map.npz").exists(): mask = arrays(run/"visual_map.npz")["active_mask"]
    else:
        # The adapter's native tomography parameters are the solid parameter grid.
        if not np.all(training["vs_fixed"]>0): raise ValueError("An explicit active map is required when the native parameter grid includes water")
        mask = np.ones_like(initial)
    first = arrays(folders[0]/"gradient/aggregate.npz")["gradient_sum"]
    last = arrays(folders[-1]/"gradient/aggregate.npz")["gradient_sum"]
    direction = arrays(folders[-1]/"direction.npz")["direction"]
    ledger = run/("optimizer_ledger.jsonl" if protocol.get("unit") else "ledger.jsonl")
    ledger_sources = {"ledger_sha256":digest(ledger)}
    if protocol.get("unit"):
        ledger_sources["separate_derivative_qualification_ledger_sha256"] = digest(run/"ledger.jsonl")
    return dict(method="DiffSPECFEM P–SV",truth=flip(truth["vp"]),initial=flip(initial),
        trajectory=trajectory,x=training["x_native"]+protocol["original_x_origin_m"],
        z=(protocol["bottom_depth_m"]-training["z_native"])[::-1],mask=flip(mask),
        gradient_initial=flip(first),gradient_last=flip(last),direction=flip(direction),
        update=trajectory[-1]-trajectory[-2],history=history,
        stage_labels={key:f"{stage['f0']:g} Hz native {key}" for key,stage in stages.items()},
        shots=len(protocol["shot_ids"]),state=read(run/"state.json"),
        gradient_label="Σ shots dJₛ/dVp (objective / [m/s])",
        geometry=f"{len(protocol['shot_ids'])} independent sources · known Vs and density · separate reflecting ROI",
        sources={"protocol_sha256":digest(run/"protocol_copy.json"),"truth_sha256_posthoc_only":digest(truth_path),
            "training_input_sha256":digest(prepared/"training_input.npz"),
            "first_gradient_container_sha256":digest(folders[0]/"gradient/aggregate.npz"),
            "last_gradient_container_sha256":digest(folders[-1]/"gradient/aggregate.npz"),
            "accepted_direction_container_sha256":digest(folders[-1]/"direction.npz"),
            **ledger_sources,"row_transform":"Exact row reversal from upward native z to depth"})


def extent(x, z):
    for value in (x, z):
        if value.ndim != 1 or value.size < 2 or not np.all(np.diff(value) > 0): raise ValueError("Need increasing 1-D coordinates")
        if not np.allclose(np.diff(value), np.diff(value)[0], atol=1e-10, rtol=0): raise ValueError("AxesImage requires the actual uniform grid")
    dx = np.diff(x)[0]; dz = np.diff(z)[0]
    return x[0]-dx/2, x[-1]+dx/2, z[-1]+dz/2, z[0]-dz/2


def export(fig, axes, bars, output, name):
    axes = np.asarray(axes)
    fig.canvas.draw()
    for image, ax, label, ticks in bars:
        box = ax.get_position(); height = fig.get_figheight()*72
        cax = fig.add_axes([box.x0, box.y0-35/height, box.width, 5/height])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=ticks,
            format=mpl.ticker.FuncFormatter(lambda value, position: f"{value:.2g}"))
        cb.ax.tick_params(labelsize=7, pad=2); cb.set_label(label, fontsize=7, labelpad=3)
    fig.canvas.draw()
    ids = np.array(list("abcdefghijklmnopqrstuvwxyz")[:axes.size]).reshape(axes.shape)
    require_matplotlib_panel_alignment(fig, axes=axes.ravel().tolist(), panel_ids=ids.ravel().tolist(),
        row_groups=[row.tolist() for row in ids if len(row)>1],
        column_groups=[col.tolist() for col in ids.T if len(col)>1],
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
        json_out=output/"qa"/(name+".alignment.json"), overlay_svg=output/"qa"/(name+".alignment.svg"))
    # PDF first avoids the Agg/PDF state issue of the previous rasterized meshes.
    fig.savefig(output/(name+".pdf"), dpi=300, bbox_inches="tight", pad_inches=.12)
    fig.savefig(output/(name+".png"), dpi=300, bbox_inches="tight", pad_inches=.12)
    fig.savefig(output/(name+".svg"), dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)


def draw(data, output, percentile):
    output.mkdir(parents=True, exist_ok=True); (output/"qa").mkdir(exist_ok=True)
    truth = data["truth"]; current = data["trajectory"][-1]; initial = data["initial"]
    x, z = data["x"]/1000, data["z"]/1000
    ext = extent(x, z); mask = data["mask"]
    if not all(value.shape == truth.shape for value in (initial, current, mask, data["gradient_initial"], data["gradient_last"], data["direction"], data["update"])):
        raise ValueError("Model/derivative grid mismatch")
    if not np.isfinite(np.stack([truth, initial, current, data["gradient_initial"], data["gradient_last"], data["direction"], data["update"]])).all():
        raise ValueError("Nonfinite figure input")
    active = mask > 0; core = mask > .999
    if not core.any(): raise ValueError("No fully active cells")
    scales = {}; count = len(data["history"])
    def image(ax, value, title, limit, label, diverging=False, region=None, low=None):
        shown = value if region is None else np.ma.masked_where(~region, value)
        lo = -limit if diverging else low
        artist = ax.imshow(shown, origin="upper", extent=ext, interpolation="nearest",
            cmap="RdBu_r" if diverging else "viridis", vmin=lo, vmax=limit)
        ax.set(xlabel="Original x (km)", ylabel="Depth (km)", xlim=(x[0],x[-1]), ylim=(z[-1],z[0]))
        ax.set_aspect("equal", adjustable="box"); ax.set_facecolor("#e9edf0"); ax.set_title(title, loc="left", pad=9)
        selected = np.ones_like(active) if region is None else region
        scales[title] = dict(vmin=float(lo), vmax=float(limit), linear=True, values_modified=False,
            saturated_fraction=float(np.mean((value[selected]<lo)|(value[selected]>limit))),
            masked_count=int((~selected).sum()), total_values=int(value.size))
        ticks = [lo, 0, limit] if diverging else [lo, (lo+limit)/2, limit]
        return artist, ax, label, ticks
    aspect = float((z[-1]-z[0])/(x[-1]-x[0]))
    fig, axes = plt.subplots(2, 3, figsize=(10.3,max(5.9,3.2+4.8*aspect))); bars=[]
    low = min(truth.min(), initial.min(), current.min())/1000
    high = max(truth.max(), initial.max(), current.max())/1000
    for column, (value,title) in enumerate(((truth,"a  True Vp: post-hoc reference"), (initial,"b  Blurred initial Vp"), (current,f"c  Vp after {count} accepted updates"))):
        bars.append(image(axes[0,column],value/1000,title,high,"Vp (km/s)",low=low))
    residuals = [truth-initial, truth-current, current-initial]
    lim = max(float(np.max(np.abs(value))) for value in residuals)/1000
    lim = max(lim, np.finfo(float).tiny)
    for column, (value,title) in enumerate(zip(residuals,("d  True − initial Vp","e  True − current Vp","f  Current − initial Vp"))):
        bars.append(image(axes[1,column],value/1000,title,lim,"ΔVp (km/s)",True))
    fig.suptitle(f"{data['method']}: actual full-shot Vp inversion\n{data['geometry']} · status: {data['state']['status']}",fontsize=10,y=.99)
    fig.subplots_adjust(left=.07,right=.975,bottom=.12,top=.91,wspace=.43,hspace=.55)
    export(fig,axes,bars,output,"model_recovery")

    fig,axes = plt.subplots(2,2,figsize=(7.2,max(5.2,2.3+5.2*aspect)));bars=[]
    shared = float(np.percentile(np.abs(np.concatenate([data["gradient_initial"][core],data["gradient_last"][core]])),percentile))
    shared = max(shared,np.finfo(float).tiny)
    first_stage=data["stage_labels"][data["history"][0]["stage"]]
    last_stage=data["stage_labels"][data["history"][-1]["stage"]]
    for ax,key,title in ((axes[0,0],"gradient_initial",f"a  Initial anchor: {first_stage}"),(axes[0,1],"gradient_last",f"b  Update {count} anchor: {last_stage}")):
        bars.append(image(ax,data[key],title,shared,data["gradient_label"],True,active))
    for ax,key,title,label in ((axes[1,0],"direction","c  Optimizer search direction","Vp direction (m/s; before trial)"),(axes[1,1],"update","d  Actual accepted model change","Accepted ΔVp (m/s)")):
        lim=max(float(np.max(np.abs(data[key][active]))),np.finfo(float).tiny)
        bars.append(image(ax,data[key],title,lim,label,True,active))
    fig.suptitle(f"Accumulated derivative → optimizer direction → accepted update\nRaw derivatives share a linear P{percentile:g} scale; no smoothing or depth gain",fontsize=9,y=.99)
    fig.subplots_adjust(left=.10,right=.97,bottom=.125,top=.91,wspace=.43,hspace=.58)
    export(fig,axes,bars,output,"gradient_direction_update")

    rows=[]
    true_derivative=np.stack(np.gradient(truth,data["z"],data["x"]))[:,core]
    true_derivative_norm=float(np.linalg.norm(true_derivative))
    for index,vp in enumerate(data["trajectory"]):
        derivative=np.stack(np.gradient(vp,data["z"],data["x"]))[:,core]
        derivative_norm=float(np.linalg.norm(derivative))
        rows.append(dict(accepted=index,vp_rmse_core_m_s=float(np.sqrt(np.mean((vp[core]-truth[core])**2))),
            vp_relative_l2_core=float(np.linalg.norm(vp[core]-truth[core])/np.linalg.norm(truth[core])),
            vp_spatial_derivative_relative_l2_core=float(np.linalg.norm(derivative-true_derivative)/true_derivative_norm) if true_derivative_norm>0 else None,
            vp_spatial_derivative_cosine_core=float(np.vdot(derivative,true_derivative)/(derivative_norm*true_derivative_norm)) if derivative_norm*true_derivative_norm>0 else None))
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.0)); palette=["#3576a6","#b17335","#427f65","#796292"]
    stages=list(dict.fromkeys(row["stage"] for row in data["history"]))
    for j,stage in enumerate(stages):
        group=[row for row in data["history"] if row["stage"]==stage]
        xx=[group[0]["accepted"]-1]+[row["accepted"] for row in group]
        yy=np.array([group[0]["loss_before"]]+[row["loss_after"] for row in group])
        if yy[0]<=0: raise ValueError("Nonpositive stage starting loss")
        axes[0].plot(xx,yy/yy[0],color=palette[j%len(palette)],lw=1.5,label=data["stage_labels"][stage])
    axes[0].set(title="a  All-shot data fit within each stage",xlabel="Accepted full-shot updates",ylabel="J / stage-start J",ylim=(0,1.06))
    axes[0].legend(loc="upper center",bbox_to_anchor=(.5,1.35),ncol=2)
    axes[1].plot([row["accepted"] for row in rows],[row["vp_rmse_core_m_s"] for row in rows],color="#3576a6",lw=1.5)
    axes[1].set(title="b  Vp error: post-hoc evaluation",xlabel="Accepted full-shot updates",ylabel="Fully active Vp RMSE (m/s)")
    for ax in axes:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_xlim(0,max(1,count));ax.grid(axis="y",color="#dde2e5",lw=.5)
    fig.subplots_adjust(left=.085,right=.98,bottom=.19,top=.70,wspace=.39)
    export(fig,axes.reshape(1,2),[],output,"data_fit_and_model_error")
    with (output/"trajectory.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(method=data["method"],accepted_updates=count,status=data["state"]["status"],shots=data["shots"],
        plotter_sha256=digest(__file__),current_vp_array_sha256=array_digest(current),
        accepted_vp_array_sha256=[array_digest(vp) for vp in data["trajectory"]],
        model_rmse=rows,history=data["history"],sources=data["sources"],display=scales,
        true_vp_used_only_for_posthoc_evaluation=True,model_error_used_by_optimizer=False,
        interpretation="Data loss is stage-normalized separately; stages have distinct objectives. Epoch caps are not convergence certificates.",
        gradient_scale=f"Shared linear P{percentile:g} of first/last accepted-update anchors' absolute summed coordinate derivative on fully active cells",
        gradient_stage_initial=data["stage_labels"][data["history"][0]["stage"]],
        gradient_stage_latest=data["stage_labels"][data["history"][-1]["stage"]],
        derivative_anchor_accepted_counts={"first":0,"last":count-1},
        terminal_model_gradient_claimed=False,
        gradient_smoothing=False,depth_gain=False,uncertainty="Deterministic single run; no replicate interval claimed",
        image_export="Nearest-neighbor raster rendering at original physical node centers; numerical fields unchanged",
        core_cells=int(core.sum()),active_cells=int(active.sum()),total_cells=int(active.size),
        array_sha256={key:array_digest(data[key]) for key in ("truth","initial","mask","gradient_initial","gradient_last","direction","update")})
    (output/"figure_sources.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True);parser.add_argument("--truth-posthoc",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--percentile",type=float,default=99.)
    parser.add_argument("--format",choices=("hybrid","native"),default="hybrid")
    parser.add_argument("--prepared",type=Path,help="Native prepared experiment directory")
    parser.add_argument("--training-input",type=Path,help="Relocated Hybrid training container; its executed hash must match")
    args=parser.parse_args()
    if not 95<=args.percentile<=100: parser.error("Declare a display percentile between95 and100")
    if args.format=="native" and args.prepared is None: parser.error("Native output needs --prepared")
    data=load_hybrid(args.run,args.truth_posthoc,args.training_input) if args.format=="hybrid" else load_native(args.run,args.prepared,args.truth_posthoc)
    report=draw(data,args.output,args.percentile)
    print(json.dumps({key:report[key] for key in ("method","accepted_updates","status","shots","model_rmse")}))


if __name__=="__main__":main()
