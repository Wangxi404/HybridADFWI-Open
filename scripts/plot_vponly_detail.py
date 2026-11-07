#!/usr/bin/env python3
"""Plot fresh known-Vs Vp derivatives, with separate physical/coordinate evidence.

Style-only inheritance from the existing scientific report. Every displayed
gradient comes from its recorded numerical bundle, without smoothing or gain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR","/tmp/hybridadfwi-vponly-mpl")
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/figure_qa"))
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
    "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"xtick.labelsize":8,"ytick.labelsize":8,
    "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.65,
    "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white","image.composite_image":False})


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def export(fig,axes,bars,output,name):
    fig.canvas.draw()
    for image,ax,label,ticks in bars:
        box=ax.get_position(); h=fig.get_figheight()*72
        cax=fig.add_axes([box.x0,box.y0-35/h,box.width,5/h])
        cb=fig.colorbar(image,cax=cax,orientation="horizontal",ticks=ticks,
                       format=mpl.ticker.FuncFormatter(lambda v,p:f"{v:.2g}"))
        cb.ax.tick_params(labelsize=7,pad=2);cb.set_label(label,fontsize=7,labelpad=3)
    fig.canvas.draw()
    labels=np.array(list("abcdef")).reshape(2,3)
    require_matplotlib_panel_alignment(fig,axes=axes.ravel().tolist(),panel_ids=labels.ravel().tolist(),
        row_groups=labels.tolist(),column_groups=labels.T.tolist(),tolerance_pt=1.5,gutter_tolerance_pt=1.5,
        json_out=output/"qa"/(name+".alignment.json"),overlay_svg=output/"qa"/(name+".alignment.svg"),strict=True)
    fig.savefig(output/(name+".pdf"),dpi=300,bbox_inches="tight",pad_inches=.1)
    fig.savefig(output/(name+".png"),dpi=300,bbox_inches="tight",pad_inches=.1)
    fig.savefig(output/(name+".svg"),dpi=300,bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def require_acceptance(evidence):
    summary=json.loads((evidence/"summary.json").read_text())
    if not summary["passed"]:
        acceptance=json.loads((evidence/"scoped_acceptance.json").read_text())
        if (not acceptance.get("coordinate_gradient_passed") or
            acceptance.get("raw_kernels_sha256")!=sha(evidence/"raw_kernels.npz") or
            acceptance.get("summary_sha256")!=sha(evidence/"summary.json")):
            raise ValueError("Missing hash-bound declared-coordinate acceptance; full gate remains failed")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence",type=Path,required=True)
    parser.add_argument("--previous-evidence",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/figures/vponly")
    parser.add_argument("--percentile",type=float,default=99.)
    args=parser.parse_args()
    if not 95<=args.percentile<=100: raise ValueError("Declare display percentile")
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/"qa").mkdir(exist_ok=True)
    summary=require_acceptance(args.evidence)
    old_summary=require_acceptance(args.previous_evidence)
    cfg=summary["config"]
    if cfg["parameters"]!=["vp"]: raise ValueError("Not a Vp-only run")
    target_file=args.evidence/"target_identity.json"
    if target_file.exists():
        target=json.loads(target_file.read_text())
        if not all(v["uint64_equal"] for v in target["checks"].values()): raise ValueError("Targets differ")
        target_check={"method":"Archived direct old/new full-array uint64 comparison","passed":True}
    else:
        previous_proof=json.loads((args.previous_evidence/"pressure_proof.json").read_text())
        fresh_proof=json.loads((args.evidence/"pressure_proof.json").read_text())
        hashes_equal=all(previous_proof["full_array_byte_hashes"][key]["sha256_c_contiguous_bytes"]==
                         fresh_proof["full_array_byte_hashes"][key]["sha256_c_contiguous_bytes"]
                         for key in ("observed_pressure","window"))
        scale_equal=np.float64(previous_proof["fixed_observed_rms"]).tobytes()==np.float64(fresh_proof["fixed_observed_rms"]).tobytes()
        if not hashes_equal or not scale_equal:
            raise ValueError("Replay target hashes/RMS differ; generate a same-environment prior reference target before claiming matched old/new targets")
        target_check={"method":"Public full-array SHA proofs plus scalar RMS byte comparison; not direct rereading of the old full arrays","passed":True}
    quality=json.loads((args.evidence/"trace_screen.json").read_text())
    if not quality["passed"]: raise ValueError("Trace screen failed")
    source=np.load(args.evidence/"raw_kernels.npz",allow_pickle=False)
    old=np.load(args.previous_evidence/"raw_kernels.npz",allow_pickle=False)
    model=np.load(args.model,allow_pickle=False)
    model_manifest=json.loads((args.evidence/"model_manifest.json").read_text())
    for meaning in ("truth","initial"):
        for p in ("vp","vs","rho"):
            key=p+("_true" if meaning=="truth" else "_initial")
            if array_sha(model[key])!=model_manifest[meaning]["fields"][p]["sha256_float64_c_bytes"]:
                raise ValueError("Local model fields differ from actually executed fields")
    x,z=source["x_m"]/1000,source["z_m"]/1000
    if not np.array_equal(model["x"]/1000,x) or not np.array_equal(model["z"]/1000,z): raise ValueError("Grid mismatch")
    if not np.allclose(np.diff(x),np.diff(x)[0],rtol=0,atol=1e-12) or not np.allclose(np.diff(z),np.diff(z)[0],rtol=0,atol=1e-12):
        raise ValueError("Image export requires the actual uniform native grid")
    dx,dz=np.diff(x)[0],np.diff(z)[0]
    extent=(x[0]-dx/2,x[-1]+dx/2,z[-1]+dz/2,z[0]-dz/2)
    mask=source["active_mask"]; active=mask>0;core=mask>.999
    if not np.array_equal(mask,old["active_mask"]): raise ValueError("P maps differ")
    gradients={(method,meaning):source[f"gradient_{method}_{'coordinate_' if meaning=='coordinate' else ''}vp"]
               for method in ("autograd","hybrid") for meaning in ("physical","coordinate")}
    for method in ("autograd","hybrid"):
        if not np.array_equal(gradients[method,"coordinate"],mask*gradients[method,"physical"]):
            raise ValueError("Saved actual coordinate gradient fails P chain")
    records={}
    def decorate(ax,title):
        ax.set(xlim=(x[0],x[-1]),ylim=(z[-1],z[0]),xlabel="Original x (km)",ylabel="Depth (km)")
        ax.set_aspect("equal",adjustable="box");ax.set_title(title,loc="left",pad=9)
    def draw(ax,g,title,limit,label,selected=None,scale_definition="P99 declared region"):
        shown=g if selected is None else np.ma.masked_where(~selected,g)
        image=ax.imshow(shown,extent=extent,origin="upper",interpolation="none",cmap="RdBu_r",vmin=-limit,vmax=limit)
        ax.set_facecolor("#e6e9ed");decorate(ax,title)
        selected=np.ones_like(active) if selected is None else selected
        records[title]={"linear_symmetric_limit":limit,"scale_definition":scale_definition,
            "saturated_fraction":float(np.mean(np.abs(g[selected])>limit)),"values_modified":False,
            "display_mask":"actual predeclared coordinate support" if selected is active else "full physical grid"}
        return image,ax,label,[-limit,0,limit]
    fig,axes=plt.subplots(2,3,figsize=(10.3,9.8));bars=[]
    vptrue,initial,fixedvs=(model[k]/1000 for k in ("vp_true","vp_initial","vs_true"))
    lo,hi=min(vptrue.min(),initial.min()),max(vptrue.max(),initial.max())
    for col,(value,title,label,vmin,vmax) in enumerate(((vptrue,"a  True Vp","Vp (km/s)",lo,hi),
         (initial,"b  Actual initial Vp","Vp (km/s)",lo,hi),
         (fixedvs,"c  Fixed true Vs (known prior)","Vs (km/s)",fixedvs.min(),fixedvs.max()))):
        image=axes[0,col].imshow(value,extent=extent,origin="upper",interpolation="none",cmap="viridis",vmin=vmin,vmax=vmax)
        decorate(axes[0,col],title);bars.append((image,axes[0,col],label,[vmin,(vmin+vmax)/2,vmax]))
    oldg=old["gradient_hybrid_coordinate_vp"]
    newg=gradients["hybrid","coordinate"]
    shared_limit=float(np.percentile(np.abs(np.concatenate((oldg[core],newg[core]))),args.percentile))
    for col,g,title in ((0,oldg,"d  Previous: blurred Vp and Vs"),(1,newg,"e  Fresh: blurred Vp, true Vs")):
        bars.append(draw(axes[1,col],g,title,shared_limit,"dJ/dθVp (objective / [m/s])",active,
                         f"Shared P{args.percentile:g} of concatenated old/new fully active |gradient|"))
    delta=vptrue-initial;limit=float(np.max(np.abs(delta)))
    bars.append(draw(axes[1,2],delta,"f  Input ΔVp (not a gradient)",limit,"True − initial Vp (km/s)",scale_definition="Full absolute maximum"))
    fig.suptitle("Known-Vs controlled experiment: fresh Vp sensitivity\n"
        "Single gradient, not an inversion · shared linear P99 scale for old/new derivatives",y=.99,fontsize=11)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.115,wspace=.43,hspace=.53)
    export(fig,axes,bars,output,"models_and_vp_gradient")
    fig,axes=plt.subplots(2,3,figsize=(10.3,9.8));bars=[]
    for row,meaning in enumerate(("physical","coordinate")):
        a,h=gradients["autograd",meaning],gradients["hybrid",meaning]
        selection=None if meaning=="physical" else active
        scale_region=np.ones_like(active) if selection is None else core
        limit=float(np.percentile(np.abs(np.concatenate((a[scale_region],h[scale_region]))),args.percentile))
        for col,g,method in ((0,a,"Native AD"),(1,h,"Hybrid")):
            title=f"{'abcdef'[row*3+col]}  {method}: {'physical gVp' if row==0 else 'actual PᵀgVp'}"
            bars.append(draw(axes[row,col],g,title,limit,"Objective / [m/s]",selection))
        diff=h-a;error_limit=float(np.max(np.abs(diff)))
        if error_limit<=0: error_limit=np.finfo(np.float64).tiny
        title=f"{'cf'[row]}  Hybrid − AD: {meaning}"
        label=f"Objective / [m/s]\nmax |Δg| = {error_limit:.2e}"
        bars.append(draw(axes[row,2],diff,title,error_limit,label,selection,"Full absolute maximum of difference"))
    gate="Full physical and coordinate gates passed" if summary["passed"] else "Full physical gate failed; only declared-coordinate scope accepted"
    fig.suptitle("Fresh Vp-only AD / Hybrid derivatives\n"+gate,y=.99,fontsize=11)
    fig.subplots_adjust(left=.075,right=.975,top=.91,bottom=.115,wspace=.43,hspace=.53)
    export(fig,axes,bars,output,"ad_hybrid_vp_gradients")
    manifest={"status":"Rendered; final PDF and panel-by-panel QA pending","backend":"python",
        "model_container_sha256":sha(args.model),"model_fields_match_executed_manifest":True,
        "model_container_identity_claimed":False,"gradient_bundle_sha256":sha(args.evidence/"raw_kernels.npz"),
        "summary_sha256":sha(args.evidence/"summary.json"),"previous_bundle_sha256":sha(args.previous_evidence/"raw_kernels.npz"),
        "target_identity_verification":target_check,
        "plotter_sha256":sha(__file__),"strict_new_gate_passed":summary["passed"],
        "strict_previous_gate_passed":old_summary["passed"],"display_scales":records,
        "gradient_postprocessing":"none; no gain, filtering, smoothing or model texture; only manifested display saturation",
        "image_export":"Uniform native-grid cells rendered with imshow interpolation=none and half-cell edge extent; no resampling/smoothing. Avoids mixed rasterized-QuadMesh export clipping of axes text.",
        "comparison":"Targets/window/RMS identical; initial Vs differs and is known exactly only in new case; density designated truth constant2450. Main old/new coordinate derivatives share P99 of concatenated fully active values; AD/H share their row scales.",
        "scope":"A controlled known-Vs single-gradient demonstrator; not recovered Vp, not blind inversion, not independent continuum/interface validation"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main()
