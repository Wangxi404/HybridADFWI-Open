#!/usr/bin/env python3
"""Independent-shot, full-batch Vp FWI around unchanged native P-SV kernels.

Training inputs physically omit unknown true Vp. True Vp is read only by the
separate observation-preparation action; optimization consumes data and priors.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"
import gradient_detail as detail
import numpy as np
import torch
from hybridadfwi.adjoint import hybrid_forward
from hybridadfwi.fwi import (array_hash,lowpass,per_shot_losses,native_no_temporal_graph,
    bounds,project,lbfgs_direction,cap_direction,accepted_curvature)

ROOT=detail.ROOT


def config(path):
    cfg=json.loads(Path(path).read_text())
    for key in ("model","source_model","observations"):
        if key in cfg and not Path(cfg[key]).is_absolute():cfg[key]=str(ROOT/cfg[key])
    if cfg["parameters"]!=["vp"]:raise ValueError("Only Vp is optimized")
    ids=cfg["selected_shots"]
    if sorted(ids)!=list(range(cfg["shots"])):raise ValueError("Every independent shot must be included exactly once")
    if len(set(cfg["shot_x_indices"]))!=cfg["shots"]:raise ValueError("Shot locations must be distinct")
    return cfg


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+".writing.npz");np.savez_compressed(tmp,**arrays);tmp.replace(path)


def prepare_inputs(cfg):
    path=Path(cfg["model"])
    if path.exists():raise FileExistsError(path)
    with np.load(cfg["source_model"]) as source:
        arrays={key:source[key] for key in ("vp_initial","x","z","nx","nz","dx","dz","ox","oz")}
        arrays.update(vs_initial=source["vs_true"],rho_initial=source["rho_true"],
                      vs_true=source["vs_true"],rho_true=source["rho_true"],
                      vs_fixed=source["vs_true"],rho_fixed=source["rho_true"])
    atomic_npz(path,**arrays)
    if "vp_true" in np.load(path).files:raise AssertionError("Unknown Vp must be absent from training input")
    detail.save(path.with_suffix(".json"),{"unknown_true_vp_present":False,
        "initial_vp_policy":"Exact prior sigma60 array; fixed water/buffer reference remains initial",
        "known_vs_density_policy":"Exact supplied true Vs and designated constant2450 density",
        "source_container_sha256":detail.digest(cfg["source_model"]),"training_container_sha256":detail.digest(path),
        "array_sha256":{key:array_hash(value) for key,value in arrays.items()},"script_sha256":detail.digest(__file__)})


def build(cfg,q=None,truth=False):
    actual=dict(cfg)
    if truth:actual["model"]=cfg["source_model"]
    model,prop,coordinates,mask=detail.build_model(actual,truth=truth,trainable=not truth)
    xs=np.asarray(cfg["shot_x_indices"],dtype=np.int64)
    if (xs<5).any() or (xs>prop.nx-6).any():raise ValueError("Sources must lie inside the physical grid")
    prop.source.loc_x=xs.tolist();prop.src_loc=prop.source.get_loc()
    prop.src_x=torch.as_tensor(xs,dtype=torch.long,device=prop.device)
    if not np.array_equal(prop.src_loc[:,0],xs):raise AssertionError("Survey/propagator geometry differs")
    if not truth:
        if q is None:q=np.zeros((prop.nz,prop.nx),dtype=np.float64)
        with torch.no_grad():
            value=torch.as_tensor(q,dtype=torch.float64,device=prop.device)
            reference=torch.as_tensor(np.load(cfg["model"])["vp_initial"],dtype=torch.float64,device=prop.device)
            coordinates["vp"].copy_(reference+cfg["velocity_scale_m_s"]*value)
    return model,prop,coordinates,mask


def field_record(model,mask,dt=None):
    vp,vs,rho=(getattr(model,p).detach().cpu().numpy() for p in ("vp","vs","rho"))
    bulk=rho*(vp*vp-4/3*vs*vs)
    if not all(np.isfinite(v).all() for v in (vp,vs,rho)) or (rho<=0).any() or (bulk<=0).any() or (vs<0).any():
        raise FloatingPointError("Nonphysical material state")
    if dt is not None and vp.max()*dt/model.dx>.5:
        raise FloatingPointError("CFL exceeded")
    result={"physical_fields_sha256":{p:array_hash(getattr(model,p).detach().cpu().numpy()) for p in ("vp","vs","rho")},
            "requires_grad":{p:bool(getattr(model,p).requires_grad) for p in ("vp","vs","rho")}}
    result["bulk_min_pa"]=float(bulk.min())
    if dt is not None:result["max_vp_dt_over_dx"]=float(vp.max()*dt/model.dx)
    if mask is not None:result["P_sha256"]=array_hash(mask.detach().cpu().numpy())
    return result


def trace_screen(pressure,dt):
    t=np.arange(pressure.shape[1])*dt
    early=np.max(np.abs(pressure[:,t<1]),axis=(1,2))
    late=np.max(np.abs(pressure[:,t>=3]),axis=(1,2)) if (t>=3).any() else np.zeros_like(early)
    ratios=late/np.maximum(early,1e-300)
    return {"all_finite":bool(np.isfinite(pressure).all()),"late_over_first_second_max":ratios.tolist(),
            "passed":bool(np.isfinite(pressure).all() and (ratios<=.1).all()),
            "scope":"Heuristic screen for the previously observed large late-time growth, not continuum/interface validation"}


def observations(cfg,path):
    model,prop,_,_=build(cfg,truth=True)
    ids=cfg["selected_shots"]
    output=native_no_temporal_graph(prop,ids,cfg["segments"])
    pressure=detail.pressure(output).detach()
    window=torch.cat([detail.data_window(prop,cfg,shot,torch) for shot in ids],dim=0)
    scales=[]
    for stage in cfg["stages"]:
        filtered=lowpass(pressure,prop.dt,stage["cutoff_hz"],cfg["filter_transition_hz"])
        scales.append((filtered*window).square().mean(dim=(1,2)).sqrt().cpu().numpy())
    arrays={"pressure":pressure.cpu().numpy(),"window":window.cpu().numpy(),"scales":np.asarray(scales),
            "shot_ids":np.asarray(ids,dtype=np.int64),"source_x_m":prop.ox+prop.src_x.cpu().numpy()*prop.dx,
            "source_z_m":prop.oz+prop.src_z.cpu().numpy()*prop.dz,
            "receiver_x_m":prop.ox+prop.rcv_x.cpu().numpy()*prop.dx,"receiver_z_m":prop.oz+prop.rcv_z.cpu().numpy()*prop.dz}
    if not np.isfinite(arrays["pressure"]).all() or (arrays["scales"]<=0).any():raise FloatingPointError("Invalid observations/normalization")
    atomic_npz(path,**arrays)
    detail.save(Path(path).with_suffix(".json"),{"shot_ids":ids,"source_encoding":False,"source_batch_dimension_is_independent":True,
        "arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"sha256":array_hash(v)} for k,v in arrays.items()},
        "model":field_record(model,None,cfg["dt"]),"filter_order":"Low-pass raw residual first, then fixed known-water time/offset window; per-stage/per-shot observed RMS fixed before optimization",
        "trace_screen":trace_screen(arrays["pressure"],cfg["dt"]),
        "source_sha256":detail.digest(__file__),"source_model_sha256":detail.digest(cfg["source_model"])})


def evaluate(cfg,prop,model,coordinates,mask,q,ids,stage,observed,method="hybrid",store_pressure=True):
    reference=torch.as_tensor(np.load(cfg["model"])["vp_initial"],dtype=torch.float64,device=prop.device)
    with torch.no_grad():coordinates["vp"].copy_(reference+cfg["velocity_scale_m_s"]*torch.as_tensor(q,dtype=torch.float64,device=prop.device))
    model.zero_grad(set_to_none=True);gc.collect()
    if cfg["device"]=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    if method=="hybrid":out,tape=hybrid_forward(prop,fd_order=8,checkpoint_segments=cfg["segments"],shot_index=ids)
    elif method=="autograd":out=detail.native(prop,cfg,ids) if isinstance(ids,int) else native_ad(prop,cfg,ids)
    else:out=native_no_temporal_graph(prop,ids,cfg["segments"])
    if cfg["device"]=="cuda":torch.cuda.synchronize()
    forward_end=time.perf_counter()
    if method=="hybrid":
        txx,tzz=(out[key].detach().requires_grad_(True) for key in ("txx","tzz"));prediction=-.5*(txx+tzz)
    else:prediction=detail.pressure(out)
    index=np.asarray(ids,dtype=np.int64)
    target=torch.as_tensor(observed["pressure"][index],device=prop.device,dtype=torch.float64)
    window=torch.as_tensor(observed["window"][index],device=prop.device,dtype=torch.float64)
    scale=torch.as_tensor(observed["scales"][stage,index,None,None],device=prop.device,dtype=torch.float64)
    losses=per_shot_losses(prediction,target,window,scale,prop.dt,cfg["stages"][stage]["cutoff_hz"],cfg["filter_transition_hz"])
    total=losses.sum()  # EACH per-shot mean has Nt*Nr denominator; no batch/shot mean here.
    arrays={}
    if method=="hybrid":
        sx,sz=torch.autograd.grad(total,(txx,tzz));gradient=tape.backward({"txx":sx,"tzz":sz})["vp"]
        coordinate=mask*gradient
        arrays.update(gradient_physical_sum=gradient.detach().cpu().numpy(),gradient_coordinate_sum=coordinate.detach().cpu().numpy(),
                      gradient_q_sum=(cfg["velocity_scale_m_s"]*coordinate).detach().cpu().numpy())
    elif method=="autograd":
        total.backward();gradient=model.vp.grad;coordinate=coordinates["vp"].grad
        arrays.update(gradient_physical_sum=gradient.detach().cpu().numpy(),gradient_coordinate_sum=coordinate.detach().cpu().numpy(),
                      gradient_q_sum=(cfg["velocity_scale_m_s"]*coordinate).detach().cpu().numpy())
    if cfg["device"]=="cuda":torch.cuda.synchronize()
    end=time.perf_counter()
    if store_pressure:arrays["predicted_pressure"]=prediction.detach().cpu().numpy()
    record={"shot_ids":list(ids),"stage":stage,"cutoff_hz":cfg["stages"][stage]["cutoff_hz"],"per_shot_loss":losses.detach().cpu().tolist(),
        "loss_sum":float(total.detach()),"q_array_sha256":array_hash(q),**field_record(model,mask,cfg["dt"]),
        "per_shot_scale":scale.detach().cpu().flatten().tolist(),"window_sha256":array_hash(window.cpu().numpy()),
        "timing":{"forward_s":forward_end-begin,"data_vjp_and_backward_s":end-forward_end,"total_s":end-begin},
        "cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if cfg["device"]=="cuda" else None,
        "cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if cfg["device"]=="cuda" else None,
        "prediction_sha256":array_hash(prediction.detach().cpu().numpy()),"method":method,
        "trace_screen":trace_screen(prediction.detach().cpu().numpy(),cfg["dt"]),
        "objective_reduction":"Sum of independent per-shot Nt*Nr means; final full-batch optimizer divides sum by Nshots once"}
    if not np.isfinite(record["per_shot_loss"]).all() or any(not np.isfinite(a).all() for a in arrays.values()):raise FloatingPointError("Nonfinite batch")
    if method=="hybrid":record["adapter_details"]=dict(tape.stats)
    return record,arrays


def native_ad(prop,cfg,ids):
    from hybridadfwi.adjoint import preserve_padding_dtype
    with preserve_padding_dtype():return prop.forward(fd_order=8,checkpoint_segments=cfg["segments"],shot_index=list(ids))


def profile(cfg,output):
    output.mkdir(parents=True,exist_ok=False)
    detail.save(output/"protocol.json",cfg)
    observations(cfg,output/"observations.npz")
    print("PROFILE_OBSERVATIONS_DONE",flush=True)
    if not json.loads((output/"observations.json").read_text())["trace_screen"]["passed"]:
        raise RuntimeError("New source geometry failed true-model trace screen")
    observed=dict(np.load(output/"observations.npz"))
    model,prop,coordinates,mask=build(cfg)
    q=np.zeros((prop.nz,prop.nx),dtype=np.float64)
    rows=[]
    initial,initial_arrays=evaluate(cfg,prop,model,coordinates,mask,q,cfg["selected_shots"],0,observed,"loss",False)
    rows.append(initial);detail.save(output/"initial_all_shot_screen.json",initial)
    print(json.dumps({"profile_initial_loss9":initial["timing"],"peak_GiB":initial["cuda_peak_allocated_bytes"]/2**30,
                      "trace_screen_passed":initial["trace_screen"]["passed"]}),flush=True)
    if not initial["trace_screen"]["passed"]:raise RuntimeError("New source geometry failed initial trace screen")
    for count in (2,3):
        if count==3 and cfg["device"]=="cuda":
            total=torch.cuda.get_device_properties(0).total_memory
            estimate=max(r["cuda_peak_reserved_bytes"] for r in rows if len(r["shot_ids"])==2)*1.5+512*1024**2
            if estimate>total-cfg["gpu_reserve_bytes"]:
                detail.save(output/"profile.json",{"complete":False,"skipped_batch3_predicted_allocation":estimate,"gpu_total_bytes":total,"rows":rows})
                return
        if cfg["device"]=="cuda":torch.cuda.empty_cache()
        ids=[0,4] if count==2 else [0,4,8]
        for repeat in range(2):
            record,arrays=evaluate(cfg,prop,model,coordinates,mask,q,ids,0,observed,"hybrid",True)
            record["repeat"]=repeat;rows.append(record)
            if repeat==1:
                atomic_npz(output/("hybrid_batch"+str(count)+"_warm.npz"),**arrays)
                detail.save(output/("hybrid_batch"+str(count)+"_warm.json"),record)
            print(json.dumps({"profile_batch":count,"repeat":repeat,**record["timing"],"peak_GiB":record["cuda_peak_allocated_bytes"]/2**30}),flush=True)
            detail.save(output/"profile.json",{"complete":False,"rows":rows})
    detail.save(output/"profile.json",{"complete":True,"rows":rows,"source_sha256":detail.digest(__file__)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare-inputs","profile","observations"))
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--output",type=Path)
    args=parser.parse_args();cfg=config(args.config)
    if args.command=="prepare-inputs":prepare_inputs(cfg);return
    detail.initialize(cfg)
    if args.command=="observations":observations(cfg,args.output)
    else:profile(cfg,args.output)


if __name__=="__main__":main()
