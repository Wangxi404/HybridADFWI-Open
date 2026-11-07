#!/usr/bin/env python3
"""Checkpointed all-shot Vp FWI: one accepted update after full reduction.

Run under fwi_supervisor.py. The optimizer never opens an unknown true-Vp
array. Observation preparation and independent true-model evaluation are
separate commands/programs. No native numerical source is modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import marmousi_fwi as native
from hybridadfwi.fwi import (array_hash,bounds,project,physical_vp,candidate,lbfgs_direction,accepted_curvature)
import numpy as np
import torch

ROOT=native.ROOT


def hashed(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def save(path,value):native.detail.save(path,value)
def read(path):return json.loads(Path(path).read_text())


def write_array(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.stem+".writing.npy");np.save(temp,value);temp.replace(path)


def model_hash(record):return hashed(record["physical_fields_sha256"])


def source_files():
    return {name:native.detail.digest(ROOT/name) for name in (
        "scripts/run_multishot_fwi.py","scripts/marmousi_fwi.py","scripts/gradient_detail.py",
        "scripts/fwi_supervisor.py","scripts/fwi_pipeline.py","scripts/fwi_ledger_check.py",
        "src/hybridadfwi/fwi.py","src/hybridadfwi/adjoint.py",
        "src/hybridadfwi/experiment.py","upstream.lock.json")}


class Engine:
    def __init__(self,cfg,observations,output):
        self.cfg=dict(cfg);self.cfg.pop("source_model",None) # Inaccessible to optimizer/build path.
        self.output=Path(output);self.observed=dict(np.load(observations))
        with np.load(cfg["model"]) as f:
            if "vp_true" in f.files:raise ValueError("Training input must physically omit true Vp")
            self.reference=f["vp_initial"];self.vs=f["vs_fixed"];self.rho=f["rho_fixed"]
            self.training_array_hashes={key:array_hash(f[key]) for key in ("vp_initial","vs_fixed","rho_fixed","x","z")}
        self.model,self.prop,self.coordinates,self.mask_gpu=native.build(self.cfg)
        self.mask=self.mask_gpu.detach().cpu().numpy()
        self.reference_gpu=torch.as_tensor(self.reference,dtype=torch.float64,device=self.prop.device)
        self.lo,self.hi=bounds(self.reference,self.vs,self.mask,cfg["velocity_scale_m_s"],cfg["upper_vp_m_s"],cfg["bulk_speed_floor_m_s"])
        self.fixed={"vs":array_hash(self.vs),"rho":array_hash(self.rho)}
        self.signature=hashed({"sources":source_files(),"config":cfg,"training_sha256":native.detail.digest(cfg["model"]),
                              "observations_sha256":native.detail.digest(observations),"torch":torch.__version__})
        self.objective_ids=[hashed({"stage":i,"cutoff_hz":stage["cutoff_hz"],"filter_transition_hz":cfg["filter_transition_hz"],
            "filter_order":"zero-padded FFT residual lowpass then fixed window","window_sha256":array_hash(self.observed["window"]),
            "scales_sha256":array_hash(self.observed["scales"][i]),"target_sha256":array_hash(self.observed["pressure"]),
            "global_normalizer":cfg["shots"],"regularizer":None}) for i,stage in enumerate(cfg["stages"])]
        self.ledger=self.output/"iterations.jsonl"
        self.last_execution_shape=None

    def event(self,event,iteration_id,**values):
        row={"event":event,"iteration_id":iteration_id,"utc":datetime.now(timezone.utc).isoformat(),**values}
        with self.ledger.open("a") as f:
            f.write(json.dumps(row,separators=(",",":"))+"\n");f.flush()

    def assign(self,q):
        with torch.no_grad():
            self.coordinates["vp"].copy_(self.reference_gpu+self.cfg["velocity_scale_m_s"]*torch.as_tensor(q,device=self.prop.device,dtype=torch.float64))
        with torch.enable_grad():self.model.forward() # Exactly the same single native model map as every solve.
        record=native.field_record(self.model,self.mask_gpu,self.cfg["dt"])
        self.check_record(record)
        return record,self.model.vp.detach().cpu().numpy().copy()

    def check_record(self,row):
        for name,value in self.fixed.items():
            if row["physical_fields_sha256"][name]!=value:raise AssertionError("Known field changed: "+name)
        if row["requires_grad"]!={"vp":True,"vs":False,"rho":False}:raise AssertionError("Active parameter flags changed")
        if row["P_sha256"]!=array_hash(self.mask):raise AssertionError("Parameter map changed")

    def batch(self,q,ids,stage,method):
        cache=self.output/"evaluations"/(self.objective_ids[stage][:16]+"_"+array_hash(q)[:20]+"_"+method)
        cache.mkdir(parents=True,exist_ok=True)
        stem=cache/("shots_"+"-".join(map(str,ids)))
        meta=stem.with_suffix(".json");data=stem.with_suffix(".npz")
        reused=False
        if meta.exists() and data.exists():
            row=read(meta)
            if (row.get("signature")==self.signature and row.get("q_array_sha256")==array_hash(q)
                and row.get("objective_id")==self.objective_ids[stage] and row.get("shot_ids")==ids
                and row.get("data_sha256")==native.detail.digest(data)):
                arrays=dict(np.load(data));reused=True
        if not reused:
            shape=(method,len(ids))
            if self.cfg["device"]=="cuda" and shape!=self.last_execution_shape:
                gc.collect();torch.cuda.empty_cache() # Avoid allocator fragmentation between B2/B1 and loss B9.
            self.last_execution_shape=shape
            row,arrays=native.evaluate(self.cfg,self.prop,self.model,self.coordinates,self.mask_gpu,q,ids,stage,self.observed,method,True)
            self.check_record(row)
            row.update(signature=self.signature,objective_id=self.objective_ids[stage])
            native.atomic_npz(data,**arrays);row["data_sha256"]=native.detail.digest(data);save(meta,row)
        self.check_record(row)
        return row,arrays,{"metadata":str(meta.relative_to(self.output)),"data":str(data.relative_to(self.output)),"reused_exact_point":reused}

    def evaluate(self,q,stage,method,iteration_id,evaluation_id,expected_model_sha,normalizer):
        objective=self.objective_ids[stage]
        phase="gradient" if method=="hybrid" else "trial"
        self.event("evaluation_start",iteration_id,evaluation_id=evaluation_id,phase=phase,model_sha256=expected_model_sha,objective_id=objective)
        batch_size=self.cfg["gradient_batch_size"] if method=="hybrid" else self.cfg["shots"]
        # Balanced first gradient batch includes both edges and centre when B=3.
        ids=list(self.cfg["selected_shots"])
        if batch_size==3 and ids==list(range(9)):batches=[[0,4,8],[1,5,7],[2,3,6]]
        else:batches=[ids[i:i+batch_size] for i in range(0,len(ids),batch_size)]
        totals={};per_shot={};proof=[];seen=[]
        for batch_ids in batches:
            row,arrays,paths=self.batch(q,batch_ids,stage,method)
            if model_hash(row)!=expected_model_sha or row["q_array_sha256"]!=array_hash(q):raise AssertionError("A shot evaluated a changed model")
            self.event("shot_batch",iteration_id,evaluation_id=evaluation_id,shot_ids=batch_ids,model_sha256=model_hash(row),
                       q_array_sha256=row["q_array_sha256"],objective_id=objective,**paths)
            seen.extend(batch_ids)
            for shot,value in zip(batch_ids,row["per_shot_loss"]):
                if shot in per_shot:raise AssertionError("Duplicate shot contribution")
                per_shot[shot]=value
            if not row["trace_screen"]["passed"]:
                self.event("evaluation_abort",iteration_id,evaluation_id=evaluation_id,reason="late_amplification_screen")
                return None,None,{"failed":"late_amplification_screen","batch":row,"proof":proof}
            for key in ("gradient_physical_sum","gradient_coordinate_sum","gradient_q_sum"):
                if key in arrays:
                    totals.setdefault(key,np.zeros_like(arrays[key]));totals[key]+=arrays[key]
            proof.append({"shot_ids":batch_ids,"model_sha256":model_hash(row),"q_array_sha256":array_hash(q),**paths})
        if Counter(seen)!=Counter(ids):raise AssertionError("Incomplete or duplicate global shot reduction")
        value=float(sum(per_shot[shot] for shot in ids)/normalizer)
        self.event("evaluation_complete",iteration_id,evaluation_id=evaluation_id,model_sha256=expected_model_sha,normalizer=normalizer,
                   loss=value,objective_id=objective,shot_ids=seen)
        return value,totals,{"per_shot_loss":per_shot,"proof":proof,"normalizer":normalizer,"model_sha256":expected_model_sha}


def load_history(output,state):
    history=[]
    if state.get("history_file"):
        with np.load(output/state["history_file"]) as f:
            for i in range(int(f["count"])):history.append((f["s_"+str(i)],f["y_"+str(i)]))
    return history


def save_history(path,history):
    arrays={"count":np.array(len(history),dtype=np.int64)}
    for i,(s,y) in enumerate(history):arrays.update({"s_"+str(i):s,"y_"+str(i):y})
    native.atomic_npz(path,**arrays)


def initial_direction_check(engine,q,gradient,stage,iteration_id,anchor_record,anchor_vp,options):
    path=engine.output/"initial_direction_check.json"
    if path.exists():
        prior=read(path)
        if prior.get("signature")==engine.signature and prior.get("q_array_sha256")==array_hash(q) and prior.get("passed"):
            return
    direction=gradient/np.linalg.norm(gradient)
    peak=float(np.max(np.abs(physical_vp(engine.reference,engine.mask,q+direction,engine.cfg["velocity_scale_m_s"])-anchor_vp)))
    epsilon=options["physical_perturbation_m_s"]/peak
    while any(np.any(q+sign*epsilon*direction<engine.lo) or np.any(q+sign*epsilon*direction>engine.hi) for sign in (-1,1)):
        epsilon*=.5
        if epsilon<1e-12:raise RuntimeError("No informative feasible initial FD direction")
    expected=float(np.vdot(gradient,direction));pairs=[]
    for scale in options["scales"]:
        h=epsilon*scale;values=[];models=[]
        for sign in (-1,1):
            point=q+sign*h*direction;record,vp=engine.assign(point)
            eid=iteration_id+f"_fd_{scale:g}_{sign:+d}"
            value,_,proof=engine.evaluate(point,stage,"loss",iteration_id,eid,model_hash(record),engine.cfg["shots"])
            if value is None:raise RuntimeError("Initial full-shot FD point failed trace screen")
            values.append(value);models.append({"q_array_sha256":array_hash(point),"physical_model_sha256":model_hash(record),
                "maximum_actual_delta_vp_m_s":float(np.max(np.abs(vp-anchor_vp))),"proof":proof})
        fd=(values[1]-values[0])/(2*h);error=abs(fd-expected)/max(abs(fd),abs(expected),1e-300)
        pairs.append({"epsilon_q":h,"minus_loss":values[0],"plus_loss":values[1],"finite_difference":fd,
            "gradient_dot_direction":expected,"relative_error":error,"models":models})
        save(path,{"passed":False,"status":"in_progress","signature":engine.signature,"q_array_sha256":array_hash(q),"pairs":pairs})
    restored,_=engine.assign(q)
    if model_hash(restored)!=model_hash(anchor_record):raise AssertionError("FD qualification did not restore anchor")
    passed=all(pair["relative_error"]<=options["relative_tolerance"] for pair in pairs)
    save(path,{"passed":passed,"status":"complete","signature":engine.signature,"q_array_sha256":array_hash(q),
        "objective_id":engine.objective_ids[stage],"normalizer":engine.cfg["shots"],"relative_tolerance":options["relative_tolerance"],
        "direction_policy":"Normalized actual globally mean-reduced q gradient; finite h chosen by declared physical0.1m/s perturbation, never true Vp",
        "pairs":pairs})
    if not passed:raise RuntimeError("Initial complete-shot gradient failed declared finite-difference qualification")


def repair_prefix(engine,state):
    if not engine.ledger.exists():return
    raw=engine.ledger.read_bytes();lines=raw.splitlines(keepends=True);events=[];valid=0
    for index,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            if index!=len(lines)-1 or line.endswith(b"\n"):raise ValueError("Malformed non-tail ledger event; fail closed")
            receipt=engine.output/"ledger_repairs";receipt.mkdir(exist_ok=True)
            digest=hashlib.sha256(line).hexdigest();(receipt/(digest+".fragment")).write_bytes(line)
            save(receipt/(digest+".json"),{"scope":"Truncated final JSONL fragment only; prior complete events retained","fragment_bytes":len(line),"fragment_sha256":digest})
            engine.ledger.write_bytes(raw[:valid]);break
        events.append(row);valid+=len(line)
    starts={row["iteration_id"]:row for row in events if row["event"]=="iteration_start"}
    ended={row["iteration_id"] for row in events if row["event"]=="iteration_end"}
    for identifier in starts.keys()-ended:
        current=[row for row in events if row["iteration_id"]==identifier]
        opened={row["evaluation_id"] for row in current if row["event"]=="evaluation_start"}
        closed={row["evaluation_id"] for row in current if row["event"] in ("evaluation_complete","evaluation_abort")}
        for evaluation in opened-closed:engine.event("evaluation_abort",identifier,evaluation_id=evaluation,reason="resumed_interrupted_worker")
        commit=engine.output/"commits"/(identifier+".json")
        if commit.exists():
            artifact=read(commit)
            if not any(row["event"]=="accepted_update" for row in current):engine.event("accepted_update",identifier,**artifact["accepted_event"])
            engine.event("iteration_end",identifier,status="accepted")
        else:engine.event("iteration_end",identifier,status="failed" if str(state.get("status","")).startswith("failed") else "interrupted")


def run(cfg,observations,output,stop_after_accepted=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    native.detail.initialize(cfg)
    engine=Engine(cfg,observations,output)
    protocol=output/"protocol.json"
    if protocol.exists():
        if read(protocol)["signature"]!=engine.signature:raise ValueError("Resume source/config/data fingerprint changed; use a fresh run")
    else:
        (output/"models").mkdir(exist_ok=True)
        native.atomic_npz(output/"models/map.npz",P=engine.mask,
            x_m=engine.prop.ox+np.arange(engine.prop.nx)*engine.prop.dx,z_m=engine.prop.oz+np.arange(engine.prop.nz)*engine.prop.dz)
        native.atomic_npz(output/"geometry.npz",source_x_m=engine.observed["source_x_m"],source_z_m=engine.observed["source_z_m"],
            receiver_x_m=engine.observed["receiver_x_m"],receiver_z_m=engine.observed["receiver_z_m"])
        target_copy=output/"observations.npz"
        if not target_copy.exists():
            try:os.link(observations,target_copy)
            except OSError:shutil.copyfile(observations,target_copy)
        training_copy=output/"training_input.npz"
        if not training_copy.exists():
            try:os.link(cfg["model"],training_copy)
            except OSError:shutil.copyfile(cfg["model"],training_copy)
        save(protocol,{"signature":engine.signature,"config":cfg,"source_hashes":source_files(),"observations_sha256":native.detail.digest(observations),
            "training_input_sha256":native.detail.digest(cfg["model"]),"training_array_hashes":engine.training_array_hashes,"torch_version":torch.__version__,
            "training_has_true_vp":False,"objective_ids":engine.objective_ids,"optimizer":"L-BFGS5 with full-shot projected Armijo",
            "mapping":"coordinate_vp=vp_initial+1000*q; actual Vp=vp_initial+P*(coordinate_vp-vp_initial); gq=1000*P*gVp",
            "reduction":f"Per-shot Nt*Nr mean; batch SUM; one global division by{cfg['shots']}; no regularizer",
            "map_sha256":native.detail.digest(output/"models/map.npz"),"geometry_sha256":native.detail.digest(output/"geometry.npz"),
            "source_encoding":False,"model_error_used_by_optimizer":False})
        (output/"executed_sources").mkdir(exist_ok=True)
        for name in source_files():
            destination=output/"executed_sources"/name.replace("/","__");shutil.copyfile(ROOT/name,destination)
    state_path=output/"state.json"
    if state_path.exists():state=read(state_path)
    else:
        q=np.zeros_like(engine.reference);qpath=output/"models/q_initial.npy";write_array(qpath,q)
        initial,vp=engine.assign(q);native.atomic_npz(output/"models/vp_initial.npz",vp=vp)
        state={"signature":engine.signature,"status":"running","accepted_total":0,"stage":0,"accepted_in_stage":0,
            "current_q":"models/q_initial.npy","current_q_sha256":array_hash(q),"model_sha256":model_hash(initial),
            "history_file":None,"last_gradient":None,"recent":[],"attempt_number":0}
        save(state_path,state)
    # A durable accepted commit outranks a state pointer interrupted during replacement.
    for commit in sorted((output/"commits").glob("*.json")) if (output/"commits").exists() else []:
        after=read(commit)["state_after"]
        if after["accepted_total"]>state["accepted_total"]:state=after
    repair_prefix(engine,state)
    if state.get("status") in ("complete_iteration_cap","converged"):return
    while state["stage"]<len(cfg["stages"]):
        if stop_after_accepted is not None and state["accepted_total"]>=stop_after_accepted:
            state["status"]="paused_requested_accepted_limit";save(state_path,state);return
        stage=state["stage"];stage_cfg=cfg["stages"][stage]
        if state["accepted_in_stage"]>=stage_cfg["epochs"]:
            state.update(stage=stage+1,accepted_in_stage=0,history_file=None,last_gradient=None,recent=[])
            save(state_path,state);continue
        q=np.load(output/state["current_q"])
        if array_hash(q)!=state["current_q_sha256"]:raise ValueError("Accepted q checkpoint changed")
        anchor_record,anchor_vp=engine.assign(q);anchor_sha=model_hash(anchor_record)
        if anchor_sha!=state["model_sha256"]:raise AssertionError("Resumed realized model differs from last accepted model")
        state["attempt_number"]+=1
        save(state_path,state) # Reserve attempt before mkdir/ledger; UUID also prevents interrupted-prefix reuse.
        identifier=f"stage{stage}_accepted{state['accepted_total']:04d}_attempt{state['attempt_number']:04d}_{uuid.uuid4().hex[:10]}"
        folder=output/"iterations"/identifier;folder.mkdir(parents=True,exist_ok=False)
        engine.event("iteration_start",identifier,shot_ids=cfg["selected_shots"],model_sha256=anchor_sha,
                     q_array_sha256=array_hash(q),reduction="mean",objective_id=engine.objective_ids[stage])
        save(state_path,state)
        evaluation=identifier+"_gradient"
        value,totals,gradient_proof=engine.evaluate(q,stage,"hybrid",identifier,evaluation,anchor_sha,cfg["shots"])
        if value is None:
            engine.event("iteration_end",identifier,status="failed");state["status"]="failed_gradient_trace_screen";save(state_path,state);return
        native.atomic_npz(folder/"raw_gradient_sums.npz",**totals)
        gradient=totals["gradient_q_sum"]/cfg["shots"]
        projected=project(q-gradient,engine.lo,engine.hi)-q
        if np.max(np.abs(projected))==0:
            engine.event("iteration_end",identifier,status="converged")
            if stage<len(cfg["stages"])-1:
                state.update(stage=stage+1,accepted_in_stage=0,history_file=None,last_gradient=None,recent=[])
                save(state_path,state);continue
            state.update(status="converged",stopping_reason="Final data-objective projected gradient and feasible step are exactly zero")
            save(state_path,state);return
        if state["accepted_total"]==0 and cfg.get("initial_direction_check",{}).get("enabled"):
            initial_direction_check(engine,q,gradient,stage,identifier,anchor_record,anchor_vp,cfg["initial_direction_check"])
        history=load_history(output,state)
        if state.get("last_gradient"):
            old=state["last_gradient"]
            previous_q=np.load(output/old["q_file"])
            previous_g=np.load(output/old["gradient_file"])["gradient_q_sum"]/cfg["shots"]
            s,y=q-previous_q,gradient-previous_g
            if accepted_curvature(s,y):history.append((s,y));history=history[-cfg["history_size"]:]
        probe=physical_vp(engine.reference,engine.mask,q-gradient,cfg["velocity_scale_m_s"])-anchor_vp
        peak=float(np.max(np.abs(probe)))
        initial_scale=cfg["max_physical_update_m_s"]/max(peak,1e-300)
        direction,lbfgs_valid=lbfgs_direction(gradient,history,initial_scale)
        native.atomic_npz(folder/"initial_optimizer_proposal.npz",direction_q=direction)
        accepted=None;trials=[]
        for fallback in (False,True):
            if fallback:
                if not history:break
                history=[];direction=-gradient*initial_scale
            for backtrack in range(cfg["max_backtracks"]):
                alpha=cfg["trial_shrink"]**backtrack
                trial,unused=candidate(q,direction,alpha,engine.lo,engine.hi,engine.reference,engine.mask,
                    cfg["velocity_scale_m_s"],cfg["max_physical_update_m_s"])
                displacement=trial-q;slope=float(np.vdot(gradient,displacement))
                record,trial_vp=engine.assign(trial);trial_sha=model_hash(record)
                maximum_change=float(np.max(np.abs(trial_vp-anchor_vp)))
                if maximum_change>cfg["max_physical_update_m_s"]*(1+1e-10):raise AssertionError("Actual mapped Vp exceeds cap")
                trial_id=identifier+f"_trial{len(trials):02d}"
                candidate_file=folder/(trial_id+".npy");write_array(candidate_file,trial)
                if slope>=0 or maximum_change==0:
                    trials.append({"evaluation_id":trial_id,"accepted":False,"reason":"non_descent_or_zero_actual_step","g_dot_actual_delta_q":slope});continue
                try:trial_value,_,proof=engine.evaluate(trial,stage,"loss",identifier,trial_id,trial_sha,cfg["shots"])
                except FloatingPointError as error:
                    engine.event("evaluation_abort",identifier,evaluation_id=trial_id,reason="nonfinite_candidate")
                    trial_value=None;proof={"failed":"nonfinite_candidate","error":str(error)}
                rhs=value+cfg["armijo_c1"]*slope
                ok=trial_value is not None and trial_value<=rhs
                entry={"evaluation_id":trial_id,"q_array_sha256":array_hash(trial),"actual_model_sha256":trial_sha,
                    "alpha":alpha,"fallback_steepest":fallback,"loss":trial_value,"anchor_loss":value,"g_dot_actual_delta_q":slope,
                    "armijo_rhs":rhs,"maximum_actual_delta_vp_m_s":maximum_change,"accepted":bool(ok),
                    "restore_model_sha256_if_rejected":anchor_sha,"proof":proof}
                trials.append(entry);save(folder/"trials.json",trials)
                if ok:accepted=(trial,trial_vp,trial_value,trial_sha,trial_id,maximum_change,fallback,alpha);break
                restored,_=engine.assign(q)
                if model_hash(restored)!=anchor_sha:raise AssertionError("Rejected trial did not restore anchor")
            if accepted:break
        if accepted is None:
            engine.event("iteration_end",identifier,status="rejected")
            if stage<len(cfg["stages"])-1:
                state.update(stage=stage+1,accepted_in_stage=0,history_file=None,last_gradient=None,recent=[],status="running")
                save(state_path,state);continue
            state["status"]="stopped_final_line_search_no_acceptable_step";save(state_path,state);return
        trial,vp,new_value,new_sha,trial_id,max_change,fallback_used,accepted_alpha=accepted
        native.atomic_npz(folder/"optimizer_direction.npz",direction_q=direction,
            accepted_alpha=np.array(accepted_alpha),fallback_steepest=np.array(fallback_used))
        count=state["accepted_total"]+1
        new_q=output/f"models/q_{count:04d}.npy";write_array(new_q,trial)
        new_vp=output/f"models/vp_{count:04d}.npz";native.atomic_npz(new_vp,vp=vp)
        native.atomic_npz(folder/"actual_update.npz",delta_q=trial-q,delta_vp=vp-anchor_vp)
        history_path=folder/"lbfgs_history.npz";save_history(history_path,history)
        recent=(state["recent"]+[{"loss_before":value,"loss_after":new_value,"max_delta_vp":max_change}])[-6:]
        after={**state,"status":"running","accepted_total":count,"accepted_in_stage":state["accepted_in_stage"]+1,
            "current_q":str(new_q.relative_to(output)),"current_q_sha256":array_hash(trial),"model_sha256":new_sha,
            "history_file":str(history_path.relative_to(output)),"last_gradient":{"q_file":state["current_q"],
            "gradient_file":str((folder/"raw_gradient_sums.npz").relative_to(output))},"recent":recent}
        accepted_event={"gradient_evaluation_id":evaluation,"candidate_evaluation_id":trial_id,
            "model_before_sha256":anchor_sha,"model_after_sha256":new_sha,"objective_id":engine.objective_ids[stage]}
        commit={"state_after":after,"accepted_event":accepted_event,"iteration_id":identifier,"loss_before":value,"loss_after":new_value,
            "maximum_actual_delta_vp_m_s":max_change,"gradient_proof":gradient_proof,"accepted_count_this_iteration":1,
            "candidate_proof":trials[-1]["proof"],
            "fallback_steepest":fallback_used,"accepted_alpha":accepted_alpha,
            "model_true_vp_used":False,"sources":source_files()}
        (output/"commits").mkdir(exist_ok=True);save(output/"commits"/(identifier+".json"),commit)
        engine.event("accepted_update",identifier,**accepted_event);engine.event("iteration_end",identifier,status="accepted")
        state=after;save(state_path,state)
        print(json.dumps({"accepted":count,"stage":stage,"loss_before":value,"loss_after":new_value,"max_delta_vp_m_s":max_change,"model_sha256":new_sha}),flush=True)
        # Decisions depend only on same-stage data decrease and actual accepted step.
        if len(recent)==6:
            decreases=[(r["loss_before"]-r["loss_after"])/max(abs(r["loss_before"]),1e-300) for r in recent]
            stagnated=max(decreases)<cfg.get("stage_relative_stagnation",1e-3)
            if stage<len(cfg["stages"])-1 and stagnated:
                state.update(stage=stage+1,accepted_in_stage=0,history_file=None,last_gradient=None,recent=[])
            elif stage==len(cfg["stages"])-1 and max(decreases)<cfg.get("final_relative_stagnation",1e-4) and max(r["max_delta_vp"] for r in recent)<cfg.get("final_step_tolerance_m_s",.5):
                state["status"]="converged";state["stopping_reason"]="Six final-band accepted steps with small data decrease and actual Vp displacement";save(state_path,state);return
            save(state_path,state)
    state["status"]="complete_iteration_cap";state["stopping_reason"]="Declared frequency-stage accepted-epoch caps reached; convergence not assumed";save(state_path,state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stop-after-accepted",type=int,help="Checkpoint qualification limit; omission resumes the same declared bounded schedule")
    args=parser.parse_args()
    try:run(native.config(args.config),args.observations.resolve(),args.output.resolve(),args.stop_after_accepted)
    except Exception as error:
        state_path=args.output/"state.json"
        if state_path.exists():
            state=read(state_path);state.update(status="failed_exception",last_error={"type":type(error).__name__,"message":str(error)})
            save(state_path,state)
        raise


if __name__=="__main__":main()
