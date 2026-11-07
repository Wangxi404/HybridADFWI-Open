#!/usr/bin/env python3
"""Explicitly reuse initial-point numerics when ONLY FD diagnostic scales change.

Original producer metadata/data and its failed gate are never edited. Consumer
metadata states that it imports a prior execution, retains the exact original
metadata and signature, and binds a compatibility proof. No solver is run.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil

import run_multishot_fwi as controller
import numpy as np
import torch
from hybridadfwi.experiment import verify_upstream
from hybridadfwi.fwi import array_hash


def read(path):return json.loads(Path(path).read_text())
def digest(path):return controller.native.detail.digest(path)


def safe_path(root,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError("Cache paths must be confined relative names")
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError("Cache path escapes producer")
    return path


def compare_config(old,new):
    left=json.loads(json.dumps(old));right=json.loads(json.dumps(new))
    old_scales=left["initial_direction_check"].pop("scales")
    new_scales=right["initial_direction_check"].pop("scales")
    if left!=right:raise ValueError("Only initial_direction_check.scales may change")
    if old_scales==new_scales:raise ValueError("No diagnostic refinement requested")
    if new_scales!=[min(old_scales),min(old_scales)*.5]:raise ValueError("Require the existing fine h and its half")
    return old_scales,new_scales


def context(producer,new_config,observations):
    verify_upstream()
    protocol=read(producer/"protocol.json");state=read(producer/"state.json")
    if state.get("accepted_total")!=0:raise ValueError("Only an initial run with no accepted update can be imported")
    if not str(state.get("status","")).startswith("failed"):raise ValueError("Producer must have stopped with its failure retained")
    diagnostic=read(producer/"initial_direction_check.json")
    if diagnostic.get("status")!="complete" or diagnostic.get("passed") is not False:
        raise ValueError("Original completed failed diagnostic must remain present")
    cfg=controller.native.config(new_config)
    old_scales,new_scales=compare_config(protocol["config"],cfg)
    sources=controller.source_files()
    if protocol["source_hashes"]!=sources:raise ValueError("Numerical/controller source hashes differ")
    if protocol["torch_version"]!=str(torch.__version__):raise ValueError("Torch version changed")
    training_sha=digest(cfg["model"]);observations_sha=digest(observations)
    if training_sha!=protocol["training_input_sha256"]:raise ValueError("Training input changed")
    if observations_sha!=protocol["observations_sha256"]:raise ValueError("Target observations changed")
    old_signature=controller.hashed({"sources":sources,"config":protocol["config"],"training_sha256":training_sha,
        "observations_sha256":observations_sha,"torch":torch.__version__})
    if protocol["signature"]!=old_signature or diagnostic["signature"]!=old_signature or state["signature"]!=old_signature:
        raise ValueError("Producer protocol/diagnostic/state signature is inconsistent")
    with np.load(cfg["model"]) as f:
        if "vp_true" in f.files:raise ValueError("Unknown true Vp appears in training input")
        expected_fields={"vp":array_hash(f["vp_initial"]),"vs":array_hash(f["vs_fixed"]),"rho":array_hash(f["rho_fixed"])}
    expected_initial_model=controller.hashed(expected_fields)
    if state["model_sha256"]!=expected_initial_model:raise ValueError("Producer accepted-anchor model is not the actual initial fields")
    map_path=producer/"models/map.npz"
    if digest(map_path)!=protocol["map_sha256"]:raise ValueError("Producer parameter map changed")
    with np.load(map_path) as mapping:P_hash=array_hash(mapping["P"])
    with np.load(observations) as obs:
        objectives=[controller.hashed({"stage":i,"cutoff_hz":stage["cutoff_hz"],"filter_transition_hz":cfg["filter_transition_hz"],
            "filter_order":"zero-padded FFT residual lowpass then fixed window","window_sha256":array_hash(obs["window"]),
            "scales_sha256":array_hash(obs["scales"][i]),"target_sha256":array_hash(obs["pressure"]),
            "global_normalizer":cfg["shots"],"regularizer":None}) for i,stage in enumerate(cfg["stages"])]
    if objectives!=protocol["objective_ids"]:raise ValueError("Actual objective/filter/window/RMS changed")
    signature=controller.hashed({"sources":sources,"config":cfg,"training_sha256":training_sha,
        "observations_sha256":observations_sha,"torch":torch.__version__})
    initial=np.load(producer/"models/q_initial.npy");initial_hash=array_hash(initial)
    if diagnostic["q_array_sha256"]!=initial_hash:raise ValueError("Diagnostic is not at initial q")
    if not np.array_equal(initial,np.zeros_like(initial)):raise ValueError("Expected unmodified initial coordinates")
    # Reuse only a producer's FINER old diagnostic pair. Coarse failed points
    # remain solely in the producer namespace and are not relabelled as passed.
    fine_index=min(range(len(old_scales)),key=lambda i:old_scales[i])
    fine_pair=diagnostic["pairs"][fine_index]
    if fine_pair["relative_error"]>cfg["initial_direction_check"]["relative_tolerance"]:
        raise ValueError("Producer fine pair did not meet the unchanged tolerance")
    coarse_index=max(range(len(old_scales)),key=lambda i:old_scales[i])
    coarse_error=diagnostic["pairs"][coarse_index]["relative_error"];fine_error=fine_pair["relative_error"]
    if coarse_error<=cfg["initial_direction_check"]["relative_tolerance"]:
        raise ValueError("Producer failure is not explained by its coarse FD step")
    if not 0<fine_error<coarse_error/2:raise ValueError("No clear improvement from diagnostic step refinement")
    allowed_loss_q={m["q_array_sha256"]:m["physical_model_sha256"] for m in fine_pair["models"]}
    proof={"scope":"Diagnostic-scale-only compatibility; imports are old executions, not new solves",
        "producer_namespace":producer.name,"producer_protocol_sha256":digest(producer/"protocol.json"),
        "producer_signature":protocol["signature"],"consumer_signature":signature,
        "producer_failed_diagnostic_sha256":digest(producer/"initial_direction_check.json"),
        "old_fd_scales":old_scales,"new_fd_scales":new_scales,"tolerance_unchanged":cfg["initial_direction_check"]["relative_tolerance"],
        "old_coarse_error":coarse_error,"old_fine_error":fine_error,
        "two_point_observed_order":math.log(coarse_error/fine_error)/math.log(max(old_scales)/min(old_scales)),
        "order_scope":"Two h values show improvement; not a formal second-order convergence certificate",
        "source_hashes_equal":sources,"training_input_sha256":training_sha,"observations_sha256":observations_sha,
        "objective_ids_identical":objectives,"initial_q_sha256":initial_hash,
        "initial_physical_model_sha256":expected_initial_model,"fixed_field_sha256":{"vs":expected_fields["vs"],"rho":expected_fields["rho"]},
        "P_array_sha256":P_hash,
        "only_configuration_difference":"initial_direction_check.scales"}
    return cfg,protocol,signature,initial_hash,allowed_loss_q,proof


def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,destination)
    except OSError:shutil.copyfile(source,destination)


def import_cache(producer,new_config,observations,consumer):
    producer=Path(producer).resolve();consumer=Path(consumer).resolve()
    cfg,protocol,signature,initial_hash,loss_hashes,proof=context(producer,new_config,observations)
    events=[json.loads(line) for line in (producer/"iterations.jsonl").read_text().splitlines()]
    complete={e["evaluation_id"]:e for e in events if e["event"]=="evaluation_complete"}
    starts={e["evaluation_id"]:e for e in events if e["event"]=="evaluation_start"}
    batches={eid:[e for e in events if e["event"]=="shot_batch" and e["evaluation_id"]==eid] for eid in complete}
    selected=[];seen_files=set();gradient_ids=[];loss_points=set()
    for eid,end in complete.items():
        rows=batches[eid]
        if end["normalizer"]!=cfg["shots"] or Counter(s for row in rows for s in row["shot_ids"])!=Counter(cfg["selected_shots"]):
            raise ValueError("Producer evaluation was not a complete exactly-once shot reduction")
        if any(row["model_sha256"]!=end["model_sha256"] for row in rows):raise ValueError("Producer mixed model points")
        for row in rows:
            meta=safe_path(producer,row["metadata"]);data=safe_path(producer,row["data"]);record=read(meta)
            if record["signature"]!=protocol["signature"]:raise ValueError("Producer metadata signature mismatch")
            if record["objective_id"]!=protocol["objective_ids"][0]:continue
            if digest(data)!=record["data_sha256"]:raise ValueError("Producer cache data checksum mismatch")
            if (controller.model_hash(record)!=row["model_sha256"] or record["shot_ids"]!=row["shot_ids"]
                or record["q_array_sha256"]!=row["q_array_sha256"]):
                raise ValueError("Producer metadata differs from its ledger")
            if record["P_sha256"]!=proof["P_array_sha256"] or record["requires_grad"]!={"vp":True,"vs":False,"rho":False}:
                raise ValueError("Producer active parameter map changed")
            if any(record["physical_fields_sha256"][key]!=value for key,value in proof["fixed_field_sha256"].items()):
                raise ValueError("Producer known physical fields changed")
            is_gradient=record["method"]=="hybrid" and record["q_array_sha256"]==initial_hash and starts[eid]["phase"]=="gradient"
            is_fine_loss=record["method"]=="loss" and record["q_array_sha256"] in loss_hashes
            if not (is_gradient or is_fine_loss):continue
            expected_model=proof["initial_physical_model_sha256"] if is_gradient else loss_hashes[record["q_array_sha256"]]
            if controller.model_hash(record)!=expected_model:raise ValueError("Imported point has a different realized model")
            rel=meta.relative_to(producer)
            if str(rel) in seen_files:continue
            seen_files.add(str(rel));selected.append((meta,data,rel,str(data.relative_to(producer)),record))
            if is_gradient:gradient_ids.extend(record["shot_ids"])
            else:loss_points.add(record["q_array_sha256"])
    if Counter(gradient_ids)!=Counter(cfg["selected_shots"]):raise ValueError("Initial gradient cache coverage incomplete")
    if loss_points!=set(loss_hashes):raise ValueError("Completed fine finite-difference pair cache incomplete")
    consumer.mkdir(parents=True,exist_ok=False)
    proof_hash=controller.hashed(proof);imports=[]
    for meta,data,relative,data_relative,record in selected:
        destination=consumer/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        original=destination.with_name(destination.stem+".producer.json")
        link(meta,original);link(data,consumer/data_relative)
        new=dict(record);new["signature"]=signature
        new["import_provenance"]={"execution_kind":"imported_same_operator_same_point","compatibility_proof_sha256":proof_hash,
            "original_producer_signature":record["signature"],"original_metadata_sha256":digest(meta),
            "original_metadata_copy":str(original.relative_to(consumer)),"producer_namespace":producer.name,
            "producer_metadata_relative_path":str(relative),"data_sha256_unchanged":digest(data),"not_new_solver_execution":True}
        controller.save(destination,new)
        imports.append({"consumer_metadata":str(relative),"original_metadata_sha256":digest(meta),
            "consumer_metadata_sha256":digest(destination),"data_relative_path":str(data_relative),"data_sha256":digest(data),
            "shot_ids":record["shot_ids"],"q_array_sha256":record["q_array_sha256"],"method":record["method"]})
    manifest={"compatibility_proof":proof,"compatibility_proof_sha256":proof_hash,"imports":imports,
        "helper_sha256":digest(__file__),"original_failure_preserved":True,"consumer_passed_flag_edited":False}
    controller.save(consumer/"diagnostic_cache_import.json",manifest)
    print(json.dumps({"consumer":str(consumer),"imported_evaluation_blocks":len(imports),"proof_sha256":proof_hash}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer",type=Path,required=True);parser.add_argument("--consumer",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True);parser.add_argument("--observations",type=Path,required=True)
    parser.add_argument("--dry-run",action="store_true",help="Validate compatibility without creating a consumer namespace")
    args=parser.parse_args()
    if args.dry_run:
        *_,proof=context(args.producer.resolve(),args.config,args.observations)
        print(json.dumps({"dry_run_passed":True,"compatibility_proof":proof}),flush=True)
    else:import_cache(args.producer,args.config,args.observations,args.consumer)


if __name__=="__main__":main()
