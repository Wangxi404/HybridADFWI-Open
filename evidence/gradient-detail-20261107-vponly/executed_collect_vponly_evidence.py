#!/usr/bin/env python3
"""Collect completed Vp-only workers into a strict model-free whitelist.

No propagation, filtering, gain or gradient correction takes place here.
Original summary bytes (including a failed gate) are retained unchanged.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ash(array): return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def save(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pressure-output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((args.run/"protocol.json").read_text())
    summary=json.loads((args.run/"summary.json").read_text())
    if cfg["parameters"]!=["vp"] or cfg["selected_shots"]!=[1]: raise ValueError("Expected this declared single-shot Vp-only case")
    rows={m:json.loads((args.run/(m+"-shot1.json")).read_text()) for m in ("autograd","hybrid")}
    manifest=json.loads((args.run/"model_manifest.json").read_text())
    expected={p:v["sha256_float64_c_bytes"] for p,v in manifest["initial"]["fields"].items()}
    for method,row in rows.items():
        records=row["physical_forward_field_evaluations"]
        if not records or not all(r["sha256_float64_c_bytes"]==expected for r in records): raise ValueError("Actual field identity failure")
        if not all(r["requires_grad"]=={"vp":True,"vs":False,"rho":False} for r in records): raise ValueError("Trainable parameter mismatch")
    canonical={}; sources={}; audit={"created_utc":datetime.now(timezone.utc).isoformat(),
        "original_summary_passed":summary["passed"],"tolerance":cfg["tolerance"],
        "actual_initial_field_hashes_match":True,"parameters":["vp"],"private_worker_source_hashes":{},"array_identity":{}}
    with np.load(args.run/"autograd-shot1.npz") as a,np.load(args.run/"hybrid-shot1.npz") as h:
        for method,values in (("autograd",a),("hybrid",h)):
            for name in ("vp","coordinate_vp"):
                key="gradient_"+method+"_"+name
                canonical[key]=values["gradient_"+name]
                audit["array_identity"][key]={"sha256_c_bytes":ash(canonical[key]),"shape":list(canonical[key].shape),"dtype":str(canonical[key].dtype),
                    "numeric_transform":"none; direct single-shot worker array, signed zeros preserved"}
            sources[method]={"physical":values["gradient_vp"],"coordinate":values["gradient_coordinate_vp"]}
            audit["private_worker_source_hashes"][method]=sha(args.run/(method+"-shot1.npz"))
        for key in ("x_m","z_m","receiver_x_m","receiver_z_m","active_mask"):
            canonical[key]=h[key]
        canonical["source_x_m"]=np.atleast_1d(h["source_x_m"])
        canonical["source_z_m"]=np.atleast_1d(h["source_z_m"])
        canonical["selected_shot_ids"]=np.array([1],dtype=np.int64)
        mask=canonical["active_mask"]
        audit["pressure_uint64_equal"]=bool(np.array_equal(a["predicted_pressure"].view(np.uint64),h["predicted_pressure"].view(np.uint64)))
        audit["pressure_sha256_float64_c_bytes"]=ash(a["predicted_pressure"])
        for method in sources:
            audit[method+"_coordinate_equals_P_times_physical_uint64"]=bool(np.array_equal(sources[method]["coordinate"].view(np.uint64),(mask*sources[method]["physical"]).view(np.uint64)))
            audit[method+"_coordinate_equals_P_times_physical_numeric"]=bool(np.array_equal(sources[method]["coordinate"],mask*sources[method]["physical"]))
    audit["agreements"]={}
    for meaning in ("physical","coordinate"):
        a,h=(sources[m][meaning] for m in ("autograd","hybrid"))
        d=h-a
        audit["agreements"][meaning]={"relative_l2":float(np.linalg.norm(d)/np.linalg.norm(a)),
            "absolute_l2":float(np.linalg.norm(d)),"absolute_linf":float(np.max(np.abs(d))),
            "regions":{}}
        z=np.broadcast_to(canonical["z_m"][:,None],mask.shape)
        for name,region in (("fixed_water",(mask==0)&(z<450)),("fixed_solid",(mask==0)&(z>=450)),
                            ("transition",(mask>0)&(mask<1)),("core",mask==1)):
            scale=np.linalg.norm(a[region]);err=np.linalg.norm(d[region])
            audit["agreements"][meaning]["regions"][name]={"cells":int(region.sum()),"reference_l2":float(scale),
                "absolute_l2":float(err),"relative_to_region_l2":float(err/scale) if scale else None}
    if not audit["pressure_uint64_equal"]: raise ValueError("AD/H data differ")
    np.savez_compressed(args.output/"raw_kernels.npz",**canonical)
    audit["canonical_raw_sha256"]=sha(args.output/"raw_kernels.npz")
    names=["protocol.json","summary.json","model_manifest.json","target_identity.json","trace_screen.json"]
    for label in ("observations","trace-screen","autograd-shot1","hybrid-shot1"):
        names.extend(label+suffix for suffix in (".json",".log",".resources.json"))
    for name in dict.fromkeys(names):
        if (args.run/name).is_file(): shutil.copyfile(args.run/name,args.output/name)
    for name in ("gradient_detail_vponly.py","gradient_detail.py"):
        shutil.copyfile(ROOT/"scripts"/name,args.output/("executed_"+name))
    save(args.output/"numerical_audit.json",audit)
    trace=json.loads((args.run/"trace_screen.json").read_text())
    save(args.output/"physical_quality.json",{"trace_screen_passed":trace["passed"],"original_full_gradient_gate_passed":summary["passed"],
        "scope":"Controlled known-true-Vs/density single-objective gradient; visual geological interpretability pending review, not inversion or globally validated physical accuracy",
        "independent_continuum_or_interface_accuracy_validated":False})
    subprocess.run([sys.executable,str(ROOT/"scripts/package_detail_pressure_proof.py"),"--run",str(args.run),
                    "--output",str(args.pressure_output)],check=True,stdout=subprocess.DEVNULL)
    for name in ("pressure_data_sampled.npz","pressure_proof.json"):
        shutil.copyfile(args.pressure_output/name,args.output/name)
    save(args.output/"array_conventions.json",{"keys":list(canonical),"source":"Fresh single-shot worker outputs; direct arrays; no aliases, no averaging",
        "physical_gradient":"gradient_{autograd|hybrid}_vp","actual_coordinate_gradient":"gradient_{autograd|hybrid}_coordinate_vp",
        "parameter_map":"m_vp=vp_initial+active_mask*(theta_vp-vp_initial); Vs/rho are fixed true physical fields",
        "excluded":"evaluation_model arrays, initial/true model arrays, source SEG-Y and previews"})
    save(args.output/"bundle_manifest.json",{"files":{p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(args.output.iterdir()) if p.is_file()},
        "collector_sha256":sha(__file__),"original_summary_preserved_byte_for_byte":sha(args.run/"summary.json")==sha(args.output/"summary.json")})
    print(json.dumps({"output":str(args.output),"raw_bytes":(args.output/"raw_kernels.npz").stat().st_size,
                      "raw_sha256":audit["canonical_raw_sha256"],"original_summary_passed":summary["passed"],"agreements":audit["agreements"]}),flush=True)


if __name__=="__main__": main()
