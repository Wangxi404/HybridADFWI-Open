#!/usr/bin/env python3
"""Fresh Vp-only elastic experiment with true fixed Vs and density.

Reuses the frozen gradient_detail implementation, never a previous gradient.
No vendor kernel, adjoint mathematics, or prior evidence is modified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import gradient_detail as detail
import numpy as np


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def require_mode(cfg):
    if cfg["parameters"] != ["vp"] or cfg.get("parameter_mode") != "vp_only_true_vs_true_rho":
        raise ValueError("This experiment requires only Vp trainable and true fixed Vs/rho")


def material_summary(fields):
    vp, vs, rho = (fields[name] for name in ("vp", "vs", "rho"))
    bulk = rho*(vp*vp-4*vs*vs/3)
    mu = rho*vs*vs
    lam = rho*(vp*vp-2*vs*vs)
    water = vs == 0
    result = {"all_fields_finite": all(bool(np.isfinite(v).all()) for v in fields.values()),
              "positive_density": bool((rho>0).all()), "positive_bulk_modulus": bool((bulk>0).all()),
              "nonnegative_shear_modulus": bool((mu>=0).all()),
              "water_shear_modulus_zero": bool((mu[water]==0).all()),
              "bulk_min_pa": float(bulk.min()), "mu_min_pa": float(mu.min()),
              "lambda_min_pa": float(lam.min()),
              "negative_lambda_cell_count": int((lam<0).sum()),
              "note": "Negative lambda alone is allowed; positive bulk and nonnegative shear are checked. No clipping or material repair.",
              "fields": {name: {"sha256_float64_c_bytes": array_hash(value), "dtype": str(value.dtype),
                                "shape": list(value.shape), "min": float(value.min()), "max": float(value.max())}
                         for name, value in fields.items()}}
    if not all(result[key] for key in ("all_fields_finite", "positive_density", "positive_bulk_modulus",
                                       "nonnegative_shear_modulus", "water_shear_modulus_zero")):
        raise ValueError("Invalid material state: "+json.dumps(result))
    return result


def audit_model(cfg, source=None):
    require_mode(cfg)
    source = source or cfg.get("parent_prepared_model")
    if source:
        source = Path(source)
        source = source.resolve() if source.exists() else detail.ROOT/source
    with np.load(cfg["model"]) as f:
        truth = {p: f[p+"_true"] for p in ("vp", "vs", "rho")}
        initial = {p: f[p+"_initial"] for p in ("vp", "vs", "rho")}
        metadata = json.loads(str(f["metadata_json"])) if "metadata_json" in f else {}
    checks = {"vs_initial_equals_true_uint64": bool(np.array_equal(initial["vs"].view(np.uint64), truth["vs"].view(np.uint64))),
              "rho_initial_equals_true_uint64": bool(np.array_equal(initial["rho"].view(np.uint64), truth["rho"].view(np.uint64))),
              "only_vp_trainable": cfg["parameters"]==["vp"]}
    if source:
        with np.load(source) as original:
            checks["vp_initial_equals_source_sigma60_uint64"] = bool(np.array_equal(initial["vp"].view(np.uint64), original["vp_initial"].view(np.uint64)))
            for p in truth:
                checks[p+"_truth_equals_source_uint64"] = bool(np.array_equal(truth[p].view(np.uint64), original[p+"_true"].view(np.uint64)))
    if not all(checks.values()):
        raise AssertionError(checks)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "parameter_mode": cfg["parameter_mode"],
            "model_sha256": detail.digest(cfg["model"]), "source_model": str(source) if source else None,
            "source_model_sha256": detail.digest(source) if source else None,
            "checks": checks, "truth": material_summary(truth), "initial": material_summary(initial),
            "density_truth_convention": "Copied from the existing controlled model; constant2450kg/m3, not the original SEG-Y density texture",
            "vp_initial_policy": "Exact existing normalized-solid Gaussian sigma60m Vp; water restored; no additional blur or clipping",
            "forward_equation": "Full native P-SV elastic forward; fixed true Vs/rho remain in material coefficients and wave propagation",
            "grid_guidance": metadata.get("dispersion"),
            "wrapper_sha256": detail.digest(__file__), "numerical_implementation_sha256": detail.digest(detail.__file__)}


def prepare(cfg, source):
    require_mode(cfg)
    source = Path(source).resolve()
    path = Path(cfg["model"])
    if path.exists():
        raise FileExistsError(path)
    with np.load(source) as f:
        arrays = {key: f[key] for key in f.files}
    arrays["vs_initial"] = arrays["vs_true"].copy()
    arrays["rho_initial"] = arrays["rho_true"].copy()
    metadata = json.loads(str(arrays.get("metadata_json", np.array("{}"))))
    metadata.update(parameter_mode=cfg["parameter_mode"], vp_initial_policy="exact source sigma60 array",
                    vs_initial_policy="exact vs_true", rho_initial_policy="exact rho_true",
                    parent_prepared_model_sha256=detail.digest(source), wrapper_sha256=detail.digest(__file__))
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    report = audit_model(cfg, source)
    report["dispersion"] = metadata.get("dispersion")
    detail.save(path.with_suffix(".json"), report)
    print(json.dumps(report), flush=True)


def worker(cfg, method, shot, result):
    require_mode(cfg)
    records = []
    original_build = detail.build_model
    with np.load(cfg["model"]) as f:
        expected = {p: array_hash(f[p+"_initial"]) for p in ("vp", "vs", "rho")}
    def monitored_build(*args, **kwargs):
        model, prop, coords, mask = original_build(*args, **kwargs)
        original_forward = model.forward
        def monitored_forward():
            output = original_forward()
            fields = {p: getattr(model, p).detach().cpu().numpy() for p in expected}
            hashes = {p: array_hash(value) for p, value in fields.items()}
            if hashes != expected:
                raise AssertionError("Realized physical fields differ from recorded Vp-only initial model")
            if model.vs.requires_grad or model.rho.requires_grad or not model.vp.requires_grad:
                raise AssertionError("Unexpected active physical fields")
            records.append({"all_initial_field_hashes_match": True, "sha256_float64_c_bytes": hashes,
                            "requires_grad": {p: bool(getattr(model,p).requires_grad) for p in expected}})
            return output
        model.forward = monitored_forward
        return model, prop, coords, mask
    detail.build_model = monitored_build
    try:
        detail.worker(cfg, method, shot, result)
    finally:
        detail.build_model = original_build
    if not records:
        raise AssertionError("No native model.forward was recorded")
    row = json.loads(Path(result).read_text())
    row.update(vponly_wrapper_sha256=detail.digest(__file__), physical_forward_field_evaluations=records,
               fixed_vs_rho_policy="Physical realized fields equal source true Vs/rho; requires_grad=False; complete elastic P-SV propagation")
    detail.save(result, row)


def trace_screen(cfg, output):
    torch = detail.initialize(cfg)
    model, prop, _, _ = detail.build_model(cfg, trainable=False)
    with torch.no_grad():
        predicted = detail.pressure(detail.native(prop, cfg, cfg["selected_shots"][0])).cpu().numpy()
    with np.load(cfg["observations"]) as f:
        observed = f["pressure_"+str(cfg["selected_shots"][0])]
    def bands(p):
        t = np.arange(cfg["nt"])*cfg["dt"]
        return [{"time_start_s":lo,"time_end_s":hi,"max_abs":float(np.max(np.abs(p[:,(t>=lo)&(t<hi)]))),
                 "rms":float(np.sqrt(np.mean(p[:,(t>=lo)&(t<hi)]**2)))}
                for lo,hi in ((0,1),(1,2),(2,3),(3,3.500501))]
    pb, ob = bands(predicted), bands(observed)
    ratio = pb[-1]["max_abs"]/pb[0]["max_abs"]
    finite = bool(np.isfinite(predicted).all() and np.isfinite(observed).all())
    passed = finite and ratio <= .1
    np.savez_compressed(output/"initial_trace_screen.npz", predicted_pressure=predicted)
    detail.save(output/"trace_screen.json", {"passed":passed,"all_finite":finite,"initial":pb,"truth":ob,
                "initial_late_max_over_first_second_max":ratio,"heuristic_upper_bound":.1,
                "scope":"Screens recurrence of the prior large late-time amplification only; not independent validation of interface/PML accuracy",
                "initial_pressure_sha256_float64_c_bytes":array_hash(predicted)})
    if not passed:
        raise RuntimeError("Late-time trace screen failed; derivative runs withheld")


def run(cfg, output):
    from hybridadfwi.resources import run_checked
    import psutil
    require_mode(cfg)
    if hasattr(os,"sched_getaffinity"):
        allowed=os.sched_getaffinity(0)
        usage=psutil.cpu_percent(interval=1,percpu=True)
        chosen=min(allowed,key=lambda cpu:(usage[cpu],cpu))
        os.sched_setaffinity(0,{chosen})
        cfg["worker_cpu_affinity"]=[chosen]
    output.mkdir(parents=True, exist_ok=False)
    cfg["observations"] = str(output/"observations.npz")
    detail.save(output/"model_manifest.json", audit_model(cfg))
    protocol = output/"protocol.json"
    detail.save(protocol, cfg)
    base = [sys.executable, str(Path(__file__).resolve())]
    jobs = [("observations",base+["observations","--config",str(protocol)]),
            ("trace-screen",base+["trace-screen","--config",str(protocol),"--output",str(output)])]
    for shot in cfg["selected_shots"]:
        for method in ("autograd","hybrid"):
            label = method+"-shot"+str(shot)
            jobs.append((label,base+["worker","--config",str(protocol),"--method",method,"--shot",str(shot),"--result",str(output/(label+".json"))]))
    for label, command in jobs:
        with (output/(label+".log")).open("w") as log:
            audit = run_checked(command, stdout=log, device=cfg["device"], min_gpu_gib=10., min_ram_gib=8.)
        detail.save(output/(label+".resources.json"),audit)
        if audit["status"] != "complete":
            raise RuntimeError("Worker blocked, failed or contaminated: "+label)
        if label=="observations" and cfg.get("reference_observations"):
            reference=Path(cfg["reference_observations"])
            if not reference.is_absolute(): reference=detail.ROOT/reference
            with np.load(reference) as old,np.load(cfg["observations"]) as new:
                checks={key: {"uint64_equal":bool(np.array_equal(old[key].view(np.uint64),new[key].view(np.uint64))),
                              "old_array_bytes_sha256":array_hash(old[key]),"new_array_bytes_sha256":array_hash(new[key])}
                        for key in ("pressure_1","window_1","scale_1")}
            detail.save(output/"target_identity.json",{"reference":str(reference),"new":cfg["observations"],
                        "checks":checks,"scope":"Same physical truth target, window and normalization; only initial Vs/rho designation and trainable coordinates change"})
            if not all(row["uint64_equal"] for row in checks.values()):
                raise AssertionError("Fresh target differs from the prior controlled physical target")
        print("DONE",label,flush=True)
    detail.compare(cfg,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare-model","audit-model","run","observations","trace-screen","worker","compare"))
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--source-model",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--method",choices=("autograd","hybrid"))
    parser.add_argument("--shot",type=int)
    parser.add_argument("--result",type=Path)
    args=parser.parse_args()
    cfg=detail.load_config(args.config)
    require_mode(cfg)
    if args.command=="prepare-model": prepare(cfg,args.source_model)
    elif args.command=="audit-model": print(json.dumps(audit_model(cfg,args.source_model)),flush=True)
    elif args.command=="run": run(cfg,args.output.resolve())
    elif args.command=="observations": detail.observations(cfg)
    elif args.command=="trace-screen": trace_screen(cfg,args.output)
    elif args.command=="worker": worker(cfg,args.method,args.shot,args.result)
    else: detail.compare(cfg,args.output)


if __name__=="__main__":
    main()
