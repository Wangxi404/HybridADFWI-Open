#!/usr/bin/env python3
"""CPU-only region/absolute-error audit; never changes gradients or thresholds."""
import argparse
import hashlib
import json
import os
from pathlib import Path
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((args.run/"protocol.json").read_text())
    rows = {m: json.loads((args.run/(m+"-shot1.json")).read_text()) for m in ("autograd", "hybrid")}
    report = {"scope": "read-only absolute/relative discrepancy localization; does not relax any tolerance",
              "source_hashes_identical": {key: rows["autograd"][key] == rows["hybrid"][key]
                   for key in ("model_sha256", "observations_sha256", "script_sha256")},
              "config_identical": rows["autograd"]["config"] == rows["hybrid"]["config"],
              "pml_profile_identical": rows["autograd"]["pml_profile"] == rows["hybrid"]["pml_profile"],
              "parameters": {}}
    with np.load(args.run/"autograd-shot1.npz") as a, np.load(args.run/"hybrid-shot1.npz") as h, np.load(cfg["model"]) as model:
        p = a["active_mask"];water=model["vs_true"]==0
        regions = {"water": water, "frozen_solid_P0": (~water)&(p==0),
                   "taper_0_P_1": (p>0)&(p<1), "interior_P1": p==1}
        pa, ph = a["predicted_pressure"], h["predicted_pressure"]
        report["pressure"] = {"numerically_exact": bool(np.array_equal(pa,ph)),
            "uint64_bits_exact": bool(np.array_equal(pa.view(np.uint64),ph.view(np.uint64))),
            "autograd_bytes_sha256": hashlib.sha256(pa.tobytes()).hexdigest(),
            "hybrid_bytes_sha256": hashlib.sha256(ph.tobytes()).hexdigest()}
        report["evaluation_fields"] = {name: {
            "ad_hybrid_bits_exact": bool(np.array_equal(a["evaluation_model_"+name].view(np.uint64),h["evaluation_model_"+name].view(np.uint64))),
            "ad_equals_prepared_initial": bool(np.array_equal(a["evaluation_model_"+name],model[name+"_initial"]))}
            for name in cfg["parameters"]}
        for name in cfg["parameters"]:
            x,y=a["gradient_"+name],h["gradient_"+name];e=y-x
            e2=float(np.vdot(e,e));x2=float(np.vdot(x,x))
            entry={"reference_l2": float(np.sqrt(x2)), "difference_absolute_l2": float(np.sqrt(e2)),
                   "relative_l2": float(np.sqrt(e2/x2)), "absolute_linf": float(np.abs(e).max()), "regions": {}}
            for label,mask in regions.items():
                xx=x[mask];ee=e[mask];norm=float(np.linalg.norm(xx));err=float(np.linalg.norm(ee))
                entry["regions"][label]={"count": int(mask.sum()), "reference_l2": norm, "difference_l2": err,
                    "relative_to_region_reference": err/norm if norm else None,
                    "absolute_linf": float(np.max(np.abs(ee))) if ee.size else None,
                    "fraction_of_total_squared_error": err*err/e2 if e2 else None}
            largest=np.argsort(np.abs(e).ravel())[-10:][::-1]
            entry["largest_differences"]=[]
            for index in largest:
                iz,ix=np.unravel_index(index,e.shape)
                entry["largest_differences"].append({"z_x_index": [int(iz),int(ix)],
                    "x_z_m": [float(a["x_m"][ix]),float(a["z_m"][iz])],"P": float(p[iz,ix]),
                    "water": bool(water[iz,ix]),"autograd":float(x[iz,ix]),"hybrid":float(y[iz,ix]),"difference":float(e[iz,ix])})
            entry["coordinate_chain_linf"]={m:float(np.abs(data["active_gradient_"+name]-p*data["gradient_"+name]).max())
                for m,data in (("autograd",a),("hybrid",h))}
            report["parameters"][name]=entry
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    concise={"pressure":report["pressure"],"source_hashes_identical":report["source_hashes_identical"],
             "pml_profile_identical":report["pml_profile_identical"],
             "parameters":{n:{k:r[k] for k in ("reference_l2","difference_absolute_l2","relative_l2","absolute_linf","regions")}
                           for n,r in report["parameters"].items()}}
    print(json.dumps(concise),flush=True)


if __name__ == "__main__":
    main()
