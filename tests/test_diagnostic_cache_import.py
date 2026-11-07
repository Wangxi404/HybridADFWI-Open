"""Metadata-only cache-import guard tests; no seismic result is fabricated.

Real batch/AD/FD solver checks are in test_multishot_fwi.py. These tiny fixtures
exercise strict import identity, producer preservation and rejection paths.
"""
from pathlib import Path
import sys,json,copy
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import import_fwi_diagnostic_cache as importer
import run_multishot_fwi as controller
from hybridadfwi.fwi import array_hash


def fixture(tmp_path):
    producer=tmp_path/"producer";producer.mkdir();(producer/"models").mkdir()
    training=tmp_path/"training.npz";observations=tmp_path/"observations.npz"
    vp=np.full((2,3),2400.);vs=np.full_like(vp,1200.);rho=np.full_like(vp,2200.)
    np.savez(training,vp_initial=vp,vs_fixed=vs,rho_fixed=rho,x=np.arange(3),z=np.arange(2))
    pressure=np.arange(24,dtype=np.float64).reshape(3,4,2)/100
    np.savez(observations,pressure=pressure,window=np.ones_like(pressure),scales=np.ones((1,3)))
    cfg={"model":str(training),"source_model":str(tmp_path/"unknown_target_not_read.npz"),"parameters":["vp"],
        "shots":3,"selected_shots":[0,1,2],"shot_x_indices":[5,16,26],"filter_transition_hz":.5,
        "stages":[{"cutoff_hz":3.,"epochs":8}],"initial_direction_check":{"enabled":True,"physical_perturbation_m_s":.1,"scales":[1.,.5],"relative_tolerance":1e-5}}
    objective=controller.hashed({"stage":0,"cutoff_hz":3.,"filter_transition_hz":.5,
        "filter_order":"zero-padded FFT residual lowpass then fixed window","window_sha256":array_hash(np.ones_like(pressure)),
        "scales_sha256":array_hash(np.ones(3)),"target_sha256":array_hash(pressure),"global_normalizer":3,"regularizer":None})
    sources=controller.source_files()
    signature=controller.hashed({"sources":sources,"config":cfg,"training_sha256":importer.digest(training),"observations_sha256":importer.digest(observations),"torch":torch.__version__})
    protocol={"signature":signature,"config":cfg,"source_hashes":sources,"torch_version":str(torch.__version__),
              "training_input_sha256":importer.digest(training),"observations_sha256":importer.digest(observations),"objective_ids":[objective]}
    controller.save(producer/"protocol.json",protocol)
    q=np.zeros_like(vp);np.save(producer/"models/q_initial.npy",q)
    np.savez(producer/"models/map.npz",P=np.ones_like(vp))
    protocol['map_sha256']=importer.digest(producer/"models/map.npz");controller.save(producer/"protocol.json",protocol)
    initial_model=controller.hashed({"vp":array_hash(vp),"vs":array_hash(vs),"rho":array_hash(rho)})
    controller.save(producer/"state.json",{"accepted_total":0,"status":"failed_exception","signature":signature,"model_sha256":initial_model})
    rows=[];fine_models=[]
    points=[("gradient",q,"hybrid",[[0,1],[2]]),("fine_minus",q-.00005,"loss",[[0,1,2]]),("fine_plus",q+.00005,"loss",[[0,1,2]])]
    for label,point,method,batches in points:
        fields={"vp":array_hash(vp+1000*point),"vs":array_hash(vs),"rho":array_hash(rho)};model_sha=controller.hashed(fields)
        eid=label
        rows.append({"event":"evaluation_start","evaluation_id":eid,"phase":"gradient" if method=="hybrid" else "trial","model_sha256":model_sha})
        for ids in batches:
            relative=Path("evaluations")/(objective[:16]+"_"+array_hash(point)[:20]+"_"+method)/("shots_"+"-".join(map(str,ids)))
            data=producer/relative.with_suffix(".npz");data.parent.mkdir(parents=True,exist_ok=True)
            np.savez(data,gradient_physical_sum=np.ones_like(vp)*len(ids),gradient_coordinate_sum=np.ones_like(vp)*len(ids),gradient_q_sum=np.ones_like(vp)*len(ids)*1000)
            meta=producer/relative.with_suffix(".json")
            record={"signature":signature,"objective_id":objective,"q_array_sha256":array_hash(point),"physical_fields_sha256":fields,
                    "shot_ids":ids,"method":method,"data_sha256":importer.digest(data),"P_sha256":array_hash(np.ones_like(vp)),
                    "requires_grad":{"vp":True,"vs":False,"rho":False}}
            controller.save(meta,record)
            rows.append({"event":"shot_batch","evaluation_id":eid,"shot_ids":ids,"model_sha256":model_sha,"q_array_sha256":array_hash(point),
                         "metadata":str(meta.relative_to(producer)),"data":str(data.relative_to(producer))})
        rows.append({"event":"evaluation_complete","evaluation_id":eid,"model_sha256":model_sha,"normalizer":3})
        if method=="loss":fine_models.append({"q_array_sha256":array_hash(point),"physical_model_sha256":model_sha})
    (producer/"iterations.jsonl").write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    diagnostic={"status":"complete","passed":False,"signature":signature,"q_array_sha256":array_hash(q),
        "pairs":[{"relative_error":1.2e-5,"models":[]},{"relative_error":3e-6,"models":fine_models}]}
    controller.save(producer/"initial_direction_check.json",diagnostic)
    refined=copy.deepcopy(cfg);refined["initial_direction_check"]["scales"]=[.5,.25]
    path=tmp_path/"refined.json";path.write_text(json.dumps(refined))
    return producer,path,observations


def test_import_keeps_original_metadata_and_records_no_new_execution(tmp_path):
    producer,cfg,obs=fixture(tmp_path);consumer=tmp_path/"consumer"
    original={str(p.relative_to(producer)):p.read_bytes() for p in producer.rglob('*') if p.is_file()}
    result=importer.import_cache(producer,cfg,obs,consumer)
    assert len(result["imports"])==4 # Two gradient blocks, two full-shot fine losses.
    assert result["original_failure_preserved"] and result["consumer_passed_flag_edited"] is False
    for relative,data in original.items():assert (producer/relative).read_bytes()==data
    for item in result["imports"]:
        row=json.loads((consumer/item["consumer_metadata"]).read_text());provenance=row["import_provenance"]
        assert provenance["not_new_solver_execution"]
        assert provenance["original_producer_signature"]!=row["signature"]
        assert importer.digest(consumer/provenance["original_metadata_copy"])==provenance["original_metadata_sha256"]
        assert importer.digest(consumer/item["data_relative_path"])==item["data_sha256"]


@pytest.mark.parametrize("mutation",["target","geometry","source"])
def test_import_rejects_changed_operator_inputs_or_source(tmp_path,mutation):
    producer,cfg,obs=fixture(tmp_path)
    if mutation=="target":
        with np.load(obs) as f:arrays={key:f[key] for key in f.files}
        arrays["pressure"]=arrays["pressure"]+.001;np.savez(obs,**arrays)
    elif mutation=="geometry":
        value=json.loads(cfg.read_text());value["shot_x_indices"][0]+=1;cfg.write_text(json.dumps(value))
    else:
        p=producer/"protocol.json";value=json.loads(p.read_text());value["source_hashes"]["scripts/marmousi_fwi.py"]="0"*64;controller.save(p,value)
    with pytest.raises(ValueError):importer.import_cache(producer,cfg,obs,tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()
