"""Independent shot-batch, full-batch mean and q-chain numerical checks."""
from pathlib import Path
import sys
import json
import subprocess
from unittest import mock
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import marmousi_fwi as fwi
from hybridadfwi.fwi import array_hash,bounds,candidate,physical_vp


def fixture(tmp_path):
    nz,nx=24,32;z,x=np.indices((nz,nx),dtype=float);water=z<5
    truth={"vp":np.where(water,1500.,2450.+12*z+2*x),"vs":np.where(water,0.,1250.+4*z+x),"rho":np.full((nz,nx),2250.)}
    arrays={}
    for p,v in truth.items():
        arrays[p+"_true"]=v;arrays[p+"_initial"]=np.where(water,v,.97*v) if p=="vp" else v.copy()
    arrays.update(x=np.arange(nx)*6.25,z=np.arange(nz)*6.25,nx=nx,nz=nz,dx=6.25,dz=6.25,ox=0.,oz=0.)
    target=tmp_path/"target.npz";np.savez(target,**arrays)
    cfg=dict(model=str(tmp_path/"training.npz"),source_model=str(target),device="cpu",nt=320,dt=.0005,f0=40.,shots=3,
             selected_shots=[0,1,2],shot_x_indices=[5,16,26],receivers=8,depth_m=12.5,nabc=8,free_surface=False,
             segments=8,parameters=["vp"],velocity_scale_m_s=1000.,filter_transition_hz=5.,
             stages=[{"cutoff_hz":20.,"epochs":1},{"cutoff_hz":None,"epochs":1}],
             fixed_pml_reference_vmax_m_s=4700.,constrained_coordinates=True,fixed_buffer_m=12.5,parameter_taper_m=12.5,
             minimum_offset_m=0.,offset_taper_m=6.25,mute_velocity_m_s=1500.,direct_mute_after_s=0.,time_taper_s=.001,end_taper_s=.005)
    fwi.prepare_inputs(cfg)
    assert "vp_true" not in np.load(cfg["model"]).files
    fwi.observations(cfg,tmp_path/"observed.npz")
    return cfg,dict(np.load(tmp_path/"observed.npz"))


def test_independent_batch_sum_equals_serial_ad_and_fd(tmp_path):
    torch.set_num_threads(1)
    cfg,observed=fixture(tmp_path)
    model,prop,coordinates,mask=fwi.build(cfg)
    q=.001*mask.detach().numpy() # Non-initial point catches stale/overwritten coordinate maps.
    records,arrays=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1,2],0,observed,"hybrid")
    serial=[];losses=[];pressures=[];hashes=[]
    for shot in (0,1,2):
        row,value=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[shot],0,observed,"hybrid")
        serial.append(value["gradient_q_sum"]);losses.extend(row["per_shot_loss"]);pressures.append(value["predicted_pressure"])
        hashes.append(row["physical_fields_sha256"]["vp"])
    np.testing.assert_allclose(arrays["gradient_q_sum"],np.sum(serial,axis=0),rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose(records["per_shot_loss"],losses,rtol=1e-13,atol=0)
    np.testing.assert_array_equal(arrays["predicted_pressure"],np.concatenate(pressures))
    assert len(set(hashes+[records["physical_fields_sha256"]["vp"]]))==1
    pair,pair_arrays=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1],0,observed,"hybrid")
    np.testing.assert_allclose(pair_arrays["gradient_q_sum"]+serial[2],arrays["gradient_q_sum"],rtol=1e-10,atol=1e-14)
    np.testing.assert_allclose((pair["loss_sum"]+losses[2])/3,records["loss_sum"]/3,rtol=1e-13)
    ar,aa=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1,2],0,observed,"autograd")
    relative=np.linalg.norm(aa["gradient_q_sum"]-arrays["gradient_q_sum"])/np.linalg.norm(aa["gradient_q_sum"])
    assert relative<1e-8
    lr,la=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1,2],0,observed,"loss")
    np.testing.assert_array_equal(la["predicted_pressure"],arrays["predicted_pressure"])
    assert records["q_array_sha256"]==array_hash(q)
    gradient=arrays["gradient_q_sum"]/3 # Objective also divides the shot sum by three ONCE.
    direction=gradient/np.linalg.norm(gradient);values=[];epsilon=1e-4
    changed_hashes=[]
    for sign in (-1,1):
        row,_=fwi.evaluate(cfg,prop,model,coordinates,mask,q+sign*epsilon*direction,[0,1,2],0,observed,"loss",False)
        values.append(row["loss_sum"]/3)
        changed_hashes.append(row["physical_fields_sha256"]["vp"])
        for fixed in ("vs","rho"):
            assert row["physical_fields_sha256"][fixed]==records["physical_fields_sha256"][fixed]
    assert len(set(changed_hashes+[records["physical_fields_sha256"]["vp"]]))==3
    derivative=(values[1]-values[0])/(2*epsilon);expected=float(np.vdot(gradient,direction))
    assert abs(derivative-expected)/abs(expected)<1e-5


def test_projected_candidate_uses_actual_physical_cap_without_true_vp():
    reference=np.full((3,4),2500.);vs=np.full((3,4),1400.);mask=np.array([[0,.1,.5,1.]]*3)
    lo,hi=bounds(reference,vs,mask,1000.)
    q=np.zeros_like(reference);direction=np.full_like(q,-100.)
    trial,delta=candidate(q,direction,1.,lo,hi,reference,mask,1000.,40.)
    assert np.max(np.abs(delta))<=40.*(1+1e-10)
    np.testing.assert_array_equal(trial[mask==0],q[mask==0])
    vp=physical_vp(reference,mask,trial,1000.)
    assert np.all(vp**2>4/3*vs**2)


def test_actual_full_loop_uneven_batch_one_commit_and_resume(tmp_path):
    cfg,_=fixture(tmp_path)
    cfg.update(stages=[{"cutoff_hz":None,"epochs":2}],gradient_batch_size=2,history_size=5,
               max_physical_update_m_s=60.,upper_vp_m_s=6000.,bulk_speed_floor_m_s=100.,
               armijo_c1=1e-4,max_backtracks=8,trial_shrink=.5,
               initial_direction_check={"enabled":True,"physical_perturbation_m_s":.1,"scales":[1.,.5],"relative_tolerance":1e-5})
    obs=tmp_path/"full_band_observations.npz";fwi.observations(cfg,obs)
    config_path=tmp_path/"config.json";config_path.write_text(json.dumps(cfg))
    output=tmp_path/"inversion"
    command=[sys.executable,str(ROOT/"scripts/run_multishot_fwi.py"),"--config",str(config_path),
             "--observations",str(obs),"--output",str(output),"--stop-after-accepted","1"]
    result=subprocess.run(command,text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    state=json.loads((output/"state.json").read_text());assert state["accepted_total"]==1
    assert state["status"]=="paused_requested_accepted_limit"
    assert json.loads((output/"initial_direction_check.json").read_text())["passed"]
    events=[json.loads(line) for line in (output/"iterations.jsonl").read_text().splitlines()]
    updates=[e for e in events if e["event"]=="accepted_update"];assert len(updates)==1
    gradient_id=updates[0]["gradient_evaluation_id"]
    batches=[e for e in events if e["event"]=="shot_batch" and e["evaluation_id"]==gradient_id]
    assert [e["shot_ids"] for e in batches]==[[0,1],[2]]
    assert len({e["model_sha256"] for e in batches})==1
    commit=json.loads(next((output/"commits").glob("*.json")).read_text())
    assert commit["loss_after"]<commit["loss_before"]
    trial=json.loads(next((output/"iterations").glob("*/trials.json")).read_text())[-1]
    assert trial["accepted"] and trial["loss"]<=trial["armijo_rhs"]
    assert trial["maximum_actual_delta_vp_m_s"]<=60*(1+1e-10)
    # Inject an interrupted prefix and a torn LAST JSONL fragment. Resumption
    # closes it, keeps the accepted model, and does not commit another update.
    ledger=output/"iterations.jsonl"
    objective=events[0]["objective_id"]
    with ledger.open("a") as f:
        f.write(json.dumps({"event":"iteration_start","iteration_id":"interrupted_fixture","shot_ids":[0,1,2],"model_sha256":state["model_sha256"],"reduction":"mean","objective_id":objective})+"\n")
        f.write(json.dumps({"event":"evaluation_start","iteration_id":"interrupted_fixture","evaluation_id":"partial_gradient","phase":"gradient","model_sha256":state["model_sha256"],"objective_id":objective})+"\n")
        f.write('{"event":"shot_batch"')
    resumed=subprocess.run(command,text=True,capture_output=True,timeout=90)
    assert resumed.returncode==0,resumed.stdout+resumed.stderr
    again=json.loads((output/"state.json").read_text());assert again["current_q_sha256"]==state["current_q_sha256"] and again["accepted_total"]==1
    repaired=[json.loads(line) for line in ledger.read_text().splitlines()]
    assert len([e for e in repaired if e["event"]=="accepted_update"])==1
    assert any(e["event"]=="iteration_end" and e["iteration_id"]=="interrupted_fixture" and e["status"]=="interrupted" for e in repaired)
    assert list((output/"ledger_repairs").glob("*.json"))
    # The orchestration selects a separate visualization interpreter without
    # changing the numerical worker's Python. Actual plotting is root's QA.
    import fwi_pipeline as pipeline
    original_run=subprocess.run;commands=[]
    def run_process(command,**kwargs):
        commands.append(command)
        if "fwi_ledger_check.py" in str(command[1]):return original_run(command,**kwargs)
        return subprocess.CompletedProcess(command,0,stdout="",stderr="")
    argv=["fwi_pipeline.py","--config",str(config_path),"--observations",str(obs),"--output",str(output),
          "--truth-posthoc",str(tmp_path/"target.npz"),"--plot-python","/unit/visualization/python"]
    with mock.patch.object(sys,"argv",argv),mock.patch.object(pipeline.subprocess,"run",side_effect=run_process):
        assert pipeline.main()==0
    assert commands[0][0]==sys.executable
    assert commands[-1][0]=="/unit/visualization/python"


def test_final_band_data_vjp_batch_matches_ad_and_fd(tmp_path):
    cfg,observed=fixture(tmp_path);model,prop,coordinates,mask=fwi.build(cfg)
    q=.002*mask.detach().numpy()
    row,hybrid=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1,2],1,observed,"hybrid")
    adrow,automatic=fwi.evaluate(cfg,prop,model,coordinates,mask,q,[0,1,2],1,observed,"autograd")
    relative=np.linalg.norm(hybrid["gradient_q_sum"]-automatic["gradient_q_sum"])/np.linalg.norm(automatic["gradient_q_sum"])
    assert relative<1e-8
    gradient=hybrid["gradient_q_sum"]/3;direction=gradient/np.linalg.norm(gradient);values=[]
    for sign in (-1,1):
        point,_=fwi.evaluate(cfg,prop,model,coordinates,mask,q+sign*1e-4*direction,[0,1,2],1,observed,"loss",False)
        values.append(point["loss_sum"]/3)
    fd=(values[1]-values[0])/2e-4;expected=float(np.vdot(gradient,direction))
    assert abs(fd-expected)/abs(expected)<1e-5
