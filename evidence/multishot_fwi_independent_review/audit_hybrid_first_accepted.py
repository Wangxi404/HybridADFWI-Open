"""Read immutable first accepted arrays and a two-shot same-point cache pair.

No solver, optimizer, Torch or CUDA context is invoked. Arrays remain on the
machine holding the experiment; only summaries and hashes are reported.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--run',type=Path,default=Path('runs/fwi-20261108-fullshot-v3'))
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
args=parser.parse_args();ROOT=args.repo_root.resolve();RUN=args.run.resolve() if args.run.is_absolute() else (ROOT/args.run).resolve()
OUT=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='hybrid-first-accepted-independent-'))
if args.output:OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def hashed(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {key:f[key].copy() for key in f.files}

protocol=read(RUN/'protocol.json');cfg=protocol['config']
commit_path=next(path for path in (RUN/'commits').glob('*.json') if read(path)['state_after']['accepted_total']==1)
commit=read(commit_path);state=commit['state_after'];iid=commit['iteration_id'];folder=RUN/'iterations'/iid
q0=np.load(RUN/state['last_gradient']['q_file'],allow_pickle=False)
q1=np.load(RUN/state['current_q'],allow_pickle=False);assert ah(q1)==state['current_q_sha256']
mapping=load(RUN/'models/map.npz');P=mapping['P'];training=load(RUN/'training_input.npz')
assert sha(RUN/'training_input.npz')==protocol['training_input_sha256']
assert {key:ah(training[key]) for key in protocol['training_array_hashes']}==protocol['training_array_hashes']
assert 'vp_true' not in training and np.count_nonzero(q0)==0
assert sha(RUN/'models/map.npz')==protocol['map_sha256']
reference=training['vp_initial'];vs=training['vs_fixed'];rho=training['rho_fixed'];scale=cfg['velocity_scale_m_s']
physical=lambda q:reference+P*((reference+scale*q)-reference)
vp0=load(RUN/'models/vp_initial.npz')['vp'];vp1=load(RUN/'models/vp_0001.npz')['vp']
assert bits(physical(q0),vp0) and bits(physical(q1),vp1)
field_hash=lambda vp:hashed({'vp':ah(vp),'vs':ah(vs),'rho':ah(rho)})
assert field_hash(vp0)==commit['accepted_event']['model_before_sha256']
assert field_hash(vp1)==commit['accepted_event']['model_after_sha256']==state['model_sha256']
with np.load(folder/'raw_gradient_sums.npz') as f:gradient=f['gradient_q_sum']/9
proposal=load(folder/'optimizer_direction.npz');direction=proposal['direction_q'];alpha=float(proposal['accepted_alpha'])
assert alpha==commit['accepted_alpha']
lower=np.sqrt(4/3*vs**2+cfg['bulk_speed_floor_m_s']**2)
lo=np.zeros_like(P);hi=np.zeros_like(P);active=P>0
lo[active]=(lower[active]-reference[active])/(scale*P[active])
hi[active]=(cfg['upper_vp_m_s']-reference[active])/(scale*P[active])
expected=np.clip(q0+alpha*direction,lo,hi)
peak=float(np.max(np.abs(physical(expected)-vp0)))
if peak>cfg['max_physical_update_m_s']:expected=q0+(expected-q0)*(cfg['max_physical_update_m_s']/peak)
assert bits(expected,q1)
delta=load(folder/'actual_update.npz');assert bits(delta['delta_q'],q1-q0) and bits(delta['delta_vp'],vp1-vp0)
slope=float(np.vdot(gradient,q1-q0));maximum=float(np.max(np.abs(vp1-vp0)))
assert slope<0 and maximum<=60*(1+1e-10)
assert np.all(vp1[active]>=lower[active]) and np.all(vp1[active]<=cfg['upper_vp_m_s'])
assert bits(vp1[~active],vp0[~active])
events=[json.loads(line) for line in (RUN/'iterations.jsonl').read_text().splitlines() if line.strip()]
first=[row for row in events if row['iteration_id']==iid]
updates=[row for row in first if row['event']=='accepted_update'];assert len(updates)==1
update_index=first.index(updates[0]);geid=commit['accepted_event']['gradient_evaluation_id'];ceid=commit['accepted_event']['candidate_evaluation_id']
for eid in (geid,ceid):
    completed=next(row for row in first if row['event']=='evaluation_complete' and row['evaluation_id']==eid)
    assert first.index(completed)<update_index and completed['normalizer']==9
    batches=[row for row in first if row['event']=='shot_batch' and row['evaluation_id']==eid]
    assert [s for row in batches for s in row['shot_ids']]==list(range(9))
    expected_model=field_hash(vp0 if eid==geid else vp1)
    assert all(row['model_sha256']==expected_model for row in batches)
candidate=commit['candidate_proof'];assert candidate['normalizer']==9 and len(candidate['proof'])==1
item=candidate['proof'][0];meta=read(RUN/item['metadata'])
assert meta['signature']==protocol['signature'] and meta['q_array_sha256']==ah(q1)
assert meta['physical_fields_sha256']=={'vp':ah(vp1),'vs':ah(vs),'rho':ah(rho)}
assert item['shot_ids']==meta['shot_ids']==list(range(9)) and item['reused_exact_point'] is False
loss=sum(meta['per_shot_loss'])/9;assert loss==commit['loss_after']
rhs=commit['loss_before']+cfg['armijo_c1']*slope
assert loss<=rhs
decision=next(row for row in read(folder/'trials.json') if row['accepted'])
assert abs(decision['g_dot_actual_delta_q']-slope)<1e-12*abs(slope)
assert abs(decision['armijo_rhs']-rhs)<1e-15
fd=read(RUN/'initial_direction_check.json');assert fd['passed'] is True and fd['relative_tolerance']==1e-5
for pair in fd['pairs']:
    for point in pair['models']:
        item_fd=point['proof']['proof'][0]
        event_fd=next(row for row in first if row['event']=='shot_batch' and row.get('metadata')==item_fd['metadata'])
        complete_fd=next(row for row in first if row['event']=='evaluation_complete' and row['evaluation_id']==event_fd['evaluation_id'])
        assert first.index(complete_fd)<update_index and complete_fd['normalizer']==9
        assert event_fd['shot_ids']==list(range(9)) and event_fd['model_sha256']==point['physical_model_sha256']

# Exactly two archived same-q pressure comparisons, not an all-nine claim.
target_q=ah(q1);hybrid_meta=None
for path in RUN.glob('evaluations/*_hybrid/shots_0-1.json'):
    row=read(path)
    if row['q_array_sha256']==target_q and row['signature']==protocol['signature']:
        hybrid_meta=(path,row);break
assert hybrid_meta is not None
hp,hm=hybrid_meta;lp=RUN/item['data']
assert hm['physical_fields_sha256']==meta['physical_fields_sha256'] and hm['objective_id']==meta['objective_id']
assert sha(hp.with_suffix('.npz'))==hm['data_sha256'] and sha(lp)==meta['data_sha256']
with np.load(hp.with_suffix('.npz')) as f:left=f['predicted_pressure']
with np.load(lp) as f:right=f['predicted_pressure']
assert left.dtype==right.dtype==np.float64
comparisons=[]
for k,sid in enumerate((0,1)):
    a=left[k];b=right[sid];assert bits(a,b)
    assert hm['per_shot_loss'][k]==meta['per_shot_loss'][sid]
    comparisons.append({'shot_id':sid,'shape':list(a.shape),'dtype':'float64',
        'actual_uint64_equal':True,'pressure_sha256':ah(a),'absolute_linf':float(np.max(np.abs(a-b))),
        'per_shot_GPU_loss_equal':True})
report={'passed':True,'scope':'Immutable first accepted nine-shot update and exactly two same-q1 archived forward pressure comparisons. No solver, optimizer, Torch or CUDA context run by reviewer.',
    'numeric_controller_sha256':protocol['source_hashes']['scripts/run_multishot_fwi.py'],'commit_sha256':sha(commit_path),
    'commit_iteration_id':iid,'accepted_updates_in_this_iteration':1,'all_gradient_and_candidate_shots_complete_before_update':True,
    'same_anchor_and_same_candidate_model_sha256_verified':True,'actual_q1_matches_projected_capped_direction_uint64':True,
    'actual_Vp0_Vp1_match_static_P_map_uint64':True,'fixed_cells_Vp_unchanged_uint64':True,
    'actual_q_delta_dot_mean_gradient':slope,'Armijo_rhs_independently_recomputed':rhs,
    'loss_before':commit['loss_before'],'loss_after_independent_nine_shot_mean':loss,
    'relative_loss_decrease':(commit['loss_before']-loss)/commit['loss_before'],
    'actual_maximum_delta_Vp_m_s':maximum,'physical_bounds_passed':True,
    'q1_sha256':ah(q1),'Vp1_sha256':ah(vp1),'actual_update_archive_sha256':sha(folder/'actual_update.npz'),
    'refined_fd_completed_before_update':True,'FD_formal_convergence_claimed':False,
    'q1_hybrid_B2_vs_loss_B9_pressure_comparisons':comparisons,
    'hybrid_B2_cache_sha256':hm['data_sha256'],'loss_B9_cache_sha256':meta['data_sha256'],
    'GPU_wavefield_or_cotensor_dtype_runtime_claimed':False,'all_nine_same_point_forward_equality_claimed':False,
    'full_inversion_completion_or_recovery_claimed':False,'output_directory':str(OUT)}
(OUT/'independent_hybrid_first_accepted_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
