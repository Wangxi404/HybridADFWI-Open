"""Read-only audit of the first complete nine-shot Hybrid gradient.

Loads only gradient/map/prior arrays and small scales/geometry. It never
loads pressure/window arrays, imports Torch, or runs a solver or optimizer.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--run',type=Path,default=Path('runs/fwi-20261108-fullshot-v2'))
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args();ROOT=options.repo_root.resolve()
RUN=options.run.resolve() if options.run.is_absolute() else (ROOT/options.run).resolve()
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='hybrid-initial9-independent-'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def hashed(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64))

protocol=read(RUN/'protocol.json');cfg=protocol['config'];state=read(RUN/'state.json')
events=[];partial_tail=False
for line in (RUN/'iterations.jsonl').read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:partial_tail=True
start=next(row for row in events if row['event']=='iteration_start')
iid=start['iteration_id'];evaluation=next(row for row in events if row['event']=='evaluation_start' and row['phase']=='gradient' and row['iteration_id']==iid)
eid=evaluation['evaluation_id'];rows=[row for row in events if row.get('evaluation_id')==eid]
complete=[row for row in rows if row['event']=='evaluation_complete'];assert len(complete)==1
assert complete[0]['normalizer']==cfg['shots']==9 and cfg['selected_shots']==list(range(9))
assert start['reduction']=='mean' and start['shot_ids']==list(range(9))
assert evaluation['model_sha256']==start['model_sha256'] and evaluation['objective_id']==start['objective_id']
source_checks={}
for name,expected in protocol['source_hashes'].items():
    original=ROOT/name;snapshot=RUN/'executed_sources'/name.replace('/','__')
    source_checks[name]={'current_matches':sha(original)==expected,'executed_snapshot_matches':sha(snapshot)==expected}
    assert all(source_checks[name].values())
assert protocol['source_hashes']['scripts/run_multishot_fwi.py']=='2a064bcfe29eb34bb1178e6d282437a6290827d9e12de9a6c9ab2b26b85ce0ef'
training_path=Path(cfg['model']);assert sha(training_path)==protocol['training_input_sha256']
with np.load(training_path,allow_pickle=False) as f:
    training_keys=f.files
    assert 'vp_true' not in f.files and 'vp' not in f.files
    training={key:f[key].copy() for key in ('vp_initial','vs_fixed','rho_fixed','x','z')}
assert {key:ah(value) for key,value in training.items()}==protocol['training_array_hashes']
q=np.load(RUN/state['current_q'],allow_pickle=False)
assert ah(q)==state['current_q_sha256']==start['q_array_sha256'] and np.count_nonzero(q)==0
with np.load(RUN/'models/map.npz',allow_pickle=False) as f:mapping={key:f[key].copy() for key in f.files}
P=mapping['P'];assert sha(RUN/'models/map.npz')==protocol['map_sha256']
assert sha(RUN/'geometry.npz')==protocol['geometry_sha256']
assert bits(training['x'],mapping['x_m']) and bits(training['z'],mapping['z_m'])
assert np.isfinite(P).all() and P.min()>=0 and P.max()<=1
water=training['vs_fixed']==0;assert np.count_nonzero(P[water])==0
x=training['x']-training['x'][0];z=training['z']-training['z'][0]
distance=np.minimum(np.minimum(x[None,:],x[-1]-x[None,:]),z[-1]-z[:,None])
u=np.clip((distance-cfg['fixed_buffer_m'])/cfg['parameter_taper_m'],0,1)
independent_P=(.5-.5*np.cos(np.pi*u))*(~water)
P_error=float(np.max(np.abs(P-independent_P)));assert P_error<1e-14
scale=cfg['velocity_scale_m_s'];assert scale==1000
expected_fields={'vp':ah(training['vp_initial']),'vs':ah(training['vs_fixed']),'rho':ah(training['rho_fixed'])}
model_hash=hashed(expected_fields);assert model_hash==start['model_sha256']==state['model_sha256']
bulk=training['rho_fixed']*(training['vp_initial']**2-4/3*training['vs_fixed']**2)
assert bulk.min()>0 and training['rho_fixed'].min()>0 and training['vs_fixed'].min()>=0
geometry={}
with np.load(RUN/'geometry.npz',allow_pickle=False) as f:
    for key in f.files:geometry[key]=f[key].copy()
assert bits(geometry['source_x_m'],cfg['grid']['ox']+np.asarray(cfg['shot_x_indices'],dtype=np.float64)*cfg['grid']['dx'])
assert len(np.unique(geometry['source_x_m']))==9
assert np.all(geometry['source_z_m']==cfg['depth_m'])

# Selective zip access: no pressure or window payload is decompressed.
observations_path=RUN/'observations.npz'
assert sha(observations_path)==protocol['observations_sha256']
with np.load(observations_path,allow_pickle=False) as f:observed_scales=f['scales'].copy()
assert observed_scales.shape==(len(cfg['stages']),9) and np.isfinite(observed_scales).all() and observed_scales.min()>0
keys=('gradient_physical_sum','gradient_coordinate_sum','gradient_q_sum')
sums={key:np.zeros_like(P) for key in keys};shot_losses={};batches=[];covered=[]
for row in rows:
    if row['event']!='shot_batch':continue
    meta=read(RUN/row['metadata']);path=RUN/row['data']
    assert sha(path)==meta['data_sha256']
    assert meta['signature']==protocol['signature'] and meta['objective_id']==start['objective_id']
    assert meta['q_array_sha256']==start['q_array_sha256'] and meta['physical_fields_sha256']==expected_fields
    assert meta['P_sha256']==ah(P) and meta['requires_grad']=={'vp':True,'vs':False,'rho':False}
    assert meta['shot_ids']==row['shot_ids'] and row['model_sha256']==model_hash and row['objective_id']==start['objective_id']
    assert bits(np.asarray(meta['per_shot_scale'],dtype=np.float64),observed_scales[0,meta['shot_ids']])
    assert meta['trace_screen']['passed'] and meta['bulk_min_pa']>0
    assert meta['method']=='hybrid' and meta['stage']==0 and meta['cutoff_hz']==3.
    assert abs(sum(meta['per_shot_loss'])-meta['loss_sum'])<=1e-14*max(abs(meta['loss_sum']),1e-300)
    with np.load(path,allow_pickle=False) as f:
        assert all(key in f.files for key in keys)
        values={key:f[key].copy() for key in keys}
    assert all(value.shape==P.shape and np.isfinite(value).all() for value in values.values())
    eP=rel(values['gradient_coordinate_sum'],P*values['gradient_physical_sum'])
    eq=rel(values['gradient_q_sum'],scale*values['gradient_coordinate_sum'])
    assert eP<1e-14 and eq<1e-14
    assert np.count_nonzero(values['gradient_coordinate_sum'][P==0])==0
    for key in keys:sums[key]+=values[key]
    for sid,loss in zip(meta['shot_ids'],meta['per_shot_loss']):
        assert sid not in shot_losses;shot_losses[sid]=loss
    covered+=meta['shot_ids']
    batches.append({'shot_ids':meta['shot_ids'],'model_sha256':row['model_sha256'],'q_sha256':row['q_array_sha256'],
        'cache_sha256':meta['data_sha256'],'metadata_sha256':sha(RUN/row['metadata']),
        'per_shot_losses':meta['per_shot_loss'],'per_shot_fixed_observed_scales':meta['per_shot_scale'],
        'P_chain_relative_l2':eP,'coordinate_scale1000_relative_l2':eq,
        'P_chain_uint64_equal':bool(bits(values['gradient_coordinate_sum'],P*values['gradient_physical_sum'])),
        'scale1000_uint64_equal':bool(bits(values['gradient_q_sum'],scale*values['gradient_coordinate_sum'])),
        'physical_fields_sha256':meta['physical_fields_sha256'],'full_gradient_shape':list(P.shape),
        'source_signature':meta['signature'],'window_sha256':meta['window_sha256'],
        'trace_screen_passed':True,'timing':meta['timing'],'adapter_details':meta['adapter_details']})
assert covered==list(range(9)) and [len(row['shot_ids']) for row in batches]==[2,2,2,2,1]
aggregate_path=RUN/'iterations'/iid/'raw_gradient_sums.npz'
with np.load(aggregate_path,allow_pickle=False) as f:aggregate={key:f[key].copy() for key in keys}
aggregate_checks={}
for key in keys:
    assert bits(sums[key],aggregate[key])
    aggregate_checks[key]={'uint64_equal_independent_five_batch_sum':True,'sum_sha256':ah(sums[key]),
        'global_mean_sha256':ah(sums[key]/9),'sum_norm':float(np.linalg.norm(sums[key])),
        'global_mean_norm':float(np.linalg.norm(sums[key]/9))}
mean_loss=sum(shot_losses[sid] for sid in range(9))/9
assert mean_loss==complete[0]['loss']
global_P_error=rel(sums['gradient_coordinate_sum'],P*sums['gradient_physical_sum'])
global_scale_error=rel(sums['gradient_q_sum'],1000*sums['gradient_coordinate_sum'])
assert global_P_error<1e-14 and global_scale_error<1e-14
assert protocol['training_has_true_vp'] is False and protocol['source_encoding'] is False and protocol['model_error_used_by_optimizer'] is False
events_before_complete=events[:events.index(complete[0])+1]
assert not any(row['event']=='accepted_update' for row in events_before_complete)
source= (RUN/'executed_sources/scripts__run_multishot_fwi.py').read_text()
helper=(RUN/'executed_sources/src__hybridadfwi__fwi.py').read_text()
assert 'self.cfg.pop("source_model",None)' in source
assert 'gradient=totals["gradient_q_sum"]/cfg["shots"]' in source
assert '.mean(dim=(1,2))' in helper
report={'all_independent_gradient_checks_passed':True,
    'scope':'First completed nine-independent-shot initial gradient only. Current production FD/Armijo/inversion completion are not certified. No pressure/window payload, Torch, GPU or solver was used by this reader.',
    'numeric_driver_sha256':protocol['source_hashes']['scripts/run_multishot_fwi.py'],
    'protocol_sha256':sha(RUN/'protocol.json'),'source_checks':source_checks,'iteration_id':iid,
    'objective_id':start['objective_id'],'all_shot_anchor_model_sha256':model_hash,
    'training_input_sha256':protocol['training_input_sha256'],'training_prior_array_hashes':protocol['training_array_hashes'],
    'no_unknown_true_Vp_in_training_container':True,'optimizer_source_drops_truth_model_path':True,
    'all_actual_forward_field_hashes_equal_fixed_initial_priors':True,
    'requires_grad_only_Vp':True,'initial_q_exact_zero':True,
    'source_x_m':geometry['source_x_m'].tolist(),'source_z_m':geometry['source_z_m'].tolist(),
    'source_ids':covered,'actual_batch_sizes':[2,2,2,2,1],'independent_recomputed_mean_objective':mean_loss,
    'global_shot_normalizer':9,'per_shot_data_normalizer':cfg['nt']*cfg['receivers'],
    'full_global_data_count':9*cfg['nt']*cfg['receivers'],
    'objective_source_contract':'Each shot: 0.5 mean_(Nt,Nr)[(window*zero-padded lowpass(predicted-observed)/fixed_observed_RMS(stage,shot))^2]; batch SUM; one global /9. No regularizer. Data AD supplies receiver cotangents.',
    'data_objective_limit':'Pressure/window payload and data cotangents were intentionally not loaded or recomputed in this lightweight audit. Full-data objective/FD validation remains a separate qualification.',
    'batches':batches,'aggregate_sum_checks':aggregate_checks,'aggregate_npz_sha256':sha(aggregate_path),
    'global_P_chain_relative_l2':global_P_error,'global_scale1000_chain_relative_l2':global_scale_error,
    'parameter_map':{'sha256':ah(P),'shape':list(P.shape),'independent_formula_max_abs_error':P_error,
        'water_cells':int(water.sum()),'fixed_zero_cells':int((P==0).sum()),'active_cells':int((P>0).sum()),'core_cells':int((P>.999).sum())},
    'material_bulk3D_min_pa':float(bulk.min()),'no_intershot_update_before_complete_gradient':True,
    'accepted_updates_in_snapshot':sum(row['event']=='accepted_update' for row in events),
    'snapshot_status':'Completed gradient; FD qualification/line search may be in progress. An open valid ledger prefix is not a failed or completed inversion.',
    'complete_gradient_events':events_before_complete,'ledger_tail_fragment_seen':partial_tail,
    'reviewer_output_directory':str(OUT),'numeric_arrays_exported':False}
(OUT/'independent_hybrid_initial9_gradient.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
