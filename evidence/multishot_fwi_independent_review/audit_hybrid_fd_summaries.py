"""Recompute the two refined nine-shot scalar FD quotients from summaries.

Reads no receiver-pressure payload and invokes no solver/GPU. This checks
reported loss reductions/points against the independently audited gradient;
it is not an independent waveform-to-objective recomputation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--bundle',type=Path,default=Path('.cache/fwi-initial9-v3'))
parser.add_argument('--gradient-bundle',type=Path,default=Path('.cache/fwi-initial9-v2'))
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
args=parser.parse_args();ROOT=args.repo_root.resolve()
resolve=lambda path:path.resolve() if path.is_absolute() else (ROOT/path).resolve()
BUNDLE=resolve(args.bundle);INITIAL=resolve(args.gradient_bundle)
OUT=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='hybrid-fd-summary-independent-'))
if args.output:OUT.mkdir(parents=True,exist_ok=False)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def model_hash(record):return hashlib.sha256(json.dumps(record['physical_fields_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest()

protocol=read(BUNDLE/'protocol.json');diagnostic=read(BUNDLE/'initial_direction_check.json')
manifest=read(BUNDLE/'diagnostic_cache_import.json');proof=manifest['compatibility_proof']
assert diagnostic['signature']==protocol['signature']==proof['consumer_signature']
assert diagnostic['objective_id']==protocol['objective_ids'][0] and diagnostic['normalizer']==9
assert diagnostic['status']=='complete' and diagnostic['relative_tolerance']==1e-5
assert protocol['config']['initial_direction_check']['scales']==[.5,.25]
ledger=[json.loads(line) for line in (BUNDLE/'iterations.jsonl').read_text().splitlines() if line.strip()]
oldledger=[json.loads(line) for line in (INITIAL/'ledger_at_gradient_complete.jsonl').read_text().splitlines() if line.strip()]
iid=next(row for row in oldledger if row['event']=='iteration_start')['iteration_id']
sumfile=INITIAL/'iterations'/iid/'raw_gradient_sums.npz'
with np.load(sumfile,allow_pickle=False) as f:gradient=f['gradient_q_sum']/9
direction=gradient/np.linalg.norm(gradient)
rhs=float(np.vdot(gradient,direction));assert rhs>0
checks=[];fixed_fields=proof['fixed_field_sha256'];scales=None
for pair,h in zip(diagnostic['pairs'],(.05,.025)):
    means=[];point_rows=[]
    for sign,model in zip(('minus','plus'),pair['models']):
        assert model['proof']['normalizer']==9 and len(model['proof']['proof'])==1
        item=model['proof']['proof'][0];meta=read(BUNDLE/item['metadata'])
        assert item['shot_ids']==meta['shot_ids']==list(range(9))
        assert meta['signature']==protocol['signature'] and meta['objective_id']==diagnostic['objective_id']
        assert item['q_array_sha256']==meta['q_array_sha256']==model['q_array_sha256']
        assert model_hash(meta)==item['model_sha256']==model['physical_model_sha256']
        assert meta['P_sha256']==proof['P_array_sha256']
        assert all(meta['physical_fields_sha256'][key]==value for key,value in fixed_fields.items())
        assert meta['method']=='loss' and meta['stage']==0 and meta['cutoff_hz']==3.
        if scales is None:scales=meta['per_shot_scale']
        assert meta['per_shot_scale']==scales and meta['trace_screen']['passed']
        assert abs(model['maximum_actual_delta_vp_m_s']-h)<1e-10
        value=sum(meta['per_shot_loss'])/9;means.append(value)
        assert abs(meta['loss_sum']/9-value)<1e-16
        assert meta['per_shot_loss']==[model['proof']['per_shot_loss'][str(s)] for s in range(9)]
        event=next(row for row in ledger if row['event']=='shot_batch' and row.get('metadata')==item['metadata'])
        complete=next(row for row in ledger if row['event']=='evaluation_complete' and row.get('evaluation_id')==event['evaluation_id'])
        assert complete['normalizer']==9 and complete['loss']==value
        if h==.05:
            assert item['reused_exact_point'] and meta['import_provenance']['not_new_solver_execution']
            execution='imported_original_fine_pair'
        else:
            assert item['reused_exact_point'] is False and 'import_provenance' not in meta
            execution='new_full_nine_shot_forward_report'
        point_rows.append({'sign':sign,'q_sha256':meta['q_array_sha256'],'model_sha256':model['physical_model_sha256'],
            'metadata_sha256':sha(BUNDLE/item['metadata']),'receiver_payload_sha256_reported':meta['data_sha256'],
            'all_nine_mean_loss_recomputed':value,'execution':execution})
    fd=(means[1]-means[0])/(2*pair['epsilon_q'])
    error=abs(fd-rhs)/max(abs(fd),abs(rhs));assert error<1e-5
    assert abs(pair['gradient_dot_direction']-rhs)<1e-12*rhs
    assert abs(pair['finite_difference']-fd)<1e-12*rhs
    assert abs(pair['relative_error']-error)<1e-12
    checks.append({'maximum_physical_perturbation_m_s':h,'epsilon_q':pair['epsilon_q'],
        'independent_gradient_dot_direction':rhs,'scalar_FD_from_all_shot_loss_summaries':fd,
        'independent_relative_error':error,'passes_original_relative_gate':True,
        'recorded_gradient_dot_direction':pair['gradient_dot_direction'],
        'relative_RHS_arithmetic_difference':abs(rhs-pair['gradient_dot_direction'])/rhs,
        'actual_loss_difference':means[1]-means[0],'points':point_rows})
assert diagnostic['passed'] is True and all(c['passes_original_relative_gate'] for c in checks)
report={'passed':True,'scope':'Independent scalar quotient, all-nine loss reduction and same-point/provenance audit only, at the initial model. The pressure payload and FFT/window objective were not independently recomputed here. No solver/GPU run.',
    'numeric_controller_sha256':protocol['source_hashes']['scripts/run_multishot_fwi.py'],
    'diagnostic_sha256':sha(BUNDLE/'initial_direction_check.json'),'protocol_sha256':sha(BUNDLE/'protocol.json'),
    'source_gradient_sum_container_sha256':sha(sumfile),'global_mean_q_gradient_sha256':ah(gradient),
    'consumer_signature':protocol['signature'],'objective_id':diagnostic['objective_id'],
    'shot_normalizer':9,'relative_tolerance_unchanged':1e-5,'checks':checks,
    'refined_errors_monotone_decreasing':bool(checks[1]['independent_relative_error']<checks[0]['independent_relative_error']),
    'formal_second_order_convergence_claimed':False,'original_v2_failed_gate_preserved_by_import_proof':True,
    'interpretation':'Both refined finite-h quotients meet the original tolerance, but the smaller-h error increases; no monotone asymptotic convergence or absence of a stable offset is established.',
    'accepted_step_or_later_iterations_certified':False,'output_directory':str(OUT)}
(OUT/'independent_hybrid_refined_fd_summary_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
