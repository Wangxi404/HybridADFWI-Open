"""Read-only actual-artifact transition audits; no solver or GPU is used."""
from pathlib import Path
import argparse
import ast
from collections import Counter
import hashlib
import json
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hybrid-root',type=Path)
parser.add_argument('--native-root',type=Path)
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
if not (options.hybrid_root or options.native_root):parser.error('Supply at least one repository root')
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='transition-binding-independent-'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def hashed(a):return hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint64),b.view(np.uint64))
def ast_functions(path):
    result={}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.FunctionDef):result[node.name]=ast.dump(node,include_attributes=False)
        elif isinstance(node,ast.ClassDef):
            for method in node.body:
                if isinstance(method,ast.FunctionDef):result[node.name+'.'+method.name]=ast.dump(method,include_attributes=False)
    return result

def hybrid(root):
    old=root/'runs/fwi-20261108-fullshot-v2';new=root/'runs/fwi-20261108-fullshot-v3'
    manifest=read(new/'diagnostic_cache_import.json');receipt=read(new/'cache_import_receipt.json')
    proof=manifest['compatibility_proof'];assert hashed(proof)==manifest['compatibility_proof_sha256']
    assert sha(root/'scripts/import_fwi_diagnostic_cache.py')==manifest['helper_sha256']
    assert receipt['producer_before_sha256']==receipt['producer_after_sha256']
    assert receipt['manifest_sha256']==sha(new/'diagnostic_cache_import.json')
    for name,expected in receipt['producer_before_sha256'].items():assert sha(old/name)==expected
    failed=read(old/'initial_direction_check.json');oldstate=read(old/'state.json')
    assert failed['passed'] is False and failed['status']=='complete' and oldstate['accepted_total']==0
    assert str(oldstate['status']).startswith('failed')
    assert sha(old/'initial_direction_check.json')==proof['producer_failed_diagnostic_sha256']
    before=read(old/'protocol.json');after=read(new/'protocol.json')
    left=json.loads(json.dumps(before['config']));right=json.loads(json.dumps(after['config']))
    old_scales=left['initial_direction_check'].pop('scales');new_scales=right['initial_direction_check'].pop('scales')
    assert left==right and old_scales==[1.,.5] and new_scales==[.5,.25]
    assert before['source_hashes']==after['source_hashes']==proof['source_hashes_equal']
    assert before['torch_version']==after['torch_version']
    assert before['training_input_sha256']==after['training_input_sha256']==proof['training_input_sha256']
    assert before['observations_sha256']==after['observations_sha256']==proof['observations_sha256']
    assert before['objective_ids']==after['objective_ids']==proof['objective_ids_identical']
    assert before['signature']==proof['producer_signature'] and after['signature']==proof['consumer_signature']
    assert before['training_array_hashes']==after['training_array_hashes']
    assert before['map_sha256']==after['map_sha256'] and before['geometry_sha256']==after['geometry_sha256']
    assert after['config']['initial_direction_check']['relative_tolerance']==1e-5
    for name,expected in proof['source_hashes_equal'].items():assert sha(root/name)==expected
    entries=[];gradient_ids=[];fine_points=set()
    for row in manifest['imports']:
        relative=row['consumer_metadata'];oldmeta=old/relative;newmeta=new/relative
        a=read(oldmeta);b=read(newmeta);p=b['import_provenance']
        original=new/p['original_metadata_copy']
        assert sha(oldmeta)==sha(original)==p['original_metadata_sha256']==row['original_metadata_sha256']
        assert sha(newmeta)==row['consumer_metadata_sha256'] and b['signature']==proof['consumer_signature']
        assert a['signature']==p['original_producer_signature']==proof['producer_signature']
        assert p['not_new_solver_execution'] and p['compatibility_proof_sha256']==manifest['compatibility_proof_sha256']
        expected=dict(a);expected['signature']=b['signature'];expected['import_provenance']=p;assert expected==b
        olddata=old/row['data_relative_path'];newdata=new/row['data_relative_path']
        assert sha(olddata)==sha(newdata)==row['data_sha256']==a['data_sha256']==b['data_sha256']
        assert b['P_sha256']==proof['P_array_sha256']
        assert b['physical_fields_sha256']['vs']==proof['fixed_field_sha256']['vs']
        assert b['physical_fields_sha256']['rho']==proof['fixed_field_sha256']['rho']
        assert b['requires_grad']=={'vp':True,'vs':False,'rho':False}
        if row['method']=='hybrid':
            gradient_ids+=row['shot_ids'];assert row['q_array_sha256']==proof['initial_q_sha256']
        else:fine_points.add(row['q_array_sha256'])
        entries.append({'shot_ids':row['shot_ids'],'method':row['method'],'q_sha256':row['q_array_sha256'],
            'old_metadata_sha256':sha(oldmeta),'consumer_metadata_sha256':sha(newmeta),'data_sha256':sha(olddata),
            'producer_original_copy_exact':True,'explicitly_not_new_solver_execution':True})
    assert len(entries)==7 and Counter(gradient_ids)==Counter(range(9)) and len(fine_points)==2
    def first_gradient(run):
        rows=[json.loads(line) for line in (run/'iterations.jsonl').read_text().splitlines() if line.strip()]
        start=next(r for r in rows if r['event']=='evaluation_start' and r['phase']=='gradient')
        blocks=[r for r in rows if r.get('evaluation_id')==start['evaluation_id'] and r['event']=='shot_batch']
        return run/'iterations'/start['iteration_id']/'raw_gradient_sums.npz',rows,blocks
    oldsum,_,_=first_gradient(old);newsum,ledger,blocks=first_gradient(new)
    sums={}
    with np.load(oldsum) as a,np.load(newsum) as b:
        for key in ('gradient_physical_sum','gradient_coordinate_sum','gradient_q_sum'):
            assert same(a[key],b[key]);sums[key]=ah(b[key])
    imported_events=[r for r in ledger if r['event']=='shot_batch' and r.get('reused_exact_point')]
    assert len(imported_events)>=7 and len(blocks)==5 and all(r['reused_exact_point'] for r in blocks)
    assert Counter(s for r in blocks for s in r['shot_ids'])==Counter(range(9))
    return {'passed':True,'scope':'Actual v2-to-v3 import lineage and cached initial-gradient identity only. Refined full-model FD and accepted update remain separate qualifications.',
        'helper_sha256':manifest['helper_sha256'],'numeric_controller_sha256':after['source_hashes']['scripts/run_multishot_fwi.py'],
        'manifest_sha256':sha(new/'diagnostic_cache_import.json'),'import_receipt_sha256':sha(new/'cache_import_receipt.json'),
        'compatibility_proof_sha256':manifest['compatibility_proof_sha256'],'producer_control_and_metadata_files_unchanged':len(receipt['producer_before_sha256']),
        'original_completed_failed_gate_retained':True,'old_coarse_error':proof['old_coarse_error'],'old_fine_error':proof['old_fine_error'],
        'observed_two_h_order_not_formal':proof['two_point_observed_order'],'only_changed_configuration':'FD scales [1,.5] -> [.5,.25]',
        'unchanged_relative_tolerance':1e-5,'imports':entries,'consumer_initial_three_gradient_sums_uint64_equal':sums,
        'observed_imported_cache_hits':len(imported_events),'actual_consumer_solver_execution_claimed_for_imports':False,
        'pressure_payload_decoded':False,'accepted_updates_at_snapshot':sum(r['event']=='accepted_update' for r in ledger)}

def native(root):
    old=root/'run_psv_fwi_roi_20261108';new=root/'run_psv_fwi_roi_v2_20261108'
    oldprep=root/'data/inversion/psv_fwi_roi_20261108';newprep=root/'data/inversion/psv_fwi_roi_v2_20261108'
    fd=read(root/'run_psv_fwi_roi_fd_20261108/verification.json');binding=read(new/'initial_anchor_binding.json')
    oldsource=old/'driver_executed.py';newsource=new/'driver_executed.py'
    assert sha(oldsource)==fd['driver_sha256']=='28eed616dacde81f48eb1d470b927f7969166fe1a74468be88e6dd3ac01716c6'
    assert sha(newsource)=='3b71ba1c8b5cd2fd30038b287ee8f120f9f3c23e4083b4c483736f9193843439'
    a=ast_functions(oldsource);b=ast_functions(newsource)
    changed=[key for key in a if a[key]!=b[key]];assert changed==['prepare','optimize','main']
    oldrt=read(old/'runtime_signature.json');newrt=read(new/'runtime_signature.json')
    assert {k:v for k,v in oldrt.items() if k!='scripts/invert_vponly_psv.py'}=={k:v for k,v in newrt.items() if k!='scripts/invert_vponly_psv_v2.py'}
    for filename in ('training_input.npz','truth_for_observations_only.npz'):assert sha(oldprep/filename)==sha(newprep/filename)
    shots=read(newprep/'protocol.json')['shot_ids'];records=[]
    for sid in shots:
        stem=f'shot_{sid:03d}';od=old/'epochs/coarse_e000/gradient';nd=new/'epochs/coarse_e000/gradient'
        x=read(od/(stem+'.json'));y=read(nd/(stem+'.json'));reference=fd['anchor_target_source_hashes'][str(sid)]
        assert sha(od/(stem+'.npz'))==x['array_sha256']==reference['anchor_array_sha256']
        assert sha(nd/(stem+'.npz'))==y['array_sha256']
        for key in ('model_sha256','vp_sha256','source_sha256','target_sha256','vs_fixed_sha256','rho_fixed_sha256','stage_id'):assert x[key]==y[key]
        assert reference['source_sha256']==y['source_sha256'] and reference['target_sha256']==y['target_sha256']
        assert y['model_sha256']==fd['model_sha256']
        modes=[c['mode'] for c in y['native_calls']];assert modes.count('record')==1 and modes.count('adjoint')==1
        assert all(str(new) in c['log'] and c['returncode']==0 for c in y['native_calls'])
        hashes={}
        with np.load(od/(stem+'.npz')) as ov,np.load(nd/(stem+'.npz')) as nv:
            for key in ('data','cotangent','gradient','loss','sample_time'):
                assert same(ov[key],nv[key]);hashes[key]=ah(nv[key])
        records.append({'shot_id':sid,'model_sha256':y['model_sha256'],'source_sha256':y['source_sha256'],'target_sha256':y['target_sha256'],
            'fresh_record_calls':1,'fresh_adjoint_calls':1,'exact_uint64_array_sha256':hashes})
    sums={}
    with np.load(old/'epochs/coarse_e000/gradient/aggregate.npz') as ov,np.load(new/'epochs/coarse_e000/gradient/aggregate.npz') as nv:
        for key in ('vp','gradient_sum','gradient','loss_sum','loss'):
            assert same(ov[key],nv[key]);sums[key]=ah(nv[key])
    assert binding['passed'] and binding['new_driver_sha256']==sha(newsource)
    return {'passed':True,'scope':'Fresh actual v2 initial coarse anchor only. Exact new data/cotangent/gradient/sum/mean/loss and unchanged operator bind the retained v1 raw-FD proof; no claim for later iterates or fine stage.',
        'old_driver_sha256':sha(oldsource),'new_driver_sha256':sha(newsource),'changed_existing_function_ASTs':changed,
        'all_other_existing_function_ASTs_equal':True,'noncontroller_runtime_hashes_equal':True,
        'identical_training_and_observation_model_bytes':True,'fresh_shots':records,'aggregate_uint64_equal_sha256':sums,
        'retained_original_raw_fd_verification_sha256':sha(root/'run_psv_fwi_roi_fd_20261108/verification.json'),
        'retained_raw_fd_checks':fd['checks'],'producer_binding_report_sha256':sha(new/'initial_anchor_binding.json'),
        'gradient_cache_copied':False,'solver_or_GPU_run_by_reviewer':False}

reports={}
if options.hybrid_root:reports['hybrid_actual_import']=hybrid(options.hybrid_root.resolve())
if options.native_root:reports['native_v2_actual_anchor']=native(options.native_root.resolve())
(OUT/'independent_transition_bindings.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps({'reports':reports,'directory':str(OUT)},indent=2))
