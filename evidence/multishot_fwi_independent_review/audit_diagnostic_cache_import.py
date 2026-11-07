"""Independent tiny cache-import/lineage checks, not a wave-solver test."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args();REPO=options.repo_root.resolve()
sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(REPO/'src'))
spec=importlib.util.spec_from_file_location('artifact_factory',REPO/'tests/test_diagnostic_cache_import.py')
factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
imp=factory.importer;ctl=factory.controller
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='hybrid-import-independent-'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)

def fixture(name):
    folder=OUT/name;folder.mkdir();return factory.fixture(folder)

def digest_tree(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}

def mutate_ledger(path,edit):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    edit(rows);path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

producer,cfg,obs=fixture('allowed');before=digest_tree(producer)
consumer=OUT/'allowed-consumer';proof=imp.import_cache(producer,cfg,obs,consumer)
assert digest_tree(producer)==before and len(proof['imports'])==4
engine=ctl.Engine.__new__(ctl.Engine)
engine.output=consumer;engine.signature=proof['compatibility_proof']['consumer_signature']
engine.objective_ids=proof['compatibility_proof']['objective_ids_identical']
engine.cfg=imp.read(cfg);engine.cfg['device']='cpu';engine.mask=np.ones((2,3))
engine.fixed=proof['compatibility_proof']['fixed_field_sha256']
saved_evaluate=ctl.native.evaluate
def forbid_solve(*args,**kwargs):raise AssertionError('Imported cache unexpectedly called a solver')
ctl.native.evaluate=forbid_solve
cache_hits=[]
try:
    for point,ids,method in ((np.zeros((2,3)),[0,1],'hybrid'),
                             (np.zeros((2,3)),[2],'hybrid'),
                             (np.full((2,3),-.00005),[0,1,2],'loss'),
                             (np.full((2,3),.00005),[0,1,2],'loss')):
        record,values,paths=engine.batch(point,ids,0,method)
        assert paths['reused_exact_point'] and record['import_provenance']['not_new_solver_execution']
        cache_hits.append({'shot_ids':ids,'method':method,'actual_consumer_cache_hit':True})
finally:ctl.native.evaluate=saved_evaluate
for row in proof['imports']:
    meta=consumer/row['consumer_metadata'];record=imp.read(meta)
    original=consumer/record['import_provenance']['original_metadata_copy']
    assert imp.digest(original)==row['original_metadata_sha256']
    assert record['import_provenance']['not_new_solver_execution']
    # Exercise the exact production writers against consumer-linked paths.
    ctl.save(meta,{'explicit_tiny_overwrite_test':True})
    ctl.native.atomic_npz(consumer/row['data_relative_path'],explicit_tiny_overwrite_test=np.array([1.]))
    assert digest_tree(producer)==before
    assert imp.digest(original)==row['original_metadata_sha256']

negatives=[]
for label in ('target','geometry','source','state-model','P','duplicate-shot',
              'missing-fine-complete','data-bytes','tolerance','accepted-before',
              'producer-running','gate-was-passed','no-refinement-improvement'):
    p,c,o=fixture(label)
    if label=='target':
        with np.load(o) as f:a={key:f[key] for key in f.files}
        a['pressure']=a['pressure']+.001;np.savez(o,**a)
    elif label=='geometry':
        a=imp.read(c);a['shot_x_indices'][0]+=1;ctl.save(c,a)
    elif label=='source':
        a=imp.read(p/'protocol.json');a['source_hashes']['scripts/marmousi_fwi.py']='0'*64;ctl.save(p/'protocol.json',a)
    elif label=='state-model':
        a=imp.read(p/'state.json');a['model_sha256']='0'*64;ctl.save(p/'state.json',a)
    elif label=='P':
        np.savez(p/'models/map.npz',P=np.zeros((2,3)))
    elif label=='duplicate-shot':
        mutate_ledger(p/'iterations.jsonl',lambda rows:next(r for r in rows if r['event']=='shot_batch' and r['evaluation_id']=='gradient').update(shot_ids=[0,0]))
    elif label=='missing-fine-complete':
        mutate_ledger(p/'iterations.jsonl',lambda rows:rows.remove(next(r for r in rows if r['event']=='evaluation_complete' and r['evaluation_id']=='fine_plus')))
    elif label=='data-bytes':
        next((p/'evaluations').rglob('*.npz')).write_bytes(b'damaged')
    elif label=='tolerance':
        a=imp.read(c);a['initial_direction_check']['relative_tolerance']=2e-5;ctl.save(c,a)
    elif label=='accepted-before':
        a=imp.read(p/'state.json');a['accepted_total']=1;ctl.save(p/'state.json',a)
    elif label=='producer-running':
        a=imp.read(p/'state.json');a['status']='running';ctl.save(p/'state.json',a)
    elif label=='gate-was-passed':
        a=imp.read(p/'initial_direction_check.json');a['passed']=True;ctl.save(p/'initial_direction_check.json',a)
    elif label=='no-refinement-improvement':
        a=imp.read(p/'initial_direction_check.json');a['pairs'][1]['relative_error']=1e-5;ctl.save(p/'initial_direction_check.json',a)
    original=digest_tree(p)
    try:imp.import_cache(p,c,o,OUT/('rejected-'+label))
    except ValueError as error:negatives.append({'case':label,'rejected':True,'message':str(error)})
    else:raise AssertionError('Incorrectly accepted '+label)
    assert digest_tree(p)==original and not (OUT/('rejected-'+label)).exists()

report={'passed':True,'scope':'Independent metadata/lineage and atomic-hardlink isolation tests using a tiny synthetic artifact factory only. No wave result is generated; no solver/GPU runs.',
    'helper_sha256':imp.digest(REPO/'scripts/import_fwi_diagnostic_cache.py'),
    'numeric_controller_sha256':imp.digest(REPO/'scripts/run_multishot_fwi.py'),
    'original_failure_and_all_producer_bytes_unchanged':True,
    'later_consumer_atomic_JSON_and_NPZ_replacements_leave_producer_bytes_unchanged':True,
    'imported_blocks':len(proof['imports']),
    'only_fine_pair_plus_initial_gradient_selected':True,
    'all_imports_explicitly_not_new_solver_execution':True,
    'actual_consumer_Engine_batch_cache_hits_without_solver':cache_hits,
    'negative_cases':negatives,'directory':str(OUT)}
(OUT/'independent_import_guard_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
