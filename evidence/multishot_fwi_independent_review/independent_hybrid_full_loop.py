"""Actual optimizer control flow with a tiny analytic observation engine only.

This does not run seismic kernels. Real native-wave derivative evidence lives
in evaluations/fullshot_forward_test and vendor qualification is separate.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--skill-root',type=Path,help='differentiate-seismic-solver skill directory')
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
REPO=options.repo_root.resolve()
SKILL=(options.skill_root or REPO.parent/'SeismicDiffSkill/skills/differentiate-seismic-solver').resolve()
sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(REPO/'src'))
SPEC=importlib.util.spec_from_file_location('real_hybrid_loop',REPO/'scripts/run_multishot_fwi.py')
RUN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(RUN)
SOURCE_SHA=hashlib.sha256((REPO/'scripts/run_multishot_fwi.py').read_bytes()).hexdigest()
RUN.native.detail.initialize=lambda cfg:None
RUN.source_files=lambda:{}
REAL_ENGINE=RUN.Engine
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='hybrid_real_loop_cpu_'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)

class PlannedInterruption(RuntimeError):pass

class AnalyticEngine(REAL_ENGINE):
    interrupt_commit=False
    seen_batches=[]
    def __init__(self,cfg,observations,output):
        self.cfg=cfg;self.output=Path(output);self.signature=RUN.hashed(cfg)
        self.reference=np.full((3,4),2500.)
        self.mask=np.array([[0.,.2,.5,1.],[0.,.4,.8,1.],[0.,.3,.7,1.]])
        self.vs=np.full((3,4),1200.);self.rho=np.full((3,4),2250.)
        self.fixed={'vs':RUN.array_hash(self.vs),'rho':RUN.array_hash(self.rho)}
        self.training_array_hashes={key:RUN.array_hash(value) for key,value in
            {'vp_initial':self.reference,'vs_fixed':self.vs,'rho_fixed':self.rho,
             'x':np.arange(4.),'z':np.arange(3.)}.items()}
        self.lo,self.hi=RUN.bounds(self.reference,self.vs,self.mask,1000.)
        self.prop=SimpleNamespace(ox=0.,oz=0.,nx=4,nz=3,dx=1.,dz=1.)
        self.objective_ids=[RUN.hashed({'stage':s,'data':'fixed-analytic-observations'}) for s in range(2)]
        self.ledger=self.output/'iterations.jsonl'
        self.observed=dict(np.load(observations))
    def assign(self,q):
        vp=RUN.physical_vp(self.reference,self.mask,q,1000.)
        return {'physical_fields_sha256':{'vp':RUN.array_hash(vp),**self.fixed},
            'requires_grad':{'vp':True,'vs':False,'rho':False},'P_sha256':RUN.array_hash(self.mask)},vp
    def batch(self,q,ids,stage,method):
        record,vp=self.assign(q)
        x=np.arange(vp.size).reshape(vp.shape)
        losses=[];gs=[]
        for sid in ids:
            target=2500.+30*np.sin(x*.4)+10*sid
            weight=(.2+(x+sid+1)**1.3)/(15.+sid)
            weight*=1.+.3*stage
            residual=(vp-target)/100.
            losses.append(float(.5*np.mean(weight*residual**2)))
            gs.append(weight*(vp-target)/(10000.*vp.size))
        g=np.sum(gs,axis=0)
        record.update(shot_ids=ids,q_array_sha256=RUN.array_hash(q),
            per_shot_loss=losses,trace_screen={'passed':True})
        arrays={} if method=='loss' else {'gradient_physical_sum':g,
            'gradient_coordinate_sum':self.mask*g,'gradient_q_sum':1000*self.mask*g}
        AnalyticEngine.seen_batches.append({'stage':stage,'method':method,'ids':ids,
            'actual_q_sha256':RUN.array_hash(q),'actual_vp_sha256':record['physical_fields_sha256']['vp']})
        return record,arrays,{'metadata':'analytic CPU only','data':'analytic CPU only'}
    def event(self,event,iteration_id,**values):
        if event=='accepted_update' and AnalyticEngine.interrupt_commit:
            assert (self.output/'commits'/(iteration_id+'.json')).exists()
            AnalyticEngine.interrupt_commit=False
            raise PlannedInterruption('Durable accepted commit exists; ledger and state pointer still old')
        return super().event(event,iteration_id,**values)

RUN.Engine=AnalyticEngine
cfg={'stages':[{'epochs':2},{'epochs':2}],'shots':3,'selected_shots':[0,1,2],
    'gradient_batch_size':2,'velocity_scale_m_s':1000.,'history_size':5,
    'max_physical_update_m_s':60.,'max_backtracks':6,'trial_shrink':.5,'armijo_c1':1e-4}
cfg['model']=str(OUT/'training.npz')
np.savez(cfg['model'],vp_initial=np.full((3,4),2500.))
observations=OUT/'observations.npz'
np.savez(observations,source_x_m=np.array([0.,1.,2.]),source_z_m=np.zeros(3),
    receiver_x_m=np.array([0.,1.]),receiver_z_m=np.zeros(2))
RUN.run(cfg,observations,OUT/'uninterrupted')
AnalyticEngine.interrupt_commit=True
try:RUN.run(cfg,observations,OUT/'resumed')
except PlannedInterruption:pass
else:raise AssertionError('Interrupted durable-commit branch was not reached')
assert json.loads((OUT/'resumed/state.json').read_text())['accepted_total']==0
assert len(list((OUT/'resumed/commits').glob('*.json')))==1
RUN.run(cfg,observations,OUT/'resumed')

spec=importlib.util.spec_from_file_location('checker',SKILL/'scripts/check_iterations.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
details={}
for name in ('uninterrupted','resumed'):
    root=OUT/name;state=json.loads((root/'state.json').read_text())
    events=[json.loads(line) for line in (root/'iterations.jsonl').read_text().splitlines()]
    audit=checker.audit_events(events)
    assert audit['passed'] and audit['accepted_updates']==4
    assert state['accepted_total']==4 and state['status']=='complete_iteration_cap'
    slopes=[];stage_first_history={}
    for commit in sorted((root/'commits').glob('*.json')):
        item=json.loads(commit.read_text());identifier=item['iteration_id']
        folder=root/'iterations'/identifier
        with np.load(folder/'raw_gradient_sums.npz') as a:g=a['gradient_q_sum']/3
        q_before=np.load(root/item['previous_pointer']['current_q']) if 'previous_pointer' in item else None
        with np.load(folder/'actual_update.npz') as a:delta=a['delta_q'];dv=a['delta_vp']
        trials=json.loads((folder/'trials.json').read_text())
        accepted=next(t for t in trials if t['accepted'])
        slope=float(np.vdot(g,delta));slopes.append(slope)
        assert slope==accepted['g_dot_actual_delta_q']
        assert accepted['loss']<=accepted['anchor_loss']+cfg['armijo_c1']*slope
        assert np.max(np.abs(dv))<=60*(1+1e-10)
        with np.load(folder/'lbfgs_history.npz') as a:count=int(a['count'])
        stage=int(identifier[5]);stage_first_history.setdefault(stage,count)
        # The real evaluate() emitted one [0,1] and one [2] batch before global /3.
        gradient_batches=[r['shot_ids'] for r in events if r['iteration_id']==identifier and r['event']=='shot_batch' and r['evaluation_id'].endswith('_gradient')]
        assert gradient_batches==[[0,1],[2]]
    assert stage_first_history=={0:0,1:0}
    details[name]={'ledger':audit,'state_status':state['status'],'stage_first_history_counts':stage_first_history,
        'accepted_actual_delta_slopes':slopes,'current_q':state['current_q']}
with np.load(OUT/'uninterrupted/models/vp_0004.npz') as a,np.load(OUT/'resumed/models/vp_0004.npz') as b:
    exact=bool(np.array_equal(a['vp'].view(np.uint64),b['vp'].view(np.uint64)))
assert exact
normal_batch=AnalyticEngine.batch
def zero_batch(self,q,ids,stage,method):
    row,values,paths=normal_batch(self,q,ids,stage,method)
    row['per_shot_loss']=[0. for _ in ids]
    for value in values.values():value.fill(0.)
    return row,values,paths
AnalyticEngine.batch=zero_batch
RUN.run(cfg,observations,OUT/'zero_gradient')
zero_state=json.loads((OUT/'zero_gradient/state.json').read_text())
zero_audit=checker.audit_events([json.loads(r) for r in (OUT/'zero_gradient/iterations.jsonl').read_text().splitlines()])
assert zero_state['status']=='converged' and zero_state['accepted_total']==0
assert zero_audit['accepted_updates']==0 and zero_audit['iteration_statuses']['converged']==2
class AtLowerEngine(AnalyticEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.lo=np.zeros_like(self.reference)
    def batch(self,q,ids,stage,method):
        row,values,paths=normal_batch(self,q,ids,stage,method)
        row['per_shot_loss']=[1. for _ in ids]
        if method!='loss':
            gp=np.ones_like(q)*len(ids)*1e-8
            values={'gradient_physical_sum':gp,'gradient_coordinate_sum':self.mask*gp,
                    'gradient_q_sum':1000*self.mask*gp}
        return row,values,paths
RUN.Engine=AtLowerEngine
RUN.run(cfg,observations,OUT/'projected_stationary')
projected_state=json.loads((OUT/'projected_stationary/state.json').read_text())
projected_audit=checker.audit_events([json.loads(r) for r in (OUT/'projected_stationary/iterations.jsonl').read_text().splitlines()])
assert projected_state['status']=='converged' and projected_state['accepted_total']==0
assert projected_audit['iteration_statuses']['converged']==2
report={'passed':True,'scope':'Actual run/evaluate/Armijo/L-BFGS/checkpoint/stage/ledger control flow using a tiny algebraic forward/gradient engine. No production seismic solver or GPU run; this does not replace AD/native derivative qualification.',
    'source_sha256':SOURCE_SHA,'directory':str(OUT),'uninterrupted_vs_durable_commit_resume_final_Vp_uint64_equal':exact,
    'checks':details,'stage_objectives_distinct_and_history_reset':True,'uneven_batches_reduced_after_all_shots':True,
    'durable_commit_precedes_old_state_recovery':True,
    'zero_gradient_no_update':{'status':zero_state['status'],'ledger':zero_audit},
    'projected_stationarity_no_update':{'status':projected_state['status'],'ledger':projected_audit},
    'limitation':'The analytic fake forward/gradient supplies known functions solely to exercise actual orchestration; source solver mathematics remains independently audited elsewhere.'}
(OUT/'independent_hybrid_full_loop.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
