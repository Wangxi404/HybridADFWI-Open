"""Exercise actual own-session supervisor using inert children and mock GPU data."""
from pathlib import Path
import argparse
import atexit
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--output',type=Path,help='New output directory; default is a fresh system temporary directory')
options=parser.parse_args()
SOURCE=options.repo_root.resolve()/'scripts/fwi_supervisor.py'
SPEC=importlib.util.spec_from_file_location('actual_supervisor',SOURCE)
SUP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SUP)
OUT=options.output.resolve() if options.output else Path(tempfile.mkdtemp(prefix='foreign_gpu_policy_cpu_'))
if options.output:OUT.mkdir(parents=True,exist_ok=False)
class TaskPath(type(Path())):
    @classmethod
    def home(cls):return cls(OUT)
SUP.Path=TaskPath # Isolate the cooperative lock; do not alter HOME or user cache.
class Process:
    def __init__(self,pid=None):self.pid=pid
    def cpu_affinity(self,values=None):return [0]
    def children(self,recursive=False):return []
SUP.psutil.Process=Process
SUP.psutil.cpu_percent=lambda **kw:[0.]
SUP.time.sleep=lambda seconds:None
SUP.os.getpriority=lambda *args:10
original_popen=subprocess.Popen
original_kill=os.kill
original_killpg=os.killpg
parent_group=os.getpgrp()
foreign_pid=999999999
spawned=[];signalled=[];children={}
def popen(*args,**kwargs):
    assert kwargs.get('start_new_session') is True,'Worker must own a new session'
    child=original_popen(*args,**kwargs);spawned.append(child.pid);children[child.pid]=child
    assert os.getpgid(child.pid)==child.pid and child.pid!=parent_group and child.pid!=foreign_pid
    return child
def kill(pid,sig):
    signalled.append({'kind':'process','pid':pid,'signal':int(sig)})
    assert pid in spawned,'Attempt to signal a foreign PID'
    return original_kill(pid,sig)
def killpg(group,sig):
    assert group in spawned and group!=parent_group and group!=foreign_pid,'Foreign/parent group forbidden'
    assert os.getpgid(group)==group,'Group must equal this Popen child PID'
    signalled.append({'kind':'process_group','pid':group,'process_group':group,'signal':int(sig)})
    return original_killpg(group,sig)
def cleanup():
    for pid,child in children.items():
        if child.poll() is None:
            assert os.getpgid(pid)==pid and pid!=parent_group and pid!=foreign_pid
            original_killpg(pid,signal.SIGTERM);child.wait(timeout=5)
atexit.register(cleanup)
SUP.subprocess.Popen=popen
SUP.os.kill=kill
SUP.os.killpg=killpg
reports={}
for scenario in ('foreign_compute','reserve'):
    samples=[0]
    def snapshot(**kw):
        samples[0]+=1
        during=samples[0]>3
        foreign=scenario=='foreign_compute' and during
        return {'eligible':not during,'compute_processes':[{'pid':foreign_pid,'executable':'external-do-not-signal'}] if foreign else [],
            'gpu':{'free_mib':1024. if scenario=='reserve' and during else 24000.}}
    SUP.gpu_snapshot=snapshot
    evidence=OUT/(scenario+'.json')
    sys.argv=[str(SOURCE),'--evidence',str(evidence),'--idle-wait-s','5','--maximum-walltime-s','60','--',
        sys.executable,'-c','import time;time.sleep(30)']
    code=SUP.main();state=json.loads(evidence.read_text())
    assert code==75
    expected='paused_foreign_compute' if scenario=='foreign_compute' else 'paused_gpu_reserve_violated'
    assert state['status']==expected
    pid=state['attempts'][0]['worker_pid']
    assert pid==spawned[-1] and signalled[-1]['pid']==pid
    assert signalled[-1]['kind']=='process_group' and children[pid].poll() is not None
    reports[scenario]={'returncode':code,'status':state['status'],'actual_own_child_pid':pid,
        'actual_own_child_group':pid,'parent_group_never_signalled':True,'actual_child_terminated':True,
        'signals':[r for r in signalled if r['pid']==pid],'foreign_pid_or_group_signalled':False}
report={'passed':True,'scope':'Actual supervisor main control flow, actual inert own process sessions terminated. GPU/process inventory and CPU affinity are mocked; no GPU, seismic solver, production worker or foreign process is used.',
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'directory':str(OUT),'scenarios':reports,
    'foreign_PID_signal_guard':'Every killpg asserts pgrp equals an actual own Popen child PID with start_new_session=True and differs from parent/foreign groups. Direct kill is also restricted to own child IDs.',
    'limitations':'Mock GPU inventory tests policy branches only. Actual remote resource telemetry and sampling gaps remain separately reported; no instantaneous or foreign-job-preemption guarantee.'}
(OUT/'independent_foreign_gpu_group_policy.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
