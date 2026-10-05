from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

from tools import compute

@pytest.mark.skipif(os.name == "nt", reason="Linux cluster controller")
@pytest.mark.parametrize("gpu,rss,passed", [(20.0,10.0,True),(24.7,10.0,False),(20.0,46.5,False),(20.0,float("nan"),False),(float("nan"),10.0,False),(None,10.0,False)])
def test_parallel_controller_gate_uses_own_pid_memory_against_declared_reservation(gpu,rss,passed,tmp_path,monkeypatch):
    from tools import compare_pipelines
    comparison=compare_pipelines.Comparison.__new__(compare_pipelines.Comparison)
    comparison.configs={"simp":{"mma":{"root":"run"}}}
    spec=compute.request("gpu_compare",["x"],".",machine="node20_21")
    comparison.data={"stages":{"simp_probe":{"gpu_limit_gb":spec["entrypoint_resources"]["gpu_gb"],"host_limit_gb":compute.JOB_TYPES["gpu_compare"]["memory_gb"]}}}
    comparison.update=lambda *args,**kwargs:None
    monkeypatch.setattr(compare_pipelines,"ROOT",tmp_path)
    out=tmp_path/"run_probe"
    (out/"fine").mkdir(parents=True)
    (out/"fine/result.json").write_text(json.dumps({"iterations":3,"mass_g":100.0,"max_violation":5.0,"status":"not_converged_iteration_cap"}))
    (out/"fine/iterations.jsonl").write_text('{}\n{}\n{}\n')
    (out/"memory.json").write_text(json.dumps({"peaks":{"probe":{"process_gpu_gb":gpu,"rss_gb":rss,"device_used_gb":50.0}}}))
    if passed:
        comparison.gate("simp")
    else:
        with pytest.raises(RuntimeError,match="gate failed"):
            comparison.gate("simp")

def saved_definition(path,problem):
    definition={'domain':{'grid':[1,2]},'problem':problem}
    path.write_text(json.dumps({'sha256':hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':')).encode()).hexdigest(),**definition},indent=2))
    return json.loads(path.read_text())['sha256']

def test_shared_pipeline_gate_rejects_changed_problem_and_retention(monkeypatch,tmp_path):
    from tools import compare_pipelines as comparison
    monkeypatch.setattr(comparison,'ROOT',tmp_path)
    tested=comparison.Comparison.__new__(comparison.Comparison)
    tested.configs={name:{'mma':{'root':name,'start_sha256':'seed'}} for name in ('simp','neural')}
    tested.config={'pipelines':{name:name+'.json' for name in tested.configs}}
    tested.update=lambda *args,**kwargs:None
    record={'problem_definition_sha256':'same','battery_retention':'ideal_press','functional_requirements':{'low_flight':{'pitch_deg':15}},'start_sha256':'seed'}
    for name in tested.configs:
        root=tmp_path/(name+'_probe')/'fine'
        root.mkdir(parents=True)
        (root/'result.json').write_text(json.dumps(record))
        production=saved_definition(root.parent/'production_definition.json',{'beta':[1,2,4]})
    tested.data={'stages':{'shared_seed':{'problem_definition_sha256':production}}}
    tested.shared()
    saved_definition(tmp_path/'neural_probe/production_definition.json',{'beta':[1,2,8]})
    with pytest.raises(RuntimeError,match='different production problem'):
        tested.shared()
    tampered=json.loads((tmp_path/'simp_probe/production_definition.json').read_text())
    tampered['problem']['beta']=[1]
    (tmp_path/'neural_probe/production_definition.json').write_text(json.dumps(tampered))
    with pytest.raises(RuntimeError,match='does not match its fingerprint'):
        tested.shared()
    saved_definition(tmp_path/'neural_probe/production_definition.json',{'beta':[1,2,4]})
    changed=deepcopy(record)
    changed['problem_definition_sha256']='different'
    target=tmp_path/'neural_probe/fine/result.json'
    target.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError,match='different problem'):
        tested.shared()
    changed=deepcopy(record)
    changed['battery_retention']='contact'
    target.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError,match='ideal retention'):
        tested.shared()
    target.write_text(json.dumps(record))
    tested.configs['neural']['mma']['start_sha256']='other'
    with pytest.raises(RuntimeError,match='shared start design'):
        tested.shared()
    tested.configs['neural']['mma']['start_sha256']='seed'
    target.write_text(json.dumps({**record,'start_sha256':'older'}))
    with pytest.raises(RuntimeError,match='shared start design'):
        tested.shared()
    tested.configs['075']={'mma':{'root':'075'}}
    assert tested.started({'neural':record})=={'seed'}
    with pytest.raises(RuntimeError,match='shared start design'):
        tested.started({'neural':{**record,'start_sha256':'older'}})

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_controller_resume_reattaches_existing_ray_job(tmp_path):
    from tools.compare_pipelines import Comparison
    tested=Comparison.__new__(Comparison)
    tested.python=sys.executable
    tested.root=tmp_path
    tested.data={'stages':{'production':{'status':'RUNNING','job':'existing-job'}}}
    tested.update=lambda stage,**values:tested.data['stages'][stage].update(values)
    calls=[]
    tested.wait=lambda job:calls.append(job)
    tested.client=type('Client',(),{'submit_job':lambda *args,**kwargs:pytest.fail('must not duplicate an existing Ray job'),'get_job_logs':lambda self,job:'reattached'})()
    path=tmp_path/'pipeline.json'
    path.write_text('{}')
    tested.ray('production','frame_mma',str(path))
    assert calls==['existing-job']
    assert tested.data['stages']['production']['status']=='SUCCEEDED'
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='inputs changed'):
        tested.ray('production','frame_mma',str(path))

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_missing_ocp_viewer_does_not_interrupt_authorized_compute_plan():
    from tools.compare_pipelines import Comparison
    tested=Comparison.__new__(Comparison)
    calls=[]
    tested.ray=lambda *args,**kwargs:None
    tested.local=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('viewer unavailable'))
    tested.update=lambda stage,**fields:calls.append((stage,fields))
    tested.display({'run':'actual_pipeline','route':'simp','suffix':'recon'})
    assert calls[0][1]['status']=='PENDING_VIEWER'
    assert calls[0][1]['frame_source']=='actual_pipeline/frame.stl'

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
@pytest.mark.parametrize('standing',[True,False])
def test_controller_acceptance_keeps_stand_and_prop_clearance_required(tmp_path,monkeypatch,standing):
    from tools import compare_pipelines as study
    monkeypatch.setattr(study,'ROOT',tmp_path)
    tested=study.Comparison.__new__(study.Comparison)
    tested.ray=lambda *args,**kwargs:None
    tested.update=lambda *args,**kwargs:None
    tested.data={'stages':{'simp_optimization':{'field_mass_g':19.67}}}
    cfg={'mma':{'root':'optimization'}}
    path=tmp_path/'pipeline.json'
    path.write_text(json.dumps(cfg))
    root=tmp_path/'optimization'
    root.mkdir()
    run=tmp_path/'run'
    (run/'functional_review').mkdir(parents=True)
    (root/'runs.json').write_text(json.dumps({'recon':str(run)}))
    (root/'functional_reviews.json').write_text(json.dumps({'recon':{'exit_code':0}}))
    (run/'manifest.json').write_text(json.dumps({'wall_rule_passed':True,'evaluation_gates':{'missed':[]}}))
    (run/'evaluation.json').write_text(json.dumps({'geometry':{'mass':{'standing':{'stability':{'reserve_passed':standing,'prop_clearance_passed':True}}}}}))
    (run/'functional_review/physical_evaluation.json').write_text(json.dumps({'stl_mass_g':20}))
    accepted=tested.physical('simp',str(path))
    assert bool(accepted) is standing
    assert not standing or (accepted[0]['mass_g'],accepted[0]['field_mass_g'])==(20,19.67)

def controller(tmp_path,monkeypatch):
    from tools import compare_pipelines as study
    monkeypatch.setattr(study,'ROOT',tmp_path)
    tested=study.Comparison.__new__(study.Comparison)
    tested.root=tmp_path/'state'
    tested.root.mkdir()
    tested.python=sys.executable
    tested.data={'stages':{},'branches':{}}
    tested.update=lambda stage,**fields:tested.data['stages'].setdefault(stage,{}).update(fields)
    tested.config={'pipelines':{'simp':'simp.json','neural':'neural.json'}}
    tested.configs={name:{'mma':{'root':name+'_opt','fine_dir':'fine2','variant':name,'start':'start','start_sha256':None}} for name in tested.config['pipelines']}
    tested.paths=dict(tested.config['pipelines'])
    return study,tested

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_controller_generates_verifies_and_passes_one_shared_seed(tmp_path,monkeypatch):
    study,tested=controller(tmp_path,monkeypatch)
    design=np.linspace(0,1,12)
    digest=hashlib.sha256(np.ascontiguousarray(design,dtype=np.float64).tobytes()).hexdigest()
    tested.config['seed']={'start':'start','sha256':digest}
    jobs=[]
    def generate(stage,action,path,**kwargs):
        jobs.append((stage,action,json.loads(Path(path).read_text())['mma']))
        (tmp_path/'start').mkdir()
        np.savez_compressed(tmp_path/'start/design.npz',design=design)
        (tmp_path/'start/result.json').write_text(json.dumps({'sha256':digest,'problem_definition_sha256':'production'}))
    tested.ray=generate
    tested.seed()
    assert [job[:2] for job in jobs]==[('shared_seed','frame_seed')] and jobs[0][2]['start_sha256']==digest
    assert {json.dumps(json.loads(Path(path).read_text())['mma']['start_sha256']) for path in tested.paths.values()}=={json.dumps(digest)}
    assert tested.data['stages']['shared_seed']['problem_definition_sha256']=='production'
    tested.seed()
    assert len(jobs)==1
    np.savez_compressed(tmp_path/'start/design.npz',design=design+1)
    with pytest.raises(RuntimeError,match='controller seed'):
        tested.seed()
    tested.config.pop('seed')
    tested.configs['neural']['mma']['start_sha256']='other'
    tested.configs['simp']['mma']['start_sha256']=digest
    with pytest.raises(RuntimeError,match='shared start design'):
        tested.seed()

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_controller_records_field_and_stl_mass_and_rejects_other_production_definition(tmp_path,monkeypatch):
    study,tested=controller(tmp_path,monkeypatch)
    fine=tmp_path/'simp_opt/fine2'
    fine.mkdir(parents=True)
    production=saved_definition(fine/'problem_definition.json',{'beta':[1,2,4]})
    record={'status':'not_converged_diverged_best_feasible','mass_g':19.67,'max_violation':0.0,'start_sha256':'seed','problem_definition_sha256':production}
    (fine/'result.json').write_text(json.dumps(record))
    tested.optimized('simp','simp_optimization',production)
    assert tested.data['stages']['simp_optimization']['stl_mass_g'] is None
    (tmp_path/'simp_opt/simp').mkdir()
    (tmp_path/'simp_opt/simp/info.json').write_text(json.dumps({'body':{'mass_g':13.73,'bodies':3,'watertight':True}}))
    tested.optimized('simp','simp_optimization',production)
    state=tested.data['stages']['simp_optimization']
    assert (state['field_mass_g'],state['stl_mass_g'],state['stl_bodies'],state['design_status'])==(19.67,13.73,3,'not_converged_diverged_best_feasible')
    with pytest.raises(RuntimeError,match='other than the verified'):
        tested.optimized('simp','simp_optimization','probe')
    (fine/'result.json').write_text(json.dumps({**record,'problem_definition_sha256':'probe'}))
    with pytest.raises(RuntimeError,match='other than the verified'):
        tested.optimized('simp','simp_optimization',production)

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
@pytest.mark.parametrize('statuses,errors,reached,reason,absent',[({'simp':'not_converged_diverged_best_feasible','neural':'not_converged_stalled'},{'simp':'simp: not_converged_diverged_best_feasible; dependent production postprocessing withheld'},(),'Optimizer not converged (simp: not_converged_diverged_best_feasible, neural: not_converged_stalled)','No reconstructed'),
                                                                  ({'simp':None,'neural':None},{'simp':'simp: three-iteration solver/memory gate failed','neural':'Optimization routes use different production problem definitions'},(),'Route failed before acceptance (simp: simp: three-iteration solver/memory gate failed, neural: Optimization routes use different production problem definitions','Optimizer not converged'),
                                                                  ({'simp':'converged','neural':'not_converged_stalled'},{'simp':'simp: production run used a problem definition other than the verified abc; results withheld'},(),'Route failed before acceptance (simp: simp: production run used a problem definition other than the verified abc','No reconstructed'),
                                                                  ({'simp':'converged','neural':None},{},('simp',),'No reconstructed frame of simp passed','Optimizer not converged'),
                                                                  ({'simp':'converged','neural':'converged'},{},('simp','neural'),'No reconstructed frame of simp, neural passed','Optimizer not converged')])
def test_continuation_names_optimizer_non_convergence_as_the_cause(tmp_path,monkeypatch,statuses,errors,reached,reason,absent):
    study,tested=controller(tmp_path,monkeypatch)
    for name,status in statuses.items():
        if status:
            tested.update(name+'_optimization',design_status=status)
    for name,error in errors.items():
        tested.data['branches'][name]={'status':'FAILED','error':error}
    for name in reached:
        tested.update(name+'_acceptance',status='FAILED',candidates=[])
    tested.continuation()
    state=tested.data['stages']['continuation']
    assert state['status']=='AWAITING_DESIGN_CORRECTION' and reason in state['reason'] and absent not in state['reason']

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_controller_records_source_and_measured_reservation_and_refuses_dirty_tree(tmp_path,monkeypatch):
    study,tested=controller(tmp_path,monkeypatch)
    submitted=[]
    tested.client=type('Client',(),{'submit_job':lambda self,**spec:submitted.append(spec) or 'job','get_job_logs':lambda self,job:''})()
    tested.wait=lambda job:None
    path=tmp_path/'pipeline.json'
    path.write_text('{}')
    monkeypatch.setattr(study,'_git',lambda worktree:{'sha':'abc','branch':'main','dirty':False})
    tested.ray('simp_optimization','frame_mma',str(path),gpu_gb=16.65)
    state=tested.data['stages']['simp_optimization']
    assert state['source']=={'sha':'abc','branch':'main','dirty':False}
    assert state['gpu_limit_gb']==submitted[0]['entrypoint_resources']['gpu_gb']==round(16.65*study.JOB_TYPES['gpu_compare']['gpu_margin'],1)
    assert state['host_limit_gb']==study.JOB_TYPES['gpu_compare']['memory_gb']
    monkeypatch.setattr(study,'_git',lambda worktree:{'sha':'abc','branch':'main','dirty':True})
    with pytest.raises(RuntimeError,match='dirty'):
        tested.ray('neural_optimization','frame_neural',str(path))
    with pytest.raises(RuntimeError,match='dirty'):
        tested.local('neural_postprocessing',['true'])
    assert len(submitted)==1 and 'neural_postprocessing' not in tested.data['stages']

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_continuation_starts_075_from_the_43_result_without_the_shared_seed_sha(tmp_path,monkeypatch):
    study,tested=controller(tmp_path,monkeypatch)
    (tmp_path/'simp_pipeline.json').write_text(json.dumps({'shape':[103,97,25],'mma':{'root':'runs/simp','fine_dir':'fine','start':'start','start_sha256':'seed','settings':{}}}))
    tested.update('simp_acceptance',candidates=[{'mass_g':20,'pipeline':'simp_pipeline.json'}])
    with pytest.raises(RuntimeError,match='design.npz missing'):
        tested.continuation()
    (tmp_path/'runs/simp/fine').mkdir(parents=True)
    np.savez_compressed(tmp_path/'runs/simp/fine/design.npz',design=np.ones(4))
    tested.display=tested.comparisons=lambda *args:None
    tested.channels=lambda best:best
    launched=[]
    def stop(stage,action,path,**kwargs):
        launched.append((stage,action,kwargs))
        raise RuntimeError('stop')
    tested.ray=stop
    with pytest.raises(RuntimeError,match='stop'):
        tested.continuation()
    cfg=json.loads((tested.root/'075_probe.json').read_text())
    assert launched==[('075_probe','frame_mma',{'kind':'gpu_075'})]
    assert cfg['mma']['start_sha256'] is None and cfg['mma']['start']==str(tmp_path/'runs/simp/fine') and cfg['shape']==[182,172,44]
    assert (cfg['mma']['fine_start_level'],cfg['mma']['until'],cfg['mma']['settings'])==(6,'fine',{'final_max_iterations':3,'final_min_iterations':10})
    assert tested.data['stages']['best_43']['design_sha256']==hashlib.sha256(np.ones(4,dtype=np.float64).tobytes()).hexdigest()

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
@pytest.mark.parametrize('resume',[True,False])
def test_controller_resumes_from_dirty_tree_but_refuses_fresh_launch_before_pid(tmp_path,monkeypatch,resume):
    study,tested=controller(tmp_path,monkeypatch)
    tested.config={'pipelines':{},'resume':resume,'continue_plan':False}
    tested.configs={}
    tested.lock=threading.RLock()
    tested.seed=lambda:None
    tested.monitor=lambda done:None
    (tested.root/'status.json').write_text('{}')
    monkeypatch.setattr(study,'_git',lambda worktree:{'sha':'abc','branch':'main','dirty':True})
    if resume:
        tested.run()
        assert 'source' not in tested.data and tested.data['resume_sources'][-1]['dirty'] is True and (tested.root/'controller.pid').exists() and 'finished' in tested.data
    else:
        with pytest.raises(RuntimeError,match='Comparison state exists'):
            tested.run()
        (tested.root/'status.json').unlink()
        with pytest.raises(RuntimeError,match='dirty'):
            tested.run()
        assert not (tested.root/'controller.pid').exists()

@pytest.mark.skipif(os.name == 'nt', reason='Linux cluster controller')
def test_controller_seed_and_shared_report_missing_records(tmp_path,monkeypatch):
    study,tested=controller(tmp_path,monkeypatch)
    tested.config['seed']={'start':'start','sha256':'seed'}
    (tmp_path/'start').mkdir()
    np.savez_compressed(tmp_path/'start/design.npz',design=np.zeros(3))
    tested.ray=lambda *args,**kwargs:pytest.fail('seed present')
    with pytest.raises(RuntimeError,match='result.json missing'):
        tested.seed()
    with pytest.raises(RuntimeError,match='production problem definition missing'):
        tested.definition(tmp_path/'simp_opt_probe/production_definition.json')
