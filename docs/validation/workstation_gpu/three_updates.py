import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from deep_frame.topology_optimization import optimize_topology
from tools.summarize_topology_study import verify_artifacts
source=Path('docs/validation/workstation_density_study/grid8over3_iter150');output=Path('exports/topology/gpu_pilot/three_updates');output.mkdir(exist_ok=False)
manifest=json.loads((source/'manifest.json').read_text());verify_artifacts(source,manifest['artifacts'],['inputs.json','fields.npz','result.json','domain_masks.npz'])
inputs=json.loads((source/'inputs.json').read_text());domain=inputs['domain'];fields=np.load(source/'fields.npz')
for key in ['allowed','preserve','forbidden']:domain[key]=fields[key]
provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path('deep_frame').glob('*.py'),Path('requirements-gpu.txt')]}
(output/'source_sha256.json').write_text(json.dumps(provenance,indent=2))
results={}
for backend in ['cpu_superlu','cuda_cudss']:
 settings={**inputs['settings'],'linear_solver':backend,'max_iterations':3,'minimum_iterations':3,'max_runtime_s':500}
 journal=(output/(backend+'.jsonl')).open('w')
 def progress(x):
  journal.write(json.dumps(x)+'\n');journal.flush();print(json.dumps({'backend':backend,**x}),flush=True)
 started=time.perf_counter();result=optimize_topology(domain,settings,progress_callback=progress);wall=time.perf_counter()-started
 journal.close();np.savez_compressed(output/(backend+'.npz'),density=result.pop('density'),design_density=result.pop('design_density'));result['wall_s']=wall
 (output/(backend+'.json')).write_text(json.dumps(result,indent=2));results[backend]=result
 if result['status']!='ok':raise RuntimeError(result['diagnostics'])
a=np.load(output/'cpu_superlu.npz');b=np.load(output/'cuda_cudss.npz')
report={'density_max_abs':float(np.max(np.abs(a['density']-b['density']))),'design_max_abs':float(np.max(np.abs(a['design_density']-b['design_density']))),'history_max_compliance_relative_error':max(abs(x['compliances_n_mm'][key]/y['compliances_n_mm'][key]-1) for x,y in zip(results['cpu_superlu']['history'],results['cuda_cudss']['history']) for key in x['compliances_n_mm']),'cpu_wall_s':results['cpu_superlu']['wall_s'],'gpu_wall_s':results['cuda_cudss']['wall_s'],'source_fields_sha256':hashlib.sha256((source/'fields.npz').read_bytes()).hexdigest()}
report['speedup']=report['cpu_wall_s']/report['gpu_wall_s'];report['passed']=report['density_max_abs']<1e-7 and report['design_max_abs']<1e-7 and report['history_max_compliance_relative_error']<1e-6
(output/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
