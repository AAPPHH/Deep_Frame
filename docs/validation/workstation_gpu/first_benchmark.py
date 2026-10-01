import json,time,hashlib
from pathlib import Path
import numpy as np
from deep_frame.topology_elasticity import HexElasticity
from deep_frame.topology_optimization import DensityMap,_settings
root=Path('docs/validation/workstation_density_study/grid4_iter300')
inputs=json.loads((root/'inputs.json').read_text());domain=inputs['domain'];data=np.load(root/'fields.npz')
for key in ['allowed','preserve','forbidden']:domain[key]=data[key]
settings=_settings(inputs['settings']);mapping=DensityMap(domain,settings);initial=mapping.physical(mapping.initial(settings['volume_fraction']*mapping.allowed.sum()))[0]
report={'source_fields_sha256':hashlib.sha256((root/'fields.npz').read_bytes()).hexdigest(),'runs':[]}
cpu=HexElasticity(domain,interface_node_policy='preserve_adjacent');gpu=HexElasticity(domain,interface_node_policy='preserve_adjacent',linear_solver='cuda_cudss')
for name,rho in [('uniform',initial),('final',data['density']),('final_repeat',data['density'])]:
 start=time.perf_counter();a=cpu.solve(rho);cpu_s=time.perf_counter()-start
 start=time.perf_counter();b=gpu.solve(rho);gpu_s=time.perf_counter()-start
 rows={key:{'compliance_relative_error':abs(b[key]['compliance_n_mm']/value['compliance_n_mm']-1),'gradient_relative_l2':float(np.linalg.norm(b[key]['derivative']-value['derivative'])/np.linalg.norm(value['derivative'])),'cpu_residual':value['relative_residual'],'gpu_residual':b[key]['relative_residual']} for key,value in a.items()}
 rec={'field':name,'cpu_s':cpu_s,'gpu_s':gpu_s,'speedup':cpu_s/gpu_s,'cases':rows};report['runs'].append(rec);report['gpu']=gpu.diagnostics();Path('exports/topology/gpu_pilot/first_4mm.json').write_text(json.dumps(report,indent=2));print(json.dumps({**{k:v for k,v in rec.items() if k!='cases'},'max_compliance':max(x['compliance_relative_error'] for x in rows.values()),'max_gradient':max(x['gradient_relative_l2'] for x in rows.values()),'max_residual':max(x['gpu_residual'] for x in rows.values())}),flush=True)
gpu.close()
