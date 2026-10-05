import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from deep_frame.topology_cables import CableRouter
from deep_frame.topology_reconstruction import stored_domain
from deep_frame.config import CABLES
import numpy as np
from deep_frame.topology_geometry import region_contains
p=Path('exports/cables/free_layout4_v3c')
graph, rods=pickle.loads((p/'graph.pkl').read_bytes())
domain=stored_domain('C:/clones/Deep_Frame-layout/exports/runs/free_layout4_v3_1/domain.json',graph.solid.shape)
r=CableRouter(graph,rods,domain)
region=next(c for c in domain['comparison_load_cases'] if c['name']=='crash_front')['loads'][0]['region']
print('camera_region',region)
for i,line in r.lines.items():
    inside=np.flatnonzero(region_contains(line,region))
    if len(inside):
        print(i,graph.members[i]['nodes'],'range',inside[[0,-1]].tolist(),'len',len(line),'ends',line[[0,-1]].round(2).tolist())
print('names',r.names)
