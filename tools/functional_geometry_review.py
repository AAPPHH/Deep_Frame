import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import trimesh
from deep_frame.frame_run import FrameLayout
from deep_frame.topology_implicit import _manifold, region_manifold
from deep_frame.frame import assembly_placements, build_components
from tools.formulation_study import configure, frame_setup, shared_definition
from deep_frame import config

class PipelineReview:
    def __init__(self,path):
        self.root=Path(__file__).resolve().parents[1]
        self.spec=json.loads(Path(path).read_text())
        self.cfg=configure(json.loads((self.root/self.spec['pipeline']).read_text()))
        self.source=self.root/self.spec['frame_source']
        self.out=self.root/self.spec['output_directory']
        self.out.mkdir(parents=True,exist_ok=True)
        self.mesh=trimesh.load_mesh(self.source,process=True)
        for key,name in [('stand','STAND_STABILITY'),('landing','LANDING'),('cable_width','CABLE_WIDTH')]:
            getattr(config,name).update(self.cfg['mma'].get(key,{}))
        self.half,self.problem=frame_setup(self.cfg,self.cfg['shape'])
        self.functions=self.half['functional_requirements']
        self.parameters=FrameLayout(self.cfg['layout']).parameters()
        self.components=build_components(self.parameters,assembly_placements(self.parameters))
        self.body=_manifold(self.mesh)
    def provenance(self):
        root=self.root/self.cfg['mma']['root']
        runs=json.loads((root/'runs.json').read_text())
        matched=[name for name,path in runs.items() if (Path(path)/'frame.stl').resolve()==self.source.resolve()]
        derived=None
        if not matched and getattr(self,'spec',{}).get('derived_cables'):
            cable_path=self.root/self.spec['derived_cables']
            cable=json.loads(cable_path.read_text())
            parent=Path(cable['body'])
            matched=[name for name,path in runs.items() if (Path(path)/'frame.stl').resolve()==parent.resolve()]
            if not cable.get('delivered') or self.source.resolve()!=cable_path.with_name('frame.stl').resolve():
                raise ValueError('Cable result lacks a delivered pipeline artifact')
            derived={'kind':'cables','report':str(cable_path),'parent_frame':str(parent)}
        if len(matched)!=1:
            raise ValueError('Review requires a frame recorded by this optimization pipeline')
        manifest=json.loads((Path(derived['parent_frame']) if derived else self.source).with_name('manifest.json').read_text())
        definition=json.loads((root/self.cfg['mma']['fine_dir']/'problem_definition.json').read_text())
        current=shared_definition(self.half,self.problem)
        passed=definition['sha256']==current['sha256'] and manifest['stages']['reconstruction']['status']=='ran'
        return {'passed':passed,'route':self.cfg['mma']['optimizer'],'result_run':matched[0],'problem_definition_sha256':definition['sha256'],'current_problem_definition_sha256':current['sha256'],'frame_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'reconstruction':manifest['stages']['reconstruction'],'derived_artifact':derived}
    def shape_body(self,shape):
        points,faces=shape.tessellate(.05)
        return _manifold(trimesh.Trimesh(np.asarray([tuple(p) for p in points]),faces,process=True))
    def overlap(self,shape):
        return float((self.body^self.shape_body(shape)).volume())
    def section(self,axis,level,points,step):
        from matplotlib.path import Path as Polygon
        origin,normal=np.zeros(3),np.zeros(3)
        origin[axis],normal[axis]=level,1
        section=self.mesh.section(plane_origin=origin,plane_normal=normal)
        remaining=[i for i in range(3) if i!=axis]
        inside=np.zeros(len(points),dtype=bool)
        for loop in section.discrete if section is not None else []:
            if len(loop)>3:
                inside^=Polygon(loop[:,remaining]).contains_points(points)
        return inside,float(inside.sum()*step**2)
    def patch(self,low,high,step=.1):
        return np.stack(np.meshgrid(*[np.arange(a+step/2,b,step) for a,b in zip(low,high)],indexing='ij'),axis=-1).reshape(-1,2)
    def battery(self):
        from build123d import Vector
        from scipy.spatial import ConvexHull
        from matplotlib.path import Path as Polygon
        guide=self.functions['battery_guide']
        part=self.parameters['components']['battery']
        frame=self.parameters['frame']
        center=np.array([0,frame['battery_y_mm'],frame['deck_top_mm']])
        half=np.array([part['width_mm']/2,part['length_mm']/2])
        points=self.patch(center[:2]-half,center[:2]+half)
        inside,seat_area=self.section(2,center[2]-.05,points,.1)
        supported=points[inside]
        stable=len(supported)>3 and np.linalg.matrix_rank(supported-supported[0])==2 and Polygon(supported[ConvexHull(supported).vertices]).contains_point(center[:2])
        faces={}
        spacing=np.asarray(self.half['grid']['spacing_mm'])
        clearance=config.TOPOLOGY_CONFIG['component_clearance_mm']
        for axis,label in [(0,'side'),(1,'end')]:
            other=1-axis
            patch=self.patch([center[other]-half[other],center[2]+.05],[center[other]+half[other],center[2]+guide['height_mm']])
            maximum_gap=clearance+spacing[axis]
            for sign in (-1,1):
                name=('side_left' if sign<0 else 'side_right') if axis==0 else ('rear' if sign<0 else 'front')
                _,area=self.section(axis,center[axis]+sign*(half[axis]+maximum_gap),patch,.1)
                minimum=guide['contact_area_mm2']['side']/2 if axis==0 else guide['contact_area_mm2'][name]
                faces[name]={'area_mm2':area,'minimum_mm2':minimum,'maximum_checked_gap_mm':maximum_gap,'passed':area>=minimum}
        insertion=[{'dz_mm':dz,'overlap_mm3':self.overlap(self.components['battery']['shape'].translate(Vector(0,0,dz)))} for dz in (0,.5,1,2,3,5,10)]
        paths=[]
        for sx in (-1,1):
            for sy in (-1,1):
                overlaps=[]
                for dz in np.linspace(guide['height_mm']+.5,0,16):
                    offset=guide['entry_relief_mm']*min(float(dz)/guide['height_mm'],1)
                    overlaps.append(self.overlap(self.components['battery']['shape'].translate(Vector(sx*offset,sy*offset,float(dz)))))
                paths.append({'initial_offset_mm':[sx*guide['entry_relief_mm'],sy*guide['entry_relief_mm']],'maximum_overlap_mm3':max(overlaps)})
        cap=next(r for r in self.half['regions'] if r['name']=='battery_guide_height_cap')
        cap_volume=float((self.body^region_manifold(cap,.02)[0]).volume())
        passed=seat_area>=guide['seat_area_mm2'] and stable and all(r['passed'] for r in faces.values()) and max(r['overlap_mm3'] for r in insertion)<=1e-4 and max(r['maximum_overlap_mm3'] for r in paths)<=1e-4 and cap_volume<=1e-4
        return {'passed':bool(passed),'seat_area_mm2':seat_area,'minimum_seat_area_mm2':guide['seat_area_mm2'],'battery_center_inside_seat_support_hull':bool(stable),'guide_faces':faces,'vertical_insertion':insertion,'guided_entry_paths':paths,'material_above_guide_height_cap_mm3':cap_volume,'height_limit_mm':guide['height_mm'],'retention':'Ideal external press; rubber geometry, stiffness and preload excluded; guide transfers handling loads only'}
    def camera(self):
        flight=self.functions['low_flight']
        normal=np.asarray(flight['vertical_axis'])
        frame_bottom=float(np.min(self.mesh.vertices@normal))
        bottoms={**flight['component_bottoms_world_z_mm'],'generated_reconstructed_frame':frame_bottom}
        lowest=min(bottoms,key=bottoms.get)
        gap=flight['lens_world_z_mm']-bottoms[lowest]
        camera_overlap=self.overlap(self.components['camera']['shape'])
        center=np.asarray(flight['camera_center_mm'])
        samples=np.vstack((self.mesh.vertices,self.mesh.triangles_center))
        guard=(np.abs(samples[:,0])>=flight['camera_width_mm']/2+.5)&(np.abs(samples[:,0])<=flight['camera_width_mm']/2+6)&(np.abs(samples[:,1]-center[1])<=flight['camera_length_mm']/2+3)
        guard_bottom=float(np.min(samples[guard]@normal)) if guard.any() else None
        drop=None if guard_bottom is None else flight['camera_bottom_world_z_mm']-guard_bottom
        body=self.half['camera']
        axes=np.asarray(body['coverage']['axes'])
        offsets=samples-np.asarray(body['coverage']['front_mm'])
        depth,across,height=(offsets@axis for axis in axes)
        fov=self.problem['camera']
        tangents=np.tan(np.radians(fov['fov_deg'])/2)
        blocked=(depth>=-fov['fov_clearance_mm'])&(np.abs(across)<fov['aperture_mm']+fov['fov_clearance_mm']+np.maximum(depth,0)*tangents[0])&(np.abs(height)<fov['aperture_mm']+fov['fov_clearance_mm']+np.maximum(depth,0)*tangents[1])
        passed=gap<=flight['maximum_gap_mm']+1e-5 and camera_overlap<=1e-4 and drop is not None and drop>=flight['guard_drop_mm']-1e-5 and not blocked.any()
        return {'passed':bool(passed),'pitch_nose_down_deg':flight['pitch_deg'],'lens_mm':flight['lens_mm'],'lens_source':flight['lens_source'],'lens_to_whole_assembly_bottom_mm':float(gap),'maximum_gap_mm':flight['maximum_gap_mm'],'lowest_part':lowest,'part_bottoms_world_z_mm':bottoms,'camera_frame_overlap_mm3':camera_overlap,'local_guard_drop_below_camera_mm':drop,'minimum_guard_drop_mm':flight['guard_drop_mm'],'fov_blocked_surface_samples':int(blocked.sum()),'fov_check':'Vertices and triangle centroids in the shared 4:3 cone; sampled geometric check, independent mechanics remains required'}
    def plot(self,report):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.spatial import ConvexHull
        flight=self.functions['low_flight']
        angle=np.radians(flight['pitch_deg'])
        rotation=np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
        minimum=min(report['camera']['part_bottoms_world_z_mm'].values())
        fig,ax=plt.subplots(figsize=(10,5))
        vertices=self.mesh.vertices@rotation.T
        vertices[:,2]-=minimum
        ax.add_collection(PolyCollection(vertices[self.mesh.faces][:,:,[1,2]],facecolors='#7993a5',edgecolors='none',alpha=.5))
        for name,item in self.components.items():
            points,_=item['shape'].tessellate(.2)
            xyz=np.asarray([tuple(p) for p in points])@rotation.T
            xyz[:,2]-=minimum
            yz=xyz[:,[1,2]]
            hull=ConvexHull(yz)
            ax.fill(yz[hull.vertices,0],yz[hull.vertices,1],color='#d9a441' if name=='battery' else '#258d59' if name=='camera' else '#565e67',alpha=.15 if name.startswith('prop_') else .6)
        lens=np.asarray(flight['lens_mm'])@rotation.T
        lens[2]-=minimum
        ax.plot(*lens[[1,2]],'o',color='#c42b32')
        ax.annotate('',(lens[1],0),(lens[1],lens[2]),arrowprops={'arrowstyle':'<->','color':'#c42b32'})
        ax.text(lens[1]+2,lens[2]/2,f'{lens[2]:.2f} mm')
        ax.axhline(0,color='#333333')
        ax.set_xlim(-90,90)
        ax.set_ylim(-2,70)
        ax.set_aspect('equal')
        ax.set_xlabel('Vorwärtsrichtung in Fluglage [mm]')
        ax.set_ylabel('Höhe über der tiefsten Stelle des bestückten Copters [mm]')
        ax.set_title(f"Tatsächliches rekonstruiertes Pipeline-Ergebnis / {flight['pitch_deg']:g}° Nase nach unten")
        fig.tight_layout()
        fig.savefig(self.out/'flight_pose_15deg.png',dpi=150)
        plt.close(fig)
    def run(self):
        provenance=self.provenance()
        camera,battery=self.camera(),self.battery()
        connected=self.mesh.is_watertight and len(self.mesh.split(only_watertight=False))==1
        report={'passed':bool(provenance['passed'] and camera['passed'] and battery['passed'] and connected),'frame_source':str(self.source),'provenance':provenance,'camera':camera,'battery':battery,'connected_watertight_frame':bool(connected),'mechanical_acceptance':'Separate shared physical-field and independent reconstructed-frame FEA required'}
        (self.out/'report.json').write_text(json.dumps(report,indent=2))
        self.plot(report)
        print(json.dumps(report),flush=True)
        return report
    def ocp(self):
        report=json.loads((self.out/'report.json').read_text())
        physics=json.loads((self.out/'physical_evaluation.json').read_text())
        digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        if not report['passed'] or not physics['passed'] or report['provenance']['frame_sha256']!=digest or physics['frame_sha256']!=digest:
            raise ValueError('OCP requires the same functionally and mechanically checked pipeline artifact')
        from build123d import Vector,import_stl
        import orjson
        from ocp_vscode import show
        from ocp_vscode.comms import comms
        from ocp_viewer_core.codec import default
        from ocp_viewer_core.websocket import NO_VIEWER_CONFIG
        models=[import_stl(self.source),import_stl(self.root/'exports/runs/reference_manafly_original/geometry.stl')]
        widths=[model.bounding_box().size.X for model in models]
        positions=[-widths[0]/2-15,widths[1]/2+15]
        models=[model.translate(Vector(positions[i]-(model.bounding_box().min.X+model.bounding_box().max.X)/2,-(model.bounding_box().min.Y+model.bounding_box().max.Y)/2,-model.bounding_box().min.Z)) for i,model in enumerate(models)]
        names=[f"Pipeline {report['provenance']['route']} frame",'ManaFly 3']
        comms.set_port(3939)
        comms._send=lambda data,message_type,port=None,timeit=False:dict(NO_VIEWER_CONFIG)
        comms.send_data=lambda data,timeit=False:(self.out/'ocp_payload.json').write_bytes(orjson.dumps(data,default=default))
        comms.send_backend=lambda data,timeit=False:None
        show(*models,names=names,colors=['#6b879a','#9b9b9b'],alphas=[1,1],axes=False,grid=False,reset_camera='reset',progress='',timeit=False)
        (self.out/'ocp_artifacts.json').write_text(json.dumps({'names':names,'frame_source':str(self.source),'frame_sha256':digest,'reference':'exports/runs/reference_manafly_original/geometry.stl'},indent=2))

def send(path):
    import time
    from ocp_vscode.comms import comms
    spec=json.loads(Path(path).read_text())
    out=Path(spec['output_directory'])
    payload=json.loads((out/'ocp_payload.json').read_text())
    expected=json.loads((out/'ocp_artifacts.json').read_text())
    comms.set_port(3939)
    comms.send_data(payload)
    time.sleep(1)
    status=comms.send_command('status')
    (out/'ocp_status.json').write_text(json.dumps({'expected':expected,'viewer_status':status},indent=2,default=str))
    states=status.get('states',{}) if isinstance(status,dict) else {}
    if len(states)!=2 or not all(any(key.rsplit('/',1)[-1]==name for key in states) for name in expected['names']):
        raise RuntimeError('OCP viewer did not confirm exactly the two requested frame objects')
    print(json.dumps({'displayed':expected['names'],'frame_sha256':expected['frame_sha256']}),flush=True)

if __name__=='__main__':
    action,path=(sys.argv[1],sys.argv[2]) if len(sys.argv)==3 else ('run',sys.argv[1])
    if action=='send':
        send(path)
    elif action=='ocp':
        PipelineReview(path).ocp()
    else:
        sys.exit(0 if PipelineReview(path).run()['passed'] else 1)
