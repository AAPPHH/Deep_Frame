import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import orjson
import trimesh
from build123d import Align, Axis, Box, Compound, Line, Plane, Polygon, Pos, Sphere, Vector, extrude, import_stl
from deep_frame.frame import assembly_placements, build_components, camera_mount_z, intersection_shape
from deep_frame.frame_run import FrameLayout
from ocp_vscode import show
from ocp_vscode.comms import comms
from ocp_viewer_core.codec import default
from ocp_viewer_core.websocket import NO_VIEWER_CONFIG

class FunctionalReview:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.config = json.loads((self.root / (sys.argv[1] if len(sys.argv)>1 else 'docs/validation/dgx/functional_geometry_review.json')).read_text())
        self.out = self.root / self.config['output_directory']
        self.out.mkdir(parents=True, exist_ok=True)
        self.source = self.root / self.config['frame_source']
        self.frame = trimesh.load_mesh(self.source, process=True)
        self.frame_shape = import_stl(self.source)
        self.pitch = float(self.config['flight_pitch_nose_down_deg'])
        self.layouts = {}
        for name, filename in [('previous', self.config['camera']['previous_layout']), ('low_camera_proposal', self.config['camera']['proposed_layout'])]:
            request = json.loads((self.root / filename).read_text())['layout']
            request['overrides']['battery'] = {'support': 'free'}
            if name == 'low_camera_proposal':
                request['overrides']['camera'].update(self.config['camera']['proposed_camera_overrides'])
            parameters = FrameLayout(request).parameters()
            components = build_components(parameters, assembly_placements(parameters))
            self.layouts[name] = (parameters, components)
        self.guide, self.band = self.battery_guide(self.layouts['low_camera_proposal'][0])
    def battery_guide(self, parameters):
        battery, frame = parameters['components']['battery'], parameters['frame']
        half_x, half_y, bottom = battery['width_mm']/2, battery['length_mm']/2, frame['deck_top_mm']
        y = frame['battery_y_mm']
        pieces = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                station = y + sy*(half_y-2.5)
                corner = Pos(sx*(half_x-2.6), y+sy*(half_y-1.8), bottom-1)*Box(10,8.4,2)
                inner = half_x + 0.4
                profile = [(sx*inner,bottom), (sx*(inner+2),bottom), (sx*(inner+2),bottom+2), (sx*(inner+0.8),bottom+2), (sx*inner,bottom+1.2)]
                wall = extrude(Plane.XZ * Polygon(*profile, align=None), amount=5, dir=(0,-1,0))
                corner += Pos(0,station+2.5,0)*wall
                inner_y = half_y + 0.4
                end_profile = [(sy*inner_y,bottom),(sy*(inner_y+2),bottom),(sy*(inner_y+2),bottom+2),(sy*(inner_y+0.8),bottom+2),(sy*inner_y,bottom+1.2)]
                end_wall = extrude(Plane.YZ * Polygon(*end_profile,align=None),amount=6,dir=(1,0,0))
                corner += Pos(sx*(half_x-3)-3,y,0)*end_wall
                pieces.append(corner)
        guide = Compound(children=pieces)
        top, low, x = bottom+battery['height_mm'], bottom-2.5, half_x+2.5
        band = Compound(children=[Pos(0,y,top+0.3)*Box(2*x+0.6,4,0.6),Pos(0,y,low-0.3)*Box(2*x+0.6,4,0.6),*[Pos(s*x,y,(top+low)/2)*Box(0.6,4,top-low) for s in (-1,1)]])
        return guide, band
    def rotation(self, pitch):
        angle = np.radians(pitch)
        return np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
    def lens(self, parameters):
        spec, frame = parameters['components']['camera'], parameters['frame']
        tilt = np.radians(spec['tilt_deg'])
        return np.array([0,frame['camera_y_mm']+spec['length_mm']/2*np.cos(tilt),camera_mount_z(parameters)+spec['length_mm']/2*np.sin(tilt)])
    def posed(self, shape, pitch, shift=(0,0,0)):
        return shape.rotate(Axis.X,-pitch).translate(Vector(*shift))
    def measurement(self, parameters, components, pitch):
        rotation = self.rotation(pitch)
        vertices = self.frame.vertices @ rotation.T
        index = int(np.argmin(vertices[:,2]))
        bottoms = {'frame_checkpoint_110': float(vertices[index,2])}
        for name, item in components.items():
            bottoms[name] = float(self.posed(item['shape'],pitch).bounding_box().min.Z)
        bottoms['proposed_battery_guide'] = float(self.posed(self.guide,pitch).bounding_box().min.Z)
        bottoms['illustrative_rubber_band'] = float(self.posed(self.band,pitch).bounding_box().min.Z)
        lowest = min(bottoms,key=bottoms.get)
        lens = self.lens(parameters) @ rotation.T
        return {'pitch_nose_down_deg':pitch,'lens_world_mm':lens.tolist(),'lowest_part':lowest,'whole_assembly_min_z_mm':bottoms[lowest],'lens_to_whole_assembly_bottom_mm':float(lens[2]-bottoms[lowest]),'part_min_z_mm':bottoms,'lowest_frame_vertex_mm':vertices[index].tolist(),'camera_optical_axis_above_horizon_deg':parameters['components']['camera']['tilt_deg']-pitch}
    def checks(self):
        parameters, components = self.layouts['low_camera_proposal']
        battery = components['battery']['shape']
        insertion = []
        for dz in (0,0.2,0.5,1,2,3,6,15):
            overlap = intersection_shape(self.guide, battery.translate(Vector(0,0,dz)))
            insertion.append({'vertical_offset_mm':dz,'guide_intersection_mm3':float(overlap.volume) if overlap is not None else 0.0})
        guided_paths = []
        for sx in (-1,1):
            for sy in (-1,1):
                maximum = 0.0
                for dz in np.linspace(3,0,13):
                    displacement = min(1.0,0.3+max(0.0,float(dz)-1.2))
                    overlap = intersection_shape(self.guide,battery.translate(Vector(sx*displacement,sy*displacement,float(dz))))
                    maximum = max(maximum,float(overlap.volume) if overlap is not None else 0.0)
                guided_paths.append({'initial_xy_offset_mm':[sx,sy],'seated_xy_offset_mm':[sx*0.3,sy*0.3],'max_intersection_mm3':maximum,'sample_count':13})
        shift = {}
        for direction, delta in [('x_plus',(0.41,0,0)),('x_minus',(-0.41,0,0)),('y_plus',(0,0.41,0)),('y_minus',(0,-0.41,0))]:
            overlap = intersection_shape(self.guide,battery.translate(Vector(*delta)))
            shift[direction] = float(overlap.volume) if overlap is not None else 0.0
        camera = parameters['components']['camera']
        center = np.array([0,parameters['frame']['camera_y_mm'],camera_mount_z(parameters)])
        a = np.radians(camera['tilt_deg'])
        camera_rotation = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        local = (self.frame.vertices-center) @ camera_rotation
        inside = np.all(np.abs(local)<np.array([camera['width_mm'],camera['length_mm'],camera['height_mm']])/2-1e-6,axis=1)
        return {'guide_geometric_screen_passed':all(row['guide_intersection_mm3']<1e-6 for row in insertion) and all(row['max_intersection_mm3']<1e-6 for row in guided_paths) and all(v>0 for v in shift.values()) and len(self.guide.solids())==4,'vertical_insertion':insertion,'guided_diagonal_insertion_paths':guided_paths,'nominal_guide_battery_intersection_mm3':insertion[0]['guide_intersection_mm3'],'lateral_contact_at_0_41mm_intersection_mm3':shift,'connected_guide_corner_count':len(self.guide.solids()),'seat_top_z_mm':parameters['frame']['deck_top_mm'],'side_gap_mm':0.4,'end_gap_mm':0.4,'guide_height_above_battery_seat_mm':2,'entry_side_gap_mm':1.2,'top_overhang':False,'retention':'Rubber band secures flight assembly; guide only positions battery during handling. Band is illustrative, no measured band specification or preload.','low_camera_geometry_fit_passed':not bool(inside.any()),'low_camera_frame_vertices_inside_camera_envelope':int(inside.sum()),'camera_interference_limit':'Vertex containment detects intersections; zero would not prove absence. Camera mount is not relocated in historical frame.','excluded_unmodelled_hardware':['battery leads','XT30 and balance connector loose placement','VTX antenna flexible placement','motor screws/shaft extensions'],'lens_definition':'Front-face midpoint of 16 x 14 x 14 mm Lux envelope; actual lens optical centre and side screw placement require hardware measurement.'}
    def scene(self, report):
        shapes, names, colors, alphas = [], [], [], []
        for name, x in [('previous',-105),('low_camera_proposal',105)]:
            parameters, components = self.layouts[name]
            measure = next(row for row in report['camera'][name] if row['pitch_nose_down_deg']==self.pitch)
            shift = (x,0,-measure['whole_assembly_min_z_mm'])
            shapes.append(self.posed(self.frame_shape,self.pitch,shift)); names.append(name+' / historical SIMP checkpoint 110 - NOT redesigned'); colors.append('#8b9caa'); alphas.append(0.65)
            for key,item in components.items():
                shapes.append(self.posed(item['shape'],self.pitch,shift)); names.append(name+' / '+key)
                colors.append('#d9a441' if key=='battery' else '#258d59' if key=='camera' else '#674a88' if key.startswith('prop_') else '#304955'); alphas.append(0.13 if key.startswith('prop_') else 0.85)
            shapes.append(self.posed(self.band,self.pitch,shift)); names.append(name+' / illustrative rubber band path - frame clearance unverified'); colors.append('#8b3946'); alphas.append(0.8)
            lens = np.array(measure['lens_world_mm'])+shift
            shapes.extend([Pos(*lens)*Sphere(0.7),Line(tuple(lens),tuple(lens*np.array([1,1,0])))])
            names.extend([name+' / LENS PROXY',name+f" / lens to entire bottom {measure['lens_to_whole_assembly_bottom_mm']:.2f} mm at {self.pitch:g} deg"])
            colors.extend(['#e73737','#e73737']); alphas.extend([1,1])
        parameters, components = self.layouts['low_camera_proposal']
        shift=(275,0,0)
        for shape,label,color,alpha in [(self.guide,'PROPOSED GUIDE - 2 mm high - no top retention','#2b96c3',1),(components['battery']['shape'],'INSERTED LiPo - separate guide concept','#d9a441',0.65),(self.band,'ILLUSTRATIVE RUBBER BAND - actual securing function','#8b3946',1)]:
            shapes.append(self.posed(shape,self.pitch,shift)); names.append(label); colors.append(color); alphas.append(alpha)
        shapes.append(Pos(70,0,-0.25)*Box(460,190,0.5)); names.append('Ground tangent plane / each complete assembly lowest point Z=0'); colors.append('#c9c9c9'); alphas.append(0.2)
        comms.set_port(3939)
        comms._send=lambda data,message_type,port=None,timeit=False:dict(NO_VIEWER_CONFIG)
        comms.send_data=lambda data,timeit=False:(self.out/'ocp_payload.json').write_bytes(orjson.dumps(data,default=default))
        comms.send_backend=lambda data,timeit=False:None
        show(*shapes,names=names,colors=colors,alphas=alphas,axes=True,grid=True,progress='',timeit=False)
    def plot(self, report):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.spatial import ConvexHull
        fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
        for ax,(name,(parameters,components)) in zip(axes,self.layouts.items()):
            row=next(row for row in report['camera'][name] if row['pitch_nose_down_deg']==self.pitch)
            offset=-row['whole_assembly_min_z_mm']
            vertices=self.frame.vertices@self.rotation(self.pitch).T; vertices[:,2]+=offset
            ax.add_collection(PolyCollection(vertices[self.frame.faces][:,:,[1,2]],facecolors='#8b9caa',edgecolors='none',alpha=.35))
            for key,item in components.items():
                points,_=item['shape'].tessellate(.15)
                xyz=np.asarray([tuple(p) for p in points])@self.rotation(self.pitch).T; xyz[:,2]+=offset
                yz=xyz[:,[1,2]]; hull=ConvexHull(yz)
                ax.fill(yz[hull.vertices,0],yz[hull.vertices,1],color='#d9a441' if key=='battery' else '#258d59' if key=='camera' else '#674a88' if key.startswith('prop_') else '#304955',alpha=.15 if key.startswith('prop_') else .65)
            lens=np.array(row['lens_world_mm']); lens[2]+=offset
            ax.plot(*lens[[1,2]],'o',color='#e73737'); ax.annotate('',(lens[1],0),(lens[1],lens[2]),arrowprops={'arrowstyle':'<->','color':'#e73737'})
            ax.text(lens[1]+2,lens[2]/2,f"{row['lens_to_whole_assembly_bottom_mm']:.2f} mm",color='#d12b2b')
            ax.axhline(0,color='#333333'); ax.set_xlim(-85,85); ax.set_ylim(-2,80); ax.set_aspect('equal'); ax.grid(alpha=.2)
            ax.set_title(('Bisherige Kamera' if name=='previous' else 'Tiefere Kamera, gleicher historischer Rahmen')+f'\n{self.pitch:g}° Nase nach unten / '+row['lowest_part']); ax.set_xlabel('Vorwärtsrichtung in Fluglage [mm]')
        axes[0].set_ylabel('Höhe über tiefster Stelle des bestückten Copters [mm]')
        fig.suptitle('Funktionsprüfung: Linsenposition = vorläufiger Mittelpunkt der Kamerafront; kein fertiger neuer Rahmen')
        fig.tight_layout(); fig.savefig(self.out/f'flight_pose_{self.pitch:g}deg.png',dpi=150); plt.close(fig)
    def run(self):
        report={'status':'FUNCTION_REVIEW_NOT_OPTIMIZATION_APPROVAL','frame_source':str(self.source.relative_to(self.root)),'frame_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'flight_pitch_nose_down_deg':self.pitch,'camera':{name:[self.measurement(parameters,components,pitch) for pitch in sorted(set((0,10,self.pitch,20,30)))] for name,(parameters,components) in self.layouts.items()},'battery_guide':self.checks(),'limitations':'Historical raw density frame, not postprocessed; new camera has no matching mounting geometry yet; guide is separate CAD concept. This does not verify a new complete design, physical insertion friction, rubber band forces or print feasibility.'}
        (self.out/'report.json').write_text(json.dumps(report,indent=2))
        self.plot(report); self.scene(report)
        print(json.dumps({'report':str(self.out/'report.json'),'camera_in_requested_flight_pose':{name:next(r for r in rows if r['pitch_nose_down_deg']==self.pitch) for name,rows in report['camera'].items()},'battery_guide':report['battery_guide']}),flush=True)
if __name__ == '__main__':
    FunctionalReview().run()
