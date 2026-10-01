from io import BytesIO
from itertools import product
from time import perf_counter

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, maximum_filter

from deep_frame.topology_geometry import _primitive_wall_checks, _trapped_voids, region_bounds
from deep_frame.topology_implicit import AXES, ImplicitField, _bounds, _envelope, _manifold, _point_distances, _surface_samples, float32_margin, frozen_gates, mesh_checks, primitive_distance, region_manifold
from deep_frame.topology_surface import _mesh_summary, _progress, _statistics
from deep_frame.topology_surface_validation import _settings as _validation_settings, _triangle_samples, surface_maturity, surface_metrics

COLUMN_JITTER = 1e-4*np.array([np.sqrt(2), np.sqrt(3)])
BUDGETS = ("maximum_wall_samples", "maximum_void_columns")

def segment_hits(mesh, origins, directions, lower, upper, batch=10000):
    triangles, tree, found = mesh.triangles, mesh.triangles_tree, []
    for start in range(0, len(origins), batch):
        origin, direction = origins[start:start+batch], directions[start:start+batch]
        ends = (origin+lower*direction, origin+upper*direction)
        faces, counts = tree.intersection_v(np.minimum(*ends)-1e-6, np.maximum(*ends)+1e-6)
        rays, faces = np.repeat(np.arange(len(origin)), counts.astype(np.int64)), faces.astype(np.int64)
        corner = triangles[faces, 0]
        first, second, ray = triangles[faces, 1]-corner, triangles[faces, 2]-corner, direction[rays]
        across = np.cross(ray, second)
        determinant = np.einsum("ij,ij->i", first, across)
        offset = origin[rays]-corner
        normal = np.cross(offset, first)
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v, t = (np.einsum("ij,ij->i", a, b)/determinant for a, b in ((offset, across), (ray, normal), (second, normal)))
            hit = (np.abs(determinant) > 1e-14*np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)) & (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9) & (t >= lower) & (t <= upper)
        found.append((rays[hit]+start, faces[hit], t[hit]))
    return [np.concatenate(values) for values in zip(*found)] if found else [np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0)]

def corner_normals(mesh, crease_deg):
    faces, normals, angles = mesh.faces, mesh.face_normals, mesh.face_angles
    star, limit, result = mesh.vertex_faces, np.cos(np.radians(crease_deg)), np.empty((len(faces), 3, 3))
    batch = max(1, 1000000//star.shape[1])
    for start in range(0, len(faces), batch):
        corner = faces[start:start+batch]
        incident = star[corner]
        valid = incident >= 0
        incident = np.where(valid, incident, 0)
        weight = (angles[incident]*(faces[incident] == corner[..., None, None])).sum(axis=-1)
        weight *= valid & (np.einsum("fcdk,fk->fcd", normals[incident], normals[start:start+batch]) >= limit)
        total = np.einsum("fcd,fcdk->fck", weight, normals[incident])
        length = np.linalg.norm(total, axis=-1, keepdims=True)
        result[start:start+batch] = np.where(length > 1e-12, total/np.maximum(length, 1e-300), normals[start:start+batch, None])
    return result

def prescribed_bores(regions, tolerance, margin):
    return [(region, region_manifold(region, tolerance, margin)[1]) for region in regions if region["role"] == "forbidden" and region["kind"] == "cylinder" and region.get("rasterize", True) is False]

def bore_allowance(points, bores, margin):
    allowance = np.full(len(points), np.nan)
    for region, polygon in bores:
        axis = AXES[region.get("axis", "z")]
        low, high = region_bounds(region)
        radial = np.sqrt(sum((points[:, i]-region["center_mm"][i])**2 for i in range(3) if i != axis))
        on = (radial >= region["radius_mm"]-margin) & (radial <= polygon["cut_radius_mm"]+margin) & (points[:, axis] >= low[axis]-margin) & (points[:, axis] <= high[axis]+margin)
        allowance[on] = np.fmax(allowance[on], polygon["oversize_mm"])
    return allowance

def wall_screen(mesh, minimum, settings, crease_deg, bores=(), margin=0.0):
    started = perf_counter()
    spacing, tolerance, guard = settings["wall_sample_spacing_mm"], settings["wall_tolerance_mm"], 1e-7
    triangles = mesh.triangles
    small = np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1) <= spacing
    points, sources = [triangles[small].mean(axis=1)], [np.flatnonzero(small)]
    count = len(sources[0])
    for face in np.flatnonzero(~small):
        samples = list(_triangle_samples(triangles[face], spacing))
        points.append(np.asarray(samples))
        sources.append(np.full(len(samples), face))
        count += len(samples)
        if count > settings["maximum_wall_samples"]:
            break
    if count > settings["maximum_wall_samples"]:
        return {"passed": False, "complete": False, "reason": "wall sample budget exceeded", "maximum_wall_samples": settings["maximum_wall_samples"]}
    points, sources = np.concatenate(points), np.concatenate(sources)
    weights = trimesh.triangles.points_to_barycentric(triangles[sources], points)
    directions = np.einsum("sc,sck->sk", weights, corner_normals(mesh, crease_deg)[sources])
    length = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.where(length > 1e-12, directions/np.maximum(length, 1e-300), mesh.face_normals[sources])
    rays, faces, distances = segment_hits(mesh, points, -directions, -guard, minimum+tolerance)
    other = faces != sources[rays]
    beyond = other & (distances > guard)
    thickness = np.full(len(points), np.inf)
    np.minimum.at(thickness, rays[beyond], distances[beyond])
    thin = thickness < minimum-tolerance
    candidates = np.flatnonzero(thin)
    allowance = bore_allowance(points[candidates], bores, margin)+bore_allowance(points[candidates]-thickness[candidates, None]*directions[candidates], bores, margin)
    web = np.isfinite(allowance)
    bore_web = np.zeros(len(points), dtype=bool)
    bore_web[candidates[web]] = True
    thin[candidates[web]] = thickness[candidates[web]] < minimum-tolerance-allowance[web]
    unresolved = np.zeros(len(points), dtype=bool)
    unresolved[rays[other & (np.abs(distances) <= guard)]] = True
    unresolved &= ~thin
    measured = np.isfinite(thickness)
    lowest = int(np.argmin(thickness)) if measured.any() else None
    return {"passed": bool(len(points) and not thin.any() and not unresolved.any()), "complete": True, "minimum_required_mm": minimum, "query_upper_bound_mm": minimum+tolerance,
            "minimum_measured_mm": float(thickness[lowest]) if lowest is not None else None,
            "minimum_sample": {"face": int(sources[lowest]), "position_mm": points[lowest].tolist(), "thickness_mm": float(thickness[lowest])} if lowest is not None else None,
            "ray_count": len(points), "rays_clear_through_upper_bound": int(np.count_nonzero(~measured)), "thin_sample_count": int(thin.sum()), "unresolved_sample_count": int(unresolved.sum()),
            "prescribed_bore_web": {"sample_count": int(bore_web.sum()), "minimum_measured_mm": float(thickness[bore_web].min()) if bore_web.any() else None, "maximum_allowance_mm": float(allowance[web].max()) if web.any() else None,
                                    "method": "Chords whose origin and hit both lie on the realised circumscribed polygon of a prescribed bore (rasterize=False keep-out cylinder, radial band [r, cut radius] within the binary32 margin) may fall short of the minimum by the sum of the two polygons' oversize from segment count and radius; all other chords keep the strict minimum"},
            "thin_samples": [{"face": int(sources[i]), "position_mm": points[i].tolist(), "thickness_mm": float(thickness[i])} for i in np.flatnonzero(thin)[:30]],
            "unresolved_samples": [{"face": int(sources[i]), "position_mm": points[i].tolist()} for i in np.flatnonzero(unresolved)[:30]],
            "ray_origin_exclusion_mm": guard, "maximum_sample_triangle_edge_mm": spacing, "normal_crease_deg": crease_deg, "elapsed_s": perf_counter()-started,
            "method": "Final mesh triangles subdivided until every sample triangle edge meets the spacing limit; float64 Moller-Trumbore along the inward surface normal (barycentric interpolation of angle-weighted corner normals over incident faces within the remeshing feature angle, so sharp creases keep the facet normal) over (1e-7, minimum + tolerance] against rtree-culled triangles; another triangle hit within +/-1e-7 of an otherwise passing origin is unresolved",
            "limitations": "Finite normal-chord screen, not a global minimum-thickness proof; sharp convex wedges can be conservatively rejected"}

def _lattice(triangles, spacing):
    counts = np.maximum(np.ceil(np.linalg.norm(triangles-np.roll(triangles, -1, axis=1), axis=2).max(axis=1)/spacing).astype(int), 1)
    points = []
    for count in np.unique(counts):
        i, j = (value.ravel() for value in np.meshgrid(np.arange(count+1), np.arange(count+1)))
        keep = i+j <= count
        weights = np.column_stack((count-i[keep]-j[keep], i[keep], j[keep]))/count
        points.append(np.einsum("kb,tbd->tkd", weights, triangles[counts == count]).reshape(-1, 3))
    return np.concatenate(points)

def _penetration(mesh, bounds, region, spacing):
    low, high = _bounds(region)
    near = np.all((bounds[0] <= high+1e-6) & (bounds[1] >= low-1e-6), axis=1)
    if not near.any():
        return None
    return max(float(primitive_distance(_lattice(mesh.triangles[near], spacing).T, region).max()), 0.0)

def _dilate(mask, spacing, radius):
    result = np.zeros(mask.shape, dtype=bool)
    if mask.any():
        pad = np.ceil(radius/spacing).astype(int)+1
        window = []
        for axis in range(3):
            present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
            window.append(slice(max(int(present[0])-pad[axis], 0), int(present[-1])+pad[axis]+1))
        window = tuple(window)
        result[window] = distance_transform_edt(~mask[window], sampling=spacing) <= radius
    return result

def stl_roundtrip(mesh):
    loaded = trimesh.load_mesh(BytesIO(trimesh.exchange.stl.export_stl(mesh)), file_type="stl")
    report = {"checks": mesh_checks(loaded), "vertices_equal": len(loaded.vertices) == len(mesh.vertices), "faces_equal": len(loaded.faces) == len(mesh.faces),
              "maximum_float32_displacement_mm": float(np.abs(np.asarray(mesh.vertices, dtype=np.float32).astype(float)-mesh.vertices).max(initial=0)),
              "method": "Binary STL as written by export_mesh, reloaded with vertex merging; closed, oriented, vertex-manifold, manifold3d and self-intersection checks repeated on its float32 coordinates"}
    report["passed"] = report["checks"]["passed"] and report["vertices_equal"] and report["faces_equal"]
    return loaded, report

def _union(solids):
    from manifold3d import Manifold, OpType
    return Manifold.batch_boolean(solids, OpType.Add) if solids else None

def _cube(low, high):
    from manifold3d import Manifold
    return Manifold.cube((np.asarray(high)-np.asarray(low)).tolist()).translate(np.asarray(low).tolist())

def _column_crossings(mesh, columns, bottom, length):
    origins = np.column_stack((columns, np.full(len(columns), bottom)))
    rays, _, distances = segment_hits(mesh, origins, np.tile([0.0, 0.0, 1.0], (len(origins), 1)), 0.0, length)
    order = np.lexsort((distances, rays))
    rays, heights = rays[order], bottom+distances[order]
    keep = np.ones(len(rays), dtype=bool)
    keep[1:] = (rays[1:] != rays[:-1]) | (np.diff(heights) > 1e-6)
    return rays[keep], heights[keep], np.bincount(rays[keep], minlength=len(columns)) % 2 == 1

class MeshAcceptance:
    def __init__(self, mesh, domain, config, settings):
        self.mesh, self.domain, self.config, self.settings = mesh, domain, config, settings
        self.tolerance = settings["volume_tolerance_mm3"]
        manufacturing = domain["manufacturing"]
        self.minimum = max(manufacturing["nozzle_width_mm"]*manufacturing["minimum_wall_nozzles"], manufacturing["minimum_feature_mm"])
        self.regions = {role: [region for region in domain["regions"] if region["role"] == role] for role in ("allowed", "preserve", "forbidden")}
        self.margin = float32_margin(domain)
        self.stl, self.stl_report = stl_roundtrip(mesh)
        self.bodies = {"float64": _manifold(mesh), "stl": _manifold(self.stl)}
        self.body, self.cavities, self.metrics = self.bodies["float64"], None, None

    def volumes(self, operation):
        values = {name: operation(body).volume() for name, body in self.bodies.items()}
        return max(values.values()), values

    def topology(self):
        checks = mesh_checks(self.mesh)
        self.intersections = checks["self_intersections"]
        bodies = self.mesh.split(only_watertight=False) if self.mesh.body_count > 1 else [self.mesh]
        self.cavities = int(sum(body.volume < 0 for body in bodies))
        topology = {**checks["topology"], "signed_volume_mm3": float(self.mesh.volume), "negative_volume_shells": self.cavities, "stl_roundtrip": self.stl_report}
        topology["passed"] = checks["topology"]["passed"] and topology["signed_volume_mm3"] > self.tolerance and not self.cavities and self.stl_report["passed"]
        return topology

    def envelope(self):
        tolerance = self.config["segment_tolerance_mm"]
        envelope = _envelope(self.domain)
        box = region_manifold(envelope, tolerance)[0]
        outside, by_body = self.volumes(lambda body: body-box)
        allowed = _union([region_manifold(region, tolerance, self.margin, inscribed=True)[0] for region in self.regions["allowed"]])
        outside_allowed, allowed_by_body = self.volumes(lambda body: body-allowed) if allowed is not None else (outside, by_body)
        stl_outside = max(float(-primitive_distance(self.stl.vertices.T, envelope).min()), 0.0)
        return {"outside_grid_box_mm3": outside, "outside_allowed_regions_mm3": outside_allowed, "outside_grid_box_by_body_mm3": by_body, "outside_allowed_regions_by_body_mm3": allowed_by_body, "stl_maximum_vertex_outside_mm": stl_outside,
                "passed": max(outside, outside_allowed) <= self.tolerance and stl_outside <= self.config["penetration_tolerance_mm"],
                "method": "manifold3d differences of the float64 mesh and of the reloaded binary STL against the exact envelope box and the inscribed allowed polyhedra, the larger gated; binary STL vertices against the exact box"}

    def forbidden(self):
        rows, spacing = {}, self.config["penetration_sample_spacing_mm"]
        bounds = [(item, (item.triangles.min(axis=1), item.triangles.max(axis=1))) for item in (self.mesh, self.stl)]
        for region in self.regions["forbidden"]:
            solid, polygon = region_manifold(region, self.config["segment_tolerance_mm"], self.margin)
            volume, by_body = self.volumes(lambda body: body ^ solid)
            penetration, stl_penetration = (_penetration(item, box, region, spacing) for item, box in bounds)
            rows[region["name"]] = {"intersection_mm3": volume, "intersection_by_body_mm3": by_body, "circumscribed": polygon, "maximum_penetration_mm": penetration, "stl_maximum_penetration_mm": stl_penetration,
                                    "passed": volume <= self.tolerance and all(value is None or value <= self.config["penetration_tolerance_mm"] for value in (penetration, stl_penetration))}
        return rows

    def preserve(self):
        rows = {}
        settings, tolerance = self.settings, self.config["segment_tolerance_mm"]
        depth, offsets = settings["attachment_probe_depth_mm"], settings["attachment_offsets_mm"]
        forbidden = _union([region_manifold(region, tolerance, self.margin, 2, 1)[0] for region in self.regions["forbidden"]])
        for region in self.regions["preserve"]:
            solid = region_manifold(region, tolerance, self.margin)[0]
            required = solid-forbidden if forbidden is not None else solid
            low, high = region_bounds(region)
            pad = max(offsets)+depth+tolerance+1e-6
            window = _cube(low-pad, high+pad)
            local = self.bodies["stl"] ^ window
            levels = []
            for offset in offsets:
                sections = {}
                for axis, side in product(range(3), (-1, 1)):
                    start, stop = low.copy(), high.copy()
                    start[axis] = low[axis]-offset-depth if side < 0 else high[axis]+offset
                    stop[axis] = start[axis]+depth
                    sections["xyz"[axis]+("-" if side < 0 else "+")] = (local ^ _cube(start, stop)).volume()/depth
                levels.append({"offset_mm": offset, "sections_mm2": sections, "summed_area_mm2": sum(sections.values())})
            area = region.get("attachment_area_min_mm2", self.domain["manufacturing"]["minimum_attachment_area_mm2"])
            attachment = {"passed": all(level["summed_area_mm2"]+1e-6 >= area for level in levels), "minimum_required_area_mm2": area, "levels": levels,
                          "binding_offset_mm": min(levels, key=lambda level: level["summed_area_mm2"])["offset_mm"],
                          "method": "Summed mean areas of manifold3d intersections of the reloaded binary STL with six slabs outside the exact preserve AABB at all prescribed offsets"}
            walls = _primitive_wall_checks(region, self.regions["forbidden"], max(self.minimum, region.get("minimum_wall_mm", self.minimum)))
            missing, by_body = self.volumes(lambda body: required-(body ^ window))
            rows[region["name"]] = {"required_volume_mm3": required.volume(), "missing_volume_mm3": missing, "missing_by_body_mm3": by_body, "walls": walls, "attachment": attachment,
                                    "required": "circumscribed preserve minus the forbidden polygons grown by two binary32 margins (the cut holes plus rounding); missing volume gated on the float64 mesh and the reloaded binary STL"}
            rows[region["name"]]["passed"] = rows[region["name"]]["required_volume_mm3"] > self.tolerance and rows[region["name"]]["missing_volume_mm3"] <= self.tolerance and walls["passed"] and attachment["passed"]
        return rows

    def features(self, field, report):
        config, h = self.config, float(field.spacing.max())
        required = self.minimum/2+h*np.sqrt(3)/2+config["ripple_sigma_mm"]**2/self.minimum+config["remesh_max_surface_distance_mm"]
        opening = {"opening_radius_mm": config["opening_radius_mm"], "required_radius_mm": required, "applied": report.get("opening") is not None,
                   "method": "Opening radius covers half the minimum wall plus the EDT error h*sqrt(3)/2, the ripple-smoothing shrink sigma^2/minimum and the remeshing surface distance"}
        opening["passed"] = bool(opening["applied"] and config["opening_radius_mm"] >= required)
        rays = wall_screen(self.mesh, self.minimum, self.settings, config["remesh_feature_deg"], prescribed_bores(self.regions["forbidden"], config["segment_tolerance_mm"], self.margin), self.margin)
        return {"minimum_wall_mm": self.minimum, "field_opening": opening, "mesh_wall_screen": rays, "passed": opening["passed"] and rays["passed"]}

    def supports(self):
        started = perf_counter()
        settings, mesh, manufacturing = self.settings, self.mesh, self.domain["manufacturing"]
        envelope = _envelope(self.domain)
        lower, upper, spacing = envelope["min_mm"], envelope["max_mm"], settings["void_grid_spacing_mm"]
        shape = np.ceil((upper-lower)/spacing).astype(int)
        needed = bool(np.any((mesh.face_normals[:, 2] < -0.1) & (mesh.triangles_center[:, 2] > mesh.bounds[0, 2]+1e-6)))
        result = {"supports_required": needed, "build_direction": manufacturing["build_direction"], "supports_allowed": bool(manufacturing["supports_allowed"])}
        if shape[0]*shape[1] > settings["maximum_void_columns"]:
            return {**result, "passed": False, "accessibility": {"passed": False, "reason": "void column budget exceeded"}}
        columns = np.stack(np.meshgrid(*(lower[axis]+(np.arange(shape[axis])+0.5)*spacing for axis in range(2)), indexing="ij"), axis=-1).reshape(-1, 2)
        bottom, length = lower[2]-spacing, upper[2]-lower[2]+2*spacing
        rays, heights, odd = _column_crossings(mesh, columns, bottom, length)
        ambiguous = np.flatnonzero(odd)
        if len(ambiguous):
            again, again_heights, still = _column_crossings(mesh, columns[ambiguous]+COLUMN_JITTER*spacing, bottom, length)
            keep = ~odd[rays]
            rays, heights = np.concatenate((rays[keep], ambiguous[again])), np.concatenate((heights[keep], again_heights))
            odd[ambiguous] = still
        toggles = np.zeros((len(columns), shape[2]+1), dtype=np.int64)
        np.add.at(toggles, (rays, np.searchsorted(lower[2]+(np.arange(shape[2])+0.5)*spacing, heights, side="right")), 1)
        occupied = np.cumsum(toggles, axis=1)[:, :shape[2]] % 2 == 1
        occupied[odd] = False
        trapped = _trapped_voids(occupied.reshape(tuple(shape)))
        accessibility = {"passed": not trapped and self.cavities == 0 and not odd.any(), "negative_volume_shells": self.cavities, "trapped_void_cells": trapped, "unresolved_columns": int(odd.sum()), "jittered_columns": len(ambiguous),
                         "grid_spacing_mm": spacing, "grid_shape": shape.tolist(), "occupied_cells": int(occupied.sum()), "elapsed_s": perf_counter()-started,
                         "method": "Vertical float64 segment-triangle crossings per column centre, deduplicated within 1e-6 mm, odd columns (rays grazing an edge or lying in a constraint plane) re-shot once at the fixed offset 1e-4*(sqrt 2, sqrt 3) cells and unresolved if still odd; six-connected exterior void flood fill of the cell-centre occupancy plus the negative-volume shell count"}
        return {**result, "accessibility": accessibility, "passed": manufacturing["build_direction"] == [0, 0, 1] and (manufacturing["supports_allowed"] or not needed) and accessibility["passed"]}

    def deviation(self, field, report):
        config = self.config
        extraction = {key: report["remesh"]["fidelity"][key] for key in ("maximum_sampled_deviation_mm", "relative_volume_change")}
        extraction["passed"] = extraction["maximum_sampled_deviation_mm"] <= config["surface_deviation_mm"] and abs(extraction["relative_volume_change"]) <= config["relative_volume_change"]
        reference = ImplicitField(field.origin, field.spacing, field.layers["reference"]).extract()[0]
        margin = float(np.linalg.norm(field.spacing))/2
        stencil = maximum_filter(np.pad(field.layers["extension_changed"], 2), size=4, origin=-1)[np.ix_(*(np.arange(n)//config["subdivisions"]+1 for n in field.values.shape))]
        excluded = _dilate(field.layers["opening_modified"] | stencil, field.spacing, config["free_zone_modified_mm"]+margin)
        del stencil
        constraint = config["constraint_offset_mm"]+config["free_zone_constraint_mm"]
        pad = max(config["free_zone_preserve_mm"], constraint)+2*margin
        excluded |= field.primitives(self.regions["preserve"], pad) > -config["free_zone_preserve_mm"]-margin
        excluded |= field.primitives(self.regions["forbidden"], pad) > -constraint-margin
        envelope = _envelope(self.domain)
        def free(points):
            index = np.clip(np.rint((points-field.origin)/field.spacing).astype(np.int64), 0, np.asarray(field.values.shape)-1)
            return ~excluded[tuple(index.T)] & (primitive_distance(points.T, envelope) > constraint)
        rows = {}
        for name, source, target in (("candidate_to_density", self.mesh, reference), ("density_to_candidate", reference, self.mesh)):
            points = _surface_samples(source)
            chosen = points[free(points)]
            rows[name] = {"sample_count": len(points), "free_sample_count": len(chosen), "surface_distance_mm": _statistics(_point_distances(chosen, target)) if len(chosen) else None}
        area = self.mesh.area_faces
        enough = all(row["free_sample_count"] >= config["free_zone_minimum_samples"] for row in rows.values())
        largest = max(row["surface_distance_mm"]["max"] for row in rows.values()) if enough else None
        free_zone = {**rows, "maximum_deviation_mm": largest, "free_area_fraction": float(area[free(self.mesh.triangles_center)].sum()/area.sum()), "excluded_sample_fraction": float(excluded.mean()),
                     "reference_mesh": _mesh_summary(reference), "lookup_margin_mm": margin, "passed": enough and largest <= config["surface_deviation_mm"],
                     "method": "Bidirectional exact MeshLab nearest-surface distances between the final mesh and marching cubes of the unextended, unsmoothed density minus threshold, on free-zone samples only: beyond the preserve and constraint distances from every exact primitive and the envelope and outside the opening- or extension-modified samples dilated by the configured distance; grid lookups add half the sample diagonal"}
        if not enough:
            free_zone["reason"] = "Too few free-zone samples; the deviation is not evaluable"
        return {"extraction": extraction, "free_zone": free_zone, "passed": extraction["passed"] and free_zone["passed"]}

    def connectivity(self, report, preserve):
        witnesses = {name: bool(report.get(name, {}).get("passed")) for name in ("witness", "density_witness", "opening_witness", "final_witness")}
        present = isinstance(preserve, dict) and "reason" not in preserve and all(row["missing_volume_mm3"] <= self.tolerance for row in preserve.values())
        result = {"witnesses": witnesses, "final_field_single_component": bool(report.get("final_witness", {}).get("single_component")), "mesh_body_count": int(self.mesh.body_count), "preserves_present": present,
                  "method": "Field witnesses on the hard density composition, the opened density, after the opening and on the final field; the final mesh is one body containing every exact preserve; mounts are never bridged"}
        result["passed"] = all(witnesses.values()) and result["final_field_single_component"] and result["mesh_body_count"] == 1 and present
        return result

    def maturity(self, reference):
        self.metrics = surface_metrics(self.mesh, self.domain, self.settings)
        if reference is None:
            return {"passed": False, "reason": "No reference geometry supplied; geometry screen alone cannot establish maturity"}
        return surface_maturity(self.metrics, reference, self.settings)

    def material_change(self, reference):
        other = _manifold(reference)
        added, removed = (self.body-other).volume(), (other-self.body).volume()
        return {"added_volume_mm3": added, "removed_volume_mm3": removed, "symmetric_difference_volume_mm3": added+removed, "net_volume_change_mm3": self.body.volume()-other.volume(),
                "method": "manifold3d float64 differences in both directions; diagnostic only"}

VALIDATION_CHECKS = ("gates", "topology", "self_intersections", "envelope", "forbidden", "preserve", "features", "supports", "deviation", "connectivity", "surface_maturity")

def validate_implicit(mesh, domain, field, report, settings=None, reference_metrics=None, reference_mesh=None, progress=None):
    started = perf_counter()
    overrides = settings or {}
    settings = _validation_settings({"maximum_wall_samples": report["settings"]["maximum_wall_samples"], **{key: value for key, value in overrides.items() if key in BUDGETS}})
    checks, timings, material = {}, {}, None
    def run(name, function, *arguments):
        clock = perf_counter()
        try:
            checks[name] = function(*arguments)
        except Exception as error:
            checks[name] = {"passed": False, "complete": False, "reason": "Not evaluable: "+type(error).__name__+": "+str(error)}
        timings[name] = perf_counter()-clock
        _progress(progress, "validation_"+name, checks[name])
    acceptance = None
    try:
        if set(overrides)-set(BUDGETS):
            raise ValueError("Only the validation sample budgets may be overridden; frozen gates cannot be weakened: " + ", ".join(sorted(set(overrides)-set(BUDGETS))))
        config, checks["gates"] = frozen_gates(report["settings"])
        acceptance = MeshAcceptance(mesh, domain, config, settings)
    except Exception as error:
        checks["topology" if "gates" in checks else "gates"] = {"passed": False, "reason": "Not evaluable: "+type(error).__name__+": "+str(error)}
    if acceptance is not None:
        run("topology", acceptance.topology)
        checks["self_intersections"] = getattr(acceptance, "intersections", {"passed": False, "reason": "Not evaluable without the topology check"})
        if checks["topology"]["passed"] and checks["self_intersections"]["passed"]:
            run("envelope", acceptance.envelope)
            run("forbidden", acceptance.forbidden)
            run("preserve", acceptance.preserve)
            run("features", acceptance.features, field, report)
            run("supports", acceptance.supports)
            run("deviation", acceptance.deviation, field, report)
            run("surface_maturity", acceptance.maturity, reference_metrics)
            if reference_mesh is not None:
                try:
                    material = acceptance.material_change(reference_mesh)
                except Exception as error:
                    material = {"reason": type(error).__name__+": "+str(error)}
        run("connectivity", acceptance.connectivity, report, checks.get("preserve"))
    violations = []
    for key in VALIDATION_CHECKS:
        value = checks.get(key)
        if value is None:
            violations.append(key+":not_evaluated")
        elif key in ("forbidden", "preserve") and "reason" not in value:
            violations.extend(key+":"+name for name, item in value.items() if not item["passed"])
        elif not value["passed"]:
            violations.append(key)
    return {"passed": not violations, "violations": violations, "checks": checks, "surface_metrics": getattr(acceptance, "metrics", None), "material_change": material, "settings": settings, "timings_s": timings, "runtime_s": perf_counter()-started,
            "acceptance_scope": "Geometry screens and surface maturity on the final mesh and its reloaded binary STL; mechanical verification and render review remain mandatory"}

def ball_curvature(mesh, radius, samples, seed=0):
    from scipy.spatial import cKDTree
    started = perf_counter()
    points = trimesh.sample.sample_surface_even(mesh, samples, seed=seed)[0]
    magnitude = np.abs(trimesh.curvature.discrete_mean_curvature_measure(mesh, points, radius))/(np.pi*radius**2)
    pairs = cKDTree(points).query_pairs(1.5*radius, output_type="ndarray")
    jumps = magnitude[pairs[:, 0]]-magnitude[pairs[:, 1]]
    return {"radius_mm": radius, "requested_samples": samples, "sample_count": len(points), "seed": seed, "absolute_mean_curvature_per_mm": _statistics(magnitude) if len(points) else None,
            "neighbour_distance_mm": 1.5*radius, "neighbour_pair_count": len(pairs), "neighbour_difference_rms_per_mm": float(np.sqrt(np.mean(jumps**2))) if len(pairs) else None, "elapsed_s": perf_counter()-started,
            "method": "Cohen-Steiner/Morvan mean curvature measure of each ball around evenly sampled surface points divided by the disc area pi r^2 (sphere 1/R, cylinder 1/(2R), plane 0); RMS of the |H| differences of sample pairs closer than 1.5 r; independent of the triangulation, record only"}
