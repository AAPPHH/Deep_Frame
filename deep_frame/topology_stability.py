import numpy as np
from scipy.spatial import ConvexHull, QhullError

from deep_frame.config import COMPONENT_DEFAULTS, STAND_STABILITY

FOUR_FEET = [(sx * 18, sy * 18, 0) for sx in (-1, 1) for sy in (-1, 1)]
STAND_CASES = {"four_feet": FOUR_FEET, "rear_feet_raised_3mm": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(sx * 18, -18, 3) for sx in (-1, 1)],
               "tripod": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(0, -18, 0)], "front_pair": [(sx * 18, 18, 0) for sx in (-1, 1)], "stepped_feet": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(sx * 18, -18, 1) for sx in (-1, 1)]}

def stand_domain(half, n=(48, 48, 12), components=None):
    nx = n[0] // 2 if half else n[0]
    grid = {"origin_mm": [0.0 if half else -n[0] / 2, -n[1] / 2, 0.0], "spacing_mm": [1.0] * 3, "shape": [nx, n[1], n[2]]}
    return {"grid": grid, "allowed": np.ones((nx, n[1], n[2]), dtype=bool), "symmetry": {"axis": 0, "plane_mm": 0.0} if half else None, "material": {"density_g_cm3": 1.0},
            "metadata": {"components": components or {"prop_a": {"center_of_mass_mm": [0.0, 0.0, 20.0], "mass_g": 0.0}}}}

def stand_design(domain, feet, low=0.02):
    from deep_frame.topology_neural import cell_centers
    c = cell_centers(domain["grid"])
    rho = np.full(len(c), low)
    rho[(c[:, 2] > 8) & (c[:, 2] < 10) & (np.abs(c[:, 0]) < 22) & (np.abs(c[:, 1]) < 22)] = 1
    for x, y, z in feet:
        rho[(np.abs(c[:, 0] - x) < 2) & (np.abs(c[:, 1] - y) < 2) & (c[:, 2] > z) & (c[:, 2] < 10)] = 1
    return rho

def voxel_reserve(domain, physical, center, threshold=0.5):
    from deep_frame.topology_neural import cell_centers
    grid = domain["grid"]
    h = np.asarray(grid["spacing_mm"], dtype=float)
    c = cell_centers(grid)[np.asarray(physical).ravel() > threshold]
    bottom = c[:, 2] - h[2] / 2
    cells = c[bottom <= bottom.min() + 1e-6, :2]
    corners = (cells[:, None, :] + np.array([[sx, sy] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)]) * h[:2]).reshape(-1, 2)
    symmetry = domain.get("symmetry")
    if symmetry is not None:
        mirror = lambda points: np.concatenate([points, np.where(np.arange(2) == symmetry["axis"], 2 * symmetry.get("plane_mm", 0.0) - points, points)])
        cells, corners = mirror(cells), mirror(corners)
    return {"cell_centres_mm": plan_reserve(cells, center)[0], "cell_faces_mm": plan_reserve(corners, center)[0], "ground_z_mm": float(bottom.min())}

def prop_bottoms(components):
    half = COMPONENT_DEFAULTS["prop"]["thickness_mm"] / 2
    return [float(entry["center_of_mass_mm"][2]) - half for name, entry in components.items() if name.startswith("prop")]

def plan_reserve(points, center):
    points, center = np.asarray(points, dtype=float), np.asarray(center, dtype=float)
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return None, None, None
    equations = hull.equations
    inside = -(equations[:, :2] @ center + equations[:, 2])
    vertices = points[hull.vertices]
    if np.all(inside >= 0):
        return float(inside.min()), vertices, float(hull.volume)
    start, end = vertices, np.roll(vertices, -1, axis=0)
    edge = end - start
    t = np.clip(np.einsum("ij,ij->i", center - start, edge) / np.einsum("ij,ij->i", edge, edge), 0, 1)
    return -float(np.min(np.linalg.norm(start + t[:, None] * edge - center, axis=1))), vertices, float(hull.volume)

def stand_measure(mesh, center, components, settings=STAND_STABILITY):
    import trimesh
    ground = float(mesh.bounds[0][2])
    level = ground + settings["contact_tolerance_mm"]
    points = mesh.vertices[mesh.vertices[:, 2] <= level][:, :2]
    lines = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, level])
    points = np.concatenate([points, lines.reshape(-1, 3)[:, :2]]) if len(lines) else points
    reserve, hull, area = plan_reserve(points, np.asarray(center)[:2])
    props = [item["center_mm"][2] - item["size_mm"][2] / 2 - ground for item in components if item["type"] == "prop"]
    clearance = min(props) if props else None
    minimum, needed = settings["prop_clearance_min_mm"], settings["reserve_min_mm"]
    return {"ground_z_mm": ground, "contact_tolerance_mm": settings["contact_tolerance_mm"], "contact_points": int(len(points)), "hull_xy_mm": None if hull is None else hull.tolist(), "support_area_mm2": area,
            "center_of_gravity_xy_mm": np.asarray(center)[:2].tolist(), "reserve_mm": reserve, "reserve_min_mm": needed, "prop_clearance_mm": clearance, "prop_clearance_min_mm": minimum,
            "reserve_passed": reserve is not None and reserve >= needed, "prop_clearance_passed": None if clearance is None or minimum is None else clearance >= minimum,
            "method": settings["definition"]["exact"] + ("" if hull is not None else "; contact set degenerate (fewer than three non-collinear points): no support polygon")}

class StandStability:
    def __init__(self, domain, settings=STAND_STABILITY):
        from scipy.special import expit
        from deep_frame.topology_neural import cell_centers
        self.expit = expit
        self.settings = settings = {**STAND_STABILITY, **settings}
        grid = domain["grid"]
        h = np.asarray(grid["spacing_mm"], dtype=float)
        self.allowed = np.flatnonzero(np.asarray(domain["allowed"]).ravel())
        centers = cell_centers(grid)[self.allowed]
        bottom = centers[:, 2] - h[2] / 2
        self.floor = float(bottom.min())
        self.contact = np.flatnonzero(bottom <= self.floor + h[2] / 2)
        copies = [centers[:, :2]]
        symmetry = domain.get("symmetry")
        if symmetry is not None:
            mirror = centers[:, :2].copy()
            mirror[:, symmetry["axis"]] = 2 * symmetry.get("plane_mm", 0.0) - mirror[:, symmetry["axis"]]
            copies.append(mirror)
        self.copies = np.stack(copies)
        self.area = float(h[0] * h[1])
        self.corners = (self.copies[:, self.contact, None, :] + np.array([[sx, sy] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5)]) * h[:2]).reshape(len(self.copies), len(self.contact), 4, 2)
        self.cell_mass = float(np.prod(h)) * domain["material"]["density_g_cm3"] / 1000
        components = domain["metadata"]["components"]
        masses = [(entry["mass_g"], entry["center_of_mass_mm"][:2]) for entry in components.values() if entry["mass_g"] > 0]
        self.component_mass = float(sum(mass for mass, _ in masses))
        self.component_moment = sum((mass * np.asarray(xy, dtype=float) for mass, xy in masses), np.zeros(2))
        props = prop_bottoms(components)
        self.prop = min(props) if props else None
        angles = 2 * np.pi * np.arange(settings["directions"]) / settings["directions"]
        self.directions = np.column_stack([np.cos(angles), np.sin(angles)])
        self.projections = self.copies[:, self.contact] @ self.directions.T
    def measure(self, physical):
        s = self.settings
        rho = np.asarray(physical, dtype=float).ravel()[self.allowed]
        mass = self.component_mass + self.cell_mass * len(self.copies) * rho.sum()
        cog = (self.component_moment + self.cell_mass * np.einsum("e,ced->d", rho, self.copies)) / mass
        width, needed = s["edge_width_mm"], s["min_contact_area_mm2"]
        t = (self.projections - (self.directions @ cog + s["reserve_min_mm"])[None, None, :]) / width
        step = self.expit(t)
        contact, delta = rho[self.contact], s["gray_delta"]
        weight, weight_slope = contact * (contact + 2 * delta) / (1 + 2 * delta), (2 * contact + 2 * delta) / (1 + 2 * delta)
        beyond = self.area * np.einsum("e,cek->k", weight, step)
        rows = 1 - beyond / needed
        top = rows.max()
        pi = np.exp(s["ks"] * (rows - top))
        g = float(top + np.log(pi.sum()) / s["ks"])
        pi /= pi.sum()
        slope = np.zeros(len(rho))
        slope[self.contact] = -self.area / needed * weight_slope * np.einsum("cek,k->e", step, pi)
        edge = self.area / width * np.einsum("e,cek,k->k", weight, step * (1 - step), pi) / needed
        slope += self.cell_mass * np.einsum("ced,dk,k->e", self.copies - cog, self.directions.T, edge) / mass
        solid = contact > 0.5
        corners = self.corners[:, solid].reshape(-1, 2)
        reserves = (corners @ self.directions.T).max(axis=0) - self.directions @ cog if len(corners) else np.full(len(self.directions), -np.inf)
        return {"ground_z_mm": self.floor, "g": g, "slope": slope, "beyond_mm2": beyond, "reserve_mm": float(reserves.min()), "reserves_mm": reserves, "center_of_gravity_xy_mm": cog, "mass_g": mass,
                "contact_area_mm2": float(self.area * len(self.copies) * contact.sum())}
    def scatter(self, values, n):
        full = np.zeros(n)
        full[self.allowed] = values
        return full
    def rows(self, physical):
        s, n = self.settings, np.asarray(physical).size
        m = self.measure(physical)
        rows = [{"name": "stand_reserve", "g": m["g"], "gradient": self.scatter(m["slope"], n), "value": m["reserve_mm"], "limit": s["reserve_min_mm"], "unit": "mm", "sense": ">=",
                 "info": {"center_of_gravity_xy_mm": m["center_of_gravity_xy_mm"].tolist(), "mass_g": m["mass_g"], "min_beyond_mm2": float(m["beyond_mm2"].min()), "contact_area_mm2": m["contact_area_mm2"]}}]
        if self.prop is not None:
            minimum = s["prop_clearance_min_mm"]
            value = self.prop - self.floor
            rows.append({"name": "stand_prop_clearance", "g": None if minimum is None else 1 - value / minimum, "gradient": np.zeros(n), "value": value, "limit": minimum, "unit": "mm", "sense": ">="})
        rows.append({"name": "stand_ground_z", "g": None, "gradient": np.zeros(n), "value": self.floor, "limit": None, "unit": "mm", "sense": ""})
        return rows
