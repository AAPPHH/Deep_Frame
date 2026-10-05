import numpy as np
from scipy.spatial import ConvexHull, QhullError

from deep_frame.config import COMPONENT_DEFAULTS, LANDING, STAND_STABILITY

FOUR_FEET = [(sx * 18, sy * 18, 0) for sx in (-1, 1) for sy in (-1, 1)]
STAND_CASES = {"four_feet": FOUR_FEET, "rear_feet_raised_3mm": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(sx * 18, -18, 3) for sx in (-1, 1)],
               "tripod": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(0, -18, 0)], "front_pair": [(sx * 18, 18, 0) for sx in (-1, 1)], "stepped_feet": [(sx * 18, 18, 0) for sx in (-1, 1)] + [(sx * 18, -18, 1) for sx in (-1, 1)]}

def stand_domain(half, n=(48, 48, 12), components=None):
    nx = n[0] // 2 if half else n[0]
    grid = {"origin_mm": [0.0 if half else -n[0] / 2, -n[1] / 2, 0.0], "spacing_mm": [1.0] * 3, "shape": [nx, n[1], n[2]]}
    plate = {"kind": "box", "min_mm": [-22.0, -22.0, 8.0], "max_mm": [22.0, 22.0, 10.0]}
    return {"grid": grid, "allowed": np.ones((nx, n[1], n[2]), dtype=bool), "symmetry": {"axis": 0, "plane_mm": 0.0} if half else None, "material": {"density_g_cm3": 1.0, "young_modulus_mpa": 2000.0, "poisson_ratio": 0.3},
            "metadata": {"components": components or {"prop_a": {"center_of_mass_mm": [0.0, 0.0, 20.0], "mass_g": 0.0}}},
            "load_cases": [{"name": "thrust_all", "analysis": "static", "fixed_regions": [], "loads": [{"region": plate, "force_n": [0.0, 0.0, 1e-3]}], "inertia_relief": {"point_masses": [{"name": "plate", "region": plate, "mass_g": 60.0}], "preserve_mass_g": 0.0}}]}

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
        self.expit = expit
        self.settings = settings = {**STAND_STABILITY, **settings}
        h, self.allowed, centers, self.floor, self.contact = floor_layer(domain)
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
    def measure(self, physical, share=None):
        s = self.settings
        rho = np.asarray(physical, dtype=float).ravel()[self.allowed]
        mass = self.component_mass + self.cell_mass * len(self.copies) * rho.sum()
        cog = (self.component_moment + self.cell_mass * np.einsum("e,ced->d", rho, self.copies)) / mass
        width, needed = s["edge_width_mm"], s["min_contact_area_mm2"]
        t = (self.projections - (self.directions @ cog + s["reserve_min_mm"])[None, None, :]) / width
        step = self.expit(t)
        contact, delta = rho[self.contact], s["gray_delta"]
        share = np.ones((len(self.copies), len(self.contact))) if share is None else share
        gray, gray_slope = contact * (contact + 2 * delta) / (1 + 2 * delta), (2 * contact + 2 * delta) / (1 + 2 * delta)
        weight = gray * share
        beyond = self.area * np.einsum("ce,cek->k", weight, step)
        rows = 1 - beyond / needed
        top = rows.max()
        pi = np.exp(s["ks"] * (rows - top))
        g = float(top + np.log(pi.sum()) / s["ks"])
        pi /= pi.sum()
        slope = np.zeros(len(rho))
        slope[self.contact] = -self.area / needed * gray_slope * np.einsum("ce,cek,k->e", share, step, pi)
        edge = self.area / width * np.einsum("ce,cek,k->k", weight, step * (1 - step), pi) / needed
        slope += self.cell_mass * np.einsum("ced,dk,k->e", self.copies - cog, self.directions.T, edge) / mass
        corners = self.corners[(contact > 0.5) & (share > 0.5)].reshape(-1, 2)
        reserves = (corners @ self.directions.T).max(axis=0) - self.directions @ cog if len(corners) else np.full(len(self.directions), -np.inf)
        return {"ground_z_mm": self.floor, "g": g, "slope": slope, "share_slope": -self.area / needed * gray[None] * np.einsum("cek,k->ce", step, pi), "beyond_mm2": beyond, "reserve_mm": float(reserves.min()), "reserves_mm": reserves,
                "center_of_gravity_xy_mm": cog, "mass_g": mass, "contact_area_mm2": float(self.area * len(self.copies) * contact.sum()), "carrying_area_mm2": float(self.area * np.sum(gray * share))}
    def scatter(self, values, n):
        full = np.zeros(n)
        full[self.allowed] = values
        return full
    def rows(self, physical, ground=None):
        s, n = self.settings, np.asarray(physical).size
        share = None if ground is None else ground.shares()
        m = self.measure(physical, None if share is None else share[0])
        gradient = self.scatter(m["slope"], n) + (0 if ground is None else ground.share_gradient(share[1] * m["share_slope"]))
        rows = [{"name": "stand_reserve", "g": m["g"], "gradient": gradient, "value": m["reserve_mm"], "limit": s["reserve_min_mm"], "unit": "mm", "sense": ">=",
                 "info": {"center_of_gravity_xy_mm": m["center_of_gravity_xy_mm"].tolist(), "mass_g": m["mass_g"], "min_beyond_mm2": float(m["beyond_mm2"].min()), "contact_area_mm2": m["contact_area_mm2"],
                          "carrying_area_mm2": m["carrying_area_mm2"], "reaction_weighted": ground is not None}}]
        if self.prop is not None:
            minimum = s["prop_clearance_min_mm"]
            value = self.prop - self.floor
            rows.append({"name": "stand_prop_clearance", "g": None if minimum is None else 1 - value / minimum, "gradient": np.zeros(n), "value": value, "limit": minimum, "unit": "mm", "sense": ">="})
        rows.append({"name": "stand_ground_z", "g": None, "gradient": np.zeros(n), "value": self.floor, "limit": None, "unit": "mm", "sense": ""})
        return rows

def floor_layer(domain):
    from deep_frame.topology_neural import cell_centers
    h = np.asarray(domain["grid"]["spacing_mm"], dtype=float)
    allowed = np.flatnonzero(np.asarray(domain["allowed"]).ravel())
    centers = cell_centers(domain["grid"])[allowed]
    bottom = centers[:, 2] - h[2] / 2
    floor = float(bottom.min())
    return h, allowed, centers, floor, np.flatnonzero(bottom <= floor + h[2] / 2)

class GroundSupport:
    def __init__(self, system, settings, bodies):
        from scipy.sparse import csr_matrix
        self.system, self.bodies = system, bodies
        self.settings = s = {**LANDING, **settings}
        h, allowed, _, self.floor, contact = floor_layer(system.domain)
        self.cells = allowed[contact]
        corners = system.connectivity[self.cells]
        bottom = np.abs(system.points[corners, 2] - self.floor) < 1e-6 * h[2]
        if not np.all(bottom.sum(axis=1) == 4):
            raise ValueError("Every floor cell needs four bottom nodes on the floor plane")
        nodes = corners[bottom].reshape(-1, 4)
        quarter = float(h[0] * h[1]) / 4
        stiffness = np.array([s["shear_n_mm3"], s["shear_n_mm3"], s["normal_n_mm3"]]) * quarter
        columns = np.repeat(np.arange(len(self.cells)), 4)
        self.springs = csr_matrix((np.tile(stiffness, 4 * len(self.cells)), ((3 * nodes[:, :, None] + np.arange(3)).ravel(), np.repeat(columns, 3))), shape=(system.ndof, len(self.cells)))
        self.normal = csr_matrix((np.full(len(columns), stiffness[2]), ((3 * nodes + 2).ravel(), columns)), shape=(system.ndof, len(self.cells)))
        relief = system.relief_operators[s["mass_case"]]
        self.masses, self.flip = (relief["direct"], relief["mirror"]), relief["flip"]
        self.total_g = float(relief["direct"].sum() + relief["mirror"].sum() + sum(body.battery["mass_g"] for body in bodies.values()))
        self.unit = 9.80665e-3 * s["factor_g"]
        self.weight_n = self.total_g * self.unit
        self.factor = 1.0 if system.symmetry is None else 2.0
        self.solvers, self.cases = {}, {}
    def loads(self, direction):
        acceleration = self.unit * np.asarray(direction, dtype=float)
        force, mirrored, states = (self.masses[0][:, None] * acceleration).ravel(), (self.masses[1][:, None] * acceleration * self.flip).ravel(), {}
        for name, body in self.bodies.items():
            vector = body.battery["mass_g"] * acceleration
            states[name] = state = body.state(np.concatenate([vector, np.cross(np.asarray(body.battery["center_mm"], dtype=float) - body.reference, vector)]))
            direct, mirror = body.forces(state)
            force, mirrored = force + direct, mirrored + mirror
        return force, mirrored, states
    def solve(self, physical, penalization, min_stiffness_ratio):
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        system, s = self.system, self.settings
        rho = np.asarray(physical, dtype=float).ravel()
        contact = rho[self.cells]
        self.gain = s["floor"] + (1 - s["floor"]) * contact ** s["exponent"]
        self.gain_slope = (1 - s["floor"]) * s["exponent"] * contact ** (s["exponent"] - 1)
        moduli = system.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * rho ** penalization)
        self.modulus_derivative = np.where(system.active_elements, system.young * (1 - min_stiffness_ratio) * penalization * rho ** (penalization - 1), 0.0)
        stiffness, spring = system.matrix(moduli).tocsr(), self.springs @ self.gain
        self.cases = {}
        for name, direction in s["directions"].items():
            force, mirrored, states = self.loads(direction)
            fields, compliance = {}, 0.0
            for fixed, part, sign in system._parts(np.zeros(0, dtype=int), force, mirrored):
                if not np.any(part):
                    continue
                free = np.setdiff1d(system.active_dofs, fixed, assume_unique=True)
                reduced, key = (stiffness[free, :][:, free] + diags(spring[free])).tocsc(), ("ground", name, sign)
                if system.linear_solver == "cpu_superlu":
                    self.solvers[key] = splu(reduced, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}).solve
                    solution = self.solvers[key](part[free])
                else:
                    solution = system._linear_solve(key, reduced, part[free][:, None])[:, 0]
                    self.solvers[key] = lambda rhs, key=key: system.gpu_solvers[key].substitute(rhs[:, None])[:, 0]
                residual = np.linalg.norm(reduced @ solution - part[free]) / np.linalg.norm(part[free])
                if not np.isfinite(residual) or residual > 1e-6:
                    raise RuntimeError(f"Ground support solve failed residual check: {residual}")
                field = np.zeros(system.ndof)
                field[free] = solution
                fields[sign] = (field, free, key)
                compliance += self.factor * float(part @ field)
            self.cases[name] = {"fields": fields, "states": states, "compliance_n_mm": compliance}
        return self.cases
    def energy(self, left, right):
        system = self.system
        return np.einsum("ei,ij,ej->e", left[system.dofs], system.ke, right[system.dofs], optimize=True) * self.modulus_derivative + self.scatter(self.springs.T @ (left * right) * self.gain_slope)
    def scatter(self, values):
        full = np.zeros(self.system.nelem)
        full[self.cells] = values
        return full
    def body_term(self, case, fields):
        return sum((body.load_term(case["states"][name], body.displacements(fields)) for name, body in self.bodies.items()), np.zeros(self.system.nelem))
    def compliance_gradient(self, name):
        case = self.cases[name]
        fields = {sign: field for sign, (field, _, _) in case["fields"].items()}
        return -self.factor * sum(self.energy(field, field) for field in fields.values()) + 2 * self.body_term(case, fields)
    def reactions(self, name=None):
        fields = self.cases[name or next(iter(self.cases))]["fields"]
        direct = sum(field for field, _, _ in fields.values())
        mirror = sum(sign * field for sign, (field, _, _) in fields.items())
        return np.stack([-(self.normal.T @ field) * self.gain / self.weight_n for field in ([direct] if self.system.symmetry is None else [direct, mirror])])
    def shares(self):
        reference = float(np.prod(self.system.spacing[:2])) / self.settings["reaction_area_mm2"]
        x = np.maximum(self.reactions(), 0.0) / reference
        return x ** 2 / (1 + x ** 2), 2 * x / (1 + x ** 2) ** 2 / reference
    def share_gradient(self, slope):
        case = self.cases[next(iter(self.cases))]
        gradient = self.scatter(np.sum(slope * self.reactions(), axis=0) / self.gain * self.gain_slope)
        coefficient, adjoints = -slope * self.gain / self.weight_n, {}
        for sign, (field, free, key) in case["fields"].items():
            rhs = self.normal @ sum(copy * row for copy, row in zip([1.0, sign], coefficient))
            adjoint = np.zeros(self.system.ndof)
            adjoint[free] = self.solvers[key](rhs[free])
            gradient -= self.energy(adjoint, field)
            adjoints[sign] = adjoint
        return gradient + self.body_term(case, adjoints) / self.factor
    def rows(self, limits):
        rows = []
        for name, case in self.cases.items():
            value, limit = case["compliance_n_mm"], limits.get(name) if isinstance(limits, dict) else limits
            rows.append({"name": name, "g": None if limit is None else float(np.log(value / limit)), "gradient": self.compliance_gradient(name) / (value if limit else 1.0), "value": value, "limit": limit, "unit": "N mm", "sense": "<=",
                         "info": {"weight_n": self.weight_n, "mass_g": self.total_g, "factor_g": self.settings["factor_g"], "floor_reaction_share": float(self.reactions(name).sum())}})
        return rows
