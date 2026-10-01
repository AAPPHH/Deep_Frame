import json
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, splu


_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.int32)


def elasticity_matrix(poisson_ratio):
    if not np.isfinite(poisson_ratio) or not -1 < poisson_ratio < 0.5:
        raise ValueError("Poisson ratio must be between -1 and 0.5")
    matrix = np.zeros((6, 6))
    matrix[:3, :3] = poisson_ratio
    np.fill_diagonal(matrix[:3, :3], 1 - poisson_ratio)
    matrix[3:, 3:] = np.eye(3) * (1 - 2 * poisson_ratio) / 2
    return matrix / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))


def hexahedron_matrices(spacing_mm, poisson_ratio):
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("Hexahedron spacing must contain three finite positive lengths")
    stiffness = np.zeros((24, 24))
    mass = np.zeros((24, 24))
    strain_matrices = []
    constitutive = elasticity_matrix(poisson_ratio)
    signs = 2 * _CORNERS - 1
    determinant = np.prod(spacing) / 8
    for location in np.ndindex(2, 2, 2):
        natural = (2 * np.array(location) - 1) / np.sqrt(3)
        factors = 1 + signs * natural
        shape = np.prod(factors, axis=1) / 8
        derivatives = np.empty((8, 3))
        for axis in range(3):
            derivatives[:, axis] = signs[:, axis] * np.prod(np.delete(factors, axis, axis=1), axis=1) / (4 * spacing[axis])
        strain = np.zeros((6, 24))
        strain[0, 0::3] = derivatives[:, 0]
        strain[1, 1::3] = derivatives[:, 1]
        strain[2, 2::3] = derivatives[:, 2]
        strain[3, 0::3] = derivatives[:, 1]
        strain[3, 1::3] = derivatives[:, 0]
        strain[4, 1::3] = derivatives[:, 2]
        strain[4, 2::3] = derivatives[:, 1]
        strain[5, 0::3] = derivatives[:, 2]
        strain[5, 2::3] = derivatives[:, 0]
        stiffness += strain.T @ constitutive @ strain * determinant
        mass += np.kron(np.outer(shape, shape), np.eye(3)) * determinant
        strain_matrices.append(strain)
    return stiffness, mass, np.asarray(strain_matrices)


def regular_grid(grid):
    shape = np.asarray(grid["shape"], dtype=int)
    spacing = np.asarray(grid["spacing_mm"], dtype=float)
    origin = np.asarray(grid["origin_mm"], dtype=float)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("Grid shape must contain three positive integers")
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("Grid spacing must contain three finite positive lengths")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("Grid origin must contain three finite coordinates")
    if grid.get("order", "C") != "C" or grid.get("axis_order", "xyz") != "xyz":
        raise ValueError("Only C-order xyz grids are supported")
    node_shape = shape + 1
    points = np.indices(tuple(node_shape)).reshape(3, -1).T * spacing + origin
    starts = np.indices(tuple(shape)).reshape(3, -1).T
    indices = starts[:, None, :] + _CORNERS[None, :, :]
    connectivity = np.ravel_multi_index(tuple(indices.transpose(2, 0, 1)), tuple(node_shape)).astype(np.int32)
    dofs = (3 * connectivity[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
    return points, connectivity, dofs


def select_nodes(points, region):
    if region.get("kind") != "box":
        raise ValueError("Topology node selectors must be boxes")
    minimum = np.asarray(region["min_mm"], dtype=float)
    maximum = np.asarray(region["max_mm"], dtype=float)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.all(np.isfinite([minimum, maximum])) or np.any(minimum > maximum):
        raise ValueError("Invalid node-selection box")
    selected = np.flatnonzero(np.all(points >= minimum - 1e-7, axis=1) & np.all(points <= maximum + 1e-7, axis=1))
    if not len(selected):
        raise ValueError(f"Empty topology node selector: {region}")
    return selected


class HexElasticity:
    def __init__(self, domain, interface_node_policy="allowed_adjacent", linear_solver="cpu_superlu"):
        if linear_solver not in ("cpu_superlu", "cuda_cudss"):
            raise ValueError("Unknown topology linear_solver")
        self.linear_solver = linear_solver
        self.gpu_solvers = {}
        self.gpu_reanalyses = 0
        self.gpu_solver_history = []
        self.domain = domain
        self.points, self.connectivity, self.dofs = regular_grid(domain["grid"])
        self.ndof = len(self.points) * 3
        self.nelem = len(self.connectivity)
        self.spacing = np.asarray(domain["grid"]["spacing_mm"], dtype=float)
        self.active_elements = np.asarray(domain["allowed"], dtype=bool).ravel()
        if self.active_elements.size != self.nelem or not np.any(self.active_elements):
            raise ValueError("Elasticity requires allowed cells matching the grid")
        self.active_nodes = np.unique(self.connectivity[self.active_elements])
        self.active_dofs = (3 * self.active_nodes[:, None] + np.arange(3)).ravel()
        self.interface_node_policy = interface_node_policy
        if interface_node_policy == "allowed_adjacent":
            self.interface_nodes = self.active_nodes
        elif interface_node_policy == "preserve_adjacent":
            preserve = np.asarray(domain["preserve"], dtype=bool).ravel()
            if preserve.size != self.nelem or not np.any(preserve) or np.any(preserve & ~self.active_elements):
                raise ValueError("Preserve-adjacent interface policy requires valid nonempty preserves")
            self.interface_nodes = np.unique(self.connectivity[preserve])
        else:
            raise ValueError("Unknown topology interface_node_policy")
        material = domain["material"]
        self.young = float(material["young_modulus_mpa"])
        self.density = float(material["density_g_cm3"])
        if not np.isfinite(self.young) or self.young <= 0 or not np.isfinite(self.density) or self.density <= 0:
            raise ValueError("Young modulus and density must be finite and positive")
        self.constitutive = elasticity_matrix(material["poisson_ratio"])
        self.ke, self.me, self.strain = hexahedron_matrices(self.spacing, material["poisson_ratio"])
        self.rows = np.repeat(self.dofs[self.active_elements], 24, axis=1).ravel()
        self.columns = np.tile(self.dofs[self.active_elements], (1, 24)).ravel()
        self.groups = defaultdict(list)
        self.cases = []
        self.selector_expansions = []
        self.selector_filtering = []
        names = set()
        for case in domain["load_cases"]:
            if case["name"] in names:
                raise ValueError("Topology load-case names must be unique")
            names.add(case["name"])
            if case["analysis"] not in ("static", "modal"):
                raise ValueError("Unknown topology load-case analysis")
            fixed_nodes = np.unique(np.concatenate([self._select(region, case["name"], "fixture") for region in case["fixed_regions"]]))
            if len(fixed_nodes) < 3 or np.linalg.matrix_rank(self.points[fixed_nodes] - self.points[fixed_nodes[0]]) < 2:
                raise ValueError("Topology fixture requires three non-collinear nodes")
            fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
            free = np.setdiff1d(self.active_dofs, fixed, assume_unique=True)
            compiled = {"name": case["name"], "analysis": case["analysis"], "free": free, "fixed": fixed, "load_regions": []}
            if case["analysis"] == "static":
                force = np.zeros(self.ndof)
                if not case.get("loads"):
                    raise ValueError("Static topology cases require loads")
                for load in case["loads"]:
                    nodes = self._select(load["region"], case["name"], "load")
                    if np.intersect1d(nodes, fixed_nodes).size:
                        raise ValueError("Topology force patch overlaps the fixture")
                    vector = np.asarray(load["force_n"], dtype=float)
                    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                        raise ValueError("Topology force must be a finite nonzero 3-vector")
                    dofs = 3 * nodes[:, None] + np.arange(3)
                    np.add.at(force, dofs.ravel(), np.tile(vector / len(nodes), len(nodes)))
                    compiled["load_regions"].append((nodes, vector))
                compiled["force"] = force
                self.groups[fixed.tobytes()].append(compiled)
            self.cases.append(compiled)
        if not self.groups:
            raise ValueError("Topology optimization requires at least one static case")

    def _select(self, region, case_name, role):
        try:
            selected = select_nodes(self.points, region)
            nodes = np.intersect1d(selected, self.interface_nodes, assume_unique=True)
            if not len(nodes):
                raise ValueError(f"Empty topology node selector: {region}")
            if len(nodes) < len(selected):
                self._record_filter(selected, nodes, region, case_name, role)
            return nodes
        except ValueError as error:
            if not str(error).startswith("Empty topology node selector"):
                raise
            expanded = {
                "kind": "box",
                "min_mm": (np.asarray(region["min_mm"]) - self.spacing / 2).tolist(),
                "max_mm": (np.asarray(region["max_mm"]) + self.spacing / 2).tolist(),
            }
            selected = select_nodes(self.points, expanded)
            nodes = np.intersect1d(selected, self.interface_nodes, assume_unique=True)
            if not len(nodes):
                raise ValueError(f"Empty topology node selector under {self.interface_node_policy}: {region}")
            self.selector_expansions.append({"case": case_name, "role": role, "original": region, "expanded": expanded, "expansion_each_side_mm": (self.spacing / 2).tolist(), "selected_nodes": len(nodes)})
            if len(nodes) < len(selected):
                self._record_filter(selected, nodes, expanded, case_name, role)
            return nodes

    def _record_filter(self, selected, nodes, region, case_name, role):
        allowed_count = len(np.intersect1d(selected, self.active_nodes, assume_unique=True))
        self.selector_filtering.append({"case": case_name, "role": role, "selector": region, "removed_forbidden_only_nodes": len(selected) - allowed_count, "removed_nonpreserve_interface_nodes": allowed_count - len(nodes), "selected_nodes": len(nodes)})

    def matrix(self, moduli):
        moduli = np.asarray(moduli, dtype=float).ravel()
        if len(moduli) != self.nelem or np.any(moduli <= 0) or not np.all(np.isfinite(moduli)):
            raise ValueError("Every element needs a finite positive modulus")
        values = (moduli[self.active_elements, None] * self.ke.ravel()[None, :]).ravel()
        stiffness = coo_matrix((values, (self.rows, self.columns)), shape=(self.ndof, self.ndof)).tocsc()
        if self.linear_solver == "cuda_cudss":
            transpose = stiffness.T.tocsc()
            if not np.array_equal(stiffness.indptr, transpose.indptr) or not np.array_equal(stiffness.indices, transpose.indices):
                raise RuntimeError("Hex8 symbolic stiffness pattern must be symmetric")
            stiffness.data = (stiffness.data + transpose.data) * 0.5
            return stiffness
        return (stiffness + stiffness.T) * 0.5

    def solve(self, physical_density, penalization=3.0, min_stiffness_ratio=1e-6, metrics=False):
        density = np.asarray(physical_density, dtype=float).ravel()
        if density.size != self.nelem or not np.all(np.isfinite(density)) or np.any(density < 0) or np.any(density > 1):
            raise ValueError("Physical densities must be finite and within [0, 1]")
        if penalization < 1 or not 0 < min_stiffness_ratio < 1:
            raise ValueError("Invalid SIMP interpolation settings")
        moduli = self.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * density ** penalization)
        derivative = self.young * (1 - min_stiffness_ratio) * penalization * density ** (penalization - 1)
        stiffness = self.matrix(moduli)
        moduli[~self.active_elements] = 0
        derivative[~self.active_elements] = 0
        results = {}
        for cases in self.groups.values():
            free = cases[0]["free"]
            forces = np.column_stack([case["force"][free] for case in cases])
            reduced = stiffness[free, :][:, free].tocsc()
            if self.linear_solver == "cpu_superlu":
                factor = splu(reduced, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
                solutions = factor.solve(forces)
            else:
                from deep_frame.topology_gpu import CudaDirectSolver

                key = cases[0]["fixed"].tobytes()
                if key in self.gpu_solvers and not self.gpu_solvers[key].same_structure(reduced):
                    previous = self.gpu_solvers.pop(key)
                    self.gpu_solver_history.append(previous.diagnostics())
                    previous.close()
                    self.gpu_reanalyses += 1
                if key not in self.gpu_solvers:
                    self.gpu_solvers[key] = CudaDirectSolver(reduced, forces)
                solutions = self.gpu_solvers[key].solve(reduced, forces)
            residual = np.linalg.norm(reduced @ solutions - forces, axis=0) / np.maximum(np.linalg.norm(forces, axis=0), 1e-30)
            tolerance = 1e-8 if self.linear_solver == "cuda_cudss" else 1e-4
            if not np.all(np.isfinite(solutions)) or np.any(residual > tolerance):
                raise RuntimeError(f"Topology linear solve failed residual check: {residual.tolist()}")
            for column, case in enumerate(cases):
                displacement = np.zeros(self.ndof)
                displacement[free] = solutions[:, column]
                element_displacement = displacement[self.dofs]
                energy = np.einsum("ei,ij,ej->e", element_displacement, self.ke, element_displacement, optimize=True)
                compliance = float(np.dot(case["force"], displacement))
                if not np.isfinite(compliance) or compliance <= 0:
                    raise RuntimeError("Topology compliance must be finite and positive")
                result = {"compliance_n_mm": compliance, "derivative": -derivative * energy, "relative_residual": float(residual[column])}
                if metrics:
                    result.update(self._metrics(displacement, element_displacement, moduli, case))
                results[case["name"]] = result
        return results

    def _metrics(self, displacement, element_displacement, moduli, case):
        nodal = displacement.reshape(-1, 3)
        loads = []
        for nodes, force in case["load_regions"]:
            mean = np.mean(nodal[nodes], axis=0)
            directional = float(mean @ force / np.linalg.norm(force))
            loads.append({"node_count": len(nodes), "force_n": force.tolist(), "mean_displacement_mm": mean.tolist(), "directional_displacement_mm": directional, "stiffness_n_per_mm": float(np.linalg.norm(force) / directional) if directional > 0 else None})
        stress_maximum = 0.0
        for strain in self.strain:
            stress = ((element_displacement @ strain.T) @ self.constitutive.T) * moduli[:, None]
            xx, yy, zz, xy, yz, xz = stress.T
            von_mises = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + xz ** 2))
            stress_maximum = max(stress_maximum, float(np.max(von_mises)))
        return {"max_displacement_mm": float(np.max(np.linalg.norm(nodal, axis=1))), "max_von_mises_mpa": stress_maximum, "loads": loads, "stiffness_n_per_mm": loads[0]["stiffness_n_per_mm"] if len(loads) == 1 else None}

    def elastic_frequencies(self, physical_density, case_name, number=3, penalization=3.0, min_stiffness_ratio=1e-6):
        if self.domain.get("point_masses"):
            raise ValueError("Point-mass modal coupling is verified by independent CalculiX, not the voxel surrogate")
        case = next((case for case in self.cases if case["name"] == case_name and case["analysis"] == "modal"), None)
        if case is None:
            raise ValueError("Modal case not found")
        density = np.asarray(physical_density, dtype=float).ravel()
        if density.size != self.nelem or not np.all(np.isfinite(density)) or np.any(density <= 0) or np.any(density > 1):
            raise ValueError("Voxel modal verification requires strictly positive densities")
        stiffness = self.matrix(self.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * density ** penalization))
        mass_values = (self.density * 1e-9 * density[self.active_elements, None] * self.me.ravel()[None, :]).ravel()
        mass = coo_matrix((mass_values, (self.rows, self.columns)), shape=(self.ndof, self.ndof)).tocsc()
        free = case["free"]
        if not 0 < number < len(free):
            raise ValueError("Invalid mode count")
        eigenvalues = eigsh(stiffness[free, :][:, free], k=number, M=mass[free, :][:, free], sigma=0.0, which="LM", return_eigenvectors=False)
        if np.any(eigenvalues <= 0) or not np.all(np.isfinite(eigenvalues)):
            raise RuntimeError("Invalid voxel modal eigenvalues")
        return (np.sqrt(np.sort(eigenvalues)) / (2 * np.pi)).tolist()

    def close(self):
        for solver in self.gpu_solvers.values():
            solver.close()

    def diagnostics(self):
        return json.loads(json.dumps({"nodes": len(self.points), "active_nodes": len(self.active_nodes), "elements": self.nelem, "active_elements": int(np.count_nonzero(self.active_elements)), "dofs": self.ndof, "active_dofs": len(self.active_dofs), "interface_node_policy": self.interface_node_policy, "linear_solver": self.linear_solver, "gpu_symbolic_reanalyses": self.gpu_reanalyses, "gpu_solver_details": self.gpu_solver_history + [solver.diagnostics() for solver in self.gpu_solvers.values()], "independent_fixtures": len(self.groups), "selector_expansions": self.selector_expansions, "selector_filtering": self.selector_filtering, "cases": [{"name": case["name"], "analysis": case["analysis"], "fixed_nodes": len(case["fixed"]) // 3, "load_nodes": [len(nodes) for nodes, _ in case["load_regions"]]} for case in self.cases]}, allow_nan=False))
