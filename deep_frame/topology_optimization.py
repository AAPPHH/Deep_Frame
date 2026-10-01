import ctypes
import os
from collections import defaultdict
from copy import deepcopy
from importlib.metadata import distribution, version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, splu
from scipy.spatial import cKDTree

from deep_frame.config import _json_copy

def _library():
    try:
        import cupy
        installed = version("nvidia-cudss-cu12")
    except ImportError as error:
        raise RuntimeError("cuda_cudss requires the optional requirements-gpu.txt environment") from error
    if not installed.startswith("0.8."):
        raise RuntimeError("cuda_cudss requires the cuDSS 0.8 ABI")
    package = distribution("nvidia-cudss-cu12")
    candidates = [Path(package.locate_file(name)) for name in package.files
                  if str(name).endswith(("cudss64_0.dll", "libcudss.so.0"))]
    if len(candidates) != 1:
        raise RuntimeError("Could not locate the installed cuDSS library")
    directories = []
    if os.name == "nt":
        locations = [candidates[0].parent]
        for dependency in ("nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12"):
            installed_dependency = distribution(dependency)
            locations.extend(Path(installed_dependency.locate_file(name)).parent
                             for name in installed_dependency.files if str(name).endswith(".dll"))
        if os.environ.get("CUDA_PATH"):
            locations.append(Path(os.environ["CUDA_PATH"]) / "bin")
        directories = [os.add_dll_directory(str(path)) for path in set(locations) if path.is_dir()]
    try:
        library = ctypes.CDLL(str(candidates[0]))
    except OSError as error:
        for directory in directories:
            directory.close()
        raise RuntimeError("cuDSS library or one of its CUDA dependencies could not be loaded") from error
    pointer = ctypes.c_void_p
    pointer_pointer = ctypes.POINTER(pointer)
    integer = ctypes.c_int
    wide = ctypes.c_int64
    signatures = {
        "cudssCreate": [pointer_pointer], "cudssDestroy": [pointer],
        "cudssConfigCreate": [pointer_pointer], "cudssConfigDestroy": [pointer],
        "cudssDataCreate": [pointer, pointer_pointer], "cudssDataDestroy": [pointer, pointer],
        "cudssSetStream": [pointer, pointer],
        "cudssMatrixCreateDn": [pointer_pointer, wide, wide, wide, pointer, integer, integer],
        "cudssMatrixCreateCsr": [pointer_pointer, wide, wide, wide, pointer, pointer, pointer,
                                 pointer, integer, integer, integer, integer, integer, integer],
        "cudssMatrixDestroy": [pointer],
        "cudssExecute": [pointer, integer, pointer, pointer, pointer, pointer, pointer],
        "cudssDataGet": [pointer, pointer, integer, pointer, ctypes.c_size_t,
                          ctypes.POINTER(ctypes.c_size_t)],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = integer
    return cupy, library, directories, installed

class CudaDirectSolver:
    def __init__(self, matrix, rhs):
        self.handles = {}
        self.closed = False
        self.cleanup_errors = []
        self.dll_directories = []
        self.stream = None
        try:
            self._initialize(matrix, rhs)
        except BaseException as error:
            issues = self.close()
            if issues and isinstance(error, Exception):
                raise RuntimeError(str(error) + "; cleanup: " + "; ".join(issues)) from error
            raise
    def _initialize(self, matrix, rhs):
        self.cp, self.library, self.dll_directories, self.version = _library()
        self.timings = []
        self.stream = self.cp.cuda.get_current_stream()
        matrix = matrix.tocsr()
        matrix.sort_indices()
        if matrix.shape[0] != matrix.shape[1] or matrix.nnz > np.iinfo(np.int32).max:
            raise ValueError("cuDSS requires a square matrix with 32-bit CSR indices")
        self.shape = matrix.shape
        if np.asarray(rhs).ndim != 2 or rhs.shape[0] != matrix.shape[0] or rhs.shape[1] < 1:
            raise ValueError("cuDSS requires a nonempty two-dimensional RHS matching the matrix")
        if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
            raise ValueError("cuDSS matrix and RHS must be finite")
        self.indptr = matrix.indptr.astype(np.int32, copy=True)
        self.indices = matrix.indices.astype(np.int32, copy=True)
        self.row_device = self.cp.asarray(self.indptr)
        self.column_device = self.cp.asarray(self.indices)
        self.values_device = self.cp.asarray(matrix.data, dtype=self.cp.float64)
        self.rhs_device = self.cp.array(rhs, dtype=self.cp.float64, order="F")
        self.solution_device = self.cp.zeros_like(self.rhs_device, order="F")
        self._create("handle", "cudssCreate")
        self._call("cudssSetStream", self.handles["handle"], self.stream.ptr)
        self._create("config", "cudssConfigCreate")
        self.handles["data"] = ctypes.c_void_p()
        self._call("cudssDataCreate", self.handles["handle"], ctypes.byref(self.handles["data"]))
        self._create("matrix", "cudssMatrixCreateCsr", matrix.shape[0], matrix.shape[1], matrix.nnz,
                     self.row_device.data.ptr, None, self.column_device.data.ptr,
                     self.values_device.data.ptr, 10, 10, 1, 3, 0, 0)
        for name, array in (("rhs", self.rhs_device), ("solution", self.solution_device)):
            self._create(name, "cudssMatrixCreateDn", array.shape[0], array.shape[1], array.shape[0],
                         array.data.ptr, 1, 0)
        self.analysis_s = self._execute(3)
    def _call(self, name, *arguments):
        status = getattr(self.library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with cuDSS status {status}")
    def _create(self, name, function, *arguments):
        self.handles[name] = ctypes.c_void_p()
        self._call(function, ctypes.byref(self.handles[name]), *arguments)
    def _execute(self, phase):
        self.stream.synchronize()
        started = perf_counter()
        self._call("cudssExecute", self.handles["handle"], phase, self.handles["config"],
                   self.handles["data"], self.handles["matrix"], self.handles["solution"], self.handles["rhs"])
        self.stream.synchronize()
        info = ctypes.c_int()
        written = ctypes.c_size_t()
        self._call("cudssDataGet", self.handles["handle"], self.handles["data"], 0,
                   ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(written))
        if info.value:
            raise RuntimeError(f"cuDSS device error {info.value} in phase {phase}")
        return perf_counter() - started
    def solve(self, matrix, rhs):
        if self.closed:
            raise RuntimeError("cuDSS solver is closed")
        started = perf_counter()
        matrix = matrix.tocsr()
        matrix.sort_indices()
        if matrix.shape != self.shape or not np.array_equal(matrix.indptr, self.indptr) or not np.array_equal(matrix.indices, self.indices):
            raise ValueError("cuDSS symbolic reuse requires unchanged sparse structure")
        if rhs.shape != self.rhs_device.shape or not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
            raise ValueError("cuDSS requires finite values and an unchanged RHS shape")
        self.values_device.set(np.asarray(matrix.data, dtype=np.float64))
        self.rhs_device.set(np.asarray(rhs, dtype=np.float64, order="F"))
        factor_s = self._execute(4)
        solve_s = self._execute(1008)
        solution = self.cp.asnumpy(self.solution_device)
        available, total = self.cp.cuda.runtime.memGetInfo()
        self.timings.append({"factor_s": factor_s, "solve_s": solve_s,
                             "transfer_factor_solve_s": perf_counter() - started,
                             "device_free_bytes_after_solve": available, "device_total_bytes": total})
        return solution
    def same_structure(self, matrix):
        matrix = matrix.tocsr()
        matrix.sort_indices()
        return matrix.shape == self.shape and np.array_equal(matrix.indptr, self.indptr) and np.array_equal(matrix.indices, self.indices)
    def diagnostics(self):
        device = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        name = device["name"]
        return {"cudss_version": self.version, "cupy_version": self.cp.__version__,
                "cuda_runtime_version": self.cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver_version": self.cp.cuda.runtime.driverGetVersion(),
                "device_name": name.decode() if isinstance(name, bytes) else name,
                "precision": "float64", "numeric_factorization": "GPU Cholesky, hybrid execution disabled",
                "analysis_s": self.analysis_s, "shape": list(self.shape), "nnz": len(self.indices),
                "solves": self.timings}
    def close(self):
        if getattr(self, "closed", True):
            return getattr(self, "cleanup_errors", []).copy()
        if self.stream is not None:
            try:
                self.stream.synchronize()
            except Exception as error:
                self.cleanup_errors.append("CUDA cleanup synchronization: " + str(error))
        def destroy(function, *arguments):
            try:
                self._call(function, *arguments)
            except Exception as error:
                self.cleanup_errors.append(str(error))
        for name in ("matrix", "solution", "rhs"):
            handle = self.handles.get(name)
            if handle and handle.value:
                destroy("cudssMatrixDestroy", handle)
        data = self.handles.get("data")
        if data and data.value:
            destroy("cudssDataDestroy", self.handles["handle"], data)
        config = self.handles.get("config")
        if config and config.value:
            destroy("cudssConfigDestroy", config)
        handle = self.handles.get("handle")
        if handle and handle.value:
            destroy("cudssDestroy", handle)
        self.closed = True
        for directory in self.dll_directories:
            try:
                directory.close()
            except Exception as error:
                self.cleanup_errors.append("CUDA library directory cleanup: " + str(error))
        for name in ("row_device", "column_device", "values_device", "rhs_device", "solution_device"):
            setattr(self, name, None)
        return self.cleanup_errors.copy()
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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
                key = cases[0]["fixed"].tobytes()
                if key in self.gpu_solvers and not self.gpu_solvers[key].same_structure(reduced):
                    previous = self.gpu_solvers.pop(key)
                    self.gpu_solver_history.append(previous.diagnostics())
                    cleanup = previous.close()
                    if cleanup:
                        raise RuntimeError("cuDSS cleanup failed: " + "; ".join(cleanup))
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
        errors = []
        for solver in self.gpu_solvers.values():
            errors.extend(solver.close())
        return errors
    def diagnostics(self):
        return _json_copy({"nodes": len(self.points), "active_nodes": len(self.active_nodes), "elements": self.nelem, "active_elements": int(np.count_nonzero(self.active_elements)), "dofs": self.ndof, "active_dofs": len(self.active_dofs), "interface_node_policy": self.interface_node_policy, "linear_solver": self.linear_solver, "gpu_symbolic_reanalyses": self.gpu_reanalyses, "gpu_solver_details": self.gpu_solver_history + [solver.diagnostics() for solver in self.gpu_solvers.values()], "independent_fixtures": len(self.groups), "selector_expansions": self.selector_expansions, "selector_filtering": self.selector_filtering, "cases": [{"name": case["name"], "analysis": case["analysis"], "fixed_nodes": len(case["fixed"]) // 3, "load_nodes": [len(nodes) for nodes, _ in case["load_regions"]]} for case in self.cases]})

DEFAULT_SETTINGS = {
    "volume_fraction": 0.20,
    "filter_radius_mm": 6.0,
    "penalization": 3.0,
    "min_stiffness_ratio": 1e-6,
    "max_iterations": 60,
    "minimum_iterations": 10,
    "change_tolerance": 0.01,
    "move_limit": 0.15,
    "projection_beta": 1.0,
    "projection_eta": 0.5,
    "minimum_design_density": 1e-3,
    "case_weights": {},
    "max_runtime_s": None,
    "interface_node_policy": "allowed_adjacent",
    "linear_solver": "cpu_superlu",
}

def _settings(settings):
    result = deepcopy(DEFAULT_SETTINGS)
    result.update(deepcopy(settings))
    positive = ("filter_radius_mm", "penalization", "min_stiffness_ratio", "change_tolerance", "move_limit", "minimum_design_density")
    for name in positive:
        if not np.isfinite(result[name]) or result[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("max_iterations", "minimum_iterations"):
        if isinstance(result[name], bool) or int(result[name]) != result[name] or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if result["minimum_iterations"] > result["max_iterations"]:
        result["minimum_iterations"] = result["max_iterations"]
    if not np.isfinite(result["volume_fraction"]) or not 0 < result["volume_fraction"] <= 1:
        raise ValueError("volume_fraction must be in (0, 1]")
    if result["penalization"] < 1 or not 0 < result["min_stiffness_ratio"] < 1 or result["minimum_design_density"] >= 1 or result["move_limit"] > 1:
        raise ValueError("Invalid SIMP or OC settings")
    if not np.isfinite(result["projection_beta"]) or result["projection_beta"] < 0 or not 0 < result["projection_eta"] < 1:
        raise ValueError("Invalid density projection settings")
    if result["max_runtime_s"] is not None and (not np.isfinite(result["max_runtime_s"]) or result["max_runtime_s"] <= 0):
        raise ValueError("max_runtime_s must be positive or None")
    if result["interface_node_policy"] not in ("allowed_adjacent", "preserve_adjacent"):
        raise ValueError("Invalid topology interface_node_policy")
    if result["linear_solver"] not in ("cpu_superlu", "cuda_cudss"):
        raise ValueError("Invalid topology linear_solver")
    return result

def validate_masks(domain):
    shape = tuple(domain["grid"]["shape"])
    masks = []
    for name in ("allowed", "preserve", "forbidden"):
        array = np.asarray(domain[name])
        if array.shape != shape or array.dtype != bool:
            raise ValueError(f"{name} must be a boolean array with grid shape")
        masks.append(array.ravel().copy())
    allowed, preserve, forbidden = masks
    if not np.array_equal(forbidden, ~allowed) or np.any(preserve & forbidden):
        raise ValueError("Forbidden must complement allowed; preserve must be allowed")
    if not np.any(allowed & ~preserve):
        raise ValueError("The design domain has no free material cells")
    return allowed, preserve, forbidden

class DensityMap:
    def __init__(self, domain, settings):
        self.settings = settings
        self.allowed, self.preserve, self.forbidden = validate_masks(domain)
        self.free = self.allowed & ~self.preserve
        self.n = self.allowed.size
        shape = tuple(domain["grid"]["shape"])
        spacing = np.asarray(domain["grid"]["spacing_mm"], dtype=float)
        coordinates = np.indices(shape).reshape(3, -1).T * spacing
        active = np.flatnonzero(self.allowed)
        tree = cKDTree(coordinates[active])
        radius = settings["filter_radius_mm"]
        distances = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
        weights = radius - distances.data
        self.filter = coo_matrix((weights, (active[distances.row], active[distances.col])), shape=(self.n, self.n)).tocsr()
        self.filter.eliminate_zeros()
        self.sums = np.asarray(self.filter.sum(axis=1)).ravel()
        self.sums[self.forbidden] = 1
        if np.any(self.sums <= 0):
            raise ValueError("Density filter contains an empty allowed-cell neighborhood")
    def physical(self, design):
        design = np.asarray(design, dtype=float).ravel()
        if design.size != self.n or not np.all(np.isfinite(design)) or np.any(design < 0) or np.any(design > 1):
            raise ValueError("Design densities must be finite and within [0, 1]")
        filtered = np.asarray(self.filter @ design).ravel() / self.sums
        beta = self.settings["projection_beta"]
        eta = self.settings["projection_eta"]
        if beta:
            denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
            physical = (np.tanh(beta * eta) + np.tanh(beta * (filtered - eta))) / denominator
            derivative = beta * (1 - np.tanh(beta * (filtered - eta)) ** 2) / denominator
        else:
            physical = filtered.copy()
            derivative = np.ones(self.n)
        physical[self.preserve] = 1
        physical[self.forbidden] = 0
        derivative[~self.free] = 0
        return np.clip(physical, 0, 1), derivative
    def pullback(self, sensitivity, projection_derivative):
        result = np.asarray(self.filter.T @ (np.asarray(sensitivity).ravel() * projection_derivative / self.sums)).ravel()
        result[~self.free] = 0
        return result
    def initial(self, target):
        design = np.zeros(self.n)
        design[self.preserve] = 1
        lower = self.settings["minimum_design_density"]
        design[self.free] = lower
        if np.sum(self.physical(design)[0]) > target + 1e-8:
            raise ValueError("Volume budget cannot contain preserves and their filtered transition")
        upper = 1.0
        for _ in range(70):
            value = (lower + upper) / 2
            design[self.free] = value
            if np.sum(self.physical(design)[0]) > target:
                upper = value
            else:
                lower = value
        design[self.free] = lower
        return design

def _oc_update(design, objective_derivative, volume_derivative, mapping, target, settings):
    free = mapping.free
    if np.any(volume_derivative[free] <= 0) or not np.all(np.isfinite(objective_derivative[free])):
        raise RuntimeError("Invalid OC sensitivities")
    ratios = np.maximum(1e-30, -objective_derivative[free] / volume_derivative[free])
    lower_density = np.maximum(settings["minimum_design_density"], design[free] - settings["move_limit"])
    upper_density = np.minimum(1.0, design[free] + settings["move_limit"])
    def proposal(multiplier):
        result = design.copy()
        result[free] = np.clip(design[free] * np.sqrt(ratios / max(multiplier, 1e-100)), lower_density, upper_density)
        return result
    upper = max(float(np.max(ratios)), 1e-12)
    for _ in range(100):
        if np.sum(mapping.physical(proposal(upper))[0]) <= target + 1e-9:
            break
        upper *= 2
    else:
        raise RuntimeError("OC could not bracket the volume multiplier")
    lower = 0.0
    candidate = proposal(upper)
    for _ in range(70):
        multiplier = (lower + upper) / 2
        proposed = proposal(multiplier)
        if np.sum(mapping.physical(proposed)[0]) > target:
            lower = multiplier
        else:
            upper = multiplier
            candidate = proposed
        if (upper - lower) / max(upper + lower, 1e-100) < 1e-8:
            break
    return candidate

def _case_scaling(solutions, settings):
    supplied = settings["case_weights"]
    if set(supplied) - set(solutions):
        raise ValueError("case_weights references an unknown static load case")
    weights = {name: float(supplied.get(name, 1.0)) for name in solutions}
    if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("Every static load case requires a finite positive weight")
    total = sum(weights.values())
    normalization = {name: result["compliance_n_mm"] for name, result in solutions.items()}
    return {name: weights[name] / total / normalization[name] for name in solutions}, normalization, {name: weight / total for name, weight in weights.items()}

def _history_entry(iteration, physical, solutions, scales, change, elapsed, final=False):
    return {
        "iteration": iteration,
        "final_evaluation": final,
        "objective": float(sum(scales[name] * result["compliance_n_mm"] for name, result in solutions.items())),
        "physical_density_sum": float(np.sum(physical)),
        "compliances_n_mm": {name: result["compliance_n_mm"] for name, result in solutions.items()},
        "maximum_design_change": change,
        "maximum_relative_residual": float(max(result["relative_residual"] for result in solutions.values())),
        "elapsed_s": elapsed,
    }

def optimize_topology(domain, settings, *, progress_callback=None):
    started = perf_counter()
    history = []
    diagnostics = []
    system = None
    outcome = None
    density = np.zeros(tuple(domain.get("grid", {}).get("shape", (0, 0, 0))))
    try:
        settings = _settings(settings)
        mapping = DensityMap(domain, settings)
        target = settings["volume_fraction"] * int(np.count_nonzero(mapping.allowed))
        if np.count_nonzero(mapping.preserve) >= target:
            raise ValueError("The volume budget must exceed the preserve-cell volume")
        design = mapping.initial(target)
        system = HexElasticity(domain, interface_node_policy=settings["interface_node_policy"], linear_solver=settings["linear_solver"])
        scales = None
        converged = False
        stop_reason = "max_iterations"
        for iteration in range(1, settings["max_iterations"] + 1):
            physical, projection_derivative = mapping.physical(design)
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"])
            if scales is None:
                scales, normalization, weights = _case_scaling(solutions, settings)
            physical_gradient = sum(scales[name] * result["derivative"] for name, result in solutions.items())
            gradient = mapping.pullback(physical_gradient, projection_derivative)
            volume_gradient = mapping.pullback(np.ones(mapping.n), projection_derivative)
            candidate = _oc_update(design, gradient, volume_gradient, mapping, target, settings)
            change = float(np.max(np.abs(candidate - design)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            design = candidate
            if iteration >= settings["minimum_iterations"] and change < settings["change_tolerance"]:
                converged = True
                stop_reason = "change_tolerance"
                break
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
        physical, _ = mapping.physical(design)
        density = physical.reshape(tuple(domain["grid"]["shape"]))
        solutions = system.solve(physical, settings["penalization"], settings["min_stiffness_ratio"], metrics=True)
        final_entry = _history_entry(iteration + 1, physical, solutions, scales, None, perf_counter() - started, final=True)
        history.append(final_entry)
        if progress_callback is not None:
            progress_callback(deepcopy(final_entry))
        volume = float(np.sum(physical) * np.prod(system.spacing))
        static_metrics = {name: {key: value for key, value in result.items() if key != "derivative"} for name, result in solutions.items()}
        summary = {
            "method": "3D Hex8 SIMP, spatial density filter, smooth projection, normalized multi-load compliance, OC volume constraint",
            "initialization": "uniform in every free cell; fixed preserves; no frame template or connectivity seed",
            "settings": settings,
            "converged": converged,
            "stop_reason": stop_reason,
            "iterations": iteration,
            "volume_fraction": float(np.sum(physical) / np.count_nonzero(mapping.allowed)),
            "target_volume_fraction": settings["volume_fraction"],
            "density_volume_mm3": volume,
            "density_frame_mass_g": volume * system.density / 1000,
            "objective_initial": history[0]["objective"],
            "objective_final": final_entry["objective"],
            "normalization_compliances_n_mm": normalization,
            "normalized_case_weights": weights,
            "static_surrogate_metrics": static_metrics,
            "modal_assessment": "independent gmsh/CalculiX candidate verification, including specified point masses; no voxel-frequency substitute",
            "stress_assessment": "Hex8 Gauss-point SIMP stresses are surrogate diagnostics only; exact-solid CalculiX governs candidate stress constraints",
            "manufacturing_assessment": "density filtering regularizes lengths; reconstructed-solid feature, connection, forbidden and manufacturing checks remain mandatory",
            "gray_fraction_free": float(np.mean((physical[mapping.free] > 0.1) & (physical[mapping.free] < 0.9))),
            "element_model": "three displacement DOFs/node, fully integrated trilinear 8-node 3D elasticity, 2x2x2 Gauss integration",
            "units": {"length": "mm", "force": "N", "stress": "MPa", "density": "g/cm3", "internal_modal_mass": "tonne"},
            "system": system.diagnostics(),
            "elapsed_s": perf_counter() - started,
        }
        if not converged:
            diagnostics.append(f"Density iteration stopped at {stop_reason}; convergence is not claimed")
        outcome = {"status": "ok", "density": density.copy(), "design_density": design.reshape(density.shape).copy(), "summary": summary, "history": history, "diagnostics": diagnostics}
        return outcome
    except ValueError as error:
        outcome = {"status": "invalid", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
        return outcome
    except (RuntimeError, MemoryError, np.linalg.LinAlgError) as error:
        outcome = {"status": "failed", "density": density, "summary": {}, "history": history, "diagnostics": [str(error)]}
        return outcome
    finally:
        if system is not None:
            cleanup = system.close()
            if cleanup and outcome is not None:
                outcome["diagnostics"].extend(cleanup)
                outcome["status"] = "failed"
