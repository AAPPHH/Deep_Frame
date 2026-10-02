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
    def __init__(self, domain, interface_node_policy="allowed_adjacent", linear_solver="cpu_superlu", gpu_solver_residency="resident"):
        if linear_solver not in ("cpu_superlu", "cuda_cudss"):
            raise ValueError("Unknown topology linear_solver")
        if gpu_solver_residency not in ("resident", "transient"):
            raise ValueError("Unknown topology gpu_solver_residency")
        self.linear_solver = linear_solver
        self.gpu_solver_residency = gpu_solver_residency
        self.gpu_solvers = {}
        self.gpu_reanalyses = 0
        self.gpu_transient_releases = 0
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
        self.symmetry = domain.get("symmetry")
        if self.symmetry is not None:
            self.axis = int(self.symmetry["axis"])
            plane = float(self.symmetry.get("plane_mm", 0.0))
            if abs(float(domain["grid"]["origin_mm"][self.axis]) - plane) > 1e-9:
                raise ValueError("Symmetric half domains must start at the symmetry plane")
            self.plane_nodes = np.intersect1d(np.flatnonzero(np.abs(self.points[:, self.axis] - plane) < 1e-7), self.active_nodes)
        names = set()
        for case in domain["load_cases"]:
            if case["name"] in names:
                raise ValueError("Topology load-case names must be unique")
            names.add(case["name"])
            if case["analysis"] not in ("static", "modal"):
                raise ValueError("Unknown topology load-case analysis")
            fixed_nodes = np.unique(np.concatenate([np.concatenate(self._select_both(region, case["name"], "fixture")) for region in case["fixed_regions"]]))
            if len(fixed_nodes) < 3 or np.linalg.matrix_rank(self.points[fixed_nodes] - self.points[fixed_nodes[0]]) < 2:
                raise ValueError("Topology fixture requires three non-collinear nodes")
            fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
            free = np.setdiff1d(self.active_dofs, fixed, assume_unique=True)
            compiled = {"name": case["name"], "analysis": case["analysis"], "free": free, "fixed": fixed, "load_regions": [], "parts": []}
            if case["analysis"] == "static":
                force, mirrored = np.zeros(self.ndof), np.zeros(self.ndof)
                if not case.get("loads"):
                    raise ValueError("Static topology cases require loads")
                for load in case["loads"]:
                    direct, mirror = self._select_both(load["region"], case["name"], "load")
                    if np.intersect1d(np.concatenate([direct, mirror]), fixed_nodes).size:
                        raise ValueError("Topology force patch overlaps the fixture")
                    vector = np.asarray(load["force_n"], dtype=float)
                    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
                        raise ValueError("Topology force must be a finite nonzero 3-vector")
                    share = vector / (len(direct) + len(mirror))
                    np.add.at(force, (3 * direct[:, None] + np.arange(3)).ravel(), np.tile(share, len(direct)))
                    if len(mirror):
                        np.add.at(mirrored, (3 * mirror[:, None] + np.arange(3)).ravel(), np.tile(share * self._flip(), len(mirror)))
                    compiled["load_regions"].append((direct, mirror, vector))
                compiled["force"] = force
                for fixed_part, part_force, sign in self._parts(fixed, force, mirrored):
                    part_free = np.setdiff1d(self.active_dofs, fixed_part, assume_unique=True)
                    if np.linalg.norm(part_force[part_free]) > 1e-12 * np.linalg.norm(force + mirrored):
                        part = {"force": part_force, "fixed": fixed_part, "free": part_free, "sign": sign}
                        compiled["parts"].append(part)
                        self.groups[fixed_part.tobytes()].append((compiled, part))
                if not compiled["parts"]:
                    raise ValueError("Topology load case has no resolvable force")
            self.cases.append(compiled)
        if not self.groups:
            raise ValueError("Topology optimization requires at least one static case")
    def _flip(self):
        flip = np.ones(3)
        flip[self.axis] = -1
        return flip
    def _parts(self, fixed, force, mirrored):
        if self.symmetry is None:
            return [(fixed, force, 1.0)]
        normal = 3 * self.plane_nodes + self.axis
        tangential = (3 * self.plane_nodes[:, None] + np.asarray([index for index in range(3) if index != self.axis])).ravel()
        symmetric, antisymmetric = (force + mirrored) / 2, (force - mirrored) / 2
        symmetric[normal], symmetric[tangential] = 0, force[tangential] / 2
        antisymmetric[tangential], antisymmetric[normal] = 0, force[normal] / 2
        return [(np.union1d(fixed, normal), symmetric, 1.0), (np.union1d(fixed, tangential), antisymmetric, -1.0)]
    def _select_both(self, region, case_name, role):
        if self.symmetry is None:
            return self._select(region, case_name, role), np.zeros(0, dtype=int)
        mirrored = deepcopy(region)
        mirrored["min_mm"][self.axis], mirrored["max_mm"][self.axis] = -region["max_mm"][self.axis], -region["min_mm"][self.axis]
        selected = []
        for candidate in (region, mirrored):
            try:
                selected.append(self._select(candidate, case_name, role))
            except ValueError as error:
                if not str(error).startswith("Empty topology node selector"):
                    raise
                selected.append(np.zeros(0, dtype=int))
        selected[1] = np.setdiff1d(selected[1], self.plane_nodes, assume_unique=True)
        if not len(selected[0]) + len(selected[1]):
            raise ValueError(f"Empty topology node selector on the symmetric half domain: {region}")
        return selected
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
        factor = 1.0 if self.symmetry is None else 2.0
        accumulated = {}
        for members in self.groups.values():
            free = members[0][1]["free"]
            forces = np.column_stack([part["force"][free] for _, part in members])
            reduced = stiffness[free, :][:, free].tocsc()
            solutions = self._linear_solve(members[0][1]["fixed"].tobytes(), reduced, forces)
            residual = np.linalg.norm(reduced @ solutions - forces, axis=0) / np.maximum(np.linalg.norm(forces, axis=0), 1e-30)
            tolerance = 1e-6 if self.linear_solver == "cuda_cudss" else 1e-4
            if not np.all(np.isfinite(solutions)) or np.any(residual > tolerance):
                raise RuntimeError(f"Topology linear solve failed residual check: {residual.tolist()}")
            for column, (case, part) in enumerate(members):
                displacement = np.zeros(self.ndof)
                displacement[free] = solutions[:, column]
                element_displacement = displacement[self.dofs]
                entry = accumulated.setdefault(case["name"], {"case": case, "compliance": 0.0, "energy": np.zeros(self.nelem), "residual": 0.0, "fields": []})
                entry["compliance"] += factor * float(np.dot(part["force"], displacement))
                entry["energy"] += factor * np.einsum("ei,ij,ej->e", element_displacement, self.ke, element_displacement, optimize=True)
                entry["residual"] = max(entry["residual"], float(residual[column]))
                entry["fields"].append((part["sign"], displacement))
        results = {}
        for name, entry in accumulated.items():
            if not np.isfinite(entry["compliance"]) or entry["compliance"] <= 0:
                raise RuntimeError("Topology compliance must be finite and positive")
            result = {"compliance_n_mm": entry["compliance"], "derivative": -derivative * entry["energy"], "relative_residual": entry["residual"]}
            if metrics:
                direct = sum(displacement for _, displacement in entry["fields"])
                mirror = sum(sign * displacement for sign, displacement in entry["fields"])
                result.update(self._metrics([direct] if self.symmetry is None else [direct, mirror], moduli, entry["case"]))
            results[name] = result
        return results
    def _linear_solve(self, key, reduced, forces):
        if self.linear_solver == "cpu_superlu":
            return splu(reduced, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}).solve(forces)
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
        if self.gpu_solver_residency == "transient":
            released = self.gpu_solvers.pop(key)
            self.gpu_solver_history.append(released.diagnostics())
            cleanup = released.close()
            if cleanup:
                raise RuntimeError("cuDSS cleanup failed: " + "; ".join(cleanup))
            self.gpu_transient_releases += 1
        return solutions
    def _metrics(self, fields, moduli, case):
        nodal = [field.reshape(-1, 3) for field in fields]
        flip = np.ones(3) if self.symmetry is None else self._flip()
        loads = []
        for direct, mirror, force in case["load_regions"]:
            values = np.concatenate([nodal[0][direct], nodal[-1][mirror] * flip])
            mean = np.mean(values, axis=0)
            directional = float(mean @ force / np.linalg.norm(force))
            loads.append({"node_count": len(values), "force_n": force.tolist(), "mean_displacement_mm": mean.tolist(), "directional_displacement_mm": directional, "stiffness_n_per_mm": float(np.linalg.norm(force) / directional) if directional > 0 else None})
        stress_maximum = 0.0
        for field in fields:
            element_displacement = field[self.dofs]
            for strain in self.strain:
                stress = ((element_displacement @ strain.T) @ self.constitutive.T) * moduli[:, None]
                xx, yy, zz, xy, yz, xz = stress.T
                von_mises = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + xz ** 2))
                stress_maximum = max(stress_maximum, float(np.max(von_mises)))
        return {"max_displacement_mm": float(max(np.max(np.linalg.norm(values, axis=1)) for values in nodal)), "max_von_mises_mpa": stress_maximum, "loads": loads, "stiffness_n_per_mm": loads[0]["stiffness_n_per_mm"] if len(loads) == 1 else None}
    def elastic_frequencies(self, physical_density, case_name, number=3, penalization=3.0, min_stiffness_ratio=1e-6):
        if self.symmetry is not None:
            raise ValueError("Voxel modal analysis is not available on symmetric half domains")
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
        return _json_copy({"nodes": len(self.points), "active_nodes": len(self.active_nodes), "elements": self.nelem, "active_elements": int(np.count_nonzero(self.active_elements)), "dofs": self.ndof, "active_dofs": len(self.active_dofs), "interface_node_policy": self.interface_node_policy, "linear_solver": self.linear_solver, "gpu_symbolic_reanalyses": self.gpu_reanalyses, "gpu_solver_residency": self.gpu_solver_residency, "gpu_transient_releases": self.gpu_transient_releases, "gpu_solver_details": self.gpu_solver_history + [solver.diagnostics() for solver in self.gpu_solvers.values()], "independent_fixtures": len(self.groups), "selector_expansions": self.selector_expansions, "selector_filtering": self.selector_filtering, "cases": [{"name": case["name"], "analysis": case["analysis"], "fixed_nodes": len(case["fixed"]) // 3, "load_nodes": [len(direct) + len(mirror) for direct, mirror, _ in case["load_regions"]], "solved_parts": len(case["parts"])} for case in self.cases], "symmetry": self.symmetry, "factorization_groups": len(self.groups)})

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
    "projection": "single",
    "robust_delta": 0.25,
    "beta_schedule": None,
    "beta_interval": 50,
    "beta_minimum_iterations": 20,
    "beta_change_tolerance": 0.01,
    "move_limit_late": None,
    "move_limit_late_beta": 8.0,
    "volume_target_relaxation": 0.2,
    "objective_window": 10,
    "gpu_solver_residency": "resident",
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
    if result["gpu_solver_residency"] not in ("resident", "transient"):
        raise ValueError("Invalid topology gpu_solver_residency")
    if result["projection"] not in ("single", "robust"):
        raise ValueError("Invalid topology projection")
    delta = result["robust_delta"]
    if result["projection"] == "robust" and (not np.isfinite(delta) or delta <= 0 or not 0 < result["projection_eta"] - delta < result["projection_eta"] + delta < 1):
        raise ValueError("Robust projection thresholds must lie within (0, 1)")
    for name in ("beta_interval", "beta_minimum_iterations"):
        if isinstance(result[name], bool) or int(result[name]) != result[name] or result[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(result["objective_window"], bool) or int(result["objective_window"]) != result["objective_window"] or result["objective_window"] < 2:
        raise ValueError("objective_window must be an integer of at least 2")
    result["beta_minimum_iterations"] = min(result["beta_minimum_iterations"], result["beta_interval"])
    if result["beta_schedule"] is not None:
        schedule = [float(beta) for beta in result["beta_schedule"]]
        if not schedule or not np.all(np.isfinite(schedule)) or schedule[0] <= 0 or np.any(np.diff(schedule) <= 0):
            raise ValueError("beta_schedule must be a nonempty strictly increasing list of positive values")
        result["beta_schedule"] = schedule
    late = result["move_limit_late"]
    if late is not None and (not np.isfinite(late) or not 0 < late <= 1):
        raise ValueError("move_limit_late must be in (0, 1] or None")
    relaxation = result["volume_target_relaxation"]
    if not np.isfinite(result["beta_change_tolerance"]) or result["beta_change_tolerance"] <= 0 or not np.isfinite(result["move_limit_late_beta"]) or not np.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("Invalid beta continuation settings")
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
        self.beta = settings["beta_schedule"][0] if settings["beta_schedule"] else settings["projection_beta"]
        self.robust = settings["projection"] == "robust"
        eta, delta = settings["projection_eta"], settings["robust_delta"]
        self.thresholds = {"eroded": eta + delta, "intermediate": eta, "dilated": eta - delta} if self.robust else {"intermediate": eta}
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
    def filtered(self, design):
        design = np.asarray(design, dtype=float).ravel()
        if design.size != self.n or not np.all(np.isfinite(design)) or np.any(design < 0) or np.any(design > 1):
            raise ValueError("Design densities must be finite and within [0, 1]")
        return np.asarray(self.filter @ design).ravel() / self.sums
    def physical(self, design):
        return self.project(self.filtered(design), self.settings["projection_eta"])
    def fields(self, design):
        filtered = self.filtered(design)
        return {name: self.project(filtered, eta) for name, eta in self.thresholds.items()}, filtered
    def volume(self, design):
        if self.robust:
            return float(np.sum(self.project(self.filtered(design), self.thresholds["dilated"])[0]))
        return np.sum(self.physical(design)[0])
    def project(self, filtered, eta):
        beta = self.beta
        if beta and self.robust:
            denominator = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
            shifted = beta * (filtered - eta)
            decay = np.exp(-2 * np.abs(shifted))
            physical = (np.tanh(beta * eta) + np.tanh(shifted)) / denominator
            derivative = beta * 4 * decay / (1 + decay) ** 2 / denominator
        elif beta:
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
        if self.volume(design) > target + 1e-8:
            raise ValueError("Volume budget cannot contain preserves and their filtered transition")
        upper = 1.0
        for _ in range(70):
            value = (lower + upper) / 2
            design[self.free] = value
            if self.volume(design) > target:
                upper = value
            else:
                lower = value
        design[self.free] = lower
        return design

def _oc_update(design, objective_derivative, volume_derivative, mapping, target, settings, move_limit):
    free = mapping.free
    volume_derivative = volume_derivative[free]
    if np.any(volume_derivative < 0) or not np.all(np.isfinite(volume_derivative)) or not np.max(volume_derivative) > 0 or not np.all(np.isfinite(objective_derivative[free])):
        raise RuntimeError("Invalid OC sensitivities")
    volume_derivative = np.maximum(volume_derivative, 1e-12 * np.max(volume_derivative))
    ratios = np.maximum(1e-30, -objective_derivative[free] / volume_derivative)
    lower_density = np.maximum(settings["minimum_design_density"], design[free] - move_limit)
    upper_density = np.minimum(1.0, design[free] + move_limit)
    def proposal(multiplier):
        result = design.copy()
        result[free] = np.clip(design[free] * np.sqrt(ratios / max(multiplier, 1e-100)), lower_density, upper_density)
        return result
    lowest = design.copy()
    lowest[free] = lower_density
    if mapping.volume(lowest) > target:
        return lowest
    upper = max(float(np.max(ratios)), 1e-12)
    for _ in range(100):
        if mapping.volume(proposal(upper)) <= target + 1e-9:
            break
        upper *= 2
    else:
        raise RuntimeError("OC could not bracket the volume multiplier")
    lower = 0.0
    candidate = proposal(upper)
    for _ in range(70):
        multiplier = (lower + upper) / 2
        proposed = proposal(multiplier)
        if mapping.volume(proposed) > target:
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

def _stage_entry(mapping, fields, volume_target, move_limit):
    entry = {"projection_beta": mapping.beta, "move_limit": move_limit}
    if mapping.robust:
        entry.update({"dilated_volume_target": volume_target, "eroded_density_sum": float(np.sum(fields["eroded"][0])),
                      "dilated_density_sum": float(np.sum(fields["dilated"][0]))})
    return entry

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
        system = HexElasticity(domain, interface_node_policy=settings["interface_node_policy"], linear_solver=settings["linear_solver"], gpu_solver_residency=settings["gpu_solver_residency"])
        scales = None
        converged = False
        stop_reason = "max_iterations"
        schedule = settings["beta_schedule"] or [mapping.beta]
        staged = mapping.robust or settings["beta_schedule"] is not None
        stiffness_name, volume_name = ("eroded", "dilated") if mapping.robust else ("intermediate", "intermediate")
        level = level_iterations = 0
        volume_target = target
        stall = None
        for iteration in range(1, settings["max_iterations"] + 1):
            fields, _ = mapping.fields(design)
            physical = fields["intermediate"][0]
            density = physical.reshape(tuple(domain["grid"]["shape"]))
            if mapping.robust:
                dilated = float(np.sum(fields[volume_name][0]))
                reference = dilated if level_iterations == 0 else volume_target
                volume_target = reference + settings["volume_target_relaxation"] * (target * dilated / float(np.sum(physical)) - reference)
            solutions = system.solve(fields[stiffness_name][0], settings["penalization"], settings["min_stiffness_ratio"])
            if scales is None:
                scales, normalization, weights = _case_scaling(solutions, settings)
            physical_gradient = sum(scales[name] * result["derivative"] for name, result in solutions.items())
            gradient = mapping.pullback(physical_gradient, fields[stiffness_name][1])
            volume_gradient = mapping.pullback(np.ones(mapping.n), fields[volume_name][1])
            late = settings["move_limit_late"] is not None and mapping.beta >= settings["move_limit_late_beta"]
            move_limit = settings["move_limit_late"] if late else settings["move_limit"]
            candidate = _oc_update(design, gradient, volume_gradient, mapping, volume_target, settings, move_limit)
            change = float(np.max(np.abs(candidate - design)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            if staged:
                recent = [entry["objective"] for entry in history[-min(level_iterations + 1, settings["objective_window"]):]]
                stall = (max(recent) - min(recent)) / min(recent) if len(recent) == settings["objective_window"] else None
                history[-1].update(_stage_entry(mapping, fields, volume_target, move_limit), objective_stall=stall)
            measure = change if stall is None else min(change, stall)
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            design = candidate
            level_iterations += 1
            if level == len(schedule) - 1:
                if level_iterations >= settings["minimum_iterations"] and measure < settings["change_tolerance"]:
                    converged = True
                    stop_reason = "change_tolerance" if change < settings["change_tolerance"] else "objective_stall"
                    break
            elif level_iterations >= settings["beta_interval"] or (level_iterations >= settings["beta_minimum_iterations"] and measure < settings["beta_change_tolerance"]):
                level += 1
                level_iterations = 0
                mapping.beta = schedule[level]
            if settings["max_runtime_s"] is not None and perf_counter() - started >= settings["max_runtime_s"]:
                stop_reason = "max_runtime_s"
                break
        fields, filtered = mapping.fields(design)
        physical = fields["intermediate"][0]
        density = physical.reshape(tuple(domain["grid"]["shape"]))
        solutions = system.solve(fields[stiffness_name][0], settings["penalization"], settings["min_stiffness_ratio"], metrics=True)
        final_entry = _history_entry(iteration + 1, physical, solutions, scales, None, perf_counter() - started, final=True)
        if staged:
            final_entry.update(_stage_entry(mapping, fields, volume_target, None))
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
        if staged:
            summary["continuation"] = {"beta_schedule": schedule, "beta_final": mapping.beta, "final_level_reached": level == len(schedule) - 1,
                                       "level_iterations_final": level_iterations}
        outcome = {"status": "ok", "density": density.copy(), "design_density": design.reshape(density.shape).copy(), "summary": summary, "history": history, "diagnostics": diagnostics}
        if mapping.robust:
            allowed = np.count_nonzero(mapping.allowed)
            summary["method"] = "3D Hex8 SIMP, spatial density filter, robust eroded/intermediate/dilated Heaviside projection with beta continuation, eroded-design normalized multi-load compliance, OC dilated-volume constraint"
            summary["robust"] = {"thresholds": mapping.thresholds, "objective_field": "eroded", "volume_constraint_field": "dilated",
                                 "reported_density_field": "intermediate", "dilated_volume_target_final": volume_target,
                                 "eroded_volume_fraction": float(np.sum(fields["eroded"][0]) / allowed),
                                 "dilated_volume_fraction": float(np.sum(fields["dilated"][0]) / allowed),
                                 "dilated_frame_mass_g": float(np.sum(fields["dilated"][0]) * np.prod(system.spacing) * system.density / 1000)}
            outcome.update({name + "_density": fields[name][0].reshape(density.shape).copy() for name in ("eroded", "dilated")})
            outcome["filtered_density"] = filtered.reshape(density.shape).copy()
        if not converged:
            diagnostics.append(f"Density iteration stopped at {stop_reason}; convergence is not claimed")
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
