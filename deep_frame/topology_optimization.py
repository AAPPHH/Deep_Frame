import ctypes
import os
from collections import defaultdict
from copy import deepcopy
from importlib.metadata import distribution, version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags
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
        self.memory_estimates = self._memory_estimates()
    def _memory_estimates(self):
        estimates = (ctypes.c_int64 * 16)()
        written = ctypes.c_size_t()
        try:
            status = self.library.cudssDataGet(self.handles["handle"], self.handles["data"], 13, ctypes.cast(estimates, ctypes.c_void_p), ctypes.sizeof(estimates), ctypes.byref(written))
        except Exception:
            return None
        return dict(zip(("permanent_device_bytes", "peak_device_bytes", "permanent_host_bytes", "peak_host_bytes"), map(int, estimates))) if status == 0 else None
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
    def substitute(self, rhs):
        if self.closed or rhs.shape != self.rhs_device.shape or not np.all(np.isfinite(rhs)):
            raise ValueError("cuDSS substitution requires an open solver and a finite RHS of unchanged shape")
        self.rhs_device.set(np.asarray(rhs, dtype=np.float64, order="F"))
        self._execute(1008)
        return self.cp.asnumpy(self.solution_device)
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
                "cudss_memory_estimates": self.memory_estimates, "solves": self.timings}
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

def orthotropic_matrix(constants):
    xy, z, nu_xy, nu_xz = constants["e_xy_mpa"], constants["e_z_mpa"], constants["nu_xy"], constants["nu_xz"]
    compliance = np.zeros((6, 6))
    compliance[:3, :3] = [[1 / xy, -nu_xy / xy, -nu_xz / xy], [-nu_xy / xy, 1 / xy, -nu_xz / xy], [-nu_xz / xy, -nu_xz / xy, 1 / z]]
    compliance[3:, 3:] = np.diag([1 / constants["g_xy_mpa"], 1 / constants["g_z_mpa"], 1 / constants["g_z_mpa"]])
    if np.any(np.linalg.eigvalsh(compliance) <= 0):
        raise ValueError("Orthotropic engineering constants are not positive definite")
    return np.linalg.inv(compliance)

def hexahedron_matrices(spacing_mm, poisson_ratio, constitutive=None):
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (3,) or np.any(spacing <= 0) or not np.all(np.isfinite(spacing)):
        raise ValueError("Hexahedron spacing must contain three finite positive lengths")
    stiffness = np.zeros((24, 24))
    mass = np.zeros((24, 24))
    strain_matrices = []
    constitutive = elasticity_matrix(poisson_ratio) if constitutive is None else constitutive
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

LINEAR_SOLVERS = ("auto", "cpu_superlu", "cuda_cudss", "multigrid")
SOLVER_CHOICE = {"multigrid_below_mm": 1.1}

def choose_solver(linear_solver, spacing, choice=None):
    if linear_solver not in LINEAR_SOLVERS:
        raise ValueError("Unknown topology linear_solver")
    if linear_solver != "auto":
        return linear_solver
    return "multigrid" if float(np.max(spacing)) < {**SOLVER_CHOICE, **(choice or {})}["multigrid_below_mm"] else "cuda_cudss"

class HexElasticity:
    def __init__(self, domain, interface_node_policy="allowed_adjacent", linear_solver="cpu_superlu", gpu_solver_residency="resident", share_static="auto", multigrid=None, solver_choice=None):
        self.requested_solver = linear_solver
        linear_solver = choose_solver(linear_solver, domain["grid"]["spacing_mm"], solver_choice)
        if gpu_solver_residency not in ("resident", "transient"):
            raise ValueError("Unknown topology gpu_solver_residency")
        self.linear_solver = linear_solver
        self.gpu_solver_residency = gpu_solver_residency
        self.share_static, self.shared, self.unshared = linear_solver != "multigrid" if share_static == "auto" else share_static, None, set()
        self.multigrid_settings, self.multigrid = multigrid, None
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
        if material.get("orthotropic"):
            self.young = float(material["orthotropic"]["e_xy_mpa"])
            self.constitutive = orthotropic_matrix(material["orthotropic"]) / self.young
        else:
            self.constitutive = elasticity_matrix(material["poisson_ratio"])
        self.ke, self.me, self.strain = hexahedron_matrices(self.spacing, material["poisson_ratio"], self.constitutive)
        self._assembly = None
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
        names, self.support, self.relief_operators = set(), None, {}
        for case in domain["load_cases"]:
            if case["name"] in names:
                raise ValueError("Topology load-case names must be unique")
            names.add(case["name"])
            if case["analysis"] not in ("static", "modal"):
                raise ValueError("Unknown topology load-case analysis")
            relief = case.get("inertia_relief")
            if relief is not None:
                if self.support is None:
                    self.support = self._relief_support(np.concatenate([np.concatenate(self._select_both(load["region"], other["name"], "load")) for other in domain["load_cases"] if "inertia_relief" in other for load in other["loads"]]))
                fixed_nodes, fixed, split = self.support
            else:
                split = None
                fixed_nodes = np.unique(np.concatenate([np.concatenate(self._select_both(region, case["name"], "fixture")) for region in case["fixed_regions"]]))
                if len(fixed_nodes) < 3 or np.linalg.matrix_rank(self.points[fixed_nodes] - self.points[fixed_nodes[0]]) < 2:
                    raise ValueError("Topology fixture requires three non-collinear nodes")
                fixed = (3 * fixed_nodes[:, None] + np.arange(3)).ravel()
            free = np.setdiff1d(self.active_dofs, fixed, assume_unique=True)
            compiled = {"name": case["name"], "analysis": case["analysis"], "free": free, "fixed": fixed, "split": split, "load_regions": [], "parts": []}
            if case["analysis"] == "static":
                force, mirrored = np.zeros(self.ndof), np.zeros(self.ndof)
                if not case.get("loads") and not any(np.any(body["force_n"]) for body in (relief or {}).get("bodies", [])):
                    raise ValueError("Static topology cases require loads")
                for load in case.get("loads", []):
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
                if relief is not None:
                    compiled["inertia_relief"] = self._inertia(relief, case["name"], fixed_nodes, force, mirrored)
                self._register(compiled, fixed, force, mirrored, split)
            self.cases.append(compiled)
        if not self.groups:
            raise ValueError("Topology optimization requires at least one static case")
    @property
    def assembly(self):
        if self._assembly is None:
            elements = self.dofs[self.active_elements]
            self._assembly = np.repeat(elements, 24, axis=1).ravel(), np.tile(elements, (1, 24)).ravel()
        return self._assembly
    def _register(self, compiled, fixed, force, mirrored, split=None):
        compiled["force"], compiled["mirrored"], self.shared = force, mirrored, None
        for fixed_part, part_force, sign in self._parts(fixed, force, mirrored, split):
            known = self.groups.get(fixed_part.tobytes())
            part_free = known[0][1]["free"] if known else np.setdiff1d(self.active_dofs, fixed_part, assume_unique=True)
            if np.linalg.norm(part_force[part_free]) > 1e-12 * np.linalg.norm(force + mirrored):
                base = fixed if split is None else split[sign]
                part = {"force": part_force, "fixed": fixed_part, "free": part_free, "sign": sign, "support": base if self.symmetry is None else np.setdiff1d(base, self._plane_dofs(sign))}
                compiled["parts"].append(part)
                self.groups[fixed_part.tobytes()].append((compiled, part))
        if not compiled["parts"]:
            raise ValueError("Topology load case has no resolvable force")
    def add_case(self, name, support, force, mirrored, keep_fields=True):
        if name in {case["name"] for case in self.cases}:
            raise ValueError("Topology load-case names must be unique")
        source = next(case for case in self.cases if case["name"] == support)
        compiled = {"name": name, "analysis": "static", "free": source["free"], "fixed": source["fixed"], "split": source.get("split"), "load_regions": [], "parts": [], "keep_fields": keep_fields}
        self._register(compiled, source["fixed"], force, mirrored, compiled["split"])
        self.cases.append(compiled)
        return compiled
    def set_force(self, case, force, mirrored):
        parts = {sign: part_force for _, part_force, sign in self._parts(case["fixed"], force, mirrored, case.get("split"))}
        for part in case["parts"]:
            part["force"] = parts[part["sign"]]
        self.shared = None
        return {part["sign"]: part["force"] for part in case["parts"]}
    def _rigid_modes(self, sign):
        center = self.points[self.active_nodes].mean(axis=0)
        if self.symmetry is not None:
            center[self.axis] = float(self.symmetry.get("plane_mm", 0.0))
        modes = np.zeros((self.ndof, 6))
        for index in range(3):
            modes[index::3, index] = 1
            modes[:, 3 + index] = np.cross(np.eye(3)[index], self.points - center).ravel()
        modes[np.setdiff1d(np.arange(self.ndof), self.active_dofs)] = 0
        if self.symmetry is None:
            return modes
        _, values, vectors = np.linalg.svd(modes[self._plane_dofs(sign)])
        return modes @ vectors[np.count_nonzero(values > 1e-9 * values[0]):].T
    def _share_plan(self):
        hosts, plan = {}, {}
        for key, members in self.groups.items():
            if self.share_static and any("inertia_relief" in case for case, _ in members):
                hosts.setdefault(members[0][1]["sign"], key)
        for key, members in self.groups.items():
            sign, fixed = members[0][1]["sign"], members[0][1]["fixed"]
            if hosts.get(sign, key) == key or key in self.unshared:
                continue
            host = self.groups[hosts[sign]][0][1]
            support = np.setdiff1d(np.intersect1d(host["fixed"], self.active_dofs), self._plane_dofs(sign) if self.symmetry is not None else [])
            constrained = np.intersect1d(np.setdiff1d(fixed, host["fixed"]), host["free"])
            modes, held = self._rigid_modes(sign), np.isin(support, fixed)
            if len(support) == modes.shape[1] and np.linalg.matrix_rank(modes[support]) == len(support) and not any(np.any(part["force"][support]) for _, part in members) and np.linalg.matrix_rank(modes[np.union1d(constrained, support[held])]) == len(support):
                plan[key] = {"host": hosts[sign], "constrained": constrained, "positions": np.searchsorted(host["free"], constrained), "modes": modes, "released": np.linalg.inv(modes[support].T)[~held], "held": modes[support[held]]}
        return plan
    def _flip(self):
        flip = np.ones(3)
        flip[self.axis] = -1
        return flip
    def _plane_dofs(self, sign):
        if sign > 0:
            return 3 * self.plane_nodes + self.axis
        return (3 * self.plane_nodes[:, None] + np.asarray([index for index in range(3) if index != self.axis])).ravel()
    def _relief_support(self, loaded):
        if self.symmetry is not None and not np.intersect1d(self.interface_nodes, self.plane_nodes).size:
            return self._mirrored_support(np.setdiff1d(self.interface_nodes, loaded))
        candidates = np.setdiff1d(np.intersect1d(self.interface_nodes, self.plane_nodes) if self.symmetry is not None else self.interface_nodes, loaded)
        points = self.points[candidates]
        a = int(np.argmin(points[:, 1] + 1e-3 * points[:, 2]))
        b = int(np.argmax(np.linalg.norm(points - points[a], axis=1)))
        normal = np.cross(points - points[a], points[b] - points[a])
        c = int(np.argmax(np.linalg.norm(normal, axis=1)))
        if np.linalg.norm(normal[c]) < 1e-9:
            raise ValueError("Inertia-relief support requires three non-collinear interface nodes")
        along, normal = np.abs(points[b] - points[a]), np.abs(np.cross(points[c] - points[a], points[b] - points[a]))
        nodes = candidates[[a, b, c]]
        dofs = [3 * nodes[0] + np.arange(3), 3 * nodes[1] + np.delete(np.arange(3), np.argmax(along)), [3 * nodes[2] + np.argmax(normal)]]
        return nodes, np.sort(np.concatenate(dofs)), None
    def _mirrored_support(self, candidates):
        plane = np.asarray([index for index in range(3) if index != self.axis])
        points = self.points[candidates][:, plane]
        a = int(np.argmin(points[:, 0] + 1e-3 * points[:, 1]))
        offset = points - points[a]
        b = int(np.argmax(np.linalg.norm(offset, axis=1)))
        area = np.abs(offset[:, 0] * offset[b, 1] - offset[:, 1] * offset[b, 0])
        c = int(np.argmax(area))
        if area[c] < 1e-9:
            raise ValueError("Inertia-relief support requires three interface nodes non-collinear in the symmetry plane projection")
        nodes = candidates[[a, b, c]]
        symmetric = np.sort(np.concatenate([3 * nodes[0] + plane, [3 * nodes[1] + plane[np.argmin(np.abs(offset[b]))]]]))
        antisymmetric = np.sort(3 * nodes + self.axis)
        return nodes, np.union1d(symmetric, antisymmetric), {1.0: symmetric, -1.0: antisymmetric}
    def _inertia(self, relief, case_name, support_nodes, force, mirrored):
        flip = np.ones(3) if self.symmetry is None else self._flip()
        direct, mirror = np.zeros(len(self.points)), np.zeros(len(self.points))
        for item in relief.get("point_masses", []):
            nodes = self._select_both(item["region"], case_name, "mass")
            share = item["mass_g"] / (len(nodes[0]) + len(nodes[1]))
            np.add.at(direct, nodes[0], share)
            np.add.at(mirror, nodes[1], share)
        if relief.get("preserve_mass_g", 0) > 0:
            preserve = np.asarray(self.domain["preserve"], dtype=bool).ravel()
            incidence = np.bincount(self.connectivity[preserve].ravel(), minlength=len(self.points)).astype(float)
            incidence *= relief["preserve_mass_g"] / (incidence.sum() * (1 if self.symmetry is None else 2))
            direct += incidence
            if self.symmetry is not None:
                mirror += incidence
                direct[self.plane_nodes] += mirror[self.plane_nodes]
                mirror[self.plane_nodes] = 0
        bodies = relief.get("bodies", [])
        positions = np.concatenate([self.points, self.points * flip, np.reshape([body["position_mm"] for body in bodies], (-1, 3))])
        masses = np.concatenate([direct, mirror, [body["mass_g"] for body in bodies]])
        applied = np.concatenate([force.reshape(-1, 3), mirrored.reshape(-1, 3) * flip, np.reshape([body["force_n"] for body in bodies], (-1, 3))])
        total = masses.sum()
        center = masses @ positions / total
        arm = positions - center
        inertia = np.eye(3) * np.sum(masses * np.sum(arm ** 2, axis=1)) - (arm * masses[:, None]).T @ arm + sum((np.asarray(body["inertia_g_mm2"]) for body in bodies), np.zeros((3, 3)))
        linear = applied.sum(axis=0) / total
        angular = np.linalg.solve(inertia, np.cross(arm, applied).sum(axis=0))
        inertial = -masses[:, None] * (linear + np.cross(angular, arm))
        count = len(self.points)
        force += inertial[:count].ravel()
        mirrored += (inertial[count:2 * count] * flip).ravel()
        loads = [{"name": body["name"], "position_mm": list(body["position_mm"]), "force_n": (inertial[2 * count + index] + body["force_n"]).tolist(), "moment_n_mm": (-np.asarray(body["inertia_g_mm2"]) @ angular).tolist()} for index, body in enumerate(bodies)]
        self.relief_operators[case_name] = {"direct": direct, "mirror": mirror, "flip": flip, "total": float(total), "center": center, "inertia": inertia, "bodies": bodies}
        return {"mass_g": float(total), "center_of_mass_mm": center.tolist(), "acceleration_n_per_g": linear.tolist(), "angular_acceleration": angular.tolist(), "support_nodes_mm": self.points[support_nodes].tolist(), "bodies": loads}
    def support_reactions(self, physical_density, case_name, penalization=3.0, min_stiffness_ratio=1e-6):
        case = next(case for case in self.cases if case["name"] == case_name)
        density = np.asarray(physical_density, dtype=float).ravel()
        stiffness = self.matrix(self.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * density ** penalization)).tocsr()
        reactions = []
        for part in case["parts"]:
            displacement = np.zeros(self.ndof)
            displacement[part["free"]] = splu(stiffness[part["free"], :][:, part["free"]].tocsc()).solve(part["force"][part["free"]])
            residual = stiffness @ displacement - part["force"]
            reactions.append({"sign": part["sign"], "support_dofs": len(part["support"]), "max_reaction_n": float(np.max(np.abs(residual[part["support"]]))), "nodal_force_sum_n": float(np.linalg.norm(part["force"].reshape(-1, 3), axis=1).sum())})
        return reactions
    def _parts(self, fixed, force, mirrored, split=None):
        if self.symmetry is None:
            return [(fixed, force, 1.0)]
        normal, tangential = self._plane_dofs(1), self._plane_dofs(-1)
        symmetric, antisymmetric = (force + mirrored) / 2, (force - mirrored) / 2
        symmetric[normal], symmetric[tangential] = 0, force[tangential] / 2
        antisymmetric[tangential], antisymmetric[normal] = 0, force[normal] / 2
        return [(np.union1d(fixed if split is None else split[1.0], normal), symmetric, 1.0), (np.union1d(fixed if split is None else split[-1.0], tangential), antisymmetric, -1.0)]
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
        stiffness = coo_matrix((values, self.assembly), shape=(self.ndof, self.ndof)).tocsc()
        if self.linear_solver != "cpu_superlu":
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
        stiffness = None if self.linear_solver == "multigrid" else self.matrix(moduli)
        moduli[~self.active_elements] = 0
        derivative[~self.active_elements] = 0
        if stiffness is None:
            self._multigrid().update(moduli)
        if self.shared is None:
            self.shared = self._share_plan()
        guests, accumulated = defaultdict(list), {}
        for key, entry in self.shared.items():
            guests[entry["host"]].append(key)
        for key, members in self.groups.items():
            if key in self.shared:
                continue
            free = members[0][1]["free"]
            reduced = None if stiffness is None else stiffness[free, :][:, free].tocsc()
            blocks = [np.column_stack([part["force"][free] for _, part in members])]
            for guest in guests[key]:
                unit = np.zeros((len(free), len(self.shared[guest]["constrained"])))
                unit[self.shared[guest]["positions"], np.arange(unit.shape[1])] = 1
                blocks += [np.column_stack([part["force"][free] for _, part in self.groups[guest]]), unit]
            solutions = self._multigrid_solve(key, members[0][1], np.hstack(blocks)) if reduced is None else self._linear_solve(key, reduced, np.hstack(blocks))
            self._collect(accumulated, members, free, reduced, solutions[:, :len(members)])
            offset = len(members)
            for guest in guests[key]:
                plan, rest = self.shared[guest], self.groups[guest]
                count, size, modes = len(rest), len(plan["constrained"]), plan["modes"]
                direct, coupling = solutions[:, offset:offset + count], solutions[:, offset + count:offset + count + size]
                offset += count + size
                rigid, released, held = modes[plan["constrained"]], plan["released"], plan["held"]
                system = np.block([[coupling[plan["positions"]], rigid], [released @ rigid.T, np.zeros((len(released), rigid.shape[1]))], [np.zeros((len(held), size)), held]])
                multipliers = np.linalg.solve(system, -np.vstack([direct[plan["positions"]], released @ modes.T @ np.column_stack([part["force"] for _, part in rest]), np.zeros((len(held), count))]))
                full = modes @ multipliers[size:]
                full[free] += direct + coupling @ multipliers[:size]
                guest_free = rest[0][1]["free"]
                guest_reduced, solved = None if stiffness is None else stiffness[guest_free, :][:, guest_free].tocsc(), full[guest_free]
                if not self._accurate(rest, guest_free, guest_reduced, solved)[0]:
                    self.unshared.add(guest)
                    forces = np.column_stack([part["force"][guest_free] for _, part in rest])
                    solved = self._multigrid_solve(guest, rest[0][1], forces) if guest_reduced is None else self._linear_solve(guest, guest_reduced, forces)
                self._collect(accumulated, rest, guest_free, guest_reduced, solved)
        if self.unshared & set(self.shared):
            self.shared = None
        if stiffness is None:
            self.multigrid.torch.cuda.empty_cache()
        results = {}
        for name, entry in accumulated.items():
            if not np.isfinite(entry["compliance"]) or entry["compliance"] <= 0:
                raise RuntimeError("Topology compliance must be finite and positive")
            result = {"compliance_n_mm": entry["compliance"], "derivative": -derivative * entry["energy"], "relative_residual": entry["residual"]}
            if entry["case"].get("keep_fields"):
                result.update(fields={sign: displacement for sign, displacement in entry["fields"]}, modulus_derivative=derivative)
            if metrics:
                direct = sum(displacement for _, displacement in entry["fields"])
                mirror = sum(sign * displacement for sign, displacement in entry["fields"])
                result.update(self._metrics([direct] if self.symmetry is None else [direct, mirror], moduli, entry["case"]))
            results[name] = result
        return results
    def _accurate(self, members, free, reduced, solutions):
        forces = np.column_stack([part["force"][free] for _, part in members])
        norms = np.linalg.norm(forces, axis=0)
        residual = np.linalg.norm((reduced @ solutions if reduced is not None else self._multigrid_product(free, solutions)) - forces, axis=0) / np.where(norms > 1e-12 * norms.max(initial=0.0), norms, max(norms.max(initial=0.0), 1e-30))
        return bool(np.all(np.isfinite(solutions)) and np.all(residual <= (1e-4 if self.linear_solver == "cpu_superlu" else 1e-6))), residual
    def _collect(self, accumulated, members, free, reduced, solutions):
        accurate, residual = self._accurate(members, free, reduced, solutions)
        if not accurate:
            raise RuntimeError(f"Topology linear solve failed residual check: {residual.tolist()}")
        factor = 1.0 if self.symmetry is None else 2.0
        for column, (case, part) in enumerate(members):
            displacement = np.zeros(self.ndof)
            displacement[free] = solutions[:, column]
            element_displacement = displacement[self.dofs]
            entry = accumulated.setdefault(case["name"], {"case": case, "compliance": 0.0, "energy": np.zeros(self.nelem), "residual": 0.0, "fields": []})
            entry["compliance"] += factor * float(np.dot(part["force"], displacement))
            entry["energy"] += factor * np.einsum("ei,ij,ej->e", element_displacement, self.ke, element_displacement, optimize=True)
            entry["residual"] = max(entry["residual"], float(residual[column]))
            entry["fields"].append((part["sign"], displacement))
    def _multigrid(self):
        if self.multigrid is None:
            from deep_frame.topology_multigrid import GeometricMultigrid
            self.multigrid = GeometricMultigrid(self.domain["grid"]["shape"], self.spacing, self.ke, self.multigrid_settings, self.me)
        return self.multigrid
    def modal_constraint(self, settings):
        if self.linear_solver != "multigrid":
            return ModalConstraint(self, settings)
        from deep_frame.topology_multigrid import MultigridModal
        return MultigridModal(self, settings, self._multigrid())
    def _multigrid_solve(self, key, part, forces):
        full = np.zeros((self.ndof, forces.shape[1]))
        full[part["free"]] = forces
        floating = self.support is not None and bool(np.isin(part["support"], self.support[1]).all())
        solutions, report = self.multigrid.solve(key, part["fixed"], full, part["support"] if floating else None)
        if not report["converged"]:
            columns = np.argsort(report["relative_residual"])[-3:]
            failed = [(int(column), report["relative_residual"][column], report["iterations"][column]) for column in columns]
            raise RuntimeError(f"Multigrid PCG did not converge: {failed}; floating={floating}; retried={report['retried_columns']}")
        return solutions[part["free"]]
    def _multigrid_product(self, free, solutions):
        full = np.zeros((self.ndof, solutions.shape[1]))
        full[free] = solutions
        return self.multigrid.product(full)[free]
    def _linear_solve(self, key, reduced, forces):
        if self.linear_solver == "cpu_superlu":
            return splu(reduced, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}).solve(forces)
        if key in self.gpu_solvers and (not self.gpu_solvers[key].same_structure(reduced) or self.gpu_solvers[key].rhs_device.shape != forces.shape):
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
        mass = coo_matrix((mass_values, self.assembly), shape=(self.ndof, self.ndof)).tocsc()
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
        if self.multigrid is not None:
            errors.extend(self.multigrid.close())
        return errors
    def diagnostics(self):
        return _json_copy({"nodes": len(self.points), "active_nodes": len(self.active_nodes), "elements": self.nelem, "active_elements": int(np.count_nonzero(self.active_elements)), "dofs": self.ndof, "active_dofs": len(self.active_dofs), "interface_node_policy": self.interface_node_policy, "linear_solver": self.linear_solver, "requested_linear_solver": self.requested_solver, "gpu_symbolic_reanalyses": self.gpu_reanalyses, "gpu_solver_residency": self.gpu_solver_residency, "gpu_transient_releases": self.gpu_transient_releases, "gpu_solver_details": self.gpu_solver_history + [solver.diagnostics() for solver in self.gpu_solvers.values()], "independent_fixtures": len(self.groups), "selector_expansions": self.selector_expansions, "selector_filtering": self.selector_filtering, "cases": [{"name": case["name"], "analysis": case["analysis"], "fixed_nodes": len(case["fixed"]) // 3, "load_nodes": [len(direct) + len(mirror) for direct, mirror, _ in case["load_regions"]], "solved_parts": len(case["parts"]), **({"inertia_relief": case["inertia_relief"]} if "inertia_relief" in case else {})} for case in self.cases], "symmetry": self.symmetry, "factorization_groups": len(self.groups) - len(self.shared or {}), "shared_static_groups": len(self.shared or {}), "unshared_static_groups": len(self.unshared), **({"multigrid": self.multigrid.diagnostics()} if self.multigrid is not None else {})})

def volume_weights(points, discs):
    weights = np.ones(len(points))
    if not discs or discs.get("mode") != "soft":
        return weights
    points = np.asarray(points, dtype=float)
    inside = np.zeros(len(points), dtype=bool)
    for x, y in discs["motors_mm"]:
        inside |= np.hypot(points[:, 0] - x, points[:, 1] - y) <= discs["radius_mm"]
    weights[inside] += discs["weight"] * np.exp(-np.abs(points[inside, 2] - discs["plane_mm"]) / discs["length_mm"])
    return weights

class AugmentedLagrangian:
    def __init__(self, settings):
        self.settings, self.multiplier, self.calls = settings, 0.0, 0
    def augment(self, violation, slope):
        penalty, multiplier = self.settings["penalty"], self.multiplier
        active = max(0.0, violation + multiplier / penalty)
        self.calls += 1
        if self.calls % self.settings["multiplier_interval"] == 0:
            self.multiplier = max(0.0, multiplier + penalty * violation)
        return penalty / 2 * active ** 2 - multiplier ** 2 / (2 * penalty), penalty * active * slope, multiplier

class StiffnessConstraint(AugmentedLagrangian):
    def __init__(self, system, settings):
        super().__init__(settings)
        case = next(case for case in system.cases if case["name"] == settings["case"])
        self.case, self.force = settings["case"], float(np.linalg.norm(case["load_regions"][0][2]))
        if len(case["load_regions"]) != 1 or "inertia_relief" in case:
            raise ValueError("The stiffness constraint needs a fixtured case with one load patch")
        self.target = settings["min_n_per_mm"] * settings["calibration"]
    def measure(self, solution):
        limit = self.force ** 2 / self.target
        violation = solution["compliance_n_mm"] / limit - 1
        return violation, solution["derivative"] / limit, {"stiffness_n_per_mm": self.force ** 2 / solution["compliance_n_mm"], "target_n_per_mm": self.target, "violation": float(violation)}
    def __call__(self, solution):
        violation, slope, info = self.measure(solution)
        penalty, gradient, multiplier = self.augment(violation, slope)
        return penalty, gradient, {**info, "multiplier": multiplier}

class ModalConstraint(AugmentedLagrangian):
    def __init__(self, system, settings):
        super().__init__(settings)
        self.system = system
        self.target = (2 * np.pi * settings["f1_min_hz"]) ** 2
        case = next(case for case in system.cases if case["name"] == settings["case"])
        self.parts = []
        for sign in (1.0,) if system.symmetry is None else (1.0, -1.0):
            fixed = case["fixed"] if system.symmetry is None else np.union1d(case["fixed"], system._plane_dofs(sign))
            self.parts.append({"sign": sign, "fixed": fixed, "free": np.setdiff1d(system.active_dofs, fixed, assume_unique=True), "vectors": None, "key": f"modal{sign:+.0f}"})
        self.lumped = np.zeros(len(system.points))
        for item in system.domain.get("point_masses", []):
            direct, mirror = system._select_both(item["attachment_region"], case["name"], "mass")
            share = np.full(len(direct), item["mass_g"] * 1e-6 / (len(direct) + len(mirror)))
            if system.symmetry is not None:
                share[np.isin(direct, system.plane_nodes)] /= 2
            np.add.at(self.lumped, direct, share)
        self.lumped = self.base_lumped = np.repeat(self.lumped, 3)
        self.lumped_slope = None
    def interpolation(self, density):
        cutoff = self.settings["mass_cutoff"]
        low = density < cutoff
        return np.where(low, density ** 6 / cutoff ** 5, density), np.where(low, 6 * density ** 5 / cutoff ** 5, 1.0)
    def matrices(self, density, penalization, min_stiffness_ratio):
        system = self.system
        stiffness = system.matrix(system.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * density ** penalization))
        mass, _ = self.interpolation(density)
        values = (system.density * 1e-9 * mass[system.active_elements, None] * system.me.ravel()[None, :]).ravel()
        return stiffness, (coo_matrix((values, system.assembly), shape=(system.ndof, system.ndof)) + diags(self.lumped)).tocsc()
    def eigenpairs(self, part, stiffness, mass, iterations):
        free = part["free"]
        reduced, reduced_mass = stiffness[free, :][:, free].tocsc(), mass[free, :][:, free].tocsc()
        vectors = part["vectors"] if part["vectors"] is not None else np.random.default_rng(0).standard_normal((len(free), self.settings["modes"]))
        if self.system.linear_solver == "cpu_superlu":
            substitute = splu(reduced, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}).solve
            solved = substitute(reduced_mass @ vectors)
        else:
            solved = self.system._linear_solve(part["key"], reduced, reduced_mass @ vectors)
            substitute = self.system.gpu_solvers[part["key"]].substitute if part["key"] in self.system.gpu_solvers else lambda rhs: self.system._linear_solve(part["key"], reduced, rhs)
        for index in range(iterations):
            if index:
                solved = substitute(reduced_mass @ vectors)
            projected_stiffness, projected_mass = solved.T @ (reduced @ solved), solved.T @ (reduced_mass @ solved)
            values, rotation = eigh((projected_stiffness + projected_stiffness.T) / 2, (projected_mass + projected_mass.T) / 2)
            vectors = solved @ rotation
        part["vectors"] = vectors
        return values, vectors
    def __call__(self, physical_density, penalization=3.0, min_stiffness_ratio=1e-6, iterations=None):
        violation, slope, info = self.measure(physical_density, penalization, min_stiffness_ratio, iterations)
        penalty, gradient, multiplier = self.augment(violation, slope)
        return penalty, gradient, {**info, "multiplier": multiplier}
    def measure(self, physical_density, penalization=3.0, min_stiffness_ratio=1e-6, iterations=None):
        system, settings = self.system, self.settings
        density = np.asarray(physical_density, dtype=float).ravel()
        stiffness, mass = self.matrices(density, penalization, min_stiffness_ratio)
        modes = []
        for part in self.parts:
            count = iterations or (settings["initial_iterations"] if part["vectors"] is None else settings["warm_iterations"])
            values, vectors = self.eigenpairs(part, stiffness, mass, count)
            for index in range(settings["tracked"]):
                full = np.zeros(system.ndof)
                full[part["free"]] = vectors[:, index]
                modes.append((part, values[index], full))
        return self.aggregate(density, modes, penalization, min_stiffness_ratio)
    def aggregate(self, density, modes, penalization, min_stiffness_ratio):
        system, settings = self.system, self.settings
        ratios = np.array([value / self.target for _, value, _ in modes])
        lowest = ratios.min()
        exponents = np.exp(-settings["ks"] * (ratios - lowest))
        aggregate = lowest - np.log(exponents.sum()) / settings["ks"]
        weights = exponents / exponents.sum()
        stiffness_slope = system.young * (1 - min_stiffness_ratio) * penalization * density ** (penalization - 1)
        mass_slope = system.density * 1e-9 * self.interpolation(density)[1]
        sensitivity = np.zeros(system.nelem)
        for weight, (part, value, vector) in zip(weights, modes):
            element = vector[system.dofs]
            strain = np.einsum("ei,ij,ej->e", element, system.ke, element, optimize=True)
            kinetic = np.einsum("ei,ij,ej->e", element, system.me, element, optimize=True)
            sensitivity += weight * (stiffness_slope * strain - value * mass_slope * kinetic - (value * self.lumped_slope(vector) if self.lumped_slope else 0.0)) / self.target
        sensitivity[~system.active_elements] = 0
        violation = 1 - aggregate
        frequencies = [[part["sign"], float(np.sqrt(max(value, 0)) / (2 * np.pi))] for part, value, _ in modes]
        return violation, -sensitivity, {"f1_hz": float(np.sqrt(max(lowest * self.target, 0)) / (2 * np.pi)), "frequencies_hz": frequencies, "aggregate_ratio": float(aggregate), "violation": float(violation)}

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
    "prop_discs": None,
    "modal": None,
    "solver_choice": SOLVER_CHOICE,
    "density_filter": "auto",
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
    if result["linear_solver"] not in LINEAR_SOLVERS:
        raise ValueError("Invalid topology linear_solver")
    if result["gpu_solver_residency"] not in ("resident", "transient"):
        raise ValueError("Invalid topology gpu_solver_residency")
    if result["density_filter"] not in DENSITY_FILTERS:
        raise ValueError("Invalid topology density_filter")
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

DENSITY_FILTERS = ("auto", "sparse", "convolution")

def choose_filter(name):
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "sparse"
    return "convolution" if torch.cuda.is_available() else "sparse"

class ConeFilter:
    def __init__(self, allowed, shape, spacing, radius):
        import torch
        self.torch, self.shape, device = torch, shape, torch.device("cuda")
        self.padding = tuple(int(np.floor(radius / h)) for h in spacing)
        offsets = np.meshgrid(*[np.arange(-k, k + 1) * h for k, h in zip(self.padding, spacing)], indexing="ij")
        self.kernel = torch.as_tensor(np.maximum(radius - np.sqrt(sum(offset ** 2 for offset in offsets)), 0.0)[None, None], dtype=torch.float64, device=device)
        self.mask = torch.as_tensor(allowed.reshape(shape), dtype=torch.float64, device=device)
    def __call__(self, vector):
        field = self.torch.as_tensor(np.asarray(vector, dtype=float).reshape(self.shape), device=self.mask.device) * self.mask
        return (self.torch.nn.functional.conv3d(field[None, None], self.kernel, padding=self.padding)[0, 0] * self.mask).cpu().numpy().ravel()

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
        radius = settings["filter_radius_mm"]
        self.filter, self.convolution = None, None
        if choose_filter(settings["density_filter"]) == "convolution":
            self.convolution = ConeFilter(self.allowed, shape, spacing, radius)
            self.sums = self.convolution(self.allowed)
        else:
            active = np.flatnonzero(self.allowed)
            tree = cKDTree(coordinates[active])
            distances = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
            weights = radius - distances.data
            self.filter = coo_matrix((weights, (active[distances.row], active[distances.col])), shape=(self.n, self.n)).tocsr()
            self.filter.eliminate_zeros()
            self.sums = np.asarray(self.filter.sum(axis=1)).ravel()
        self.sums[self.forbidden] = 1
        if np.any(self.sums <= 0):
            raise ValueError("Density filter contains an empty allowed-cell neighborhood")
        self.weights = np.ones(self.n)
        self.weights[self.free] = volume_weights((coordinates + 0.5 * spacing + np.asarray(domain["grid"].get("origin_mm", [0.0, 0.0, 0.0])))[self.free], settings["prop_discs"])
    def filtered(self, design):
        design = np.asarray(design, dtype=float).ravel()
        if design.size != self.n or not np.all(np.isfinite(design)) or np.any(design < 0) or np.any(design > 1):
            raise ValueError("Design densities must be finite and within [0, 1]")
        return (self.convolution(design) if self.convolution else np.asarray(self.filter @ design).ravel()) / self.sums
    def physical(self, design):
        return self.project(self.filtered(design), self.settings["projection_eta"])
    def fields(self, design):
        filtered = self.filtered(design)
        return {name: self.project(filtered, eta) for name, eta in self.thresholds.items()}, filtered
    def volume(self, design):
        if self.robust:
            return float(np.sum(self.weights * self.project(self.filtered(design), self.thresholds["dilated"])[0]))
        return np.sum(self.weights * self.physical(design)[0])
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
        scaled = np.asarray(sensitivity).ravel() * projection_derivative / self.sums
        result = self.convolution(scaled) if self.convolution else np.asarray(self.filter.T @ scaled).ravel()
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
        system = HexElasticity(domain, interface_node_policy=settings["interface_node_policy"], linear_solver=settings["linear_solver"], gpu_solver_residency=settings["gpu_solver_residency"], solver_choice=settings["solver_choice"])
        modal = system.modal_constraint(settings["modal"]) if settings["modal"] else None
        modal_log = []
        scales = None
        converged = False
        stop_reason = "max_iterations"
        schedule = settings["beta_schedule"] or [mapping.beta]
        staged = mapping.robust or settings["beta_schedule"] is not None
        stiffness_name, volume_name = ("eroded", "dilated") if mapping.robust else ("intermediate", "intermediate")
        minimum_iterations = max(settings["minimum_iterations"], settings["beta_minimum_iterations"]) if staged else settings["minimum_iterations"]
        level = level_iterations = 0
        volume_target = target
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
            modal_penalty, modal_info = 0.0, {}
            if modal is not None:
                modal_penalty, modal_gradient, modal_info = modal(fields[stiffness_name][0], settings["penalization"], settings["min_stiffness_ratio"])
                physical_gradient = physical_gradient + modal_gradient
                modal_log.append({"iteration": iteration, **modal_info})
            gradient = mapping.pullback(physical_gradient, fields[stiffness_name][1])
            volume_gradient = mapping.pullback(mapping.weights, fields[volume_name][1])
            late = settings["move_limit_late"] is not None and mapping.beta >= settings["move_limit_late_beta"]
            move_limit = settings["move_limit_late"] if late else settings["move_limit"]
            candidate = _oc_update(design, gradient, volume_gradient, mapping, volume_target, settings, move_limit)
            change = float(np.max(np.abs(candidate - design)))
            history.append(_history_entry(iteration, physical, solutions, scales, change, perf_counter() - started))
            history[-1].update(modal_penalty=modal_penalty, f1_hz=modal_info.get("f1_hz"))
            if staged:
                recent = [entry["objective"] for entry in history[-min(level_iterations + 1, settings["objective_window"]):]]
                stall = (max(recent) - min(recent)) / min(recent) if len(recent) == settings["objective_window"] else None
                history[-1].update(_stage_entry(mapping, fields, volume_target, move_limit), objective_stall=stall)
            if progress_callback is not None:
                progress_callback(deepcopy(history[-1]))
            design = candidate
            level_iterations += 1
            if level == len(schedule) - 1:
                if level_iterations >= minimum_iterations and change < settings["change_tolerance"]:
                    converged = True
                    stop_reason = "change_tolerance"
                    break
            elif level_iterations >= settings["beta_interval"] or (level_iterations >= settings["beta_minimum_iterations"] and change < settings["beta_change_tolerance"]):
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
            "modal": None if modal is None else {"settings": settings["modal"], "final": modal(fields[stiffness_name][0], settings["penalization"], settings["min_stiffness_ratio"], settings["modal"]["initial_iterations"])[2], "history": modal_log},
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
