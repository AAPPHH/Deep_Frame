import ctypes
import os
from importlib.metadata import distribution, version
from pathlib import Path
from time import perf_counter

import numpy as np


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
