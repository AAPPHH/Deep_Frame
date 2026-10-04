from time import perf_counter
import numpy as np
from scipy.linalg import qr
from scipy.sparse import coo_matrix

from deep_frame.topology_optimization import _CORNERS, CudaDirectSolver, regular_grid

MULTIGRID = {"coarse": "galerkin", "coarsest_dofs": 80000, "max_levels": 8, "smoother": "chebyshev", "sweeps": 2, "damping": 1.0, "chebyshev_degree": 3, "chebyshev_ratio": 20.0, "growth": 2.0, "cycle": "V",
             "power_iterations": 20, "precision": "float32", "tolerance": 1e-8, "max_iterations": 2000, "batch": 12, "projection": True, "coarsest_projection": True, "dead_ratio": 1e-12, "device": "cuda"}
PRECISIONS = {"float64": ("float64", None), "float32": ("float32", None), "bfloat16": ("float32", "bfloat16"), "float16": ("float32", "float16")}
_CHILDREN = np.array(list(np.ndindex(2, 2, 2)))

def _torch():
    import torch
    import torch.nn.functional as functional
    return torch, functional

def _interpolation():
    local = np.zeros((8, 24, 24))
    for child, offset in enumerate(_CHILDREN):
        for row, corner in enumerate(_CORNERS):
            position = (offset + corner) / 2
            weights = np.prod(np.where(_CORNERS == 1, position, 1 - position), axis=1)
            for column, weight in enumerate(weights):
                local[child, 3 * row:3 * row + 3, 3 * column:3 * column + 3] = weight * np.eye(3)
    return local

def _rigid(shape, spacing, center):
    coordinates = np.stack(np.meshgrid(*[np.arange(n + 1) * h for n, h in zip(shape, spacing)], indexing="ij")) - np.asarray(center)[:, None, None, None]
    modes = np.zeros((6, 3) + coordinates.shape[1:])
    for axis in range(3):
        modes[axis, axis] = 1
    x, y, z = coordinates
    modes[3, 1], modes[3, 2] = -z, y
    modes[4, 0], modes[4, 2] = z, -x
    modes[5, 0], modes[5, 1] = -y, x
    return modes

class Level:
    def __init__(self, shape, spacing):
        self.shape, self.spacing = tuple(int(n) for n in shape), np.asarray(spacing, dtype=float)
        self.nodes = tuple(n + 1 for n in self.shape)
        self.moduli = self.blocks = self.kernel = None

class GeometricMultigrid:
    def __init__(self, shape, spacing, ke, settings=None):
        self.torch, self.F = _torch()
        torch = self.torch
        self.settings = {**MULTIGRID, **(settings or {})}
        if self.settings["precision"] not in PRECISIONS or self.settings["coarse"] not in ("galerkin", "rediscretize") or self.settings["smoother"] not in ("jacobi", "chebyshev") or self.settings["cycle"] not in ("V", "W"):
            raise ValueError("Invalid multigrid settings")
        self.device = torch.device(self.settings["device"])
        work, autocast = PRECISIONS[self.settings["precision"]]
        self.work, self.autocast = getattr(torch, work), autocast and getattr(torch, autocast)
        self.shape, self.spacing = tuple(int(n) for n in shape), np.asarray(spacing, dtype=float)
        self.ke64 = np.asarray(ke, dtype=float)
        count = 2
        while count < self.settings["max_levels"] and np.all(np.ceil(np.asarray(self.shape) / 2 ** count) >= 1) and np.prod(np.ceil(np.asarray(self.shape) / 2 ** (count - 1)) + 1) * 3 > self.settings["coarsest_dofs"]:
            count += 1
        factor = 2 ** (count - 1)
        self.padded = tuple(int(np.ceil(n / factor) * factor) for n in self.shape)
        self.levels = [Level(np.asarray(self.padded) // 2 ** level, self.spacing * 2 ** level) for level in range(count)]
        self.center = np.asarray(self.shape) * self.spacing / 2
        select = np.zeros((24, 3, 2, 2, 2))
        for corner, offset in enumerate(_CORNERS):
            for component in range(3):
                select[3 * corner + component, component, offset[0], offset[1], offset[2]] = 1
        stencil = np.einsum("i,j,k->ijk", *[np.array([0.5, 1.0, 0.5])] * 3)
        tensor = lambda array, dtype: torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=self.device)
        self.tensor = tensor
        self.select = {dtype: tensor(select, dtype) for dtype in {torch.float64, self.work}}
        self.weight = {dtype: tensor(np.einsum("ij,jdabc->idabc", self.ke64, select), dtype) for dtype in {torch.float64, self.work}}
        self.diagonal_ke = tensor(np.diag(self.ke64), torch.float64)
        self.transfer = tensor(np.broadcast_to(stencil, (3, 1, 3, 3, 3)), self.work)
        self.local = tensor(_interpolation(), torch.float64)
        self.moduli = None
        self.hierarchies, self.coarsest_solvers, self.factored, self.statistics = {}, {}, set(), []
    def grid(self, flat, dtype=None):
        torch = self.torch
        flat = torch.as_tensor(flat, device=self.device, dtype=dtype or torch.float64)
        flat = flat[:, None] if flat.ndim == 1 else flat
        count = flat.shape[1]
        nodes = tuple(n + 1 for n in self.shape)
        out = torch.zeros((count, 3) + self.levels[0].nodes, dtype=flat.dtype, device=self.device)
        out[:, :, :nodes[0], :nodes[1], :nodes[2]] = flat.T.reshape((count,) + nodes + (3,)).permute(0, 4, 1, 2, 3)
        return out
    def flat(self, grid):
        nodes = tuple(n + 1 for n in self.shape)
        return grid[:, :, :nodes[0], :nodes[1], :nodes[2]].permute(0, 2, 3, 4, 1).reshape(grid.shape[0], -1).T
    def element_field(self, values, dtype=None):
        torch = self.torch
        field = torch.zeros(self.padded, dtype=dtype or torch.float64, device=self.device)
        field[:self.shape[0], :self.shape[1], :self.shape[2]] = torch.as_tensor(np.asarray(values, dtype=float).reshape(self.shape), dtype=field.dtype, device=self.device)
        return field
    def update(self, moduli):
        self.moduli = self.element_field(moduli)
        self.hierarchies.clear()
        self.factored.clear()
    def _scatter(self, values, dtype):
        return self.F.conv_transpose3d(values, self.select[dtype])
    def _gather(self, grid, dtype):
        return self.F.conv3d(grid, self.select[dtype])
    def operator(self, grid, moduli=None, mask=None):
        dtype = grid.dtype
        moduli = self.moduli if moduli is None else moduli
        result = self._scatter(self.F.conv3d(grid if mask is None else grid * mask, self.weight[dtype]) * moduli.to(dtype), dtype)
        return result if mask is None else result * mask
    def _apply(self, level, grid):
        if level.blocks is None:
            values = self.F.conv3d(grid, self.weight[self.work]) * level.moduli
        else:
            gathered = self._gather(grid, self.work)
            count = gathered.shape[0]
            values = self.torch.bmm(level.blocks, gathered.reshape(count, 24, -1).permute(2, 1, 0)).permute(2, 1, 0).reshape(gathered.shape)
        return self._scatter(values, self.work) * level.mask
    def _diagonal(self, level, moduli=None, blocks=None):
        torch = self.torch
        if blocks is None:
            values = moduli[None, None] * self.diagonal_ke[None, :, None, None, None]
        else:
            values = torch.diagonal(blocks, dim1=-2, dim2=-1).permute(3, 0, 1, 2)[None]
        return self._scatter(values, torch.float64)[0]
    def _children(self, previous, mask, slab):
        torch = self.torch
        def split(field):
            shape = field.shape
            coarse = (shape[1] // 2, shape[2] // 2)
            field = field.reshape((2, coarse[0], 2, coarse[1], 2) + shape[3:]).permute((1, 3, 0, 2, 4) + tuple(range(5, 2 + len(shape))))
            return field.reshape((coarse[0] * coarse[1], 8) + shape[3:])
        if previous.blocks is None:
            moduli = split(previous.moduli64[2 * slab:2 * slab + 2])
            gathered = self._gather(mask[None, :, 2 * slab:2 * slab + 3], torch.float64)[0].permute(1, 2, 3, 0)
            gathered = split(gathered)
            return moduli[..., None, None] * gathered[..., :, None] * gathered[..., None, :] * torch.as_tensor(self.ke64, device=self.device)
        return split(previous.blocks64[2 * slab:2 * slab + 2])
    def mask(self, fixed):
        torch = self.torch
        active = (self._scatter((self.moduli > 0).to(torch.float64)[None, None].expand(1, 24, *self.padded).contiguous(), torch.float64)[0] > 0).to(torch.float64)
        constrained = torch.zeros_like(active)
        if len(fixed):
            constrained = self.grid(np.isin(np.arange(3 * np.prod(np.asarray(self.shape) + 1)), fixed).astype(float))[0] * active
        return active * (1 - constrained), constrained
    def hierarchy(self, key, fixed):
        if key in self.hierarchies:
            return self.hierarchies[key]
        torch, F = self.torch, self.F
        started = perf_counter()
        levels = [Level(level.shape, level.spacing) for level in self.levels]
        fine = levels[0]
        fixed = np.asarray(fixed, dtype=np.int64)
        fine.mask64, constrained = self.mask(fixed)
        fine.moduli64 = self.moduli
        modes = _rigid(fine.shape, fine.spacing, self.center)
        fixed_modes = modes[:, constrained.cpu().numpy() > 0]
        gram = fixed_modes @ fixed_modes.T
        values, vectors = np.linalg.eigh(gram)
        coefficients = vectors[:, values <= 1e-10 * max(values.max(), 1.0)] if fixed_modes.size else np.eye(6)
        rediscretize = self.settings["coarse"] == "rediscretize"
        for index, level in enumerate(levels):
            if index == 0:
                level.diagonal = self._diagonal(level, moduli=self.moduli)
            else:
                previous = levels[index - 1]
                if rediscretize:
                    level.moduli64 = F.avg_pool3d(previous.moduli64[None, None], 2)[0, 0]
                    stiffness = torch.as_tensor(self._coarse_ke(level.spacing), device=self.device)
                    injected = previous.mask64[:, ::2, ::2, ::2]
                    gathered = self._gather(injected[None], torch.float64)[0].permute(1, 2, 3, 0)
                    level.blocks64 = level.moduli64[..., None, None] * gathered[..., :, None] * gathered[..., None, :] * stiffness
                    level.diagonal = self._diagonal(level, blocks=level.blocks64)
                    level.mask64 = injected * (level.diagonal > self.settings["dead_ratio"] * level.diagonal.max())
                else:
                    blocks = torch.empty(level.shape + (24, 24), dtype=torch.float64, device=self.device)
                    for slab in range(level.shape[0]):
                        children = self._children(previous, previous.mask64, slab)
                        blocks[slab] = torch.einsum("cia,ecij,cjb->eab", self.local, children, self.local).reshape(level.shape[1], level.shape[2], 24, 24)
                    level.blocks64 = (blocks + blocks.transpose(-1, -2)) / 2
                    level.diagonal = self._diagonal(level, blocks=level.blocks64)
                    level.mask64 = (level.diagonal > self.settings["dead_ratio"] * level.diagonal.max()).to(torch.float64)
            if index == 0:
                level.diagonal = level.diagonal * fine.mask64
            mask = level.mask64
            level.mask = mask.to(self.work)
            level.inverse = torch.where(mask > 0, 1 / torch.where(level.diagonal > 0, level.diagonal, 1), 0).to(self.work)
            if index == 0:
                level.moduli = self.moduli.to(self.work)
            else:
                level.blocks = level.blocks64.reshape(-1, 24, 24).to(self.work)
            rigid = np.einsum("mk,m...->k...", coefficients, _rigid(level.shape, level.spacing, self.center)) * mask.cpu().numpy()[None]
            level.kernel = None
            if coefficients.shape[1] and self.settings["projection"]:
                basis, _ = np.linalg.qr(rigid.reshape(coefficients.shape[1], -1).T)
                level.kernel = torch.as_tensor(basis.T.reshape(rigid.shape), device=self.device)
        for index, level in enumerate(levels[:-1]):
            level.lambda_max = self._spectrum(level)
        hierarchy = {"key": key, "levels": levels, "coefficients": coefficients, "fixed": fixed}
        self._coarsest(hierarchy)
        for index, level in enumerate(levels):
            if index:
                level.moduli64 = None
                if index < len(levels) - 1:
                    level.blocks64 = None
        hierarchy["setup_s"] = perf_counter() - started
        hierarchy["kernel_check"] = self._kernel_check(hierarchy)
        self.hierarchies[key] = hierarchy
        return hierarchy
    def _coarse_ke(self, spacing):
        scale = spacing / self.spacing
        if not np.allclose(scale, scale[0]):
            raise ValueError("Rediscretization needs uniform coarsening")
        return self.ke64 * scale[0]
    def _spectrum(self, level):
        torch = self.torch
        generator = torch.Generator(device=self.device).manual_seed(0)
        vector = (torch.rand((1, 3) + level.nodes, generator=generator, device=self.device, dtype=self.work) - 0.5) * level.mask
        estimate = 1.0
        for _ in range(self.settings["power_iterations"]):
            vector = level.inverse * self._apply(level, vector)
            estimate = float(torch.linalg.vector_norm(vector))
            vector = vector / estimate
        return estimate
    def _remove(self, kernel, grid):
        if kernel is None:
            return grid
        basis = kernel.to(grid.dtype).reshape(kernel.shape[0], -1)
        values = grid.reshape(grid.shape[0], -1)
        return (values - (values @ basis.T) @ basis).reshape(grid.shape)
    def _coarsest(self, hierarchy):
        level = hierarchy["levels"][-1]
        dofs = regular_grid({"shape": list(level.shape), "spacing_mm": [1.0] * 3, "origin_mm": [0.0] * 3})[2]
        blocks = level.blocks64.reshape(-1, 24, 24).cpu().numpy()
        size = 3 * int(np.prod(level.nodes))
        matrix = coo_matrix((blocks.ravel(), (np.repeat(dofs, 24, axis=1).ravel(), np.tile(dofs, (1, 24)).ravel())), shape=(size, size)).tocsr()
        live = np.flatnonzero(level.mask64.permute(1, 2, 3, 0).reshape(-1).cpu().numpy() > 0)
        pins = np.zeros(0, dtype=int)
        kernel = None
        if level.kernel is not None and self.settings["coarsest_projection"]:
            kernel = level.kernel.permute(2, 3, 4, 1, 0).reshape(-1, level.kernel.shape[0]).cpu().numpy()[live]
            pins = qr(kernel.T, pivoting=True)[2][:kernel.shape[1]]
        keep = np.setdiff1d(np.arange(len(live)), pins)
        reduced = matrix[live[keep]][:, live[keep]]
        reduced = ((reduced + reduced.T) / 2).tocsr()
        index = lambda array: self.torch.as_tensor(array, dtype=self.torch.int64, device=self.device)
        hierarchy["coarsest"] = {"matrix": reduced, "live": index(live), "keep": index(keep), "kernel": None if kernel is None else self.torch.as_tensor(kernel, device=self.device), "dofs": len(keep), "pins": len(pins)}
    def _coarsest_solver(self, hierarchy):
        key, matrix = hierarchy["key"], hierarchy["coarsest"]["matrix"]
        solver = self.coarsest_solvers.get(key)
        if key in self.factored:
            return solver
        zeros = np.zeros((matrix.shape[0], self.settings["batch"]))
        if solver is not None and not solver.same_structure(matrix):
            solver.close()
            solver = None
        if solver is None:
            solver = CudaDirectSolver(matrix, zeros)
        solver.solve(matrix, zeros)
        self.coarsest_solvers[key] = solver
        self.factored.add(key)
        return solver
    def _solve_coarsest(self, hierarchy, grid):
        torch = self.torch
        info, level = hierarchy["coarsest"], hierarchy["levels"][-1]
        count = grid.shape[0]
        values = grid.to(torch.float64).permute(0, 2, 3, 4, 1).reshape(count, -1)[:, info["live"]]
        if info["kernel"] is not None:
            values = values - (values @ info["kernel"]) @ info["kernel"].T
        solver = self._coarsest_solver(hierarchy)
        rhs = torch.zeros((self.settings["batch"], len(info["keep"])), dtype=torch.float64, device=self.device)
        rhs[:count] = values[:, info["keep"]]
        torch.cuda.current_stream().synchronize()
        solver.rhs_device[...] = solver.cp.from_dlpack(rhs).T
        solver._execute(1008)
        result = torch.zeros_like(values)
        result[:, info["keep"]] = torch.from_dlpack(solver.solution_device.T)[:count]
        if info["kernel"] is not None:
            result = result - (result @ info["kernel"]) @ info["kernel"].T
        full = torch.zeros((count, 3 * int(np.prod(level.nodes))), dtype=torch.float64, device=self.device)
        full[:, info["live"]] = result
        return full.reshape((count,) + level.nodes + (3,)).permute(0, 4, 1, 2, 3).to(grid.dtype).contiguous()
    def _smooth(self, level, rhs, solution, index):
        count = max(1, int(round((self.settings["sweeps"] if self.settings["smoother"] == "jacobi" else self.settings["chebyshev_degree"]) * self.settings["growth"] ** index)))
        if self.settings["smoother"] == "jacobi":
            step = self.settings["damping"] / level.lambda_max
            for sweep in range(count):
                solution = step * level.inverse * rhs if solution is None else solution + step * level.inverse * (rhs - self._apply(level, solution))
            return solution
        upper = 1.1 * level.lambda_max
        lower = upper / self.settings["chebyshev_ratio"]
        theta, delta = (upper + lower) / 2, (upper - lower) / 2
        sigma = theta / delta
        rho = 1 / sigma
        residual = rhs if solution is None else rhs - self._apply(level, solution)
        direction = level.inverse * residual / theta
        solution = direction if solution is None else solution + direction
        for _ in range(count - 1):
            residual = residual - self._apply(level, direction)
            updated = 1 / (2 * sigma - rho)
            direction = updated * rho * direction + 2 * updated / delta * level.inverse * residual
            rho = updated
            solution = solution + direction
        return solution
    def _cycle(self, hierarchy, index, rhs):
        levels = hierarchy["levels"]
        if index == len(levels) - 1:
            return self._solve_coarsest(hierarchy, rhs)
        level, coarse = levels[index], levels[index + 1]
        solution = self._smooth(level, rhs, None, index)
        for _ in range(2 if self.settings["cycle"] == "W" and index + 2 < len(levels) else 1):
            residual = (rhs - self._apply(level, solution)) * level.mask
            restricted = self.F.conv3d(residual.to(self.work), self.transfer, stride=2, padding=1, groups=3) * coarse.mask
            correction = self._cycle(hierarchy, index + 1, restricted)
            solution = solution + self.F.conv_transpose3d(correction.to(self.work), self.transfer, stride=2, padding=1, groups=3) * level.mask
        return self._smooth(level, rhs, solution, index)
    def precondition(self, hierarchy, residual):
        torch = self.torch
        scale = residual.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-300)
        rhs = (residual / scale).to(self.work)
        with torch.autocast(device_type=self.device.type, dtype=self.autocast or torch.bfloat16, enabled=self.autocast is not None):
            result = self._cycle(hierarchy, 0, rhs)
        return self._remove(hierarchy["levels"][0].kernel, result.to(torch.float64) * scale)
    def _kernel_check(self, hierarchy):
        torch = self.torch
        result = []
        for level in hierarchy["levels"]:
            if level.kernel is None:
                continue
            kernel = level.kernel.to(self.work)
            applied = self._apply(level, kernel)
            reference = self._apply(level, torch.rand_like(kernel) * level.mask)
            result.append(float(torch.linalg.vector_norm(applied) / torch.linalg.vector_norm(reference)))
        return result
    def solve(self, key, fixed, forces, support=None):
        torch = self.torch
        started = perf_counter()
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        floating = support is not None and self.settings["projection"]
        dirichlet = np.setdiff1d(fixed, support) if floating else np.asarray(fixed)
        hierarchy = self.hierarchy((key, floating), dirichlet)
        fine = hierarchy["levels"][0]
        forces = np.array(forces, dtype=float).reshape(len(forces), -1)
        if floating:
            rigid = np.einsum("mk,m...->k...", hierarchy["coefficients"], _rigid(fine.shape, fine.spacing, self.center))
            rigid = self.flat(torch.as_tensor(rigid, device=self.device) * fine.mask64).cpu().numpy()
            support = np.asarray(support)
            forces[support] += np.linalg.lstsq(rigid[support].T, -(rigid.T @ forces), rcond=None)[0]
        pieces, reports, consistency = [], [], []
        for start in range(0, forces.shape[1], self.settings["batch"]):
            rhs = self.grid(forces[:, start:start + self.settings["batch"]]) * fine.mask64
            if fine.kernel is not None:
                basis = fine.kernel.reshape(fine.kernel.shape[0], -1)
                consistency += (torch.linalg.vector_norm(rhs.reshape(rhs.shape[0], -1) @ basis.T, dim=1) / torch.linalg.vector_norm(rhs.reshape(rhs.shape[0], -1), dim=1).clamp_min(1e-300)).cpu().tolist()
                rhs = self._remove(fine.kernel, rhs)
            solution, part = self._pcg(hierarchy, rhs)
            pieces.append(self.flat(solution).cpu().numpy())
            reports.append(part)
            del rhs, solution
        flat = np.hstack(pieces)
        report = {"iterations": sum((part["iterations"] for part in reports), []), "relative_residual": sum((part["relative_residual"] for part in reports), []), "converged": all(part["converged"] for part in reports), "history": [part["history"] for part in reports], "batches": len(reports)}
        if floating:
            shift = np.linalg.lstsq(rigid[support], flat[support], rcond=None)[0]
            flat = flat - rigid @ shift
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        report.update(seconds=perf_counter() - started, setup_s=hierarchy.pop("setup_s", 0.0), floating=floating, levels=len(hierarchy["levels"]), coarsest_dofs=hierarchy["coarsest"]["dofs"],
                      kernel=int(hierarchy["coefficients"].shape[1]) if floating else 0, rhs_kernel_component=consistency or None, kernel_check=hierarchy["kernel_check"], lambda_max=[level.lambda_max for level in hierarchy["levels"][:-1]])
        self.statistics.append(report)
        return flat, report
    def _pcg(self, hierarchy, rhs):
        torch = self.torch
        fine = hierarchy["levels"][0]
        tolerance, limit = self.settings["tolerance"], self.settings["max_iterations"]
        apply = lambda grid: self.operator(grid, mask=fine.mask64)
        norm = torch.linalg.vector_norm(rhs.flatten(1), dim=1)
        solution = torch.zeros_like(rhs)
        count = rhs.shape[0]
        iterations, relative = np.zeros(count, dtype=int), np.ones(count)
        active = torch.arange(count, device=self.device)
        residual = rhs.clone()
        preconditioned = self.precondition(hierarchy, residual)
        direction = preconditioned.clone()
        product = (residual * preconditioned).flatten(1).sum(1)
        step, history = 0, []
        while active.numel() and step < limit:
            applied = apply(direction)
            alpha = product / (direction * applied).flatten(1).sum(1)
            solution[active] += alpha[:, None, None, None, None] * direction
            residual = residual - alpha[:, None, None, None, None] * applied
            step += 1
            current = torch.linalg.vector_norm(residual.flatten(1), dim=1) / norm[active]
            history.append(current.max().item())
            converged = current < tolerance
            if bool(converged.any()):
                true = rhs[active] - apply(solution[active])
                if fine.kernel is not None:
                    true = self._remove(fine.kernel, true)
                current = torch.linalg.vector_norm(true.flatten(1), dim=1) / norm[active]
                converged = current < tolerance
                residual = torch.where(converged[:, None, None, None, None], residual, true)
            done = active[converged].cpu().numpy()
            iterations[done], relative[done] = step, current[converged].cpu().numpy()
            keep = ~converged
            if not bool(keep.any()):
                active = active[keep]
                break
            active, residual, direction, preconditioned, product = active[keep], residual[keep], direction[keep], preconditioned[keep], product[keep]
            updated = self.precondition(hierarchy, residual)
            beta = (residual * (updated - preconditioned)).flatten(1).sum(1) / product
            product = (residual * updated).flatten(1).sum(1)
            direction = updated + beta[:, None, None, None, None] * direction
            preconditioned = updated
        if active.numel():
            final = rhs[active] - apply(solution[active])
            relative[active.cpu().numpy()] = (torch.linalg.vector_norm(final.flatten(1), dim=1) / norm[active]).cpu().numpy()
            iterations[active.cpu().numpy()] = step
        return solution, {"iterations": iterations.tolist(), "relative_residual": relative.tolist(), "converged": bool(np.all(relative < tolerance)), "history": history}
    def product(self, flat):
        batch = self.settings["batch"]
        return np.hstack([self.flat(self.operator(self.grid(flat[:, start:start + batch]))).cpu().numpy() for start in range(0, flat.shape[1], batch)])
    def element_energy(self, grid):
        return (self.F.conv3d(grid, self.weight[grid.dtype]) * self._gather(grid, grid.dtype)).sum(1)
    def diagnostics(self):
        return {"settings": self.settings, "levels": [{"shape": list(level.shape), "spacing_mm": level.spacing.tolist()} for level in self.levels], "padded_shape": list(self.padded), "solves": self.statistics[-64:]}
    def close(self):
        errors = []
        for solver in self.coarsest_solvers.values():
            errors.extend(solver.close())
        self.coarsest_solvers.clear()
        return errors

def floating(system, part):
    return system.support is not None and bool(np.isin(part["support"], system.support[1]).all())

def solve_elasticity(system, multigrid, physical_density, penalization=3.0, min_stiffness_ratio=1e-6):
    torch = multigrid.torch
    density = np.asarray(physical_density, dtype=float).ravel()
    moduli = system.young * (min_stiffness_ratio + (1 - min_stiffness_ratio) * density ** penalization)
    derivative = system.young * (1 - min_stiffness_ratio) * penalization * density ** (penalization - 1)
    moduli[~system.active_elements] = 0
    derivative[~system.active_elements] = 0
    multigrid.update(moduli)
    factor = 1.0 if system.symmetry is None else 2.0
    shape = multigrid.shape
    results = {}
    for members in system.groups.values():
        part = members[0][1]
        forces = np.column_stack([member["force"] for _, member in members])
        solutions, report = multigrid.solve(part["fixed"].tobytes(), part["fixed"], forces, part["support"] if floating(system, part) else None)
        grid = multigrid.grid(solutions)
        energies = multigrid.element_energy(grid)[:, :shape[0], :shape[1], :shape[2]].reshape(len(members), -1).cpu().numpy()
        for column, (case, member) in enumerate(members):
            entry = results.setdefault(case["name"], {"compliance_n_mm": 0.0, "energy": np.zeros(system.nelem), "relative_residual": 0.0, "iterations": [], "fields": {}})
            entry["compliance_n_mm"] += factor * float(member["force"] @ solutions[:, column])
            entry["energy"] += factor * energies[column]
            entry["relative_residual"] = max(entry["relative_residual"], report["relative_residual"][column])
            entry["iterations"].append(report["iterations"][column])
            entry["fields"][member["sign"]] = solutions[:, column]
        del grid
    for entry in results.values():
        entry["derivative"] = -derivative * entry.pop("energy")
    torch.cuda.empty_cache() if multigrid.device.type == "cuda" else None
    return results
