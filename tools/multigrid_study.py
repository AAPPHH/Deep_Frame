import json
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deep_frame.config import PRINT_MATERIAL, command_line, configure
from deep_frame.topology_multigrid import GeometricMultigrid, solve_elasticity
from deep_frame.topology_optimization import HexElasticity
from deep_frame.topology_problem import PROBLEM, cantilever_domain, orthotropic_material

INTERPOLATION = PROBLEM["interpolation"]
OPERATOR = {"output": "docs/validation/multigrid_operator.json", "grids": ["cantilever", "cantilever_fine", "frame_coarse", "frame"], "fields": ["uniform", "binary", "smooth"], "batch": 8, "repeats": 20}
OPERATOR_KINDS = {"output": "text", "grids": ["text"], "fields": ["text"], "batch": "int", "repeats": "int"}
GRIDS = {"cantilever": {"kind": "cantilever", "shape": [32, 6, 12], "spacing": 1.0}, "cantilever_fine": {"kind": "cantilever", "shape": [64, 12, 24], "spacing": 0.5},
         "frame_coarse": {"kind": "frame", "shape": [68, 64, 24]}, "frame": {"kind": "frame", "shape": [102, 96, 24]}}

def domain_of(name):
    spec = GRIDS[name]
    if spec["kind"] == "cantilever":
        return cantilever_domain(tuple(spec["shape"]), spec["spacing"])
    from tools.formulation_study import FORMULATION, frame_setup
    return frame_setup(FORMULATION, spec["shape"])[0]

def density_field(kind, shape, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0, 1, shape)
    if kind == "binary":
        return (rng.uniform(0, 1, shape) < 0.3).astype(float)
    from scipy.ndimage import gaussian_filter
    return np.clip(gaussian_filter(rng.normal(size=shape), 2.0) * 6 + 0.3, 0, 1)

def moduli_of(system, density):
    moduli = system.young * (INTERPOLATION["min_stiffness_ratio"] + (1 - INTERPOLATION["min_stiffness_ratio"]) * np.asarray(density, dtype=float).ravel() ** INTERPOLATION["penalization"])
    active = moduli.copy()
    active[~system.active_elements] = 0
    return moduli, active

def multigrid_of(system, settings=None):
    return GeometricMultigrid(system.domain["grid"]["shape"], system.spacing, system.ke, settings)

def timed(function, repeats):
    import torch
    function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    started = perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (perf_counter() - started) / repeats, (torch.cuda.max_memory_allocated() - base) / 2 ** 20

def operator_check(cfg):
    import cupy
    import cupyx.scipy.sparse as device_sparse
    import torch
    rows = []
    for grid in cfg["grids"]:
        domain = domain_of(grid)
        system = HexElasticity(domain, linear_solver="cuda_cudss")
        part = next(iter(system.groups.values()))[0][1]
        free = part["free"]
        for seed, field in enumerate(cfg["fields"]):
            moduli, active = moduli_of(system, density_field(field, domain["grid"]["shape"], seed))
            matrix = system.matrix(moduli).tocsr()
            mg = multigrid_of(system)
            mg.update(active)
            rng = np.random.default_rng(100 + seed)
            vectors = rng.normal(size=(system.ndof, 3))
            vectors[np.setdiff1d(np.arange(system.ndof), system.active_dofs)] = 0
            full = mg.flat(mg.operator(mg.grid(vectors))).cpu().numpy()
            reference = matrix @ vectors
            masked = vectors.copy()
            masked[np.setdiff1d(np.arange(system.ndof), free)] = 0
            mask, _ = mg.mask(part["fixed"])
            reduced = mg.flat(mg.operator(mg.grid(masked), mask=mask)).cpu().numpy()[free]
            expected = matrix[free][:, free] @ masked[free]
            error = lambda a, b: float(np.max(np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)))
            row = {"grid": grid, "shape": domain["grid"]["shape"], "spacing_mm": domain["grid"]["spacing_mm"][0], "field": field, "dofs": system.ndof, "free_dofs": len(free),
                   "relative_error_full": error(full, reference), "relative_error_free": error(reduced, expected), "case": part.get("sign")}
            if seed == 0:
                for precision in ("float64", "float32"):
                    variant = multigrid_of(system, {"precision": precision})
                    variant.update(active)
                    for count in (1, cfg["batch"]):
                        grid_vectors = variant.grid(vectors[:, :1].repeat(count, axis=1)).to(getattr(torch, precision))
                        seconds, memory = timed(lambda: variant.operator(grid_vectors), cfg["repeats"])
                        row[f"matvec_{precision}_rhs{count}_ms"], row[f"matvec_{precision}_rhs{count}_peak_mib"] = 1000 * seconds, memory
                device = device_sparse.csr_matrix(matrix)
                device_vectors = cupy.asarray(vectors[:, :1].repeat(cfg["batch"], axis=1))
                for count in (1, cfg["batch"]):
                    block = device_vectors[:, :count]
                    device @ block
                    cupy.cuda.Device().synchronize()
                    started = perf_counter()
                    for _ in range(cfg["repeats"]):
                        device @ block
                    cupy.cuda.Device().synchronize()
                    row[f"csr_spmv_float64_rhs{count}_ms"] = 1000 * (perf_counter() - started) / cfg["repeats"]
                row["csr_matrix_mib"] = (matrix.nnz * 12 + matrix.indptr.nbytes) / 2 ** 20
                row["operator_storage_mib"] = mg.moduli.numel() * 8 / 2 ** 20
                del device, device_vectors
                cupy.get_default_memory_pool().free_all_blocks()
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("grid", "field", "relative_error_full", "relative_error_free")}), flush=True)
        system.close()
    record = {"tolerance": 1e-12, "passed": all(row["relative_error_full"] < 1e-12 and row["relative_error_free"] < 1e-12 for row in rows), "rows": rows, "versions": versions()}
    Path(cfg["output"]).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({"passed": record["passed"]}))
    return 0 if record["passed"] else 1

CANTILEVER = {"output": "docs/validation/multigrid_cantilever.json", "scales": [1, 2, 3, 4], "fields": ["solid", "smooth"],
              "variants": {"float64": {"precision": "float64"}, "float32": {"precision": "float32"}, "bfloat16": {"precision": "bfloat16"}, "float16": {"precision": "float16"},
                           "jacobi_bfloat16": {"precision": "bfloat16", "smoother": "jacobi"}, "float32_deep": {"precision": "float32", "coarsest_dofs": 1500}}, "reference_variant": "bfloat16"}
CANTILEVER_KINDS = {"output": "text", "scales": ["int"], "fields": ["text"], "variants": "object", "reference_variant": "text"}
TRUSS = {"output": "docs/validation/multigrid_truss.json", "spacing_mm": 1.25, "shape": [128, 16, 32], "panel_cells": 16, "member_mm": 2.5,
         "frame_density": "C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt/fine/density_half.npz", "problems": ["sweep", "truss", "frame"],
         "frame_cases": ["stiffness_arm_tip", "thrust_all", "crash_front", "crash_side_left"],
         "variants": {"galerkin": {"precision": "float32"}, "galerkin_float64": {"precision": "float64"}, "galerkin_float16": {"precision": "float16"}, "galerkin_bfloat16": {"precision": "bfloat16"},
                      "rediscretize": {"precision": "float32", "coarse": "rediscretize"}, "galerkin_pinned": {"precision": "float32", "projection": False}, "jacobi": {"precision": "float32", "smoother": "jacobi", "growth": 1.0},
                      "galerkin_deep": {"precision": "float32", "coarsest_dofs": 1000}, "rediscretize_deep": {"precision": "float32", "coarse": "rediscretize", "coarsest_dofs": 1000},
                      "pinned_deep": {"precision": "float32", "projection": False, "coarsest_dofs": 1000}, "w_cycle_deep": {"precision": "float32", "cycle": "W", "coarsest_dofs": 1000},
                      "power100_deep": {"precision": "float32", "coarsest_dofs": 1000, "power_iterations": 100}, "galerkin_levels3": {"precision": "float32", "max_levels": 3, "coarsest_dofs": 1},
                      "rediscretize_levels3": {"precision": "float32", "coarse": "rediscretize", "max_levels": 3, "coarsest_dofs": 1},
                      "chebyshev6": {"precision": "float32", "chebyshev_degree": 6}, "unprojected_coarsest": {"precision": "float32", "coarsest_projection": False, "max_iterations": 500}},
         "frame_variants": ["galerkin", "galerkin_float64", "galerkin_float16", "rediscretize", "galerkin_pinned", "galerkin_levels3", "rediscretize_levels3", "galerkin_deep", "unprojected_coarsest"],
         "sweep": {"min_stiffness": [1e-6, 1e-4, 1e-2], "levels": [2, 3, 4, 5], "settings": {"precision": "float32", "coarsest_dofs": 1}}}
TRUSS_KINDS = {"output": "text", "spacing_mm": "float", "shape": ["int"] * 3, "panel_cells": "int", "member_mm": "float", "frame_density": "text", "problems": ["text"], "frame_cases": ["text"], "variants": "object",
               "frame_variants": ["text"], "sweep": "object"}

def device_used():
    import cupy
    free, total = cupy.cuda.runtime.memGetInfo()
    return (total - free) / 2 ** 20

def relative(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / max(np.linalg.norm(b), 1e-300))

def release(system):
    import cupy
    system.close()
    system.gpu_solvers.clear()
    cupy.get_default_memory_pool().free_all_blocks()

def reference_solve(system, density, e_min):
    import cupy
    import torch
    release(system)
    torch.cuda.empty_cache()
    base = device_used()
    started = perf_counter()
    result = system.solve(density, INTERPOLATION["penalization"], e_min)
    seconds = perf_counter() - started
    row = {"solver": "cudss", "seconds": seconds, "factor_solve_s": sum(t["factor_s"] + t["solve_s"] for solver in system.gpu_solvers.values() for t in solver.timings),
           "device_mib": device_used() - base, "dofs": len(system.active_dofs), "relative_residual": max(value["relative_residual"] for value in result.values())}
    release(system)
    return result, row

def compare(system, density, variants, label, fields=False, e_min=INTERPOLATION["min_stiffness_ratio"]):
    import torch
    for case in system.cases:
        case["keep_fields"] = fields
    reference, row = reference_solve(system, density, e_min)
    solid = np.zeros(system.ndof, dtype=bool)
    solid[system.dofs[np.asarray(density) >= 0.5].ravel()] = True
    rows = [{"problem": label, **row}]
    for name, settings in variants.items():
        mg = multigrid_of(system, settings)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base, allocated = device_used(), torch.cuda.memory_allocated()
        started = perf_counter()
        try:
            results = solve_elasticity(system, mg, density, INTERPOLATION["penalization"], e_min)
        except Exception as error:
            rows.append({"problem": label, "solver": name, "settings": settings, "error": f"{type(error).__name__}: {error}"[:500], "solves": mg.statistics})
            mg.close()
            torch.cuda.empty_cache()
            print(json.dumps({key: rows[-1][key] for key in ("problem", "solver", "error")}), flush=True)
            continue
        torch.cuda.synchronize()
        seconds = perf_counter() - started
        peak, used = (torch.cuda.max_memory_allocated() - allocated) / 2 ** 20, device_used() - base
        cases = {}
        for case, value in results.items():
            expected = reference[case]
            cases[case] = {"iterations": value["iterations"], "relative_residual": value["relative_residual"],
                           "compliance_rel_diff": abs(value["compliance_n_mm"] - expected["compliance_n_mm"]) / expected["compliance_n_mm"],
                           "sensitivity_rel_diff": relative(value["derivative"], expected["derivative"])}
            if fields:
                cases[case]["displacement_rel_diff"] = max(relative(value["fields"][sign], field) for sign, field in expected["fields"].items())
                cases[case]["displacement_solid_rel_diff"] = max(relative(value["fields"][sign][solid], field[solid]) for sign, field in expected["fields"].items())
        statistics = mg.statistics
        rows.append({"problem": label, "solver": name, "settings": settings, "seconds": seconds, "setup_s": sum(entry["setup_s"] for entry in statistics), "torch_peak_mib": peak, "device_mib": used,
                     "levels": statistics[0]["levels"], "coarsest_dofs": statistics[0]["coarsest_dofs"], "max_iterations": max(max(entry["iterations"]) for entry in statistics),
                     "converged": all(entry["converged"] for entry in statistics), "max_relative_residual": max(max(entry["relative_residual"]) for entry in statistics),
                     "kernel_check": max([value for entry in statistics for value in entry["kernel_check"]] or [0.0]), "lambda_max": statistics[0]["lambda_max"], "kernel_checks": next((entry["kernel_check"] for entry in statistics if entry["kernel_check"]), []),
                     "max_compliance_rel_diff": max(value["compliance_rel_diff"] for value in cases.values()), "max_sensitivity_rel_diff": max(value["sensitivity_rel_diff"] for value in cases.values()),
                     "max_displacement_rel_diff": max(value["displacement_rel_diff"] for value in cases.values()) if fields else None,
                     "max_displacement_solid_rel_diff": max(value["displacement_solid_rel_diff"] for value in cases.values()) if fields else None, "cases": cases,
                     "solves": [{key: entry[key] for key in ("iterations", "relative_residual", "floating", "kernel", "setup_s", "seconds", "rhs_kernel_component")} for entry in statistics]})
        mg.close()
        del mg, results
        torch.cuda.empty_cache()
        print(json.dumps({key: rows[-1][key] for key in ("problem", "solver", "seconds", "max_iterations", "converged", "max_compliance_rel_diff", "max_sensitivity_rel_diff")}), flush=True)
    return rows

def cantilever_study(cfg):
    rows = []
    for scale in cfg["scales"]:
        domain = cantilever_domain((32 * scale, 6 * scale, 12 * scale), 1.0 / scale)
        for seed, field in enumerate(cfg["fields"]):
            system = HexElasticity(domain, linear_solver="cuda_cudss")
            density = np.ones(system.nelem) if field == "solid" else np.kron(density_field("smooth", (32, 6, 12), seed), np.ones((scale,) * 3)).ravel()
            result = compare(system, density, cfg["variants"], f"cantilever_s{scale}_{field}", fields=True)
            for row in result:
                row.update(scale=scale, field=field, shape=domain["grid"]["shape"], spacing_mm=domain["grid"]["spacing_mm"][0])
            rows += result
            Path(cfg["output"]).write_text(json.dumps({"rows": rows, "versions": versions()}, indent=1, default=float), encoding="utf-8")
    return 0

def segment_mask(centers, a, b, radius):
    direction = np.asarray(b, dtype=float) - a
    t = np.clip((centers - a) @ direction / (direction @ direction), 0, 1)
    return np.linalg.norm(centers - a - t[:, None] * direction, axis=1) <= radius + 1e-9

def truss_domain(cfg, floating):
    shape, h, panel = np.asarray(cfg["shape"]), cfg["spacing_mm"], cfg["panel_cells"]
    length, half, height = shape * h
    centers = (np.indices(tuple(shape)).reshape(3, -1).T + 0.5) * h
    radius = cfg["member_mm"] / 2
    plane, low, high = half - radius, radius, height - radius
    members = np.zeros(len(centers), dtype=bool)
    points = np.arange(0, shape[0] + 1, panel) * h
    points[0], points[-1] = radius, length - radius
    for chord in (low, high):
        members |= segment_mask(centers, [radius, plane, chord], [length - radius, plane, chord], radius)
    for index, x in enumerate(points):
        for chord in (low, high):
            members |= segment_mask(centers, [x, 0.0, chord], [x, plane, chord], radius)
        if index + 1 < len(points):
            a, b = (low, high) if index % 2 == 0 else (high, low)
            members |= segment_mask(centers, [x, plane, a], [points[index + 1], plane, b], radius)
    for x in (points[0], points[-1]):
        members |= segment_mask(centers, [x, plane, low], [x, plane, high], radius)
    members = members.reshape(tuple(shape))
    allowed = np.ones(tuple(shape), dtype=bool)
    box = lambda lo, hi: {"kind": "box", "min_mm": [float(v) for v in lo], "max_mm": [float(v) for v in hi]}
    tip = box([length - 2 * radius - 1e-6, plane - radius, -1e-6], [length + 1e-6, half + 1e-6, 2 * radius + 1e-6])
    case = {"name": "tip", "analysis": "static", "loads": [{"region": tip, "force_n": [0.0, 0.0, -10.0]}]}
    if floating:
        masses = [{"name": f"mass_{i}", "region": box([x - radius, plane - radius, high - radius], [x + radius, half + 1e-6, high + radius]), "mass_g": 10.0} for i, x in enumerate(points[:-1])]
        case.update(fixed_regions=[], inertia_relief={"point_masses": masses, "preserve_mass_g": 0.0})
    else:
        case["fixed_regions"] = [box([-1e-6, plane - radius - 1e-6, -1e-6], [1e-6, half + 1e-6, height + 1e-6])]
    domain = {"grid": {"origin_mm": [0.0, 0.0, 0.0], "spacing_mm": [h] * 3, "shape": shape.tolist(), "axis_order": "xyz", "order": "C"}, "allowed": allowed, "preserve": members, "forbidden": ~allowed,
              "symmetry": {"axis": 1, "plane_mm": 0.0}, "material": orthotropic_material({"density_g_cm3": PRINT_MATERIAL["density_g_cm3"], "poisson_ratio": PRINT_MATERIAL["poisson_ratio"]}),
              "load_cases": [case], "point_masses": [], "optimizer_settings": {"interface_node_policy": "preserve_adjacent"}}
    return domain, members.ravel().astype(float)

def truss_study(cfg):
    rows = []
    for problem in cfg["problems"]:
        if problem == "sweep":
            domain, density = truss_domain(cfg, False)
            system = HexElasticity(domain, interface_node_policy="preserve_adjacent", linear_solver="cuda_cudss")
            sweep = cfg["sweep"]
            for e_min in sweep["min_stiffness"]:
                variants = {f"levels{count}": {**sweep["settings"], "max_levels": count} for count in sweep["levels"]}
                variants.update({f"rediscretize_levels{count}": {**sweep["settings"], "max_levels": count, "coarse": "rediscretize"} for count in sweep["levels"][1:]})
                result = compare(system, density, variants, f"truss_sweep_emin{e_min:g}", e_min=e_min)
                for row in result:
                    row.update(min_stiffness=e_min, shape=domain["grid"]["shape"], spacing_mm=cfg["spacing_mm"])
                rows += result
            release(system)
        elif problem == "truss":
            for floating in (False, True):
                domain, density = truss_domain(cfg, floating)
                system = HexElasticity(domain, interface_node_policy="preserve_adjacent", linear_solver="cuda_cudss")
                result = compare(system, density, cfg["variants"], "truss_floating" if floating else "truss_clamped", fields=True)
                for row in result:
                    row.update(member_fraction=float(density.mean()), shape=domain["grid"]["shape"], spacing_mm=cfg["spacing_mm"])
                rows += result
        else:
            from tools.formulation_study import FORMULATION, frame_setup
            domain = frame_setup(FORMULATION, [102, 96, 24])[0]
            domain["load_cases"] = [case for case in domain["load_cases"] if case["name"] in cfg["frame_cases"]]
            system = HexElasticity(domain, interface_node_policy=domain.get("optimizer_settings", {}).get("interface_node_policy", "allowed_adjacent"), linear_solver="cuda_cudss")
            density = np.load(cfg["frame_density"])["density"].astype(float).ravel()
            result = compare(system, density, {name: cfg["variants"][name] for name in cfg["frame_variants"]}, "frame_4over3", fields=True)
            for row in result:
                row.update(shape=domain["grid"]["shape"], spacing_mm=domain["grid"]["spacing_mm"][0], density=cfg["frame_density"])
            rows += result
        Path(cfg["output"]).write_text(json.dumps({"rows": rows, "versions": versions()}, indent=1, default=float), encoding="utf-8")
    return 0

def versions():
    import cupy
    import torch
    from importlib.metadata import version
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cupy": cupy.__version__, "cudss": version("nvidia-cudss-cu12"),
            "device": torch.cuda.get_device_name(), "python": sys.version.split()[0]}

def main(argv=None):
    return command_line({"operator": lambda overrides: operator_check(configure(OPERATOR, OPERATOR_KINDS, overrides)),
                         "cantilever": lambda overrides: cantilever_study(configure(CANTILEVER, CANTILEVER_KINDS, overrides)),
                         "truss": lambda overrides: truss_study(configure(TRUSS, TRUSS_KINDS, overrides))}, argv)

if __name__ == "__main__":
    raise SystemExit(main())
