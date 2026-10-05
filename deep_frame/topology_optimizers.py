from time import perf_counter
import numpy as np

from deep_frame.topology_neural import Adam, SharedNeuralField
from deep_frame.topology_optimization import continuation_decision

MMA = {"package": "mmapy==0.3.1 (Deetman, Python port of Svanberg's MMA)", "move": 0.1, "move_late": 0.05, "move_late_beta": 32.0, "scale": 100.0, "asyinit": 0.5, "asydecr": 0.7, "asyincr": 1.2, "raa0": 1e-5, "c": 1e4, "d": 1.0,
       "start_level": 0, "level_window": 5, "level_mass_change": 1e-2, "level_design_change": 1e-2, "level_violation": 1e-2, "level_min_iterations": 10, "level_max_iterations": 80, "final_min_iterations": 10, "final_max_iterations": 300,
       "stall_window": 40, "stall_improvement": 1e-3, "diverge_window": 10, "diverge_grace": 10, "diverge_violation": 1.0, "checkpoint_interval": 10, "objective": None,
       "rules": {"continuation": "an intermediate beta level advances when, after level_min_iterations, its last level_window iterations are all feasible (max g <= level_violation) with max design change < level_design_change (reason change_tolerance); "
                                 "at level_max_iterations it advances anyway, recorded as reason iteration_cap with quiet false and feasible = last iterate feasible, restarting from the lowest-f0 feasible iterate of that level when one exists (continued_from_best); "
                                 "a capped level never makes the result converged, only the final level decides",
                 "convergence": "final beta level: Termination window of iterations with relative mass_g spread < mass_change, design change < design_change and max g <= violation, after final_min_iterations",
                 "best_feasible": "only final-level iterates that close a window of Termination-window iterations, all feasible with design change < design_change, are candidates; the lowest f0 candidate is returned with status not_converged_<reason>_best_feasible, otherwise the last iterate with not_converged_<reason>",
                 "divergence": "after diverge_grace iterations of a level: a feasible iterate earlier in the level and a violation above diverge_violation within the last diverge_window iterations",
                 "stall": "relative mass_g spread < level_mass_change and violation improvement < stall_improvement over stall_window iterations while infeasible"}}

def stationary(rows, window, change, violation):
    tail = rows[-window:]
    return len(tail) == window and all(row["change"] is not None and np.isfinite(row["change"]) and row["change"] < change and row["max_violation"] <= violation for row in tail)

class Termination:
    def __init__(self, settings):
        self.settings, self.masses, self.violations = settings, [], []
    def __call__(self, mass, violation, final_level=True, change=None, minimum_iterations=0):
        if not final_level:
            self.masses.clear()
            self.violations.clear()
            return False
        self.masses.append(float(mass))
        self.violations.append(float(violation))
        window = self.masses[-self.settings["window"]:]
        if len(window) < self.settings["window"]:
            return False
        stable = (max(window) - min(window)) / max(min(window), 1e-30) < self.settings["mass_change"]
        return stable and continuation_decision(len(self.masses), True, change, violation, minimum_iterations, float("inf"), self.settings.get("design_change", 1e-3), self.settings["violation"]) == "converged"

class MMAOptimizer:
    def __init__(self, problem, settings=MMA):
        from mmapy import mmasub
        self.problem, self.settings, self.subproblem = problem, {**MMA, **settings}, mmasub
        self.free = problem.map.free
    def start(self, design):
        x = np.clip(np.asarray(design, dtype=float).ravel(), 0, 1)
        x[self.problem.map.preserve] = 1
        x[~self.problem.map.allowed] = 0
        return x
    def terms(self, result):
        name = self.settings.get("objective")
        if not name:
            return result["objective"], result["objective_gradient"], result["constraints"], result["constraint_gradients"]
        index = result["names"].index(name)
        keep = np.arange(len(result["names"])) != index
        return result["constraints"][index] + 1, result["constraint_gradients"][index], result["constraints"][keep], result["constraint_gradients"][keep]
    def step(self, x, result, state):
        settings, free = self.settings, self.free
        f0, df0, g, dg = self.terms(result)
        n, m = int(np.count_nonzero(free)), len(g)
        value = x[free][:, None]
        zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
        moved = self.subproblem(m, n, state["iteration"], value, zeros, ones, state["old1"], state["old2"], settings["scale"] * f0, settings["scale"] * df0[free][:, None],
                                settings["scale"] * g[:, None], settings["scale"] * dg[:, free], state["low"], state["upp"], 1.0, np.zeros((m, 1)), np.full((m, 1), settings["c"]), np.full((m, 1), settings["d"]),
                                move=self.move(), asyinit=settings["asyinit"], asydecr=settings["asydecr"], asyincr=settings["asyincr"], raa0=settings["raa0"])
        state.update(old2=state["old1"], old1=value.copy(), low=moved[9], upp=moved[10], iteration=state["iteration"] + 1)
        x = x.copy()
        x[free] = np.clip(moved[0].ravel(), 0, 1)
        return x
    def move(self):
        return self.settings["move_late"] if self.problem.beta >= self.settings["move_late_beta"] else self.settings["move"]
    def diverged(self, level, tolerance):
        window, grace = self.settings["diverge_window"], self.settings["diverge_grace"]
        return len(level) >= max(grace + window, window + 1) and any(row["max_violation"] <= tolerance for row in level[:-window]) and max(row["max_violation"] for row in level[-window:]) > self.settings["diverge_violation"]
    def fresh(self, x):
        value = x[self.free][:, None]
        return {"iteration": 1, "old1": value.copy(), "old2": value.copy(), "low": np.zeros_like(value), "upp": np.ones_like(value)}
    def stalled(self, history):
        window = self.settings["stall_window"]
        if len(history) < window:
            return False
        masses, violations = [row["mass_g"] for row in history[-window:]], [row["max_violation"] for row in history[-window:]]
        return (max(masses) - min(masses)) / max(min(masses), 1e-30) < self.settings["level_mass_change"] and violations[0] - min(violations) < self.settings["stall_improvement"]
    def advance_ready(self, level):
        settings = self.settings
        return len(level) >= max(settings["level_min_iterations"], settings["level_window"]) and stationary(level, settings["level_window"], settings["level_design_change"], settings["level_violation"])
    @staticmethod
    def candidate(row, x, result):
        best = {key: row[key] for key in ("iteration", "level", "beta", "f0", "mass_g", "max_violation")}
        best.update(x=x.copy(), result={key: value for key, value in result.items() if "gradient" not in key})
        return best
    @staticmethod
    def summary(best):
        return {key: value for key, value in best.items() if key not in ("x", "result")} if best else None
    def run(self, design, progress=None, checkpoint=None, keep=None):
        problem, settings = self.problem, self.settings
        while problem.level < settings["start_level"] and problem.advance():
            pass
        criteria = problem.problem["termination"]
        x, termination, history, levels, level, best, level_best, kept = self.start(design), Termination(criteria), [], [], [], None, None, []
        state, initial = self.fresh(x), problem.problem["modal"]["initial_iterations"] if problem.modal is not None else None
        tolerance, design_change = criteria["violation"], criteria.get("design_change", 1e-3)
        started = perf_counter()
        previous = None
        while True:
            clock = perf_counter()
            result = problem.evaluate(x, initial if not level else None)
            f0, _, g, _ = self.terms(result)
            row = {"iteration": len(history), "level": problem.level, "beta": problem.beta, "objective": result["objective"], "f0": float(f0), "mass_g": result["mass_g"], "max_violation": float(np.max(g)),
                   "move": self.move(), "change": None if previous is None else float(np.max(np.abs(x[self.free]-previous))), "constraints": dict(zip(result["names"], result["constraints"].tolist()))}
            history.append(row)
            level.append(row)
            row["objective_stalled"] = self.stalled(level)
            feasible = row["max_violation"] <= tolerance
            if feasible and (level_best is None or row["f0"] < level_best["f0"]):
                level_best = self.candidate(row, x, result)
            row["stationary"] = problem.final_level and stationary(level, criteria["window"], design_change, tolerance)
            if row["stationary"] and (best is None or row["f0"] < best["f0"]):
                best = self.candidate(row, x, result)
                if keep:
                    keep(x, best)
            converged = termination(row["mass_g"], row["max_violation"], problem.final_level, row["change"], settings["final_min_iterations"])
            reason = "converged" if converged else None
            if problem.final_level and not converged:
                reason = "iteration_cap" if len(level) >= settings["final_max_iterations"] else "stalled" if len(level) >= settings["final_min_iterations"] and not feasible and self.stalled(level) else "diverged" if self.diverged(level, tolerance) else None
            elif not problem.final_level:
                quiet = self.advance_ready(level)
                if self.diverged(level, tolerance):
                    reason = "diverged"
                elif quiet or len(level) >= settings["level_max_iterations"]:
                    restart = not quiet and level_best is not None
                    levels.append({"beta": problem.beta, "iterations": len(level), "reason": "change_tolerance" if quiet else "iteration_cap", "quiet": quiet, "feasible": feasible, "mass_g": row["mass_g"], "max_violation": row["max_violation"],
                                   "best_feasible": self.summary(level_best), "continued_from_best": restart})
                    x = level_best["x"].copy() if restart else x
                    problem.advance()
                    level, state, level_best = [], self.fresh(x), None
                    previous = None
            if reason:
                fallback = reason != "converged" and best is not None
                levels.append({"beta": problem.beta, "iterations": len(level), "reason": reason, "feasible": feasible, "mass_g": row["mass_g"], "max_violation": row["max_violation"],
                               "best_feasible": self.summary(best), "returned_best": fallback})
                row["seconds"] = perf_counter() - clock
                if fallback:
                    x, result, kept = best["x"], {**result, **best["result"]}, self.summary(best)
                break
            if level:
                previous = x[self.free].copy()
                x = self.step(x, result, state)
            row["seconds"] = perf_counter() - clock
            if progress:
                progress(row)
            if checkpoint and len(history) % settings["checkpoint_interval"] == 0:
                checkpoint(x, history, levels)
        if progress:
            progress(row)
        runtime = perf_counter() - started
        status = "converged" if reason == "converged" else "not_converged_" + reason + ("_best_feasible" if kept else "")
        return {"design": x, "status": status, "iterations": len(history), "runtime_s": runtime, "seconds_per_iteration": runtime / len(history), "history": history, "levels": levels, "result": result,
                "best_feasible": kept or None, "capped_levels": [entry["beta"] for entry in levels if entry["reason"] == "iteration_cap" and "quiet" in entry], "settings": settings}

class NeuralALOptimizer(MMAOptimizer):
    def __init__(self, problem, settings=None, neural=None):
        super().__init__(problem, settings or {})
        self.neural = {"penalty": 10.0, "penalty_growth": 2.0, "penalty_max": 1e4, "multiplier_interval": 10, "learning_rate": 0.002, "initial_fit_iterations": 100, **(neural or {})}
        if self.neural["penalty"] <= 0 or self.neural["learning_rate"] <= 0 or self.neural["multiplier_interval"] < 1:
            raise ValueError("Neural AL needs positive penalty, learning rate and multiplier interval")
        self.mapping = SharedNeuralField(problem.domain, self.free, self.neural)
        self.fit_rmse = []
    def start(self, design):
        x = super().start(design)
        self.fit_rmse.append(self.mapping.fit(x[self.free]))
        x[self.free] = self.mapping.values()[0]
        return x
    def fresh(self, x):
        mismatch = float(np.max(np.abs(self.mapping.values()[0] - x[self.free])))
        if mismatch > 1e-8:
            self.fit_rmse.append(self.mapping.fit(x[self.free]))
            x[self.free] = self.mapping.values()[0]
        return {"iteration": 1, "multipliers": None, "penalty": self.neural["penalty"], "previous_violation": None, "optimizer": Adam(self.mapping.field.parameters, self.neural["learning_rate"])}
    def move(self):
        return None
    def step(self, x, result, state):
        _, df0, g, dg = self.terms(result)
        if state["multipliers"] is None:
            state["multipliers"] = np.zeros_like(g)
        penalty = state["penalty"]
        coefficients = np.maximum(0.0, state["multipliers"] + penalty * g)
        gradient = df0[self.free] + coefficients @ dg[:, self.free]
        _, cache = self.mapping.values()
        state["optimizer"].step(self.mapping.field.parameters, self.mapping.gradient(cache, gradient))
        if state["iteration"] % self.neural["multiplier_interval"] == 0:
            violation = max(float(np.max(g)), 0.0)
            state["multipliers"] = coefficients
            if state["previous_violation"] is not None and violation > 0.75 * state["previous_violation"]:
                state["penalty"] = min(penalty * self.neural["penalty_growth"], self.neural["penalty_max"])
            state["previous_violation"] = violation
        state["iteration"] += 1
        moved = x.copy()
        moved[self.free] = self.mapping.values()[0]
        return moved
