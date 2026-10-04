import json
import sys
from copy import deepcopy
from math import cos, radians, sin
from pathlib import Path

import numpy as np

from deep_frame.config import LAYOUT_DEFAULT, LAYOUT_OPTIMIZATION, LAYOUT_REFERENCES
from deep_frame.frame_evaluation import component_inertia, rigid_assembly
from deep_frame.frame_run import FrameLayout, LayoutModel, layout_setup

HERE = Path(__file__).resolve().parent
RUNS = HERE / "constraint_check_runs"
LUX = {"spec": "https://docs.hd-zero.com/camera-lux (abgerufen 2026-10-04): Max FOV D 155 / H 126 / V 94 deg (4:3), D 170 / H 145 / V 82 deg (16:9), 1/2 inch sensor",
       "modes": {"model_union": [145.0, 94.0], "lux_4x3": [126.0, 94.0], "lux_16x9": [145.0, 82.0]}}
VARIANTS = {
    "baseline": {},
    "cg_band_m5": {"cg_band_mm": [-5.0, 5.0]},
    "cg_open": {"cg_band_mm": [-15.0, 5.0]},
    "fov_4x3": {"fov_deg": [126.0, 94.0]},
    "fov_16x9": {"fov_deg": [145.0, 82.0]},
    "fov_4x3_edge5": {"fov_deg": [116.0, 84.0]},
    "fov_4x3_edge10": {"fov_deg": [106.0, 74.0]},
    "cg_m5_fov_4x3": {"cg_band_mm": [-5.0, 5.0], "fov_deg": [126.0, 94.0]},
    "cg_m5_fov_4x3_edge5": {"cg_band_mm": [-5.0, 5.0], "fov_deg": [116.0, 84.0]},
    "rank3": {"rank_tolerance": 0.03},
    "cg_m5_fov_4x3_rank3": {"cg_band_mm": [-5.0, 5.0], "fov_deg": [126.0, 94.0], "rank_tolerance": 0.03},
    "cg_m5_fov_4x3_edge5_rank3": {"cg_band_mm": [-5.0, 5.0], "fov_deg": [116.0, 84.0], "rank_tolerance": 0.03},
}
EVALUATOR = {
    "2inch": {"file": "C:/clones/Deep_Frame/Examples/Frames/2inch/_eval/evaluation/evaluation.json", "prop_thickness_mm": 5.0},
    "tbs_source_one_v5": {"file": "C:/clones/Deep_Frame/Examples/Frames/TBS_Source_One_V5/_eval/evaluation/evaluation.json", "prop_thickness_mm": 5.0},
    "aether4": {"file": "C:/clones/Deep_Frame-eval/exports/evaluation/aether4/evaluation.json", "prop_thickness_mm": 0.8},
    "manafly3": {"file": "C:/clones/Deep_Frame-eval/exports/evaluation/ref_manafly3/evaluation.json", "prop_thickness_mm": 0.8},
}
REAL = {
    "manafly3_142g": {"source": "ANNAHME: Abfluggewicht 142,7 g (Foto, datasheet Feld 9) mit 3-Zoll-Hardware: 4 x 8,5 g 14xx-Motoren (Hoehe 14,6 mm wie Modell), 4 x 2,5 g Dreiblatt-3-Zoll, Akku = Rest 59 g (4S 450-550), 31 x 60 x 22 mm, Rest wie Modell",
                      "frame": "manafly3", "set": {"motor.mass_g": 8.5, "prop.mass_g": 2.5, "battery.mass_g": 59.0, "battery.height_mm": 22.0, "battery.length_mm": 60.0, "battery.width_mm": 31.0}},
    "tbs_5in_6s": {"source": "ANNAHME typische 5-Zoll-Hardware: Rahmen Carbon 118 g (Herstellerangabe) im STL-Schwerpunkt z 9,5; 4 x 33 g 2207-Motoren, Sitz z 7,5, Hoehe 19 mm; 4 x 4,5 g 5-Zoll-Props, Nabe 7 mm; 6S 1100 185 g, 35 hoch, auf dem Deck z 34,85; Stack 20 g z 15,5; Kamera+VTX 20 g z 17",
                   "parts": [(118.0, 9.5), (4 * 33.0, 7.5 + 9.5), (4 * 4.5, 7.5 + 19.0 + 3.5), (185.0, 34.85 + 17.5), (20.0, 15.5), (20.0, 17.0)], "rotor_z": 7.5 + 19.0 + 3.5},
}

LIMITS = {"cg_above_rotor_plane": ("Schwerpunkt ueber Rotorebene", "mm", 0.0, ">="), "camera_sees_no_prop": ("Propscheiben ausserhalb Bildfeld 145/94 (Union), Abstand zum Bildrand", "deg", 0.0, ">="),
          "battery_over_stack": ("Spalt Deckunterseite - Stackoberkante inkl. ELRS-Draht", "mm", 3.0, ">="), "battery_over_camera": ("Spalt Deckunterseite - Kameraoberkante (Kamera unter dem Akku)", "mm", 3.0, ">="),
          "camera_under_hoop": ("Buegelscheitel - Kameraoberkante (geneigte Huellbox)", "mm", 0.0, ">="), "cg_y": ("|Schwerpunkt y - Motormitte|", "mm", 1.0, "<=")}
VERDICTS = {
    ("manafly3", "cg_above_rotor_plane"): "nicht robust: kippt mit realer Hardware (Abfluggewicht 142,7 g -> +0,3 mm; Akku >= 83 g bei unseren Motoren, >= 56 g bei 8,5-g-Motoren und 22 mm Akkuhoehe) und mit Motorsitz z 8 statt 10 (-2,3 mm); die Unterschreitung entsteht durch unseren leichten 2S-Akku auf einem 29-g-Rahmen",
    ("manafly3", "camera_sees_no_prop"): "robust: bei keiner Kameraposition im Rahmen erfuellbar (bestes 4:3 -15,1 Grad bei 20 Grad Neigung, -9,4 bei 40 Grad); 3-Zoll-True-X mit Kamera zwischen den Vorderprops; im Bild 3,9 % (4:3) bzw. 7,6 % (16:9) Propscheibe; der reale Frame fliegt so",
    ("tadpole", "cg_above_rotor_plane"): "unbestimmt: -0,14 mm liegt im Rauschen des Handmodells (Seitenteil-Box z 13 -> 0,0; Akku 1 mm hoeher -> +0,3)",
    ("tadpole", "battery_over_stack"): "nicht robust: -0,5 mm verschwindet ohne den 3-mm-ELRS-Zuschlag oder mit Akku 1 mm hoeher; Boxhoehe 15-18 mm aus Fotos (+-10 %)",
    ("tadpole", "battery_over_camera"): "Modellannahme: Kamera-y 28 und 0 mm Bodenabstand sind geschaetzt; die Pruefung verlangt 3 mm Luft unter dem Akku, die reale Kamera sitzt zwischen den Alu-Seitenteilen unter der Top-Platte",
    ("tadpole", "camera_sees_no_prop"): "robust fuer die angenommene Kameraposition (4:3 -18,1 Grad, 3,0 % Bild), aber nicht fuer den Frame: eine weiter vorn und hoeher sitzende Kamera erreicht +13 Grad (4:3, Erreichbarkeit)",
    ("tadpole", "camera_under_hoop"): "Modellartefakt: die 20 Grad geneigte 14-mm-Huellbox ist 17,9 mm hoch und passt fuer keine Lage unter den geschaetzten 18-mm-Buegel; mit Buegel +10 % (19,8 mm) -0,14 mm; reale Nano-Kamera mit Drehachse passt",
    ("ours_rule", "cg_y"): "robust (Akku mittig, Kamera vorn: Schwerpunkt 2 mm vor der Motormitte); echtes Trimmproblem",
    ("ours_rule", "camera_sees_no_prop"): "abhaengig vom Bildmodus: Union -17,9 Grad, 4:3 -8,4 Grad (1,8 % Bild), 16:9 -17,9 Grad (5,5 %)",
}

def table(refs, rows):
    entries = []
    for frame, label in (("ours", "rule"), ("manafly3", "own"), ("tadpole", "own")):
        data = refs[frame][label]
        for name, item in data["violated"].items():
            text, unit, limit, sense = LIMITS[name]
            margin = item["margin"]
            value = limit + margin if sense == ">=" else limit - margin
            if name == "cg_above_rotor_plane":
                value = data["cg_above_rotor_plane_mm"]
            key = "ours_rule" if frame == "ours" else frame
            entries.append({"frame": key, "constraint": name, "meaning": text, "value": value, "limit": f"{sense} {limit:g} {unit}", "margin": margin, "unit": unit,
                            "fov_4x3_margin_deg": data["fov"]["lux_4x3"]["margin_deg"] if name == "camera_sees_no_prop" else None,
                            "image_fraction_4x3": data["fov"]["lux_4x3"]["image_fraction_in_prop_discs"] if name == "camera_sees_no_prop" else None,
                            "robustness": VERDICTS[(key, name)], "sweep": [{"case": row["case"], "margin": row["constraints"].get(name, 0.0)} for row in rows.get(key, [])]})
    return entries

def settings_for(name):
    settings = deepcopy(LAYOUT_OPTIMIZATION)
    settings.update(deepcopy(VARIANTS[name]))
    return settings

def ours_setup(settings):
    return {**FrameLayout({}).setup(settings), "own": settings["frame_share"]["layout"]}

def setup_for(frame, settings):
    return ours_setup(settings) if frame == "ours" else layout_setup(frame, settings=settings)

def optimize(name):
    settings = settings_for(name)
    model = LayoutModel(ours_setup(settings), settings)
    optimum = model.optimize()
    result = {"variant": name, "override": VARIANTS[name], "optimum": optimum, "rounded": model.rounded(optimum["layout"]), "default": model.evaluate(LAYOUT_DEFAULT["layout"]),
              "rule": model.evaluate(settings["frame_share"]["layout"])}
    RUNS.mkdir(exist_ok=True)
    (RUNS / (name + ".json")).write_text(json.dumps(result, indent=1))
    print(json.dumps({"variant": name, "rounded": result["rounded"] and result["rounded"]["layout"], "alpha": result["rounded"] and result["rounded"]["alpha_rad_s2"]}))

def coverage(model, vector, fov, samples=161):
    lens = np.asarray(model.placement(vector)["lens_mm"])
    u, v = np.meshgrid(np.radians(np.linspace(-fov[0] / 2, fov[0] / 2, samples)), np.radians(np.linspace(-fov[1] / 2, fov[1] / 2, samples)))
    theta, phi = np.hypot(u, v), np.arctan2(v, u)
    local = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1).reshape(-1, 3)
    world = local @ model.axes
    motors = np.asarray(model.setup["motors_xy"], dtype=float)
    prop = model.setup["prop"]
    inner, outer = prop.get("hub_diameter_mm", 0.0) / 2, prop["diameter_mm"] / 2
    hit = np.zeros(len(world), dtype=bool)
    for z in (model.prop_bottom, model.prop_top):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - lens[2]) / world[:, 2]
        points = lens[:2] + t[:, None] * world[:, :2]
        distance = np.linalg.norm(points[:, None, :] - motors[None], axis=2)
        hit |= (t > 0) & ((distance >= inner) & (distance <= outer)).any(axis=1)
    return float(hit.mean())

def fov_report(model, layout):
    vector, report = model.vector(layout), {}
    for mode, fov in LUX["modes"].items():
        model.settings = {**model.settings, "fov_deg": fov}
        report[mode] = {"margin_deg": model.fov_margin(vector), "image_fraction_in_prop_discs": coverage(model, vector, fov)}
    return report

def tilt_sweep(frame, layout, tilts=(20.0, 30.0, 40.0)):
    result = {}
    for tilt in tilts:
        setup = setup_for(frame, LAYOUT_OPTIMIZATION)
        setup["camera"] = {**setup["camera"], "tilt_deg": tilt}
        model = LayoutModel(setup, LAYOUT_OPTIMIZATION)
        result[f"{tilt:g}"] = {"layout": fov_report(model, layout), "reachable_margin_deg": {mode: best_fov(setup, fov) for mode, fov in LUX["modes"].items()}}
    return result

def best_fov(setup, fov):
    from scipy.optimize import differential_evolution
    settings = {**LAYOUT_OPTIMIZATION, "fov_deg": fov}
    model = LayoutModel(setup, settings)
    return float(-differential_evolution(lambda vector: -model.fov_margin(vector), model.bounds, popsize=12, maxiter=120, seed=1, polish=True).fun)

def patched(frame, changes):
    setup = layout_setup(frame)
    for key, value in changes.items():
        if key.startswith("frame_part."):
            index, field = key.split(".")[1:]
            setup["frame"]["parts"][int(index)][field] = value
            continue
        if "." not in key:
            setup[key] = value
            continue
        group, field = key.split(".")
        setup[group] = {**setup[group], field: value}
    if "parts" in setup["frame"]:
        total, center, inertia = rigid_assembly([(part["mass_g"], part["center_mm"], component_inertia(part)) for part in setup["frame"]["parts"]])
        setup["frame"] = {**setup["frame"], "mass_g": float(total), "center_mm": center.tolist(), "inertia_g_mm2": inertia.tolist()}
    return setup

def robustness(frame, cases):
    rows = []
    for label, changes in cases:
        layout_override = {key.split(".", 1)[1]: value for key, value in changes.items() if key.startswith("layout.")}
        setup = patched(frame, {key: value for key, value in changes.items() if not key.startswith("layout.")})
        model = LayoutModel(setup, LAYOUT_OPTIMIZATION)
        result = model.evaluate({**setup["own"], **layout_override})
        rows.append({"case": label, "changes": changes, "mass_g": result["mass_g"], "cg_above_rotor_plane_mm": result["cg_above_rotor_plane_mm"], "violated": result["violated"],
                     "constraints": {name: result["constraints"][name] for name in result["violated"] or []} | {"cg_above_rotor_plane": result["constraints"]["cg_above_rotor_plane"]}})
    return rows

def crossover(frame, field, low, high, base=None):
    from scipy.optimize import brentq
    value = lambda x: LayoutModel(patched(frame, {**(base or {}), field: x}), LAYOUT_OPTIMIZATION).evaluate(layout_setup(frame)["own"])["cg_above_rotor_plane_mm"]
    return float(brentq(value, low, high)) if value(low) * value(high) < 0 else None

def evaluator_cog(name, entry):
    data = json.loads(Path(entry["file"]).read_text())["geometry"]["mass"]
    frame = (data["frame_mass_g"], data["frame_center_of_mass_mm"])
    motors = {item["name"].split("_", 1)[1]: item for item in data["components"] if item["type"] == "motor"}
    props = {item["name"].split("_", 1)[1]: item for item in data["components"] if item["type"] == "prop"}
    rotor, parts = {}, [frame] + [(item["mass_g"], item["center_mm"]) for item in data["components"] if item["type"] not in ("prop",)]
    for key, motor in motors.items():
        up = 1.0 if props[key]["center_mm"][2] > motor["center_mm"][2] else -1.0
        z = motor["center_mm"][2] + up * (motor["size_mm"][2] / 2 + entry["prop_thickness_mm"] / 2)
        rotor[key] = z
        parts.append((props[key]["mass_g"], [*props[key]["center_mm"][:2], z]))
    total = sum(mass for mass, _ in parts)
    cog = sum(mass * center[2] for mass, center in parts) / total
    planes = sorted(set(round(value, 2) for value in rotor.values()))
    mean = float(np.mean(list(rotor.values())))
    return {"source": entry["file"], "components": "unsere Hardware (K) auf den Aufnahmen der Referenz, wie Bewertungswerkzeug", "mass_g": total, "cog_z_mm": cog, "evaluator_cog_z_mm": data["center_of_mass_mm"][2],
            "rotor_planes_z_mm": planes, "rotor_plane_mean_z_mm": mean, "cg_above_rotor_plane_mm": cog - mean, "rotor_rule": "Motoroberkante + halbe Propnabe (Konvention LayoutModel / LOAD_COVARIANCE prop_height_mm)",
            "evaluator_prop_z_mm": sorted(set(round(item["center_mm"][2], 2) for item in props.values()))}

def analysis():
    runs = {name: json.loads((RUNS / (name + ".json")).read_text()) for name in VARIANTS if (RUNS / (name + ".json")).exists()}
    base = LAYOUT_OPTIMIZATION
    refs = {}
    for frame in ("ours", "manafly3", "tadpole"):
        setup = setup_for(frame, base)
        model = LayoutModel(setup, base)
        layouts = {"own": setup["own"]} if frame != "ours" else {"rule": setup["own"], "default": LAYOUT_DEFAULT["layout"]}
        refs[frame] = {}
        for label, layout in layouts.items():
            result = model.evaluate(layout)
            refs[frame][label] = {"layout": layout, "mass_g": result["mass_g"], "cg_above_rotor_plane_mm": result["cg_above_rotor_plane_mm"], "rotor_z_mm": model.rotor_z, "alpha_rad_s2": result["alpha_rad_s2"],
                                  "violated": {name: {"margin": result["constraints"][name]} for name in result["violated"]}, "constraints": result["constraints"], "fov": fov_report(LayoutModel(setup, base), layout)}
    mana = [("Modell (K)", {}), ("Motorsitz z 8 (Datenblatt Feld 3: 8-9,2)", {"pad_z_mm": 8.0}), ("Akku 50 g", {"battery.mass_g": 50.0}), ("Akku 60 g, 22 hoch", {"battery.mass_g": 60.0, "battery.height_mm": 22.0}),
            ("Akku 75 g 4S 650, 26 hoch", {"battery.mass_g": 75.0, "battery.height_mm": 26.0}), ("Motoren 8,5 g", {"motor.mass_g": 8.5}), ("Props 2,5 g", {"prop.mass_g": 2.5}),
            ("Real 142,7 g (ANNAHME)", REAL["manafly3_142g"]["set"]), ("Akku 3 mm tiefer", {"layout.deck_top_mm": 24.8}), ("Kamera Neigung 30", {"camera.tilt_deg": 30.0}), ("Kamera Neigung 40", {"camera.tilt_deg": 40.0})]
    mana_rows = robustness("manafly3", mana)
    tad = [("Handmodell", {}), ("Seitenteile 2 g", {"frame_part.5.mass_g": 2.0}), ("Seitenteile 6 g", {"frame_part.5.mass_g": 6.0}), ("Seitenteil-Box z 7", {"frame_part.5.center_mm": [0.0, 5.0, 7.0]}),
           ("Seitenteil-Box z 13", {"frame_part.5.center_mm": [0.0, 5.0, 13.0]}), ("Akku 1 mm tiefer", {"layout.deck_top_mm": 18.0}), ("Akku 1 mm hoeher", {"layout.deck_top_mm": 20.0}),
           ("Buegel 10 % hoeher (19,8)", {"hoop.top_mm": 19.8}), ("Kamera 2 mm hoeher", {"layout.camera_bottom_clearance_mm": 2.0}), ("Kamera y 33", {"layout.camera_y_mm": 33.0}),
           ("Kamera Neigung 30", {"camera.tilt_deg": 30.0}), ("ohne ELRS-Draht (3 mm)", {"aio.elrs_mm": 0.0}), ("Props 2,5 g", {"prop.mass_g": 2.5})]
    tad_rows = robustness("tadpole", tad)
    cross = {"manafly3_battery_mass_g_at_cg_0": crossover("manafly3", "battery.mass_g", 37.0, 200.0), "manafly3_battery_mass_g_at_cg_0_with_8.5g_motors_2.5g_props": crossover("manafly3", "battery.mass_g", 37.0, 300.0, {"motor.mass_g": 8.5, "prop.mass_g": 2.5}),
             "manafly3_battery_mass_g_at_cg_0_with_8.5g_motors_2.5g_props_22mm": crossover("manafly3", "battery.mass_g", 37.0, 300.0, {"motor.mass_g": 8.5, "prop.mass_g": 2.5, "battery.height_mm": 22.0})}
    cog = {name: evaluator_cog(name, entry) for name, entry in EVALUATOR.items()}
    for name in ("manafly3", "tadpole"):
        cog[name + "_layout_model"] = {"cg_above_rotor_plane_mm": refs[name]["own"]["cg_above_rotor_plane_mm"], "rotor_planes_z_mm": [refs[name]["own"]["rotor_z_mm"]], "source": "LayoutModel, LAYOUT_REFERENCES (K)"}
    cog["ours_default"] = {"cg_above_rotor_plane_mm": refs["ours"]["default"]["cg_above_rotor_plane_mm"], "rotor_planes_z_mm": [refs["ours"]["default"]["rotor_z_mm"]], "source": "LayoutModel, LAYOUT_DEFAULT"}
    cog["ours_rule"] = {"cg_above_rotor_plane_mm": refs["ours"]["rule"]["cg_above_rotor_plane_mm"], "rotor_planes_z_mm": [refs["ours"]["rule"]["rotor_z_mm"]], "source": "LayoutModel, Regel-Layout (frame_share.layout)"}
    tbs = REAL["tbs_5in_6s"]
    total = sum(mass for mass, _ in tbs["parts"])
    cog["tbs_5in_6s_real_estimate"] = {"source": tbs["source"], "mass_g": total, "cg_above_rotor_plane_mm": sum(mass * z for mass, z in tbs["parts"]) / total - tbs["rotor_z"], "rotor_planes_z_mm": [tbs["rotor_z"]]}
    real = robustness("manafly3", [("Real 142,7 g (ANNAHME)", REAL["manafly3_142g"]["set"])])[0]
    cog["manafly3_142g_real_estimate"] = {"source": REAL["manafly3_142g"]["source"], "mass_g": real["mass_g"], "cg_above_rotor_plane_mm": real["cg_above_rotor_plane_mm"], "rotor_planes_z_mm": [refs["manafly3"]["own"]["rotor_z_mm"]]}
    tilts = {frame: tilt_sweep(frame, layout) for frame, layout in (("manafly3", layout_setup("manafly3")["own"]), ("tadpole", layout_setup("tadpole")["own"]), ("ours", LAYOUT_DEFAULT["layout"]))}
    variants = {}
    for name, run in runs.items():
        best = run["rounded"] or run["optimum"]
        model = LayoutModel(ours_setup(settings_for(name)), settings_for(name))
        variants[name] = {"override": run["override"], "layout": best["layout"], "alpha_rad_s2": best["alpha_rad_s2"], "alpha_min_rad_s2": best["alpha_min_rad_s2"], "cg_above_rotor_plane_mm": best["cg_above_rotor_plane_mm"],
                          "feasible": best["feasible"], "active": sorted(name for name, value in best["constraints"].items() if abs(value) < 0.1), "unsatisfiable": run["optimum"]["unsatisfiable"],
                          "delta_to_default_mm": {key: best["layout"][key] - LAYOUT_DEFAULT["layout"][key] for key in best["layout"]}, "default_feasible": run["default"]["feasible"], "default_violated": run["default"]["violated"],
                          "rule_feasible": run["rule"]["feasible"], "rule_violated": run["rule"]["violated"], "rule_alpha_rad_s2": run["rule"]["alpha_rad_s2"], "fov": fov_report(model, best["layout"])}
    result = {"table": table(refs, {"manafly3": mana_rows, "tadpole": tad_rows}), "lux": LUX, "constraints": refs, "robustness": {"manafly3": mana_rows, "tadpole": tad_rows, "crossover": cross}, "cg_band_references": cog, "camera_tilt_sweep": tilts, "variants": variants,
              "variant_settings": VARIANTS, "code": "Deep_Frame-layout a37f1bb (git archive, sauberer Stand ohne die offenen Phase-2-Aenderungen)"}
    (HERE / "layout_constraint_check.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"variants": {name: {"layout": {k: round(v, 2) for k, v in entry["layout"].items()}, "alpha": {k: round(v) for k, v in entry["alpha_rad_s2"].items()}, "cg": round(entry["cg_above_rotor_plane_mm"], 2),
                                          "active": entry["active"], "default_violated": entry["default_violated"], "rule_violated": entry["rule_violated"]} for name, entry in variants.items()}}, indent=1))

if __name__ == "__main__":
    optimize(sys.argv[2]) if sys.argv[1] == "optimize" else analysis()
