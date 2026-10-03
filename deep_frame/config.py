from copy import deepcopy
import hashlib
import json
from math import atan2, degrees
from pathlib import Path
import sys

COMPONENT_LIBRARY = {
    "GTS V3 1203": {
        "type": "motor",
        "dimensions_mm": {"diameter": 15.76, "height": 9.9, "shaft_diameter": 1.5},
        "mass_g": 4.5,
        "hole_pattern": {"layout": "bolt_circle", "count": 4, "pitch_mm": 9.0, "screw_diameter_mm": 2.0, "clearance_diameter_mm": 2.2},
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "screws",
        "data": {"kv": 8000, "thrust_n": 1.80},
        "source": "Hardware/GTS V3 1203/GTSV31203chanpinyemian301-54f4e.jpg (RCinPower GTS V3 1203 8000KV data sheet)",
        "model": "RCinPower GTS V3 1203 8000KV (user decision)",
        "parameter_sources": {
            "diameter_mm": "data sheet: motor dimension 15.76 x 9.9 mm",
            "height_mm": "data sheet: body length 9.9 mm, shaft tip excluded",
            "shaft_diameter_mm": "data sheet drawing: shaft tip 1.5 mm",
            "mount_layout": "data sheet drawing: 4 x M2 on 9 mm bolt circle, cross pattern",
            "mass_g": "data sheet: 4.5 g with 3 cm wire",
            "thrust_n": "data sheet: GF 65R, 7.4 V, 100 % throttle, 184 g pull = 1.80 N",
            "screw_clearance_mm": "design: 0.2 mm diametral clearance for M2",
        },
    },
    "GEPRC GR1105": {
        "type": "motor",
        "dimensions_mm": {"diameter": 14.2, "height": 14.6},
        "mass_g": 5.9,
        "hole_pattern": {"layout": "bolt_circle", "count": 4, "pitch_mm": 9.0, "screw_diameter_mm": 2.0, "clearance_diameter_mm": 2.2},
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "screws",
        "source": "https://geprc.com/product/gep-gr1105-motor/",
        "model": "PROVISIONAL: GEPRC GR1105 envelope, not a selected motor",
        "parameter_sources": {
            "diameter_mm": "https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg",
            "height_mm": "https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg; includes upper shaft",
            "mount_layout": "PROVISIONAL: drawing shows four M2 on 9 mm bolt circle; not 9 x 9 square",
            "mass_g": "https://geprc.com/wp-content/uploads/2019/05/22-8095453337.jpg; includes pictured leads",
            "screw_diameter_mm": "https://geprc.com/wp-content/uploads/2019/05/22-6199766706.jpg",
            "screw_clearance_mm": "design: 0.2 mm diametral clearance for M2",
        },
    },
    "HQProp T2.5X2X3V2S": {
        "type": "prop",
        "dimensions_mm": {"diameter": 63.5, "hub_diameter": 9.8, "hub_height": 5.0, "shaft_diameter": 1.5},
        "mass_g": 1.2,
        "hole_pattern": None,
        "keep_out": {"clearance_mm": 2.0},
        "mounting": "shaft",
        "data": {"size_in": 2.5, "pitch_in": 2.0, "blades": 3, "material": "polycarbonate", "rotation": "2 CW + 2 CCW", "adaptor_rings": False, "thrust_g_estimate": 206.0, "thrust_n_estimate": 2.02},
        "source": "user decision: HQProp T2.5X2X3V2S (HQ Durable Prop T2.5X2X3 V2S), 2 CW + 2 CCW, polycarbonate, 1.5 mm shaft, no adaptor rings",
        "model": "HQProp T2.5X2X3V2S 2.5 x 2 x 3 (user decision)",
        "parameter_sources": {
            "diameter_mm": "HQProp data: 2.5 inch = 63.5 mm",
            "hub_diameter_mm": "HQProp data: hub diameter 9.8 mm",
            "hub_height_mm": "HQProp data: hub thickness 5 mm; used as the axial extent of the swept prop disc",
            "shaft_diameter_mm": "HQProp data: 1.5 mm shaft bore, matches GTS V3 1203 shaft tip",
            "mass_g": "HQProp data: 1.2 g",
            "thrust_n_estimate": "ESTIMATE: Hardware/GTS V3 1203/GTSV31203chanpinyemian301-54f4e.jpg, GTS V3 1203 8000KV table, row HQ T65R, 7.4 V, 100 % throttle: 206 g pull = 2.02 N; T65R is a 65 mm 3-blade HQ prop, not the T2.5X2X3V2S itself",
            "keep_out_clearance_mm": "design: radial and axial prop clearance as TOPOLOGY_CONFIG prop_clearance_mm",
        },
    },
    "HDZero AIO15": {
        "type": "aio",
        "dimensions_mm": {"width": 31.3, "length": 31.3, "stack_height": 6.0, "grommet_height": 3.0},
        "mass_g": 7.2,
        "hole_pattern": {"layout": "square", "count": 4, "pitch_mm": 25.5, "screw_diameter_mm": 2.0},
        "keep_out": {"clearance_mm": 0.5, "elrs_antenna_mm": 3.0},
        "mounting": "grommets",
        "source": "user: HDZero AIO15 dimensions, M2 mounting pattern and mass",
        "parameter_sources": {
            "stack_height_mm": "PROVISIONAL: complete populated board envelope; measure actual stack",
            "grommet_height_mm": "PROVISIONAL: HDZero manual, soft mount on the 4 included rubber grommets; height not given, typical whoop grommet",
            "elrs_antenna_clearance_mm": "HDZero AIO15 manual: lift the ELRS antenna at least 3 mm off the board",
            "vtx_antenna": "HDZero AIO15 manual: UFL VTX antenna mounted outward; rear antenna eyelet keeps it outboard",
        },
    },
    "HDZero Lux": {
        "type": "camera",
        "dimensions_mm": {"length": 14.0, "width": 16.0, "height": 14.0},
        "mass_g": 2.3,
        "hole_pattern": {"layout": "side_pair", "count": 2, "screw_diameter_mm": 2.0, "clearance_diameter_mm": 2.2},
        "keep_out": {"clearance_mm": 0.5, "side_mm": 2.0, "bottom_mm": 2.0},
        "mounting": "screws",
        "source": "user: HDZero Lux dimensions and mass",
        "parameter_sources": {"clearance_diameter_mm": "frame v0: 2.2 mm side screw bore", "keep_out": "frame v0: 2 mm side and bottom camera clearance"},
    },
    "GNB5502S120A": {
        "type": "battery",
        "dimensions_mm": {"length": 63.0, "width": 30.0, "height": 11.0},
        "mass_g": 37.0,
        "hole_pattern": None,
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "strap",
        "data": {"cells": 2, "capacity_mah": 550, "power_connector": "AMASS XT30U-F", "balance_connector": "JST XHP-3"},
        "source": "user: GNB5502S120A dimensions and mass",
        "mass_scope": "battery including leads and connectors; represented at battery center",
    },
    "HDZero VTX + ELRS": {
        "type": "antennas",
        "dimensions_mm": {"bore": 3.0, "holder_height": 10.0},
        "mass_g": 0.0,
        "hole_pattern": None,
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "eyelet",
        "data": {"vtx": "UFL VTX antenna mounted outward through the rear eyelet", "elrs": "wire antenna lifted at least 3 mm above the AIO board"},
        "source": "frame v0 antenna eyelet; HDZero AIO15 manual for VTX and ELRS routing",
        "mass_scope": "PROVISIONAL: antenna mass not measured, neglected",
    },
    "AMASS XT30U-F": {
        "type": "connector",
        "dimensions_mm": {"width": 10.2, "length": 12.4, "height": 5.2},
        "mass_g": 0.0,
        "hole_pattern": None,
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "strap",
        "source": "https://images.100y.com.tw/pdf_file/AMASS-XT30U.pdf#page=2",
        "model": "AMASS XT30U-F bounding envelope",
        "mass_scope": "already_in_battery",
        "parameter_sources": {"mass_g": "battery mass includes leads and connectors; avoid double counting"},
    },
    "JST XHP-3": {
        "type": "connector",
        "dimensions_mm": {"width": 9.8, "length": 7.5, "height": 5.7},
        "mass_g": 0.0,
        "hole_pattern": None,
        "keep_out": {"clearance_mm": 0.5},
        "mounting": "strap",
        "data": {"pins": 3},
        "source": "https://www.jst-mfg.com/product/pdf/eng/eXH.pdf#page=4",
        "model": "PROVISIONAL: JST XHP-3 housing envelope; verify actual GNB connector",
        "mass_scope": "already_in_battery",
        "parameter_sources": {"pins": "user: 3-pin balance connector", "mass_g": "battery mass includes leads and connectors; avoid double counting"},
    },
}

LIBRARY_FIELDS = {
    "common": ["type", "dimensions_mm", "mass_g", "hole_pattern", "keep_out.clearance_mm", "mounting", "source"],
    "motor": ["dimensions_mm.diameter", "dimensions_mm.height", "hole_pattern.layout", "hole_pattern.pitch_mm", "hole_pattern.screw_diameter_mm", "hole_pattern.clearance_diameter_mm", "data.thrust_n"],
    "aio": ["dimensions_mm.width", "dimensions_mm.length", "dimensions_mm.stack_height", "dimensions_mm.grommet_height", "hole_pattern.layout", "hole_pattern.pitch_mm", "hole_pattern.screw_diameter_mm", "keep_out.elrs_antenna_mm"],
    "camera": ["dimensions_mm.length", "dimensions_mm.width", "dimensions_mm.height", "hole_pattern.clearance_diameter_mm", "keep_out.side_mm", "keep_out.bottom_mm"],
    "battery": ["dimensions_mm.length", "dimensions_mm.width", "dimensions_mm.height", "data.power_connector", "data.balance_connector"],
    "antennas": ["dimensions_mm.bore", "dimensions_mm.holder_height"],
    "connector": ["dimensions_mm.width", "dimensions_mm.length", "dimensions_mm.height"],
    "prop": ["dimensions_mm.diameter", "dimensions_mm.hub_diameter", "dimensions_mm.hub_height", "dimensions_mm.shaft_diameter", "data.size_in", "data.pitch_in", "data.blades"],
}
MOUNTING_TYPES = ("grommets", "screws", "strap", "eyelet", "shaft")
PROP_RULE = {"swept_margin_mm": 1.5, "thickness_mm": 0.8, "mass_g": 0.7, "source": "PROVISIONAL layout rule for prop sizes without a library entry: swept disk = prop size x 25.4 mm + 1.5 mm margin",
             "parameter_sources": {"thickness_mm": "PROVISIONAL: swept disk thickness; excludes blade flex", "mass_g": "PROVISIONAL: prop model not selected; equivalent uniform disk inertia"}}
LEGACY_HOLE_KEYS = {"layout": "mount_layout", "pitch_mm": "mount_pitch_mm", "screw_diameter_mm": "screw_diameter_mm", "clearance_diameter_mm": "screw_clearance_mm"}

def component_spec(entry):
    spec = {key + "_mm": value for key, value in entry["dimensions_mm"].items()}
    spec.update({LEGACY_HOLE_KEYS[key]: value for key, value in (entry["hole_pattern"] or {}).items() if key in LEGACY_HOLE_KEYS})
    if "elrs_antenna_mm" in entry["keep_out"]:
        spec["elrs_antenna_clearance_mm"] = entry["keep_out"]["elrs_antenna_mm"]
    spec.update(deepcopy(entry.get("data", {})))
    spec["mass_g"] = entry["mass_g"]
    spec.update({key: deepcopy(entry[key]) for key in ("source", "model", "mass_scope", "parameter_sources") if key in entry})
    return spec

def prop_spec(size_in, entry=None, rule=PROP_RULE):
    if entry is not None and entry["data"]["size_in"] == size_in:
        return {**component_spec(entry), "thickness_mm": entry["dimensions_mm"]["hub_height"]}
    return {"diameter_mm": round(size_in * 25.4 + rule["swept_margin_mm"], 3), **{key: deepcopy(value) for key, value in rule.items() if key != "swept_margin_mm"}}

DEFAULT_SELECTION = {"aio15": "HDZero AIO15", "camera": "HDZero Lux", "battery": "GNB5502S120A", "motor": "GTS V3 1203", "xt30": "AMASS XT30U-F", "balancer": "JST XHP-3"}
DEFAULT_PROP = "HQProp T2.5X2X3V2S"
COMPONENT_DEFAULTS = {name: component_spec(COMPONENT_LIBRARY[part]) for name, part in DEFAULT_SELECTION.items()}
COMPONENT_DEFAULTS["camera"]["tilt_deg"] = 20.0
COMPONENT_DEFAULTS["camera"]["parameter_sources"]["tilt_deg"] = "design: adjustable initial camera tilt"
COMPONENT_DEFAULTS["prop"] = prop_spec(2.5, COMPONENT_LIBRARY[DEFAULT_PROP])

FRAME_DEFAULTS = {'wheelbase_mm': 135.0,
 'lateral_longitudinal_ratio': 1.3658536585365855,
 'minimum_wall_mm': 2.0,
 'base_thickness_mm': 2.5,
 'body_width_mm': 40.0,
 'body_length_mm': 40.0,
 'base_window_mm': 20.0,
 'arm_width_mm': 6.5,
 'arm_height_mm': 4.0,
 'arm_root_mm': 12.0,
 'motor_pad_radius_mm': 9.5,
 'motor_shaft_hole_mm': 2.8,
 'hole_clearance_mm': 0.2,
 'aio_standoff_mm': 3.0,
 'deck_width_mm': 40.0,
 'deck_length_mm': 70.0,
 'deck_top_mm': 28.0,
 'deck_thickness_mm': 2.5,
 'battery_margin_mm': 5.0,
 'support_length_mm': 56.0,
 'wall_window_length_mm': 16.0,
 'wall_window_height_mm': 10.0,
 'wall_window_bottom_mm': 12.0,
 'deck_window_width_mm': 22.0,
 'deck_window_length_mm': 46.0,
 'strap_slot_width_mm': 2.0,
 'strap_slot_length_mm': 12.0,
 'strap_slot_x_mm': 16.5,
 'strap_slot_y_mm': 12.0,
 'component_clearance_mm': 1.0,
 'camera_side_clearance_mm': 2.0,
 'camera_bottom_clearance_mm': 2.0,
 'camera_y_mm': 35.0,
 'cage_length_mm': 34.0,
 'cage_height_mm': 26.0,
 'camera_screw_diameter_mm': 2.2,
 'tail_width_mm': 40.0,
 'tail_length_mm': 40.0,
 'tail_y_mm': 38.0,
 'tail_window_width_mm': 22.0,
 'tail_window_length_mm': 14.0,
 'tail_window_y_mm': 29.0,
 'connector_offset_x_mm': 7.5,
 'connector_y_mm': 45.0,
 'connector_clearance_mm': 0.5,
 'connector_holder_height_mm': 9.0,
 'antenna_bore_mm': 3.0,
 'antenna_holder_height_mm': 10.0,
 'antenna_y_mm': 56.0,
 'cable_slot_width_mm': 4.0,
 'cable_slot_height_mm': 3.0,
 'prop_motor_gap_mm': 0.0}

FRAME_DEFAULT_SOURCES = {'wheelbase_mm': {'value': 135.0,
                  'kind': 'design_assumption',
                  'frame_ids': ['gecko_3', 'tadpole_2_5', 'tadpole_hd_3'],
                  'principle_ids': ['wide_x', 'component_driven'],
                  'rationale': '135mm liegt zwischen Tadpole3(131.25) und '
                               'TadpoleHD(137); aus groesserem30-mm-Akku plus '
                               'Propellerfreigang gewaehlt, kein '
                               'Herstellerwert.'},
 'lateral_longitudinal_ratio': {'value': 1.3658536585365855,
                                'kind': 'dimensionless_reference',
                                'frame_ids': ['gecko_3',
                                              'tadpole_2_5',
                                              'tadpole_hd_3'],
                                'principle_ids': ['wide_x', 'component_driven'],
                                'rationale': 'Gecko3 nennt112mm quer und82mm '
                                             'laengs; uebernommen wird '
                                             'ausschliesslich112/82, keine '
                                             'Kontur.'},
 'minimum_wall_mm': {'value': 2.0,
                     'kind': 'design_assumption',
                     'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                     'principle_ids': ['deck_load_path',
                                       'functional_voids',
                                       'component_driven'],
                     'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                  'und2-mm-Mindestwand abgeleitete eigene '
                                  'Druckabmessung.'},
 'base_thickness_mm': {'value': 2.5,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'},
 'body_width_mm': {'value': 40.0,
                   'kind': 'design_assumption',
                   'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                   'principle_ids': ['deck_load_path',
                                     'functional_voids',
                                     'component_driven'],
                   'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                'und2-mm-Mindestwand abgeleitete eigene '
                                'Druckabmessung.'},
 'body_length_mm': {'value': 40.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                    'principle_ids': ['deck_load_path',
                                      'functional_voids',
                                      'component_driven'],
                    'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                 'und2-mm-Mindestwand abgeleitete eigene '
                                 'Druckabmessung.'},
 'base_window_mm': {'value': 20.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                    'principle_ids': ['deck_load_path',
                                      'functional_voids',
                                      'component_driven'],
                    'rationale': '20-mm-Fenster haelt '
                                 'um25.5-mm-M2-Montageachsen '
                                 'mindestens2-mm-Material bis zum Fenster.'},
 'arm_width_mm': {'value': 6.5,
                  'kind': 'design_assumption',
                  'frame_ids': ['gecko_3', 'tadpole_2_5', 'tadpole_hd_3'],
                  'principle_ids': ['wide_x', 'component_driven'],
                  'rationale': 'Eigene Dimensionierung fuer65-mm-Props '
                               'und30-mm-Akku; Querschnitte sind vorlaeufige '
                               'PA6-CF-Annahmen.'},
 'arm_height_mm': {'value': 4.0,
                   'kind': 'design_assumption',
                   'frame_ids': ['gecko_3', 'tadpole_2_5', 'tadpole_hd_3'],
                   'principle_ids': ['wide_x', 'component_driven'],
                   'rationale': 'Eigene Dimensionierung fuer65-mm-Props '
                                'und30-mm-Akku; Querschnitte sind vorlaeufige '
                                'PA6-CF-Annahmen.'},
 'arm_root_mm': {'value': 12.0,
                 'kind': 'design_assumption',
                 'frame_ids': ['gecko_3', 'tadpole_2_5', 'tadpole_hd_3'],
                 'principle_ids': ['wide_x', 'component_driven'],
                 'rationale': 'Eigene Dimensionierung fuer65-mm-Props '
                              'und30-mm-Akku; Querschnitte sind vorlaeufige '
                              'PA6-CF-Annahmen.'},
 'motor_pad_radius_mm': {'value': 9.5,
                         'kind': 'design_assumption',
                         'frame_ids': ['gecko_3',
                                       'tadpole_2_5',
                                       'tadpole_hd_3'],
                         'principle_ids': ['wide_x', 'component_driven'],
                         'rationale': 'Eigene Dimensionierung fuer65-mm-Props '
                                      'und30-mm-Akku; Querschnitte sind '
                                      'vorlaeufige PA6-CF-Annahmen.'},
 'motor_shaft_hole_mm': {'value': 2.8,
                         'kind': 'design_assumption',
                         'frame_ids': ['gecko_3',
                                       'tadpole_2_5',
                                       'tadpole_hd_3'],
                         'principle_ids': ['wide_x', 'component_driven'],
                         'rationale': '2.8-mm-Zentralbohrung laesst '
                                      'bei4.5-mm-Lochkreisradius '
                                      'und1.1-mm-Schraubradius '
                                      'genau2-mm-Netzsteg; reale '
                                      'Motorunterseite pruefen.'},
 'hole_clearance_mm': {'value': 0.2,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'},
 'aio_standoff_mm': {'value': 3.0,
                     'kind': 'design_assumption',
                     'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                     'principle_ids': ['deck_load_path',
                                       'functional_voids',
                                       'component_driven'],
                     'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                  'und2-mm-Mindestwand abgeleitete eigene '
                                  'Druckabmessung.'},
 'deck_width_mm': {'value': 40.0,
                   'kind': 'design_assumption',
                   'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                   'principle_ids': ['deck_load_path',
                                     'functional_voids',
                                     'component_driven'],
                   'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                'und2-mm-Mindestwand abgeleitete eigene '
                                'Druckabmessung.'},
 'deck_length_mm': {'value': 70.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                    'principle_ids': ['deck_load_path',
                                      'functional_voids',
                                      'component_driven'],
                    'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                 'und2-mm-Mindestwand abgeleitete eigene '
                                 'Druckabmessung.'},
 'deck_top_mm': {'value': 28.0,
                 'kind': 'design_assumption',
                 'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                 'principle_ids': ['deck_load_path',
                                   'functional_voids',
                                   'component_driven'],
                 'rationale': 'Eigene OberkanteZ28 auf einer gemeinsamen '
                              'Knotenebene der 4-, 2-, 4/3- und 1-mm-Gitter; '
                              'Boardunterseite5.5 plus Stack6, '
                              'Deckunterseite25.5. Gecko22mm hat unklaren '
                              'Hoehenbezug und wird nicht kopiert.'},
 'deck_thickness_mm': {'value': 2.5,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'},
 'battery_margin_mm': {'value': 5.0,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'},
 'support_length_mm': {'value': 56.0,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'},
 'wall_window_length_mm': {'value': 16.0,
                           'kind': 'design_assumption',
                           'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                           'principle_ids': ['deck_load_path',
                                             'functional_voids',
                                             'component_driven'],
                           'rationale': 'Aus AIO15- und Akkuhuelle, '
                                        'Stegbreiten und2-mm-Mindestwand '
                                        'abgeleitete eigene Druckabmessung.'},
 'wall_window_height_mm': {'value': 10.0,
                           'kind': 'design_assumption',
                           'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                           'principle_ids': ['deck_load_path',
                                             'functional_voids',
                                             'component_driven'],
                           'rationale': 'Aus AIO15- und Akkuhuelle, '
                                        'Stegbreiten und2-mm-Mindestwand '
                                        'abgeleitete eigene Druckabmessung.'},
 'wall_window_bottom_mm': {'value': 12.0,
                           'kind': 'design_assumption',
                           'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                           'principle_ids': ['deck_load_path',
                                             'functional_voids',
                                             'component_driven'],
                           'rationale': 'Aus AIO15- und Akkuhuelle, '
                                        'Stegbreiten und2-mm-Mindestwand '
                                        'abgeleitete eigene Druckabmessung.'},
 'deck_window_width_mm': {'value': 22.0,
                          'kind': 'design_assumption',
                          'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                          'principle_ids': ['deck_load_path',
                                            'functional_voids',
                                            'component_driven'],
                          'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                       'und2-mm-Mindestwand abgeleitete eigene '
                                       'Druckabmessung.'},
 'deck_window_length_mm': {'value': 46.0,
                           'kind': 'design_assumption',
                           'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                           'principle_ids': ['deck_load_path',
                                             'functional_voids',
                                             'component_driven'],
                           'rationale': 'Aus AIO15- und Akkuhuelle, '
                                        'Stegbreiten und2-mm-Mindestwand '
                                        'abgeleitete eigene Druckabmessung.'},
 'strap_slot_width_mm': {'value': 2.0,
                         'kind': 'design_assumption',
                         'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                         'principle_ids': ['deck_load_path',
                                           'functional_voids',
                                           'component_driven'],
                         'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                      'und2-mm-Mindestwand abgeleitete eigene '
                                      'Druckabmessung.'},
 'strap_slot_length_mm': {'value': 12.0,
                          'kind': 'design_assumption',
                          'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                          'principle_ids': ['deck_load_path',
                                            'functional_voids',
                                            'component_driven'],
                          'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                       'und2-mm-Mindestwand abgeleitete eigene '
                                       'Druckabmessung.'},
 'strap_slot_x_mm': {'value': 16.5,
                     'kind': 'design_assumption',
                     'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                     'principle_ids': ['deck_load_path',
                                       'functional_voids',
                                       'component_driven'],
                     'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                  'und2-mm-Mindestwand abgeleitete eigene '
                                  'Druckabmessung.'},
 'strap_slot_y_mm': {'value': 12.0,
                     'kind': 'design_assumption',
                     'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                     'principle_ids': ['deck_load_path',
                                       'functional_voids',
                                       'component_driven'],
                     'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                  'und2-mm-Mindestwand abgeleitete eigene '
                                  'Druckabmessung.'},
 'component_clearance_mm': {'value': 1.0,
                            'kind': 'design_assumption',
                            'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                            'principle_ids': ['deck_load_path',
                                              'functional_voids',
                                              'component_driven'],
                            'rationale': 'Aus AIO15- und Akkuhuelle, '
                                         'Stegbreiten und2-mm-Mindestwand '
                                         'abgeleitete eigene Druckabmessung.'},
 'camera_side_clearance_mm': {'value': 2.0,
                              'kind': 'design_assumption',
                              'frame_ids': ['tadpole_2_5', 'gecko_3'],
                              'principle_ids': ['camera_cage',
                                                'component_driven'],
                              'rationale': 'Lux-Huelle mit20Grad-Tilt, '
                                           'Freiraum und tragendem Schutzrand; '
                                           'Schraubensitz ist vorlaeufig.'},
 'camera_bottom_clearance_mm': {'value': 2.0,
                                'kind': 'design_assumption',
                                'frame_ids': ['tadpole_2_5', 'gecko_3'],
                                'principle_ids': ['camera_cage',
                                                  'component_driven'],
                                'rationale': 'Lux-Huelle mit20Grad-Tilt, '
                                             'Freiraum und tragendem '
                                             'Schutzrand; Schraubensitz ist '
                                             'vorlaeufig.'},
 'camera_y_mm': {'value': 35.0,
                 'kind': 'design_assumption',
                 'frame_ids': ['tadpole_2_5', 'gecko_3'],
                 'principle_ids': ['camera_cage', 'component_driven'],
                 'rationale': 'Lux-Huelle mit20Grad-Tilt, Freiraum und '
                              'tragendem Schutzrand; Schraubensitz ist '
                              'vorlaeufig.'},
 'cage_length_mm': {'value': 34.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['tadpole_2_5', 'gecko_3'],
                    'principle_ids': ['camera_cage', 'component_driven'],
                    'rationale': 'Lux-Huelle mit20Grad-Tilt, Freiraum und '
                                 'tragendem Schutzrand; Schraubensitz ist '
                                 'vorlaeufig.'},
 'cage_height_mm': {'value': 26.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['tadpole_2_5', 'gecko_3'],
                    'principle_ids': ['camera_cage', 'component_driven'],
                    'rationale': 'Lux-Huelle mit20Grad-Tilt, Freiraum und '
                                 'tragendem Schutzrand; Schraubensitz ist '
                                 'vorlaeufig.'},
 'camera_screw_diameter_mm': {'value': 2.2,
                              'kind': 'design_assumption',
                              'frame_ids': ['tadpole_2_5', 'gecko_3'],
                              'principle_ids': ['camera_cage',
                                                'component_driven'],
                              'rationale': 'Lux-Huelle mit20Grad-Tilt, '
                                           'Freiraum und tragendem Schutzrand; '
                                           'Schraubensitz ist vorlaeufig.'},
 'tail_width_mm': {'value': 40.0,
                   'kind': 'design_assumption',
                   'frame_ids': ['odonata_40mm', 'gecko_3'],
                   'principle_ids': ['tail_routing'],
                   'rationale': '40-mm-Heckfuss ueberlappt die Deckwaende '
                                'flaechig;36mm erzeugte eine '
                                'nichtmannigfaltige Linienberuehrung im STL.'},
 'tail_length_mm': {'value': 40.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['odonata_40mm', 'gecko_3'],
                    'principle_ids': ['tail_routing'],
                    'rationale': 'Aus Steckerhuellen, Clearance und '
                                 'Propellerfreigang abgeleitete eigene '
                                 'Heckabmessung.'},
 'tail_y_mm': {'value': 38.0,
               'kind': 'design_assumption',
               'frame_ids': ['odonata_40mm', 'gecko_3'],
               'principle_ids': ['tail_routing'],
               'rationale': 'Aus Steckerhuellen, Clearance und '
                            'Propellerfreigang abgeleitete eigene '
                            'Heckabmessung.'},
 'tail_window_width_mm': {'value': 22.0,
                          'kind': 'design_assumption',
                          'frame_ids': ['odonata_40mm', 'gecko_3'],
                          'principle_ids': ['tail_routing'],
                          'rationale': 'Aus Steckerhuellen, Clearance und '
                                       'Propellerfreigang abgeleitete eigene '
                                       'Heckabmessung.'},
 'tail_window_length_mm': {'value': 14.0,
                           'kind': 'design_assumption',
                           'frame_ids': ['odonata_40mm', 'gecko_3'],
                           'principle_ids': ['tail_routing'],
                           'rationale': 'Aus Steckerhuellen, Clearance und '
                                        'Propellerfreigang abgeleitete eigene '
                                        'Heckabmessung.'},
 'tail_window_y_mm': {'value': 29.0,
                      'kind': 'design_assumption',
                      'frame_ids': ['odonata_40mm', 'gecko_3'],
                      'principle_ids': ['tail_routing'],
                      'rationale': 'Aus Steckerhuellen, Clearance und '
                                   'Propellerfreigang abgeleitete eigene '
                                   'Heckabmessung.'},
 'connector_offset_x_mm': {'value': 7.5,
                           'kind': 'design_assumption',
                           'frame_ids': ['odonata_40mm', 'gecko_3'],
                           'principle_ids': ['tail_routing'],
                           'rationale': 'Aus Steckerhuellen, Clearance und '
                                        'Propellerfreigang abgeleitete eigene '
                                        'Heckabmessung.'},
 'connector_y_mm': {'value': 45.0,
                    'kind': 'design_assumption',
                    'frame_ids': ['odonata_40mm', 'gecko_3'],
                    'principle_ids': ['tail_routing'],
                    'rationale': 'Aus Steckerhuellen, Clearance und '
                                 'Propellerfreigang abgeleitete eigene '
                                 'Heckabmessung.'},
 'connector_clearance_mm': {'value': 0.5,
                            'kind': 'design_assumption',
                            'frame_ids': ['odonata_40mm', 'gecko_3'],
                            'principle_ids': ['tail_routing'],
                            'rationale': 'Aus Steckerhuellen, Clearance und '
                                         'Propellerfreigang abgeleitete eigene '
                                         'Heckabmessung.'},
 'connector_holder_height_mm': {'value': 9.0,
                                'kind': 'design_assumption',
                                'frame_ids': ['odonata_40mm', 'gecko_3'],
                                'principle_ids': ['tail_routing'],
                                'rationale': 'Aus Steckerhuellen, Clearance '
                                             'und Propellerfreigang '
                                             'abgeleitete eigene '
                                             'Heckabmessung.'},
 'antenna_bore_mm': {'value': 3.0,
                     'kind': 'design_assumption',
                     'frame_ids': ['odonata_40mm', 'gecko_3'],
                     'principle_ids': ['tail_routing'],
                     'rationale': 'Aus Steckerhuellen, Clearance und '
                                  'Propellerfreigang abgeleitete eigene '
                                  'Heckabmessung.'},
 'antenna_holder_height_mm': {'value': 10.0,
                              'kind': 'design_assumption',
                              'frame_ids': ['odonata_40mm', 'gecko_3'],
                              'principle_ids': ['tail_routing'],
                              'rationale': 'Aus Steckerhuellen, Clearance und '
                                           'Propellerfreigang abgeleitete '
                                           'eigene Heckabmessung.'},
 'antenna_y_mm': {'value': 56.0,
                  'kind': 'design_assumption',
                  'frame_ids': ['odonata_40mm', 'gecko_3'],
                  'principle_ids': ['tail_routing'],
                  'rationale': 'Aus Steckerhuellen, Clearance und '
                               'Propellerfreigang abgeleitete eigene '
                               'Heckabmessung.'},
 'cable_slot_width_mm': {'value': 4.0,
                         'kind': 'design_assumption',
                         'frame_ids': ['odonata_40mm', 'gecko_3'],
                         'principle_ids': ['tail_routing'],
                         'rationale': 'Aus Steckerhuellen, Clearance und '
                                      'Propellerfreigang abgeleitete eigene '
                                      'Heckabmessung.'},
 'cable_slot_height_mm': {'value': 3.0,
                          'kind': 'design_assumption',
                          'frame_ids': ['odonata_40mm', 'gecko_3'],
                          'principle_ids': ['tail_routing'],
                          'rationale': 'Aus Steckerhuellen, Clearance und '
                                       'Propellerfreigang abgeleitete eigene '
                                       'Heckabmessung.'},
 'prop_motor_gap_mm': {'value': 0.0,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'HQProp T2.5X2X3V2S ohne Adapterringe: die '
                                    '5-mm-Nabe sitzt direkt auf der Motorglocke, '
                                    'die Propscheibe beginnt an der Motoroberkante.'}}

FEA_CONFIG = {
    "material": {
        "name": "PA6-CF, preliminary isotropic dry XY surrogate",
        "young_modulus_mpa": 4430.0,
        "poisson_ratio": 0.30,
        "density_g_cm3": 1.09,
        "source": "https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf",
        "assumptions": [
            "Bambu PA6-CF TDS: XY Young modulus 4430 +/- 310 MPa, density 1.09 g/cm3.",
            "Poisson ratio 0.30 is an unmeasured modeling assumption.",
            "Isotropic full-density surrogate; real printed Z modulus is lower (2170 +/- 230 MPa).",
            "No moisture, voids, plasticity, fatigue, damage or strain-rate dependence.",
        ],
    },
    "settings": {
        "mesh_size_mm": 3.0,
        "mesh_min_size_mm": 0.5,
        "mesh_curvature_points": 12,
        "mesh_threads": 1,
        "element_order": 2,
        "num_modes": 6,
        "work_dir": "exports/fea",
        "solver_path": None,
        "mesh_timeout_s": 180.0,
        "solver_timeout_s": 180.0,
        "threads": 2,
        "minimum_elastic_frequency_hz": 0.1,
        "stiffness_load_case": "arm_tip",
    },
}

PRINT_MATERIAL = {
    "name": "Bambu PA6-CF, printed, transversely isotropic (layer plane XY, build direction Z)",
    "young_modulus_mpa": 4430.0,
    "poisson_ratio": 0.30,
    "density_g_cm3": 1.09,
    "orthotropic": {"e_xy_mpa": 4430.0, "e_z_mpa": 2170.0, "nu_xy": 0.30, "nu_xz": 0.30, "g_xy_mpa": 1703.8, "g_z_mpa": 834.6},
    "strength_xy_mpa": 102.0,
    "strength_z_mpa": 48.0,
    "source": "https://store.bblcdn.eu/s8/default/a64af9edb0f64095ad18bc4ad4faf1ec/Bambu_PA6-CF_Technical_Data_Sheet-v2.pdf (Technical Data Sheet V3.0, ISO 527 / ISO 1183, specimens 100 % infill, annealed and dried 80 C 12 h)",
    "measured": ["E_xy 4430 +/- 310 MPa", "E_z 2170 +/- 230 MPa", "tensile strength XY 102 +/- 7 MPa", "tensile strength Z 48 +/- 6 MPa", "density 1.09 g/cm3"],
    "assumptions": {"nu_xy": "0.30 not in the TDS", "nu_xz": "0.30 not in the TDS (load in the layer plane, contraction along Z)", "g_xy_mpa": "E_xy/(2(1+nu)), in-plane isotropy assumed",
                    "g_z_mpa": "E_z/(2(1+nu)), interlayer shear modulus not in the TDS", "state": "dry, annealed, 100 % infill; no moisture, voids, plasticity, fatigue or strain rate"},
}

EVALUATION_CONFIG = {
    "name": None,
    "stl": None,
    "output": None,
    "print_axis": [0.0, 0.0, 1.0],
    "prop_diameter_mm": 65.0,
    "motors": None,
    "motor_up": {},
    "ours": False,
    "domain": None,
    "datasheet": None,
    "components": [],
    "mount_patterns": [],
    "keep_outs": [],
    "connectors": [],
    "selectors": None,
    "domain_grid": [102, 96, 24],
    "parts": ["geometry", "walls", "fea", "slicer"],
    "python": sys.executable,
    "compute": "C:/clones/Deep_Frame-int/tools/compute.py",
    "voxel_mm": 0.4,
    "loop_closing_mm": 3.0,
    "overhang_deg": 45.0,
    "bed_tolerance_mm": 0.2,
    "fit_tolerance_mm3": 1.0,
    "hub_radius_mm": COMPONENT_DEFAULTS["motor"]["diameter_mm"] / 2,
    "section_voxel_mm": 0.05,
    "hole_tolerance_mm": 0.6,
    "screw_head_radius_mm": 1.9,
    "tool_skip_mm": 3.0,
    "connector_radius_mm": 2.0,
    "surface_samples": 5000,
    "curvature_radius_mm": 1.0,
    "seed": 0,
    "loads": {"arm_tip_force_n": 3.6, "all_up_mass_g": 125.0, "standard_gravity_m_s2": 9.80665, "crash_front_g": 25.0, "crash_arm_g": 12.5, "crash_back_g": 12.5, "safety_factor": 2.0,
              "sources": {"arm_tip": "motor thrust 1.80 N (GTS V3 1203 8000KV, GF65R, 7.4 V) x 2, upward on the front-left motor seat, centre mounts fixed",
                          "crash_front": "all-up mass 125 g x 25 g = 30.6 N rearward on the camera region, centre mounts fixed",
                          "crash_arm": "all-up mass 125 g x 12.5 g = 15.3 N on the front-left motor seat, oblique (inward, tangential, downward 2:2:1 normalised), centre mounts fixed",
                          "crash_back": "ASSUMPTION: all-up mass 125 g x 12.5 g = 15.3 N on the battery deck towards the frame (landing on the back), four motor seats fixed",
                          "safety_factor": "2.0 against TDS tensile strength, linear static equivalent load, no impact dynamics"}},
    "fea_settings": {"threads": 4, "mesh_threads": 4, "mesh_timeout_s": 900.0, "solver_timeout_s": 900.0, "fea_memory_budget_mb": 9728.0, "mesh_minimum_sicn": 0.005, "fea_remesh_targets_mm": [2.0, 1.5], "num_modes": 6},
    "fea_surface": {"targets_mm": [0.5, 0.6], "taubin": 10, "feature_degs": [40.0, 60.0, 89.0]},
    "slicer": {"executable": "C:/clones/prusaslicer/PrusaSlicer-2.9.6/prusa-slicer-console.exe", "version": "PrusaSlicer 2.9.6 portable (github.com/prusa3d/PrusaSlicer/releases/tag/version_2.9.6)",
               "options": ["--nozzle-diameter", "0.4", "--layer-height", "0.2", "--first-layer-height", "0.2", "--perimeters", "2", "--fill-density", "15%", "--filament-diameter", "1.75", "--filament-density", "1.09",
                           "--bed-shape", "0x0,256x0,256x256,0x256", "--max-print-height", "256", "--center", "128,128"],
               "support": ["--support-material", "--support-material-auto"], "timeout_s": 1800.0,
               "profile": "PrusaSlicer built-in defaults (generic FFF printer, default speeds) with nozzle 0.4, layer 0.2 mm, 2 perimeters, 15 % infill, automatic supports on a 256 x 256 x 256 mm bed"},
    "targets": {"airflow": ["geometry.airflow.prop_ring_share", None, 0.15], "overhang": ["geometry.printability.overhang_share", None, 0.25],
                "support": ["scaled.support_per_volume", None, 1.5], "strut_min": ["geometry.form.strut_width_mm.p10", 1.19, None], "symmetry": ["scaled.symmetry_per_wheelbase", None, 0.0025],
                "arm_tip_slope": ["scaled.arm_tip_slope", None, 0.007], "f1": ["fea.eigenfrequencies_hz.0", 330.0, None]},
    "warnings": {"strut_max": ["geometry.form.strut_width_mm.p90", None, 6.5], "section_ratio": ["geometry.form.section_ratio.p50", 1.0, 1.4], "top_view_material": ["geometry.airflow.bbox_share", None, 0.45],
                 "loops": ["geometry.form.loops.loops", 20, None], "roughness": ["geometry.form.roughness.curvature_neighbour_rms_per_mm", None, 0.18], "height": ["scaled.height_per_wheelbase", None, 0.4],
                 "cog_offset": ["scaled.cog_offset_per_wheelbase", None, 0.01], "inertia_z": ["scaled.izz_per_mass_arm2", None, 0.45], "print_time": ["scaled.print_min_per_g", None, 16.0]},
}

EVALUATION_KINDS = {"name": "text", "stl": "path", "output": "path", "print_axis": ["float", "float", "float"], "prop_diameter_mm": "float", "motors": "object", "motor_up": "object", "ours": "flag", "domain": "path", "datasheet": "path", "components": "list",
                    "mount_patterns": "list", "keep_outs": "list", "connectors": "list", "selectors": "object", "domain_grid": ["int", "int", "int"], "parts": [("geometry", "walls", "fea", "slicer")], "python": "text", "compute": "text", "targets": "object", "warnings": "object",
                    **{key: "float" for key in ("voxel_mm", "loop_closing_mm", "overhang_deg", "bed_tolerance_mm", "fit_tolerance_mm3", "hub_radius_mm", "section_voxel_mm", "hole_tolerance_mm", "screw_head_radius_mm", "tool_skip_mm", "connector_radius_mm", "curvature_radius_mm")},
                    "surface_samples": "int", "seed": "int", "loads": "object", "fea_settings": "object", "fea_surface": "object", "slicer": "object"}

INTEGRATION_CONFIG = {
    "model_version": "frame-v0-linear-fixtures-v1",
    "arm_tip_force_n": 3.6,
    "arm_tip_motor": "front_left",
    "thrust_safety_factor": 2.0,
    "all_up_mass_g": 125.0,
    "crash_front_g_factor": 25.0,
    "crash_arm_g_factor": 12.5,
    "crash_directions": [],
    "load_sources": {
        "arm_tip_force_n": "motor thrust 1.80 N (GTS V3 1203 8000KV, GF65R, 7.4 V, 100 %) x safety factor 2",
        "thrust_all": "all four motors at full thrust x 2 upward against the AIO mounts",
        "crash_front": "all-up mass ~125 g x 25 g equivalent static deceleration = 30.7 N rearward on the camera hoops",
        "crash_arm": "half the all-up mass x 25 g = 15.3 N on one motor ring, oblique (inward, tangential, downward)",
        "crash_directions": "opt-in optimizer set replacing crash_front and crash_arm: every listed direction carries the crash_front magnitude (30.7 N); front -Y on the camera hoops, side +-X on the two motor rings of one side, arm oblique on each motor ring, below +Z on the camera hoops, back -Z on the battery band",
    },
    "battery_impact_g_factor": 10.0,
    "standard_gravity_m_s2": 9.80665,
    "camera_side_force_n": 5.0,
    "central_fixture_fraction": 0.9,
    "selection_tolerance_mm": 0.01,
    "motor_pad_margin_mm": 0.1,
    "battery_attachment_band_width_mm": 4.0,
    "battery_attachment_y_mm": 0.0,
    "battery_attachment_margin_mm": 0.1,
    "camera_upper_height_fraction": 0.25,
    "camera_length_fraction": 0.5,
}

CRASH_DIRECTIONS = ["front", "side_left", "side_right", "arm_front_left", "arm_front_right", "arm_rear_left", "arm_rear_right", "below", "back"]

OPTIMIZATION_CONFIG = {
    "n_trials": 6,
    "seed": 42,
    "population_size": 8,
    "storage": "sqlite:///exports/optimization/study.sqlite3",
    "study_name": "frame-v0",
    "evaluation_id": "frame-v0-fea-v1",
    "output_dir": "exports/optimization",
    "relative_constraints": {
        "mass_ratio_max": 1.05,
        "stiffness_ratio_min": 0.95,
        "frequency_ratio_min": 0.95,
        "displacement_ratio_max": 1.05,
        "stress_ratio_max": 1.10,
    },
    "printability": {
        "nozzle_width_mm": 0.4,
        "minimum_wall_nozzles": 4,
        "wall_thickness_paths": [
            "frame.minimum_wall_mm",
            "frame.base_thickness_mm",
            "frame.deck_thickness_mm",
            "frame.arm_height_mm",
        ],
        "clamp_sections": [
            {
                "name": "arm_root",
                "width_path": "frame.arm_width_mm",
                "height_path": "frame.arm_height_mm",
                "minimum_area_mm2": 20.0,
            }
        ],
    },
    "initial_candidates": [
        {"frame.arm_height_mm": 4.2, "frame.arm_width_mm": 6.5},
        {"frame.arm_height_mm": 4.0, "frame.arm_width_mm": 6.7},
    ],
}

SEARCH_SPACE = {
    "frame.arm_height_mm": {"low": 3.8, "high": 4.4, "step": 0.2},
    "frame.arm_width_mm": {"low": 6.3, "high": 6.9, "step": 0.2},
}

TOPOLOGY_CONFIG = {
    "grid": {
        "origin_mm": [-68.0, -64.0, 0.0],
        "spacing_mm": [4.0, 4.0, 4.0],
        "shape": [34, 32, 8],
        "axis_order": "xyz",
        "order": "C",
    },
    "component_clearance_mm": 0.5,
    "prop_clearance_mm": 2.0,
    "motor_contact_radius_mm": 7.7,
    "aio_contact_radius_mm": 3.2,
    "aio_boss_radius_mm": 3.1,
    "camera_mount_radius_mm": 4.1,
    "antenna_eyelet_radius_mm": 4.3,
    "flush_overlap_mm": 0.5,
    "prescribed_wall_margin_mm": 0.1,
    "contact_depth_mm": 4.0,
    "camera_contact_width_mm": 6.0,
    "camera_tool_radius_mm": 2.0,
    "camera_contact_length_mm": 8.0,
    "battery_contact_width_mm": 3.5,
    "battery_contact_length_mm": 50.0,
    "battery_contact_y_mm": 0.0,
    "battery_rail_edge_inset_mm": 0.0,
    "connection_proof_force_n": 0.05,
    "manufacturing": {
        "nozzle_width_mm": 0.4,
        "minimum_wall_nozzles": 5.0,
        "minimum_feature_mm": 2.0,
        "minimum_attachment_area_mm2": 8.0,
        "supports_allowed": True,
        "build_direction": [0.0, 0.0, 1.0],
        "assumptions": [
            "External and accessible internal supports are allowed; not a support-free design.",
            "Coarse density cells do not resolve mounting bores; exact reconstruction cuts them.",
            "Nozzle width and track count are provisional manufacturing inputs.",
            "Feature and attachment checks are geometric screens, not print-process certification.",
        ],
    },
    "optimizer": {
        "interface_node_policy": "preserve_adjacent",
        "volume_fraction": 0.10,
        "filter_radius_mm": 6.0,
        "penalization": 3.0,
        "min_stiffness_ratio": 1e-6,
        "max_iterations": 45,
        "change_tolerance": 0.015,
        "move_limit": 0.12,
        "projection_beta": 1.0,
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
    },
    "reconstruction": {
        "density_threshold": 0.35,
        "minimum_feature_mm": 2.0,
    },
    "additional_regions": [],
}

SURFACE_FIDELITY = {"maximum_surface_deviation_mm": 0.20, "maximum_relative_volume_change": 0.01}

IMPLICIT_CONFIG = {
    "subdivisions": 10,
    "interpolation_method": "pchip",
    "thresholds": [0.25, 0.35],
    "extensions": ["preserve", "preserve_forbidden"],
    "density_sigma_mm": 0.4,
    "transition_radius_mm": 2.0,
    "preserve_inflation_mm": 0.18,
    "constraint_offset_mm": 0.3,
    "opening_radius_mm": 1.35,
    "protected_opening": False,
    "diagnostic_fea": False,
    "wall_screen_blocking": False,
    "wall_voxel_mm": 0.1,
    "wall_opening_radius_mm": 1.0,
    "wall_deep_mm": 0.45,
    "wall_deep_max_fraction": 0.005,
    "wall_deep_component_max_mm3": 5.0,
    "wall_motor_zone_margin_mm": 2.0,
    "wall_tile_voxels": 120,
    "wall_thin_max_fraction": 0.01,
    "wall_very_thin_mm": 1.5,
    "wall_very_thin_max_fraction": 0.0005,
    "detached_volume_max_fraction": 0.005,
    "ripple_sigma_mm": 0.25,
    "reinit_band_cells": 1.5,
    "remesh_target_mm": 0.6,
    "remesh_iterations": 5,
    "remesh_feature_deg": 60.0,
    "remesh_max_surface_distance_mm": 0.05,
    "segment_tolerance_mm": 0.01,
    "surface_deviation_mm": SURFACE_FIDELITY["maximum_surface_deviation_mm"],
    "relative_volume_change": SURFACE_FIDELITY["maximum_relative_volume_change"],
    "free_zone_preserve_mm": 3.0,
    "free_zone_constraint_mm": 2.0,
    "free_zone_minimum_samples": 1000,
    "free_zone_modified_mm": 1.0,
    "free_zone_opening_cells": 0.5,
    "penetration_tolerance_mm": 1e-4,
    "penetration_sample_spacing_mm": 0.25,
    "maximum_wall_samples": 2000000,
    "curvature_radius_mm": 1.0,
    "curvature_samples": 5000,
    "render_faces": 30000,
    "tet_attempts": ["remesh_hxt", "remesh_delaunay", "refine_hxt", "classify_hxt", "classify_delaunay", "direct_hxt"],
    "mesh_minimum_sicn": 0.01,
    "mesh_boundary_deviation_mm": 0.05,
    "fea_remesh_targets_mm": [2.0, 1.5, 1.2, 1.0],
    "fea_memory_budget_mb": 9728.0,
    "fea_memory_per_element_kb": 38.0,
    "fea_remesh_iterations": 5,
    "fea_remesh_feature_deg": 40.0,
    "fea_remesh_max_surface_distance_mm": 0.05,
    "fea_sliver_target_mm": 0.4,
    "fea_sliver_iterations": 8,
    "fea_sliver_collapse_mm": 0.02,
    "fea_refine_edge_mm": 1.0,
    "fea_refine_max_surface_distance_mm": 0.01,
    "fea_merge_relative_tolerance": 1e-9,
    "fea_t_vertex_ratio": 20.0,
    "fea_classify_angle_deg": 40.0,
    "fea_direct_minimum_angle_deg": 10.0,
    "fea_fallback_timeout_s": 120.0,
    "mesh_timeout_s": 900.0,
    "solver_timeout_s": 900.0,
    "candidate_timeout_s": 1800.0,
}

IMPLICIT_KINDS = {
    **{key: "float" if isinstance(value, float) else "int" for key, value in IMPLICIT_CONFIG.items() if isinstance(value, (int, float))},
    "interpolation_method": ("pchip", "cubic"),
    "protected_opening": "flag",
    "diagnostic_fea": "flag",
    "wall_screen_blocking": "flag",
    "thresholds": ["float"],
    "extensions": [("none", "preserve", "preserve_forbidden")],
    "tet_attempts": [("remesh_hxt", "remesh_delaunay", "refine_hxt", "classify_hxt", "classify_delaunay", "direct_hxt")],
    "fea_remesh_targets_mm": ["float"],
}

DESIGN_RECONSTRUCTION_CONFIG = {
    "source": None,
    "output": None,
    "fine_shape": [204, 192, 48],
    "density_sigma_cells": 1.0,
    "threshold": 0.5,
    "spur_factor": 2.0,
    "spur_minimum_mm": 3.0,
    "prune_passes": 3,
    "path_sigma_samples": 4.0,
    "section_sigma_samples": 6.0,
    "minimum_radius_mm": 1.0,
    "maximum_aspect": 2.0,
    "joint_blend_factor": 0.5,
    "anchor_reach_mm": 3.0,
    "volume_match": True,
    "shell_aspect": 2.2,
    "shell_sigma_mm": 1.0,
    "shell_outline_sigma_mm": 1.0,
    "transition_radius_mm": 1.5,
    "voxel_mm": 0.25,
    "target_volume_mm3": 0.0,
    "calibration_steps": 4,
    "calibration_voxel_mm": 0.5,
    "selector_half_band_mm": 1.0,
    "fea_surface_mm": 0.5,
    "fea_surface_taubin": 10,
    "boolean_offset_mm": 0.3,
    "closing_radius_mm": 0.5,
    "reference_subdivisions": 2,
    "preserve_blend_mm": 2.5,
    "preserve_round_mm": 0.8,
    "preserve_flush_mm": -0.3,
    "member_smooth_mm": 0.5,
    "root_preserves": "motor_contact",
    "root_distance_mm": 3.0,
    "root_taper_mm": 6.0,
    "root_taper_slope": 0.5,
    "root_slope_floor_mm": 1.0,
    "minimum_scale": 0.7,
    "maximum_scale": 1.15,
    "calibration_tolerance": 0.03,
    "load_path_voxel_mm": 0.3,
    "load_path_core_mm": [0.5, 0.9],
    "load_path_mounts": ["motor_contact", "aio_contact", "battery_rail", "camera_mount"],
}

DESIGN_RECONSTRUCTION_KINDS = {
    **{key: "float" if isinstance(value, float) else "int" for key, value in DESIGN_RECONSTRUCTION_CONFIG.items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
    "source": "path",
    "output": "path",
    "fine_shape": ["int", "int", "int"],
    "volume_match": "flag",
    "root_preserves": "text",
    "load_path_core_mm": ["float"],
    "load_path_mounts": ["text"],
}

SPLINE_RECONSTRUCTION_CONFIG = {
    **DESIGN_RECONSTRUCTION_CONFIG,
    "section_body": None,
    "body_subdivisions": 3,
    "spline_lengths_mm": [10.0, 20.0],
    "spline_sample_mm": 0.6,
    "profile_degree": 2,
    "profile_sigma_samples": 1.0,
    "minimum_radius_mm": 1.25,
    "maximum_aspect": 2.2,
    "transition_radius_mm": 1.0,
    "preserve_blend_mm": 1.0,
    "shell_outline_sigma_mm": 2.5,
    "member_smooth_mm": 0.0,
    "closing_radius_mm": 0.0,
    "root_distance_mm": 0.0,
    "calibration_steps": 0,
    "volume_match": False,
    "bump_minimum_mm": 0.3,
    "bump_pad_mm": 1.5,
    "loop_factor": 6.0,
    "clearance_iterations": 4,
    "clearance_margin_mm": 0.1,
    "clearance_tolerance_mm": 0.05,
    "anchor_inset_mm": 0.5,
    "anchor_snap_mm": 3.0,
    "bridge_gap_mm": 3.0,
}

SPLINE_RECONSTRUCTION_KINDS = {**DESIGN_RECONSTRUCTION_KINDS, **{key: "float" if isinstance(value, float) else "int" for key, value in SPLINE_RECONSTRUCTION_CONFIG.items() if isinstance(value, (int, float)) and not isinstance(value, bool)},
                               "section_body": "path", "spline_lengths_mm": ["float"]}

MATERIALS = {
    "PA6-CF": {
        **{key: PRINT_MATERIAL[key] for key in ("name", "density_g_cm3", "young_modulus_mpa", "poisson_ratio", "strength_xy_mpa", "strength_z_mpa", "source")},
        "e_z_mpa": PRINT_MATERIAL["orthotropic"]["e_z_mpa"],
        "value_sources": {"density_g_cm3": "TDS ISO 1183", "young_modulus_mpa": "TDS ISO 527 XY 4430 +/- 310 MPa", "e_z_mpa": "TDS ISO 527 Z 2170 +/- 230 MPa",
                          "strength_xy_mpa": "TDS XY 102 +/- 7 MPa", "strength_z_mpa": "TDS Z 48 +/- 6 MPa", "poisson_ratio": "ASSUMPTION: not in the TDS"},
    },
}

LAYOUT_RULES = {
    "x_types": {"compressed_x": {"arm_angle_deg": degrees(atan2(112, 82)), "source": "Gecko3 112 mm lateral x 82 mm longitudinal motor spacing"},
                "true_x": {"arm_angle_deg": 45.0, "source": "square X"},
                "stretched_x": {"arm_angle_deg": degrees(atan2(82, 112)), "source": "Gecko3 spacing rotated: 82 mm lateral x 112 mm longitudinal"}},
    "prop_tip_gap_mm": 14.7,
    "wheelbase_step_mm": 0.5,
    "envelope": {"origin_mm": [-68.0, -64.0, 0.0], "size_mm": [136.0, 128.0, 32.0]},
    "battery_mounts": {"top": {"deck_top_mm": 28.0, "headroom_mm": 3.0}, "bottom": {"gap_mm": 1.0}},
    "battery_prop_clearance_mm": 2.0,
    "camera": {"stack_gap_mm": 12.35, "top_clearance_mm": 3.0},
    "antennas": {"angle_deg": 0.0, "connector_clearance_mm": 0.5, "eyelet_radius_mm": 4.3, "envelope_margin_mm": 3.7},
    "connectors": {"stack_gap_mm": 13.75},
    "cg_tolerance_mm": 3.0,
    "pad": {"top_mm": 28 / 3, "thickness_mm": 8 / 3},
    "hoop": {"side_gap_mm": 4.0, "radius_mm": 1.6, "path_yz_mm": [[-11.0, 27.5], [-1.0, 26.0], [7.0, 23.5], [11.5, 18.0], [12.5, 11.0], [10.5, 5.0], [6.0, 2.0], [-2.0, 1.5]],
             "load_y_min_mm": 9.0, "load_z_mm": [4.0, 22.0]},
    "neural": {"half_wavelength_per_width": 2.0},
}

LAYOUT_OVERRIDES = {"motors": {"arm_angle_deg": "float", "wheelbase_mm": "float"}, "camera": {"tilt_deg": "float", "y_mm": "float"},
                    "antennas": {"angle_deg": "float", "y_mm": "float"}, "battery": {"deck_top_mm": "float"}, "stack": {"standoff_mm": "float"},
                    "optimizer": {"volume_fraction": "float", "max_frequency_per_mm": "float", "prop_discs": ("soft", "hard"), "f1_min_hz": "float", "method": ("neural", "simp"),
                                  "arm_tip_stiffness_min_n_per_mm": "float", "stiffness_calibration": "float"}}

STYLES = {
    "freestyle": {"crash_directions": ["front", "side_left", "side_right", "arm_front_left", "arm_front_right", "arm_rear_left", "arm_rear_right", "below", "back"],
                  "flight_cases": ["arm_tip", "thrust_all"], "torsion": True, "hoops": True},
}

DURABILITY = {
    "crash_resistant": {"crash_weight": 2.0, "safety_factor": 2.5, "minimum_width_mm": 2.4, "volume_fraction": 0.06},
    "standard": {"crash_weight": 1.0, "safety_factor": 2.0, "minimum_width_mm": 2.0, "volume_fraction": 0.05},
    "light": {"crash_weight": 0.5, "safety_factor": 1.5, "minimum_width_mm": 2.0, "volume_fraction": 0.04},
}

FRAME_REQUEST = {"name": None, "style": "freestyle", "durability": "standard", "prop_size_in": 2.5, "layout": {"x_type": "compressed_x", "battery_mount": "top"},
                 "components": {"motor": "GTS V3 1203", "aio": "HDZero AIO15", "camera": "HDZero Lux", "battery": "GNB5502S120A", "antennas": "HDZero VTX + ELRS", "prop": "HQProp T2.5X2X3V2S"},
                 "material": "PA6-CF", "print": {"nozzle_mm": 0.4, "layer_mm": 0.2}, "overrides": {}, "grid": "coarse", "reconstruction": True}
FRAME_REQUEST_KINDS = {"name": "text", "style": tuple(STYLES), "durability": tuple(DURABILITY), "prop_size_in": "float", "layout": "object", "components": "object",
                       "material": tuple(MATERIALS), "print": "object", "overrides": "object", "grid": ("coarse", "fine"), "reconstruction": "flag"}
FRAME_LAYOUT_KINDS = {"x_type": tuple(LAYOUT_RULES["x_types"]), "battery_mount": tuple(LAYOUT_RULES["battery_mounts"])}
FRAME_PRINT_KINDS = {"nozzle_mm": "float", "layer_mm": "float"}
FRAME_COMPONENT_KINDS = {"motor": "text", "aio": "text", "camera": "text", "battery": "text", "antennas": "text", "prop": "text"}

RUN_GRIDS = {
    "coarse": {"shape": [68, 64, 24], "fine_shape": [136, 128, 48], "neural": {"max_iterations": 60, "minimum_iterations": 30, "sharpness_iterations": 45, "max_runtime_s": 900.0},
               "reconstruction": {"voxel_mm": 0.5, "calibration_voxel_mm": 1.0}, "evaluation": {"voxel_mm": 0.5},
               "compute": {"reconstruction": "geometry"}},
    "fine": {"shape": [102, 96, 24], "fine_shape": [204, 192, 48], "neural": {}, "reconstruction": {}, "evaluation": {"fea_settings": {"fea_remesh_targets_mm": [2.0, 1.5, 1.2, 1.0]}}, "compute": {}},
}

STAGES = {
    "optimization": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "tools/neural_study.py", "argv": ["run"], "compute": "density_neural", "python": "C:/clones/Deep_Frame-gpu-venv/Scripts/python.exe"},
    "reconstruction": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "tools/reconstruction_study.py", "argv": ["build"], "compute": "reconstruction", "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe"},
    "geometry": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "deep_frame/topology_implicit_validation.py", "argv": ["wall_rule"], "compute": "wall_check", "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe"},
    "evaluation": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "tools/evaluate_frame.py", "argv": ["run"], "compute": None, "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe"},
    "datasheet": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "run.py", "argv": ["datasheet"], "compute": "cpu", "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe"},
    "renders": {"worktree": "C:/clones/Deep_Frame-r4", "tool": "tools/neural_study.py", "argv": ["render_views"], "compute": "render", "python": "C:/clones/Deep_Frame/.venv/Scripts/python.exe"},
}
RUN_SETTINGS = {"root": "exports/runs", "compute": "C:/clones/Deep_Frame-int/tools/compute.py", "domain_stage": "optimization", "domain_compute": "cpu",
                "views": {"iso": [[0.55, -0.85, -0.62], [0, 0, 1]], "top": [[0, 0, -1], [0, 1, 0]], "side": [[-1, 0, 0], [0, 0, 1]], "front": [[0, -1, 0], [0, 0, 1]]},
                "datasheet_voxel_mm": 0.5, "deck_band_mm": 1.0, "fixture_band_mm": 0.5}

CONFIG = {
    "length_mm": 30.0,
    "width_mm": 20.0,
    "thickness_mm": 3.0,
    "stl_path": Path(__file__).resolve().parent.parent / "exports" / "smoke.stl",
    "viewer_port": 3939,
    "frame": FRAME_DEFAULTS,
    "components": COMPONENT_DEFAULTS,
    "default_sources": FRAME_DEFAULT_SOURCES,
    "material": FEA_CONFIG["material"],
    "fea": FEA_CONFIG,
    "optimization": OPTIMIZATION_CONFIG,
    "optimization_search_space": SEARCH_SPACE,
    "integration": INTEGRATION_CONFIG,
    "checks": {
        "minimum_clearance_mm": 0.5,
        "prop_clearance_mm": 2.0,
        "intersection_tolerance_mm3": 1e-6,
        "distance_tolerance_mm": 1e-6,
        "maximum_battery_prop_overlap_percent": 0.0,
    },
    "frame_export_stem": "exports/frame_v0",
}

def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))

def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def _convert(kind, value, key):
    if isinstance(kind, tuple):
        if value in kind:
            return value
    elif isinstance(kind, list):
        if isinstance(value, (list, tuple)) and (len(value) == len(kind) if len(kind) > 1 else len(value) > 0):
            return [_convert(kind[0], item, key) for item in value]
    elif kind == "flag":
        if isinstance(value, bool):
            return value
    elif kind == "object":
        if isinstance(value, dict):
            return value
    elif kind == "text":
        if isinstance(value, str):
            return value
    elif kind in ("object", "list"):
        if isinstance(value, dict if kind == "object" else list):
            return value
    elif kind == "path":
        if isinstance(value, (str, Path)):
            return Path(value)
    elif not isinstance(value, bool) and isinstance(value, (str, int) if kind == "int" else (str, int, float)):
        try:
            return {"int": int, "float": float}[kind](value)
        except ValueError:
            pass
    raise ValueError(f"Invalid configuration value for {key}: {value!r}")

def configure(defaults, kinds, overrides, required=()):
    if not isinstance(overrides, dict):
        raise ValueError("Configuration overrides must be a JSON object")
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError("Unknown configuration keys: " + ", ".join(unknown))
    config = deepcopy(defaults)
    config.update({key: _convert(kinds[key], value, key) for key, value in overrides.items()})
    missing = [key for key in required if config[key] is None]
    if missing:
        raise ValueError("Missing required configuration: " + ", ".join(missing))
    return config

def command_line(commands, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not 1 <= len(argv) <= 2 or argv[0] not in commands:
        raise SystemExit("usage: {" + ",".join(commands) + "} [config.json]")
    return commands[argv[0]](json.loads(Path(argv[1]).read_text(encoding="utf-8-sig")) if len(argv) == 2 else {})
