from copy import deepcopy
import json
from pathlib import Path
import sys

COMPONENT_DEFAULTS = {
    "aio15": {
        "width_mm": 31.3,
        "length_mm": 31.3,
        "stack_height_mm": 6.0,
        "mount_pitch_mm": 25.5,
        "screw_diameter_mm": 2.0,
        "mass_g": 7.2,
        "source": "user: HDZero AIO15 dimensions, M2 mounting pattern and mass",
        "parameter_sources": {
            "stack_height_mm": "PROVISIONAL: complete populated board envelope; measure actual stack",
        },
    },
    "camera": {
        "length_mm": 14.0,
        "width_mm": 16.0,
        "height_mm": 14.0,
        "tilt_deg": 20.0,
        "mass_g": 2.3,
        "source": "user: HDZero Lux dimensions and mass",
        "parameter_sources": {"tilt_deg": "design: adjustable initial camera tilt"},
    },
    "battery": {
        "length_mm": 63.0,
        "width_mm": 30.0,
        "height_mm": 11.0,
        "mass_g": 37.0,
        "source": "user: GNB5502S120A dimensions and mass",
        "mass_scope": "battery including leads and connectors; represented at battery center",
    },
    "motor": {
        "diameter_mm": 14.2,
        "height_mm": 14.6,
        "mount_pitch_mm": 9.0,
        "mount_layout": "bolt_circle",
        "screw_diameter_mm": 2.0,
        "screw_clearance_mm": 2.2,
        "mass_g": 5.9,
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
    "prop": {
        "diameter_mm": 65.0,
        "thickness_mm": 0.8,
        "mass_g": 0.7,
        "source": "user: 65 mm swept disk; intentionally larger than exact 2.5 inch (63.5 mm)",
        "parameter_sources": {
            "thickness_mm": "PROVISIONAL: swept disk thickness; excludes blade flex",
            "mass_g": "PROVISIONAL: prop model not selected; equivalent uniform disk inertia",
        },
    },
    "xt30": {
        "width_mm": 10.2,
        "length_mm": 12.4,
        "height_mm": 5.2,
        "mass_g": 0.0,
        "source": "https://images.100y.com.tw/pdf_file/AMASS-XT30U.pdf#page=2",
        "model": "AMASS XT30U-F bounding envelope",
        "mass_scope": "already_in_battery",
        "parameter_sources": {
            "mass_g": "battery mass includes leads and connectors; avoid double counting",
        },
    },
    "balancer": {
        "width_mm": 9.8,
        "length_mm": 7.5,
        "height_mm": 5.7,
        "pins": 3,
        "mass_g": 0.0,
        "source": "https://www.jst-mfg.com/product/pdf/eng/eXH.pdf#page=4",
        "model": "PROVISIONAL: JST XHP-3 housing envelope; verify actual GNB connector",
        "mass_scope": "already_in_battery",
        "parameter_sources": {
            "pins": "user: 3-pin balance connector",
            "mass_g": "battery mass includes leads and connectors; avoid double counting",
        },
    },
}

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
 'deck_top_mm': 29.0,
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
 'prop_motor_gap_mm': 2.0}

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
 'deck_top_mm': {'value': 29.0,
                 'kind': 'design_assumption',
                 'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                 'principle_ids': ['deck_load_path',
                                   'functional_voids',
                                   'component_driven'],
                 'rationale': 'Eigene OberkanteZ29; Boardunterseite5.5 plus '
                              'Stack6, Deckunterseite26.5. Gecko22mm hat '
                              'unklaren Hoehenbezug und wird nicht kopiert.'},
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
 'prop_motor_gap_mm': {'value': 2.0,
                       'kind': 'design_assumption',
                       'frame_ids': ['tadpole_hd_3', 'tadpole_2_5'],
                       'principle_ids': ['deck_load_path',
                                         'functional_voids',
                                         'component_driven'],
                       'rationale': 'Aus AIO15- und Akkuhuelle, Stegbreiten '
                                    'und2-mm-Mindestwand abgeleitete eigene '
                                    'Druckabmessung.'}}

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

INTEGRATION_CONFIG = {
    "model_version": "frame-v0-linear-fixtures-v1",
    "arm_tip_force_n": 1.0,
    "arm_tip_motor": "front_left",
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
    "motor_contact_radius_mm": 9.5,
    "aio_contact_radius_mm": 3.2,
    "contact_depth_mm": 4.0,
    "camera_contact_width_mm": 6.0,
    "camera_tool_radius_mm": 2.0,
    "camera_contact_length_mm": 8.0,
    "battery_contact_width_mm": 8.0,
    "battery_contact_length_mm": 8.0,
    "battery_contact_y_mm": 20.0,
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
    },
    "reconstruction": {
        "density_threshold": 0.35,
        "minimum_feature_mm": 2.0,
    },
    "additional_regions": [],
}

IMPLICIT_CONFIG = {
    "subdivisions": 10,
    "interpolation_method": "pchip",
    "thresholds": [0.25, 0.35],
    "extensions": ["preserve", "preserve_forbidden"],
    "density_sigma_mm": 0.4,
    "transition_radius_mm": 2.0,
    "preserve_inflation_mm": 0.3,
    "constraint_offset_mm": 0.3,
    "opening_radius_mm": 1.35,
    "ripple_sigma_mm": 0.25,
    "reinit_band_cells": 1.5,
    "remesh_target_mm": 0.6,
    "remesh_iterations": 5,
    "remesh_feature_deg": 60.0,
    "remesh_max_surface_distance_mm": 0.05,
    "segment_tolerance_mm": 0.01,
    "surface_deviation_mm": 0.20,
    "relative_volume_change": 0.01,
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
    "tet_attempts": ["classify_hxt", "classify_delaunay", "direct_hxt"],
    "mesh_minimum_sicn": 0.01,
    "mesh_boundary_deviation_mm": 0.05,
    "mesh_timeout_s": 900.0,
    "solver_timeout_s": 900.0,
    "candidate_timeout_s": 1800.0,
}

IMPLICIT_KINDS = {
    **{key: "float" if isinstance(value, float) else "int" for key, value in IMPLICIT_CONFIG.items() if isinstance(value, (int, float))},
    "interpolation_method": ("pchip", "cubic"),
    "thresholds": ["float"],
    "extensions": [("none", "preserve", "preserve_forbidden")],
    "tet_attempts": [("classify_hxt", "classify_delaunay", "direct_hxt")],
}

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
    elif kind == "text":
        if isinstance(value, str):
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
