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
