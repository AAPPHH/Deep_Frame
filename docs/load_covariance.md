# Lastmodell als Verteilung (Sigma)

Ersetzt die Einzelsteifigkeiten (Armspitze, thrust/torsion/twist als eigene Fälle) durch eine Lastverteilung an den Schnittstellen. Crash-Richtungen sind nicht enthalten, sie bleiben eigene Nebenbedingungen.

## Definition

- 7 Schnittstellen × 6 DOF = 42: `motor_front_left`, `motor_front_right`, `motor_rear_left`, `motor_rear_right`, `stack`, `battery`, `camera`, je `Fx Fy Fz Mx My Mz`. Reihenfolge = `LOAD_COVARIANCE["interfaces"]` × `["dofs"]`.
- Einheiten N und N·mm (wie K). Koordinaten x rechts, y vorne, z oben; Lasten wirken auf den Rahmen, Momente um den Referenzpunkt der Schnittstelle (Motor: Padoberseite auf der Achse; Stack: Grommetsitz; Akku: Deckoberseite unter dem Akkuschwerpunkt; Kamera: Mitte der Seitenschraubenachse).
- Generatives Modell f = μ + B ξ, ξ unabhängig mit Mittel 0, Varianz 1. Jede Spalte von B ist ein dokumentierter Faktor (Tabelle unten); Momente entstehen aus den Kräften über die Hebel aus `COMPONENT_LIBRARY` (r × F), nicht als freie Werte.
- **Sigma = E[f fᵀ] = B Bᵀ + μ μᵀ** (zweites Moment). Nur damit gilt E[fᵀ K⁻¹ f] = tr(K⁻¹ Σ); bei zentrierter Kovarianz fiele der mittlere Schub (größte Last) heraus. Zentrierte Kovarianz `covariance` und Mittel `mean` werden getrennt mitgeliefert.
- Σ hängt nicht von der Rahmengeometrie ab (nur Komponenten und Hebel), dasselbe Σ gilt für ManaFly und Aether4 im selben Evaluator.

## API (`deep_frame/topology_problem.py`)

- `LOAD_COVARIANCE`: Konfigurations-Dict mit Faktoren, Werten, Einheiten und Quellen.
- `LoadCovariance(config)`: `.sigma`, `.mean`, `.covariance`, `.scatter` (B, 42 × 33), `.labels` [(Schnittstelle, DOF)], `.eigenvalues` (absteigend), `.rank`, `.directions` (42 × rank, Spalten = Eigenvektor · √λ, `directions @ directions.T = sigma`), `.correlation(a, b, centered)`, `.dominant()`, `.markdown()`.
- `load_covariance(config)`: Dict mit `sigma, mean, covariance, directions, eigenvalues, rank, labels, model`.
- Nutzung: Spalten von `directions` als gewichtete Lastfälle (Multi-RHS auf der vorhandenen Faktorisierung). Mittlere Compliance = Σₖ lₖᵀ K⁻¹ lₖ. Worst Case: λ_max(Lᵀ F L) hat dieselben Eigenwerte wie Σ^½ F Σ^½ (L = Σ^½ Q, Q orthogonal).

## Annahmen und Vorzeichen

- Drehrichtung Betaflight-Standard props in: FL, RR CW; FR, RL CCW (von oben). Reaktionsmoment auf den Rahmen +z bei CW, k_Q = 8,47 N·mm / 2,02 N = 4,19 mm (Schätzkette wie `FORMULATION["loads"]`). Props out kehrt nur Vorzeichen der Mz-Kopplung um.
- Diagonalpaare: `diagonal_fl_rr` (FL +, RR −) und `diagonal_fr_rl` (FR +, RL −) für Twist; `yaw_saddle` (FL, RR + / FR, RL −) für Gieren. Korrelationen Fz (Σ / zentriert): FL–RR 0,78 / 0,47; FL–FR 0,68 / 0,22; FL–Akku −0,40 / −0,17; Mx FL–FR −0,99 (Kreiselmoment, gegenläufige Rotoren).
- Kreiselmoment H × Ω mit H = J ω = 1,94 N·mm·s je Rotor, Körperrate 6 rad/s RMS → 11,6 N·mm RMS an jedem Pad.
- Akku: Nutzervorgabe 5–10 g, hier 7,5 g RMS je Achse (2,72 N); Bereich 5–10 g skaliert diesen Block mit (5/7,5)² … (10/7,5)².
- Rang 29 von 42: Varianz 0 bei stack/Mz, battery/Mz, camera/Mx, My, Mz (keine Quelle); Akku- und Stack-Momente Mx/My sind starr an Fy/Fx gekoppelt (Hebel 5,5 bzw. 6 mm).
- Eigenwerte von Σ mischen N² und (N·mm)², ihre Rangfolge ist einheitenabhängig (Momente dominieren numerisch). tr(K⁻¹Σ) und λ_max(K⁻¹Σ) sind einheiteninvariant; die Faktorisierung ist exakt.

## Spektrum

Spur Σ = 1761 (zentriert 1685). Eigenwerte (Rang 29): 543,4 ×2 (Kreiselmomente My bzw. Mx, Vorzeichen nach Drehrichtung), 235,6 ×2 (Akku Mx/My mit Fy/Fx), 116,6 (Mz ± nach Drehrichtung: Reaktionsmoment aus Mittel- und Kollektivschub), 24,8 (Mz gleichsinnig: Gier-Kommando yaw_saddle), 20,4 ×2 (Mz Diagonale), 7,57, 2,16 ×2, 1,55 ×4, … bis 1,2·10⁻³.

## Tabellen (aus `LoadCovariance().markdown()` erzeugt)

| Faktor | Wert | Einheit | Mittel | Begruendung / Quelle |
|---|---|---|---|---|
| collective | 0.583 | N je Motor | 1.01 | Schub je Motor gleichverteilt auf [0; 2,02 N] (HQProp-Schaetzung 2,02 N): Mittel 1,01 N, Streuung 2,02/sqrt(12); Fz am Pad, Reaktionsmoment Mz, Traegheit aller Massen mit a_z = 4 T / m_ges |
| diagonal_fl_rr | 0.35 | N | 0 | ANNAHME: Steuerung um die Diagonalachse FR-RL (Roll+Nick kombiniert): FL +, RR - (Twist); 0,35 N ~ 17 % Maximalschub, 3 sigma ~ halber Schub |
| diagonal_fr_rl | 0.35 | N | 0 | ANNAHME: wie diagonal_fl_rr fuer die andere Diagonale: FR +, RL - |
| yaw_saddle | 0.35 | N | 0 | ANNAHME: Gier-Kommando, gleichsinnige Diagonalpaare gemeinsam +/- (Sattel); koppelt ueber k_Q in alle vier Mz |
| thrust_scatter | 0.1 | N | 0 | ANNAHME: unabhaengige Streuung je Motor (Unwucht, Turbulenz, ESC), 5 % Maximalschub |
| h_force_x | 0.1 | N | 0 | ANNAHME: Rotor-H-Kraft/Blattschlag ~5 % Maximalschub in der Propebene, je Motor unabhaengig |
| h_force_y | 0.1 | N | 0 | ANNAHME: wie h_force_x in y |
| torque_transient | 4 | N mm | 0 | ANNAHME: Hochlauf-Reaktionsmoment je Motor; Grenze kt I = 1,19 N mm/A x ~10 A ~ 12 N mm = 3 sigma |
| body_rate_p | 6 | rad/s | 0 | ANNAHME: Rollrate RMS 6 rad/s (Spitzen ~17 rad/s = 1000 deg/s); Kreiselmoment H x Omega am Pad |
| body_rate_q | 6 | rad/s | 0 | ANNAHME: wie body_rate_p fuer die Nickrate |
| body_accel_x | 1 | g | 0 | ANNAHME: seitliche spezifische Kraft im Koerpersystem (Luftwiderstand, Boeen) 1 g RMS, gemeinsam fuer alle Massen |
| body_accel_y | 1 | g | 0 | ANNAHME: wie body_accel_x in y |
| battery_x | 7.5 | g | 0 | Nutzervorgabe Akkumasse x 5-10 g in alle Richtungen: Mitte 7,5 g als RMS je Achse, 37 g -> 2,72 N am Akkuschwerpunkt, unabhaengig je Achse; z zusaetzlich zum Schubanteil aus collective |
| battery_y | 7.5 | g | 0 | Nutzervorgabe Akkumasse x 5-10 g in alle Richtungen: Mitte 7,5 g als RMS je Achse, 37 g -> 2,72 N am Akkuschwerpunkt, unabhaengig je Achse; z zusaetzlich zum Schubanteil aus collective |
| battery_z | 7.5 | g | 0 | Nutzervorgabe Akkumasse x 5-10 g in alle Richtungen: Mitte 7,5 g als RMS je Achse, 37 g -> 2,72 N am Akkuschwerpunkt, unabhaengig je Achse; z zusaetzlich zum Schubanteil aus collective |
| stack_x | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |
| stack_y | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |
| stack_z | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |
| camera_x | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |
| camera_y | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |
| camera_z | 2 | g | 0 | Nutzervorgabe Masse x Flugbeschleunigung: lokale Manoever-/Vibrationsbeschleunigung 2 g RMS je Achse (ANNAHME), zusaetzlich zu collective und body_accel |

| Schnittstelle | DOF | Mittel | Streuung | RMS sqrt(Sigma_ii) |
|---|---|---|---|---|
| motor_front_left | Fx | 0 | 0.115 | 0.115 |
| motor_front_left | Fy | 0 | 0.115 | 0.115 |
| motor_front_left | Fz | 0.826 | 0.694 | 1.08 |
| motor_front_left | Mx | 0 | 11.7 | 11.7 |
| motor_front_left | My | 0 | 11.7 | 11.7 |
| motor_front_left | Mz | 4.23 | 5.13 | 6.65 |
| motor_front_right | Fx | 0 | 0.115 | 0.115 |
| motor_front_right | Fy | 0 | 0.115 | 0.115 |
| motor_front_right | Fz | 0.826 | 0.694 | 1.08 |
| motor_front_right | Mx | 0 | 11.7 | 11.7 |
| motor_front_right | My | 0 | 11.7 | 11.7 |
| motor_front_right | Mz | -4.23 | 5.13 | 6.65 |
| motor_rear_left | Fx | 0 | 0.115 | 0.115 |
| motor_rear_left | Fy | 0 | 0.115 | 0.115 |
| motor_rear_left | Fz | 0.826 | 0.694 | 1.08 |
| motor_rear_left | Mx | 0 | 11.7 | 11.7 |
| motor_rear_left | My | 0 | 11.7 | 11.7 |
| motor_rear_left | Mz | -4.23 | 5.13 | 6.65 |
| motor_rear_right | Fx | 0 | 0.115 | 0.115 |
| motor_rear_right | Fy | 0 | 0.115 | 0.115 |
| motor_rear_right | Fz | 0.826 | 0.694 | 1.08 |
| motor_rear_right | Mx | 0 | 11.7 | 11.7 |
| motor_rear_right | My | 0 | 11.7 | 11.7 |
| motor_rear_right | Mz | 4.23 | 5.13 | 6.65 |
| stack | Fx | 0 | 0.158 | 0.158 |
| stack | Fy | 0 | 0.158 | 0.158 |
| stack | Fz | -0.233 | 0.195 | 0.304 |
| stack | Mx | 0 | 0.947 | 0.947 |
| stack | My | 0 | 0.947 | 0.947 |
| stack | Mz | 0 | 0 | 0 |
| battery | Fx | 0 | 2.75 | 2.75 |
| battery | Fy | 0 | 2.75 | 2.75 |
| battery | Fz | -1.2 | 2.81 | 3.05 |
| battery | Mx | 0 | 15.1 | 15.1 |
| battery | My | 0 | 15.1 | 15.1 |
| battery | Mz | 0 | 0 | 0 |
| camera | Fx | 0 | 0.0504 | 0.0504 |
| camera | Fy | 0 | 0.0504 | 0.0504 |
| camera | Fz | -0.0743 | 0.0623 | 0.097 |
| camera | Mx | 0 | 0 | 0 |
| camera | My | 0 | 0 | 0 |
| camera | Mz | 0 | 0 | 0 |

| Eigenwert | Anteil | dominante Eintraege |
|---|---|---|
| 543.4 | 30.9% | motor_front_left/My +0.49, motor_rear_right/My +0.49, motor_rear_left/My -0.49, motor_front_right/My -0.49 |
| 543.4 | 30.9% | motor_rear_left/Mx -0.49, motor_front_left/Mx +0.49, motor_front_right/Mx -0.49, motor_rear_right/Mx +0.49 |
| 235.6 | 13.4% | battery/Mx -0.98, battery/Fy +0.18, stack/Mx -0.00, motor_front_left/Mx -0.00 |
| 235.6 | 13.4% | battery/My +0.98, battery/Fx +0.18, stack/My +0.00, motor_rear_right/My +0.00 |
| 116.6 | 6.6% | motor_front_left/Mz -0.49, motor_rear_left/Mz +0.49, motor_front_right/Mz +0.49, motor_rear_right/Mz -0.49 |
| 24.79 | 1.4% | motor_front_left/Mz +0.50, motor_front_right/Mz +0.50, motor_rear_left/Mz +0.50, motor_rear_right/Mz +0.50 |
| 20.36 | 1.2% | motor_rear_left/Mz +0.71, motor_front_right/Mz -0.71, motor_rear_left/Fz -0.04, motor_front_right/Fz +0.04 |
| 20.36 | 1.2% | motor_rear_right/Mz +0.71, motor_front_left/Mz -0.71, motor_front_left/Fz -0.04, motor_rear_right/Fz +0.04 |
| 7.567 | 0.4% | battery/Fz +0.99, motor_front_left/Mz +0.07, motor_rear_right/Mz +0.07, motor_front_right/Mz -0.07 |
| 2.158 | 0.1% | motor_front_left/Mx -0.48, motor_front_right/Mx -0.48, motor_rear_left/Mx -0.48, motor_rear_right/Mx -0.48 |

## Grenzen und Kalibrierung

- Maß: voll gekoppelte Evaluator-Flexibilität (42 × 42, `frame_evaluation.sigma`), Gruppe `stack_fixed` (36 × 36), unskaliert. Nutzerentscheidung 03.10.: **ManaFly 3 mit 20 % Reserve** (`LOAD_COVARIANCE_LIMITS["reference"]`, `["reserve"]`); Aether4 nur Vergleichsspalte. Erzeugt von `tools/evaluate_frame.py limits` als Variante `evaluator_full` (`exports/cov/limits.md`).
- Grenzen: tr(ΣF) ≤ 0,8 × 0,8590 = **0,687 N mm**, λmax(Σ^½ F Σ^½) ≤ 0,8 × 0,2858 = **0,229 N mm**. Die Neun-Kriterien-Zeile des Evaluators vergleicht damit dasselbe volle Maß (vorher Grenze aus dem diagonalen Gap-Maß).
- Optimierer: Grenze × Kalibrierfaktor je Schlüssel (`topology_problem.COVARIANCE["limits"]["calibration"]`) = Optimierer-Maß (erodiertes Voxelfeld) / Evaluator-Maß (Rohkörper) am selben Entwurf. Iteration: Lauf → Evaluator → Faktor neu; weiterer Lauf, solange sich ein Faktor um mehr als 10 % ändert.

| Iteration | Entwurf | Optimierer tr / λmax N mm | Evaluator tr / λmax N mm | Faktor tr / λmax | Evaluator-Grenze | Optimierer-Grenze |
|---|---|---|---|---|---|---|
| 0 | SIMP+MMA 17,2 g (Feld) / 14,6 g roh | 0,8551 / 0,1967 | diag. 1,437 / 0,501 (voll 1,163 / 0,287) | 0,5952 / 0,3928 (voll 0,735 / 0,685) | 0,390 / 0,168 (Aether4 skaliert, diag.) | 0,232 / 0,066 |
| 1 | Lastmodell 29,4 g (Feld) / 26,6 g roh (baff3e1, noch mit Zubehörsitzen) | 0,2322 / 0,05851 | voll 0,1862 / 0,04739 | 1,247 / 1,235 | 0,687 / 0,229 | 0,857 / 0,283 |
| 2 | Lastmodell ohne Sitze, Lauf mit Faktor 1: 18,5 g (Feld) / 15,9 g roh (simp_mma_cov2) | 0,8563 / 0,2574 | voll 0,6189 / 0,1873 | **1,384 / 1,374** | **0,687 / 0,229** | **0,951 / 0,315** |
| 3 | Lastmodell ohne Sitze, Lauf mit Faktor 2: 18,2 g (Feld) / 15,7 g roh (simp_mma_cov3) | 0,9467 / 0,3081 | voll 0,6870 / 0,2197 | 1,378 / 1,402 | 0,687 / 0,229 | (0,947 / 0,321) |

Faktor 0 → 1 gegen das volle Maß: +70 % / +80 % (> 10 %, nächster Lauf mit Faktor 1 nötig, danach erneut kalibrieren). Faktor 1 → 2: +11,0 % / +11,2 % (knapp > 10 %, dritter und letzter Lauf mit Faktor 2). Der 15,9-g-Rohkörper liegt im Evaluator bei 90 % / 82 % der Grenzen. Faktor 2 → 3: −0,4 % / +2,1 % (< 10 %): **Kalibrierung stabil**, Faktor 2 bleibt im Code (er hat den Endlauf erzeugt). Der 15,7-g-Rohkörper des dritten Laufs trifft die Grenzen mit 100,0 % / 96 % (tr aktiv, wie im Optimierer). Der 26,6-g-Rahmen liegt im Evaluator bei 27 % / 21 % der neuen Grenzen, war also etwa 3,7× / 4,8× steifer als nötig.

## Freie Akkulagerung (`BATTERY_SUPPORT`, Schalter `TOPOLOGY_CONFIG["battery_support"]`)

Standard bleibt `"rails"` (zwei Schienen als feste Bereiche). Mit `"free"` (Auftrag: `overrides.battery.support = "free"`, Formulierung: `FORMULATION["layout"]`) entfallen die Schienen; der Akku ist ein Starrkörper, der Rahmen trägt ihn über dichteabhängige Federn an seiner Unterseite. Einziges Haltemodell ist `retention = "ideal_press"` (Nutzerentscheidung 5. Oktober): Der Akku gilt unter jeder Last als ideal auf seinen Sitz gepresst. Gummiband, Vorspannung, Reibung und Einlegehilfe sind nicht modelliert; das frühere einseitige Kontaktmodell mit Bandvorspannung ist entfernt. `camera.near_ground` setzt diese freie Lagerung voraus (`FrameLayout.patch`).

- Körper: 37 g, Quader 30 × 63 × 11 mm, Eigenträgheit als Quader, Volumen = Keep-out `battery_envelope`. Referenzpunkt Unterseitenmitte (0, y_b, deck_top) wie der Sigma-Akkublock; Schwerpunkt 5,5 mm darüber. Das senkrechte Einschubvolumen `battery_insertion` bleibt verboten (kein Dach).
- Federn nur unten: Kontaktknoten sind die Knoten zwischen erlaubten Zellen und der Unterseite von `battery_envelope`. Je Knoten wirken eine Normalenfeder z und Schubfedern x/y, alle bilateral (ideal angepresst, kein Active-Set). Steifigkeit: Knotenfläche × Pad (Normal 1,0 N/mm³, Schub 0,3 N/mm³, ANNAHME Haftpad) × (floor + (1 − floor) ρ³). ρ ist das Mittel der anliegenden erlaubten Zellen im erodierten Feld.
- Designabhängige Last: Akkuwrench w → s = D⁻¹ w mit D = Σ T_jᵀ K_j T_j; die Knotenkräfte sind f_j = K_j T_j s (Einweg-Kopplung: Verteilung auf starrem Rahmen, Rahmenantwort mit diesen Kräften). Das gilt für die sechs Akkuspalten von Sigma und für jeden Inertia-Relief-Fall, alle sechs Wrenchkomponenten einschließlich Abheben und Kippen über die Unterseite. Die Akkuträgheit und die crash_back-Decklast wirken auf den Körper, nicht auf Rahmenknoten.
- Nebenbedingungen:
  - Federweg am Akkuschwerpunkt in x/y ≤ 0,5 mm: im Flug 3σ aus dem Sigma-Akkublock, im Crash |δ_xy| je Crashfall;
  - Auflagefläche unten Σ A_j ρ_j ≥ 350 mm² (`min_area_mm2`, Fläche der Schienenvariante 2 × 3,5 × 50 mm). Dieselbe Zahl gilt in der Abnahme (`tools/functional_geometry_review.py`: STL-Schnitt 0,05 mm unter der Akkuunterseite, 0,1-mm-Raster, Schwerpunkt innerhalb der konvexen Hülle der Auflage, freies senkrechtes Einlegen); beide Maße sind verschieden diskretisiert;
  - der Massenanteil des Akkus in der Modalanalyse liegt nach k_z verteilt auf den Unterseitenknoten.
- Sensitivitäten: d(uᵀf)/dk_jd = [T_j s]_d ([u_j]_d − [T_j r]_d) mit r = D⁻¹ Σ T_jᵀ K_j u_j. Die Kompatibilitäten erhalten 2 × diesen Term; f1 erhält −λ φᵀ(dM_lumped)φ.
- Prüfung:
  - `tests/test_topology_problem.py::test_ideal_press_battery_support_equilibrium_and_gradients` (Gleichgewicht Σ T_jᵀ f_j = w, FD aller Zeilen auf dem Minigitter);
  - `tools/formulation_study.py battery_fd` (FD auf dem Rahmengitter 68 × 64 × 24 → `docs/validation/battery_support_fd.json`; der gespeicherte Datensatz stammt noch vom entfernten Kontaktmodell und ist bei der nächsten Ausführung neu zu erzeugen).

## Freier Kamerakäfig (`CAMERA_SUPPORT`, Schalter `TOPOLOGY_CONFIG["camera_support"]`)

Standard bleibt `"prescribed"` (Bügelpfad, Laschen, Aufprallkontakte als feste Bereiche). Mit `"free"` (Auftrag: `overrides.camera.support = "free"`) entfallen Bügel, Laschen und Aufprallkontakte; die Kamera ist ein Starrkörper wie der freie Akku.

- Körper: Lux 2,3 g, Quader 16 × 14 × 14 mm, um `tilt_deg` gekippt, Eigenträgheit gedreht. Volumen = Keep-out `camera_envelope`, Referenzpunkt Mitte der Schraubachse (wie der Sigma-Kamerablock).
- Kopplung: bilaterale Federn (normal x, Schub y/z) an den Knoten zwischen erlaubten Zellen und Kamera-Keep-out auf beiden Seitenflächen innerhalb 4 mm um die Schraubachse. k = Fläche × Pad (ANNAHME 1000 / 400 N/mm³) × (floor + (1 − floor) ρ³).
- Sichtfeld 4:3 (126° × 94°) als Keep-out: Pyramidenstumpf ab der Kamerafront (Öffnung ±3 mm) entlang der gekippten Achse, jede Zelle mit einer Ecke im Stumpf + 0,5 mm. `camera_front_access` (Einschub) bleibt.
- Crash als designabhängige Last: `crash_front` (−y), `crash_below` (+z) und neu `crash_camera_oblique` (45° von vorn links) wirken mit voller Crashkraft auf die Zone vor und über der Kamera (x ± (Keep-out + 6 mm), y ab Kameramitte bis zur Hüllenfront, z ab Kameramitte bis Keep-out-Oberkante + 6 mm). Verteilung f_e = F w_e / Σw mit w_e = floor + (1 − floor) ρ_e. Das Inertia Relief wird je Auswertung aus Kraft und Moment der aktuellen Verteilung neu gebildet (linear im Lastvektor, Massen fest), einschließlich der Körperwrenches von Akku und Kamera.
- Nebenbedingungen:
  - Kameraverschiebung relativ zum Rahmen |δ| ≤ Grenze je Fall `crash_front`, `crash_camera_oblique`. δ ist die Starrkörperbewegung der Kamera auf ihren Federn am verformten Rahmen, s = D⁻¹(w + Σ Tᵀ K u), minus dem Starrkörper-Ausgleich der Stack-Montage (`aio_contact_`), ausgewertet am Kameraschwerpunkt. Die Sensitivität kommt aus drei adjungierten Lastfällen je Crashfall.
  - Frontabdeckung ≥ Grenze: projizierte Materialfläche vor der Kamerafront entlang achsparalleler Strahlen der gekippten Achse im Fenster Front ± (B/2 + 6 mm) × [−H/2, H/2 + 6 mm]. Je Strahl gilt 1 − Π(1 − ρ_s) (Abtastung h/2, trilinear, Zwischenfeld), bezogen auf die Frontfläche B × H. Werte über 1 sind möglich, weil das Fenster größer ist als die Front.
  - Montagefläche Σ A ρ ≥ Grenze.
  - Die Kameramasse liegt in der Modalanalyse nach k verteilt auf den Federknoten.
  - Die Kamera-Sigma-Spalten wirken wie beim Akku über die Federn.
- Grenzen: aus ManaFly mit denselben Definitionen gemessen (Abschnitt unten). Ohne gesetzte Grenze werden die Zeilen nur überwacht.
- Prüfung:
  - `tests/test_topology_problem.py::test_free_camera_support_equilibrium_and_gradients` (Gleichgewicht der Zonenlast mit Relief, FD aller Zeilen);
  - `tools/formulation_study.py camera_fd` (Rahmengitter).

### Grenzen des freien Kamerakäfigs aus ManaFly (`formulation_study.py camera_limits` → `docs/validation/camera_limits_manafly.json`)

Messaufbau für ManaFly 3:
- dieselben Definitionen und derselbe Code wie im Optimierer;
- Halbmodell, binäre 4/3-mm-Voxel (Volumenanteil ≥ 50 %, größte flächenverbundene Komponente; 11 Inselzellen verworfen);
- Inertia Relief über Rahmenmasse (29,3 g), AIO, Motoren + Props, Akku (Punktmasse am Deck) und Kamera;
- Lux in der Evaluator-Platzierung (0, 50, 16), 20° gekippt;
- Federn an den Innenflächen der Seitenplatten (x = 10,67 mm); der Spalt zwischen Kameraseite und Platte gilt als starr überbrückt (Schraube/Distanz).

| Größe | ManaFly 20° | ManaFly 0° | Grenze | Akku-Analogon |
|---|---|---|---|---|
| Kameraverschiebung frontal (mm) | 0,0866 | 0,0873 | ≤ 0,0866 | 0,5 |
| Kameraverschiebung schräg 45° (mm) | 0,3816 | 0,3926 | ≤ 0,3816 | 0,5 |
| Frontabdeckung (× Frontfläche 16 × 14 mm) | 0,4227 | 0,4071 | ≥ 0,4227 | – |
| Montagefläche (Anteil des 4-mm-Patches) | 1,00 | 1,00 | ≥ 1,0 | – |

Im reinen Silhouettenfenster der Kamera hat ManaFly 0 % Abdeckung, weil das Sichtfeld und der Einschubkorridor das verbieten. Das Maß zählt deshalb das Fenster mit 6 mm Rand seitlich und oben. Werte über 1 sind möglich.

Crash-Referenzen der Zonenfälle: Nachgiebigkeit der Formulierungs-Referenzdichte (r4_neural_v06_f1) im Freikamera-Gebiet unter den Zonenlasten. Ergebnis: `crash_front` 10,09 N mm (alte Patch-Definition 6,18), `crash_below` 103,0 N mm (alt 27,52). Die Grenze ist wie bisher 1,5 × Referenz. `crash_camera_oblique` hat keine Nachgiebigkeitszeile, nur die Kameraverschiebung.
