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
