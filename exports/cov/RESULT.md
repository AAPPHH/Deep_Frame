# Frame mit Lastmodell als Verteilung, Kalibrierschleife (SIMP+MMA, 4/3 mm)

Stand 04.10. 04:10. Kalibrierschleife abgeschlossen und stabil (3 Läufe). Ergebnis ist der **Rohkörper 15,7 g** (`simp_mma_cov3_raw_1`). **Rekonstruktion v3 steht aus:** recon3 hat um 02:13 den Nachweis 879713d committet, aber SIMP+MMA liegt dort nicht innerhalb 10 % vom Rohkörper (Masse +12,4 %, Armspitze −19,0 %, f1 −9,4 %; nur neural_v06_f1 besteht). Nach Regel wird gewartet; der 1:1-Körper ist **nicht** das Ergebnis, nur Vergleich. Bilder mit v3 und das v3-STL fehlen deshalb.

## Zur Frage "wieso sind die Halter noch dran?"

Der 26,6-g-Rahmen stammt aus Lauf baff3e1. Der wurde vor dem Merge 2bafe8f gestartet und hatte die Sitze für XT30, Balancer und VTX-Antenne noch als feste Bereiche. Alle Läufe hier laufen ohne Sitze:

- Freie Zellen grob 33 908 statt 32 524.
- In den Renders gibt es kein Antennenauge hinten in der Mitte und keine Steckersitze mehr.

Nebenwirkung: Das Antennenauge war der einzige feste Bereich auf der Symmetrieebene. Daran hing die Lagerung der Inertia-Relief-Fälle. Ohne Auge brach der Aufbau ab (leeres argmin).

- Ein Ausweichen auf Knoten im Leerbereich lief zwar, der cuDSS-Lauf scheiterte aber an der Residuenprüfung (Residuum 1e-6).
- Ersetzt durch eine Lagerung abseits der Ebene: drei Preserve-Knoten mit eigenen Freiheitsgraden für den symmetrischen und den antisymmetrischen Teil (aa782ed, 86b9480).
- Getestet: Halb- und Vollmodell stimmen auf 1e-8 überein, Reaktionen ≈ 0, unter `preserve_adjacent`.
- Die zusätzliche Invarianzprüfung am ganzen Rahmen (Ray-Job `relief_check`) wurde vom Hintergrund-Zeitlimit abgebrochen. Sie prüfte noch die verworfene Leerbereichsvariante.

## Kalibrierschleife

Grenzen im Evaluator (volles Maß, ManaFly 3 × 0,8): tr ≤ 0,687 N mm, λmax ≤ 0,229 N mm. Faktor = Optimierer-Maß / Evaluator-Maß am selben Entwurf. Alle Läufe starten vom 26,6-g-Feld (baff3e1). Gleicher Start in jeder Iteration, damit sich nur die Grenze ändert. Das 26,6-g-Feld ist zulässig und liegt über dem Ziel, MMA trägt also nur Material ab.

| Lauf | Faktor im Lauf tr / λmax | Optimierergrenze N mm | Feld / roh | Optimierer tr / λmax | Evaluator voll tr / λmax | neuer Faktor | Änderung |
|---|---|---|---|---|---|---|---|
| baff3e1 (mit Sitzen) | 0,5952 / 0,3928 (diag.) | 0,232 / 0,066 | 29,4 / 26,6 g | 0,2322 / 0,0585 | 0,1862 / 0,0474 | 1,247 / 1,235 | +70 % / +80 % |
| 2: simp_mma_cov2 | 1,247 / 1,235 | 0,857 / 0,283 | 18,45 / 15,9 g | 0,8563 / 0,2574 | 0,6189 / 0,1873 | 1,384 / 1,374 | +11,0 % / +11,2 % |
| **3: simp_mma_cov3** | 1,384 / 1,374 | 0,951 / 0,315 | **18,22 / 15,7 g** | 0,9467 / 0,3081 | **0,6870 / 0,2197** | 1,378 / 1,402 | **−0,4 % / +2,1 %: stabil** |

Lauf 2 lag knapp über der 10-%-Schwelle, daher Lauf 3. Lauf 3 bestätigt den Faktor; im Code bleibt 1,384 / 1,374 (er hat den Endlauf erzeugt). Der Rohkörper trifft tr mit 100,0 % der Grenze (0,6870 bei 0,6872) und λmax mit 96 %. Damit ist die Kalibrierung geschlossen.

- Lauf 3: grob 68x64x24, 183 Iterationen; fein 102x96x24, 87 Iterationen, beide `converged`, Graufraktion 0,09 %.
- Laufzeit Lauf 3 etwa 1 h 50 min. Fein 45 s/It. statt 13 s/It. im 26,6-g-Lauf, weil der Rechner während der Läufe 0 GB freien Arbeitsspeicher hatte (Ray-Typ `density_simp`, 4 Kerne, 8 GB, 10 GPU-GB).
- Dazu kamen zwei gescheiterte Starts wegen der Lagerung (siehe oben). Der Zeitrahmen von 3,5 h (bis 02:50) wurde deshalb um gut 1 h überschritten.

## Masse

| | SIMP+MMA 17,2 g (heute früh) | Lastmodell 1 (baff3e1, mit Sitzen) | Iteration 2 | **Endlauf (Iteration 3)** |
|---|---|---|---|---|
| Optimierer (Feld) | 17,18 g | 29,36 g | 18,45 g | **18,22 g** |
| Rohkörper (STL) | 14,56 g | 26,56 g | 15,88 g | **15,72 g** |
| 1:1 (nur Vergleich) | 16,39 g | 26,34 g | – | 16,28 g |
| recon v3 | – | – | – | ausstehend |

## Nebenbedingungen Endlauf (Optimierer, fein)

| constraint | value | limit | unit | status | margin |
|---|---|---|---|---|---|
| load_mean (tr) | 0.9467 | <= 0.9508 | N mm | active | +0.4 % |
| load_worst (λmax) | 0.3081 | <= 0.3146 | N mm | satisfied | +2.1 % |
| crash_front | 4.085 | <= 9.268 | N mm | satisfied | +55.9 % |
| crash_side_left/right | 13.78 | <= 20.44 | N mm | satisfied | +32.6 % |
| crash_arm_front_left | 102.1 | <= 211.6 | N mm | satisfied | +51.8 % |
| crash_arm_front_right | 33.84 | <= 33.81 | N mm | active | -0.1 % |
| crash_arm_rear_left | 65.64 | <= 65.64 | N mm | active | +0.0 % |
| crash_arm_rear_right | 96.69 | <= 108.2 | N mm | satisfied | +10.6 % |
| crash_below | 14.07 | <= 41.28 | N mm | satisfied | +65.9 % |
| crash_back | 7.693 | <= 10.41 | N mm | satisfied | +26.1 % |
| f1 | 310.6 | >= 300 | Hz | satisfied | +7.1 % |
| volume | 0.0446 | <= 0.1 | - | satisfied | +55.4 % |
| shadow | 0.2552 | <= 0.3137 | mm | satisfied | +18.7 % |

Aktiv sind tr und zwei Arm-Crash-Fälle. f1 hat 7 % Reserve; die robuste Mindestbreite 2,5 mm und die Spiegelsymmetrie sind im Feld erzwungen. Die Arm-Crash-Fälle bremsen die Massenabnahme: Lauf 2 → 3 hebt die tr-Grenze um 11 %, die Masse sinkt aber nur um 0,2 g.

Überwacht (alte Lastfälle), 17,2 g → 26,6 g → Endlauf: thrust_all 2,77 → 0,96 → 4,25 N mm, torsion_yaw 0,130 → 0,039 → 0,082 N mm, twist 8,49 → 1,64 → 5,18 N mm, f1 intermediär 332 → 337 → 375 Hz.

## Lastmodell-Maße gegen die ManaFly-Grenzen (Evaluator, Gruppe stack_fixed)

| Körper | Masse | tr voll | λmax voll | tr diag. | λmax diag. | gegen 0,687 / 0,229 |
|---|---|---|---|---|---|---|
| SIMP+MMA 17,2 g roh | 14,6 g | 1,163 | 0,287 | 1,437 | 0,501 | verfehlt (169 % / 125 %) |
| Lastmodell 26,6 g roh | 26,6 g | 0,186 | 0,047 | 0,232 | 0,086 | erfüllt (27 % / 21 %) |
| Iteration 2 roh | 15,9 g | 0,619 | 0,187 | 0,728 | 0,266 | erfüllt (90 % / 82 %) |
| **Endlauf roh** | **15,7 g** | **0,687** | **0,220** | **0,809** | **0,305** | **erfüllt (100,0 % / 96 %)** |
| Endlauf 1:1 (Vergleich) | 16,3 g | – | – | – | – | nicht gerechnet: Tet-Vernetzung scheiterte in allen 6 Versuchen |
| Endlauf recon v3 | | ausstehend | | | | |
| ManaFly 3 | | 0,859 | 0,286 | 1,248 | 0,579 | Referenz (Grenze = × 0,8) |
| Aether4 (unskaliert) | | 0,391 | 0,143 | 0,528 | 0,242 | nur Vergleich |

## Steifigkeit je Schnittstelle und Richtung gegen ManaFly

Quelle ist der Evaluator-Teil `sigma`, also dieselben Einheitslasten und dieselbe Lagerung wie der Gap-Finder. Er liefert bitgleich die Gap-Finder-Werte; der Gap-Finder selbst wurde nicht neu gestartet. Einheiten: F in N/mm, M in N mm/rad. Die rechten Motoren sind spiegelgleich. Die volle Tabelle mit allen Körpern steht in `comparison.json` → `stiffness_table`.

| Schnittstelle/Richtung | SIMP 17,2 g roh | Lastmodell 26,6 g roh | Endlauf 15,7 g roh | ManaFly 3 | Aether4 | Endlauf/ManaFly |
|---|---|---|---|---|---|---|
| motor_front_left Fx | 44.48 | 137.8 | 59.31 | 64.89 | 88.82 | 0.91 |
| motor_front_left Fy | 3.589 | 20 | 11.68 | 54.81 | 62.9 | 0.21 |
| motor_front_left Fz | 8.466 | 51.77 | 9.724 | 6.564 | 12.43 | 1.48 |
| motor_front_left Mx | 1503 | 11730 | 3432 | 2917 | 20040 | 1.18 |
| motor_front_left My | 2953 | 17770 | 4238 | 4710 | 21050 | 0.90 |
| motor_front_left Mz | 2307 | 13540 | 7714 | 21470 | 126700 | 0.36 |
| motor_rear_left Fx | 32.2 | 75.96 | 29.89 | 31.8 | 125.6 | 0.94 |
| motor_rear_left Fy | 10.33 | 19.11 | 9.271 | 14.72 | 135 | 0.63 |
| motor_rear_left Fz | 7.431 | 44.69 | 19.21 | 6.295 | 21.79 | 3.05 |
| motor_rear_left Mx | 2310 | 12660 | 3737 | 5673 | 35130 | 0.66 |
| motor_rear_left My | 2920 | 19780 | 4709 | 8700 | 29150 | 0.54 |
| motor_rear_left Mz | 5087 | 10890 | 6208 | 13870 | 69600 | 0.45 |
| stack Fx | 308.7 | 2581 | 407 | 433.7 | 2606 | 0.94 |
| stack Fy | 208.4 | 1526 | 162.6 | 683.5 | 1874 | 0.24 |
| stack Fz | 113.3 | 459.2 | 209.8 | 94.32 | 310.4 | 2.22 |
| stack Mx | 21370 | 193900 | 40120 | 34340 | 92730 | 1.17 |
| stack My | 24390 | 275900 | 58390 | 18980 | 113100 | 3.08 |
| stack Mz | 104100 | 500800 | 72830 | 405100 | 1943000 | 0.18 |
| battery Fx | 50.14 | 446.1 | 117.5 | 49.73 | 123.7 | 2.36 |
| battery Fy | 89.83 | 692.4 | 193.7 | 88.2 | 132.4 | 2.20 |
| battery Fz | 229.7 | 1405 | 439.1 | 406.9 | 246.1 | 1.08 |
| battery Mx | 21490 | 37640 | 24910 | 21920 | 8273 | 1.14 |
| battery My | 49490 | 280900 | 88680 | 43910 | 74550 | 2.02 |
| battery Mz | 40480 | 301900 | 71600 | 140900 | 130700 | 0.51 |
| camera Fx | 88.83 | 19.84 | 48.6 | 27.62 | 46.88 | 1.76 |
| camera Fy | 113.4 | 387.4 | 210.1 | 60.16 | 42.03 | 3.49 |
| camera Fz | 92.69 | 78.67 | 148.2 | 42.01 | 41.06 | 3.53 |
| camera Mx | 29310 | 13080 | 14000 | 25290 | 30570 | 0.55 |
| camera My | 27300 | 10460 | 29040 | 19130 | 58110 | 1.52 |
| camera Mz | 42320 | 99440 | 58440 | 55870 | 86930 | 1.05 |

Die Lasten, die Σ stark gewichtet, liegen bei oder über ManaFly:

- Motor-Fz 1,5–3,1×.
- Akku Fx/Fy/My mindestens 2×.

Weit unter ManaFly bleiben die Richtungen, die Σ kaum gewichtet:

- Motor-Fy 0,2–0,6×.
- Motor-Mz 0,4×.
- Stack Fy/Mz 0,2×.
- Akku Mz 0,5×.

## Verwindung (Diagonalpaare ±1 N Fz, Stack fest)

| Körper | Nachgiebigkeit N mm | Steifigkeit N/mm |
|---|---|---|
| SIMP 17,2 g roh | 0,589 | 6,8 |
| Lastmodell 26,6 g roh | 0,086 | 46,3 |
| Iteration 2 roh | 0,282 | 14,2 |
| **Endlauf roh** | **0,328** | **12,2** |
| ManaFly 3 | 0,236 | 16,9 |
| Aether4 (unskaliert) | 0,120 | 33,2 |

Der Endlauf ist um den Faktor 1,8 verwindungssteifer als der 17,2-g-Rahmen, liegt aber bei 72 % von ManaFly. Verwindung ist im Σ-Modell nur ein Teil der Differenzlast; die Grenze gilt für tr und λmax, nicht für diese Einzelzahl.

## Neun Kriterien (Rohkörper = Ergebnis; 1:1 nur Vergleich)

- **Endlauf roh** (`exports/runs/simp_mma_cov3_raw_1`):
  - Werte: 15,7 g; Armspitze 10,9 N/mm (17,2 g: 8,6); Σ tr 0,687 / λmax 0,220 N mm; f1 407 Hz; Spannungen front 6,5 / arm 14,2 / back 1,8 MPa (SF 2).
  - Verfehlt: bolt_patterns, keep_outs_free, wall_deep_fraction, wall_deep_component, target:symmetry.
  - Warnungen: loops, cog_offset.
  - Wandregel: tiefer Anteil 1,14 %, größte Komponente 49,3 mm³, Motorzonen 0 (17,2 g: 3,09 % / 230 mm³; 26,6 g: 0,86 % / 19,3 mm³).
  - FEA-Oberfläche: Warnung, 0,27 mm > 0,20 mm.
  - Die Fehlpunkte sind dieselben wie beim 26,6-g-Rohkörper. Bohrbilder und Keep-outs verfehlt der Rohkörper grundsätzlich; das soll die Rekonstruktion (v3) liefern.
- **Endlauf 1:1** (`simp_mma_cov3_recon_1`, nur Vergleich):
  - 16,3 g; Bohrbilder 20/20.
  - FEA nicht lösbar: gmsh scheiterte in allen 6 Vernetzungsversuchen. Deshalb fehlen fea_solved, die Crash-Festigkeiten und die Zielwerte (Armspitze, f1, Σ).
- **Datenblätter** (10 Felder + Neun-Kriterien-Zeile + Oberflächenwarnung): `exports/runs/simp_mma_cov3_raw_1/datasheet.md`, `exports/runs/simp_mma_cov3_recon_1/datasheet.md`. Das v3-Datenblatt steht aus.

## Form (Sichtprüfung der Renders, Rohkörper)

- **Seitenansicht:** geschlossener Rahmen aus Untergurt, Deckschienen als Obergurt, Pfosten und Schrägstreben vorn und hinten. Ein geschlossener Querschnitt in der Seitenebene, also ein Fachwerkträger mit Diagonalen.
- **Draufsicht:** Die Arme laufen in einen Leiterrahmen aus den zwei Akkuschienen mit drei Querriegeln (vorn, Mitte, hinten). Es gibt kein durchgehendes X über die Rumpfmitte; Torsion übernimmt der geschlossene Ring um den Akkuschacht.
- **Rückseite:** keine Sitze mehr; das Antennenauge und die Steckersitze sind weg.
- **Arme:** einzelne, im Querschnitt gefüllte Holme; die Fenster im Steg des 26,6-g-Laufs sind verschwunden.

## Bilder und STLs

- Rohkörper-STL: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov2/geometry.stl` (= `simp_mma_cov3_raw_1/frame.stl`, Endlauf).
- 1:1 nur als Vergleich: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov2_1to1/geometry.stl`.
- v3: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov2_recon/geometry.stl`, **ausstehend**.
- `exports/cov/cov_4views.png` (roh | v3 | ManaFly) und `exports/cov/recon_1to1_vs_v3.png`: **ausstehend** bis v3 bewiesen ist. Es gibt bewusst kein Ersatzbild mit 1:1 an der v3-Stelle.
- Renders des Rohkörpers: `exports/runs/simp_mma_cov3_raw_1/renders/{iso,top,side,front}.png`.

## Abschluss, sobald recon3 einen bestandenen Nachweis committet hat

Bedingung: SIMP+MMA und neural_v06_f1 beide innerhalb 10 % vom Rohkörper.

1. Override `cov3_post.json` (wie hier, `mma.bodies` = `["raw", "recon", "v3"]`, `v3.ref` = SHA des Nachweis-Commits) und `python tools/formulation_study.py frame_runs <override>`. Fertige raw- und 1:1-Läufe werden wiederverwendet. v3 entsteht aus einem git archive des recon3-Commits mit Evaluator inkl. Σ, Datenblatt und Renders; das STL geht nach `simp_mma_cov2_recon`.
2. Danach `compose` mit derselben Override-Datei (beide Bilder) und `cov_compare`; dann v3 hier eintragen, auch die Abweichung v3 zu roh bei f1 und Armspitze.
