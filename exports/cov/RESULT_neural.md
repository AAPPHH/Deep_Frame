# Neural+AL gegen SIMP+MMA mit identischer Formulierung (Lastmodell, 4/3 mm)

Stand 04.10. 07:20. Branch feature/neural-cov (Worktree /c/clones/Deep_Frame-ncov) = feature/load-covariance + Merge feature/neural-al. Beide Optimierer rechnen auf derselben `frame_setup`-Formulierung:

- Σ-Zeilen tr / λmax mit den ManaFly-×-0,8-Grenzen 0,687 / 0,229 N mm und dem Kalibrierfaktor 1,384 / 1,374, also Optimierergrenze 0,951 / 0,315 N mm;
- robuste Mindestbreite 2,5 mm, f1 ≥ 300 Hz, radialer Propschatten, Volumen ≤ 0,10, Spiegelsymmetrie, neun Crash-Richtungen;
- keine Zubehörsitze.

Rekonstruktion für beide: v3 = recon-v3 879713d mit der Sitz-Entfernung (af19376, `recon-v3-noseats`), Standardwerte.

## Ergebnis in einem Satz

**SIMP+MMA ist unter denselben Nebenbedingungen deutlich leichter.**

- SIMP: Feld 18,2 g, Rohkörper 15,7 g, v3 17,6 g.
- Neural+AL: Feld 29,9 g, Rohkörper 27,5 g, v3 27,9 g.
- Das sind +64 % (Feld), +75 % (roh) und +58 % (v3) für Neural+AL.

Neural+AL ist zudem **nicht konvergiert**. Der Lauf stoppte am Laufzeitwächter (5400 s), eine Iteration nach dem Sprung auf β = 64. In dieser letzten Iteration verletzt f1 die Grenze um 13,5 % (280 Hz < 300 Hz, Optimierermaß am erodierten Feld). Das letzte β-32-Iterat lag noch fast zulässig, mit tr +1,9 % über der Grenze und f1 erfüllt.

Gegen die Lastgrenzen sieht es umgekehrt aus. Der neuronale Rohkörper erfüllt sie im Evaluator klar, mit 72 % / 70 %. SIMP v3 verfehlt λmax (108 %), neural v3 hält beide Grenzen ein (78 % / 85 %). Die Reserve kommt aber über die Masse.

## Haupttabelle

| Größe | SIMP+MMA roh | SIMP+MMA recon v3 | Neural+AL roh | Neural+AL recon v3 |
|---|---|---|---|---|
| Masse Optimierer (Feld, intermediär) | 18,22 g | – | 29,91 g | – |
| Masse Körper (STL) | 15,72 g | 17,63 g | 27,48 g | 27,92 g |
| Evaluator voll tr (Grenze 0,687 N mm) | 0,687 (100,0 %, erfüllt) | 0,646 (94 %, erfüllt) | 0,494 (72 %, erfüllt) | 0,535 (78 %, erfüllt) |
| Evaluator voll λmax (Grenze 0,229 N mm) | 0,220 (96 %, erfüllt) | **0,247 (108 %, verfehlt)** | 0,160 (70 %, erfüllt) | 0,195 (85 %, erfüllt) |
| Evaluator diag. tr / λmax | 0,809 / 0,305 | 0,747 / 0,297 | 0,546 / 0,205 | 0,591 / 0,246 |
| Armspitze | 10,9 N/mm | 9,4 N/mm | 19,4 N/mm | 19,7 N/mm |
| f1 (Evaluator) | 407 Hz | 521 Hz | 481 Hz | 450 Hz |
| Verwindung (Diagonalpaare ±1 N Fz) | 12,2 N/mm | 13,0 N/mm | 18,5 N/mm | 15,4 N/mm |
| Iterationen (grob + fein) | 270 (183 + 87) | | 240 (138 + 102) | |
| s/Iteration grob / fein | 5,9 / 45,4 s | | 4,5 / 43,1 s | |
| Gesamtzeit Optimierung | 5174 s (1 h 26 min) | | 5535 s (1 h 32 min), Wächter | |
| Abbruch | konvergiert (Kriterien, beide Stufen) | | **guard** (max_runtime_s 5400), nicht konvergiert | |
| Neun-Kriterien-Ergebnis (verfehlt) | bolt_patterns, keep_outs_free, wall_deep_fraction, wall_deep_component, target:symmetry | target:symmetry, target:sigma_worst | bolt_patterns, keep_outs_free, wall_deep_component | wall_deep_component, wall_motor_zones |
| Warnungen | loops, cog_offset | loops, cog_offset | loops, cog_offset | loops, cog_offset |
| Wandregel: tiefer Anteil / größte Komponente / Motorzonen (Grenzen 0,5 % / 5 mm³ / 0) | 1,14 % / 49,3 mm³ / 0 | 0,00 % / 0,0 mm³ / 0 | 0,43 % / 88,0 mm³ / 0 | 0,23 % / 11,6 mm³ / 4 |
| Gesamturteil | nicht gut | nicht gut | nicht gut | nicht gut |

Zur FEA-Route:

- Neural roh und neural v3 liefen mit der Standard-Evaluator-FEA; beide ließen sich vernetzen, eine Ersatz-FEA war nicht nötig.
- Für SIMP v3 gilt die Ersatz-FEA aus dem SIMP-Lauf (fea_memory_budget_mb 16384, Remesh 2,0/1,5/1,2/1,0 mm). SIMP roh wurde mit Standard- und Ersatz-FEA gerechnet; die Werte liegen auf 0,6 % beieinander (Armspitze 10,88 / 10,82 N/mm, tr 0,687 / 0,686).
- Die Wandregel ist vorläufig, kalibriert an ManaFly 3 (Tiefe 0,45 mm, Öffnungsradius 1,0 mm).

## Nebenbedingungen im Optimierer (feines Gitter, Endentwurf)

| Nebenbedingung | Grenze | SIMP+MMA Wert | Status | Reserve | Neural+AL Wert | Status | Reserve |
|---|---|---|---|---|---|---|---|
| load_mean (tr) | ≤ 0,9508 N mm | 0,9467 | aktiv | +0,4 % | 0,9249 | erfüllt | +2,7 % |
| load_worst (λmax, KS) | ≤ 0,3146 N mm | 0,3081 | erfüllt | +2,1 % | 0,2746 | erfüllt | +12,7 % |
| crash_front | ≤ 9,268 N mm | 4,085 | erfüllt | +55,9 % | 6,642 | erfüllt | +28,3 % |
| crash_side_left / right | ≤ 20,44 N mm | 13,78 | erfüllt | +32,6 % | 16,28 | erfüllt | +20,4 % |
| crash_arm_front_left | ≤ 211,6 N mm | 102,1 | erfüllt | +51,8 % | 116,6 | erfüllt | +44,9 % |
| crash_arm_front_right | ≤ 33,81 N mm | 33,84 | aktiv | −0,1 % | 21,5 | erfüllt | +36,4 % |
| crash_arm_rear_left | ≤ 65,64 N mm | 65,64 | aktiv | +0,0 % | 59,1 | erfüllt | +10,0 % |
| crash_arm_rear_right | ≤ 108,2 N mm | 96,69 | erfüllt | +10,6 % | 92,27 | erfüllt | +14,7 % |
| crash_below | ≤ 41,28 N mm | 14,07 | erfüllt | +65,9 % | 15,29 | erfüllt | +63,0 % |
| crash_back | ≤ 10,41 N mm | 7,693 | erfüllt | +26,1 % | 4,858 | erfüllt | +53,4 % |
| f1 (erodiert) | ≥ 300 Hz | 310,6 | erfüllt | +7,1 % | 280,4 | **verletzt** | **−13,5 %** |
| volume | ≤ 0,10 | 0,0446 | erfüllt | +55,4 % | 0,0732 | erfüllt | +26,8 % |
| shadow (radial) | ≤ 0,3137 mm | 0,2552 | erfüllt | +18,7 % | 0,2218 | erfüllt | +29,3 % |
| robuste Mindestbreite 2,5 mm, Symmetrie | im Feld erzwungen | ja | | | ja | | |
| überwacht: f1 intermediär | – | 375 Hz | | | 330 Hz | | |
| überwacht: thrust_all / torsion_yaw / twist | – | 4,25 / 0,082 / 5,19 N mm | | | 2,15 / 0,130 / 2,33 N mm | | |

## Neural v3 gegen den neuronalen Rohkörper (Entscheidung: v3 ist die Rekonstruktion, Abweichung ehrlich)

| Größe | roh | recon v3 | Abweichung | innerhalb 10 %? |
|---|---|---|---|---|
| Masse | 27,48 g | 27,92 g | +1,6 % | ja |
| Armspitze | 19,39 N/mm | 19,68 N/mm | +1,5 % | ja |
| f1 | 481,1 Hz | 449,6 Hz | −6,6 % | ja |
| Σ tr voll | 0,494 N mm | 0,535 N mm | +8,2 % | ja |
| Σ λmax voll | 0,160 N mm | 0,195 N mm | **+22,2 %** | **nein** |
| Σ tr / λmax diag. | 0,546 / 0,205 | 0,591 / 0,246 | +8,3 % / +20,1 % | ja / nein |
| Verwindung | 18,5 N/mm | 15,4 N/mm | **−16,6 %** | **nein** |

Für den dickeren neuronalen Rahmen bleiben Masse, Armspitze und f1 innerhalb 10 %; bei SIMP verfehlten Masse, Armspitze und f1 diese Schwelle. Die schwächste Richtung (λmax) und die Verwindung verschlechtert v3 aber um mehr als 10 %.

Zu den Kriterien:

- Neu erfüllt gegenüber roh: bolt_patterns (20/20) und keep_outs_free.
- Neu verfehlt: wall_motor_zones (4 tiefe Stellen in Motorzonen).
- wall_deep_component bleibt verfehlt (11,6 mm³ > 5 mm³).

## Kalibrierung (gemessen auf SIMP-Feldern)

Neural roh im vollen Evaluatormaß: tr 0,494 / λmax 0,160 N mm. Das Optimierermaß am selben Entwurf liegt bei 0,9249 / 0,2746 N mm. Daraus ergibt sich das Verhältnis Optimierer/Evaluator 1,873 / 1,718, gegen 1,384 / 1,374 auf SIMP. Der Faktor überträgt sich also nicht 1:1 auf das neuronale Feld; dessen Rohkörper ist im Evaluator steifer, als der Optimierer annimmt.

Die Grenzen werden trotzdem eingehalten (72 % / 70 %), nicht um mehr als 10 % verfehlt. Daher gibt es laut Auftrag **keinen Kalibrier-Neulauf**. Ein eigener Faktor für Neural würde die Optimierergrenze lockern und die Masse senken; das wäre aber eine andere Formulierung als bei SIMP.

## Lauf und Start

- Neural+AL: Start r4_neural_v06_f1_1 (Auftrag), über 800 Fit-Iterationen ins Netz gebracht; Fit-Fehler 0,0011, Volumen 0,0583 statt 0,0574.
- Grobe Stufen β 1–4 auf 68x64x24, fein ab β 8 auf 102x96x24 (Halbgebiet), cuDSS.
- Ray-Typ `density_simp` (4 Kerne, 8 GB, 10 GPU-GB) wie SIMP.
- AL-Strafen blieben bei 1, weil Wachstum nur auf der letzten β-Stufe erlaubt ist. Multiplikatoren am Ende: tr 1,51, crash_arm_rear_left 0,013, Schatten 2,62.
- SIMP+MMA startete dagegen vom 26,6-g-Lastmodellfeld der Kalibrierschleife (zulässig, MMA trägt nur ab). Der neuronale Start (r4) verletzte tr / λmax anfangs um +413 % / +469 %. Die Starts sind also verschieden, wie vorgegeben.

Grund für die höhere Masse: Bis β 4 auf dem groben Gitter kämpfte Adam/AL vor allem gegen Schatten und f1. Die Masse pendelte dabei bei 32–34 g. Auf dem feinen Gitter sank sie in 102 Iterationen nur von 31,7 auf 29,9 g. Bei 43 s/Iteration reicht das Zeitbudget nicht, um die Masse wie MMA bis ~18 g abzubauen. Dass das Verfahren bei längerer Laufzeit SIMP erreicht, ist damit **nicht belegt**. Der Lauf zeigt nur, dass es unter identischen Nebenbedingungen und ähnlicher Rechenzeit deutlich schwerer endet.

Wie bei nal_frame_1 endet der Lauf zudem kurz nach dem letzten β-Sprung. Ausgewertet wird das letzte Iterat (β 64) mit f1-Verletzung; berichtet, nicht gelockert.

## Kragträger-Nachweis mit Kovarianz-Nebenbedingung (d211422)

Gleiche Aufgabe wie der MMA-Nachweis: min Masse bei tr(FΣ) ≤ 4 × Vollblock.

- Neural+AL endet aktiv: 0,01311 bei einer Grenze von 0,01315 N mm.
- Volumen 0,524 gegen MMA 0,484, also +8,4 %.
- Abbruch durch die gemeinsame Terminierung nach 418 Iterationen.
- Richtungs-FD des Lagrange-Terms in den Netzparametern 3,2e-8.

## Dateien

- Vergleichsbild SIMP v3 | Neural v3 | ManaFly (iso/top/side/front, gleicher Renderer): `exports/cov/compare_simp_neural_manafly.png`
- STLs: `C:/clones/Deep_Frame-neural/exports/neural_cov/geometry.stl` (roh), `C:/clones/Deep_Frame-neural/exports/neural_cov_v3/geometry.stl` (v3)
- Läufe: `exports/runs/neural_cov_opt/neural_cov` (Optimierung, info.json, iterations.jsonl, network.npz), `exports/runs/neural_cov_raw_1`, `exports/runs/neural_cov_v3_1` (Evaluation, Datenblatt `datasheet.md`, Renders)
- Zahlen: `exports/cov/comparison_neural.json`; Overrides `exports/ncov/frame_run.json`, `exports/ncov/post.json`
