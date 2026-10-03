# 0,75-mm-Feinläufe auf der DGX

Vier neuronale Feinläufe (TOuNN, `tools/neural_study.py`) auf je einer A100-80GB, halbe Domäne, 40 Iterationen, warm gestartet aus einem hochskalierten 4/3-mm-R3-Feld. Gerechnet am 03.10.2026 auf `dgxruth` über Ray in einer Slurm-Zuteilung (4 GPUs, 64 Kerne, 256 GB), Job-Typ `gpu_a100` in `tools/compute.py`.

Stand: alle vier Läufe fertig. Vergleichsbild: `compare/side_by_side.png` (Zeilen iso, oben, seitlich; Spalten R3 7 % auf 4/3 mm, Läufe 1–4, ManaFly).

## Ergebnisse

Compliance in N·mm aus der Ersatzrechnung des jeweiligen Rasters (`info.json` → `static_surrogate_metrics`); Masse aus dem STL-Volumen × 1,09 g/cm³.

| Lauf | Masse | arm_tip | thrust_all | schlechtester Crash | Streben < 2 mm | s/Iteration | GPU-Speicher | Host-RSS |
|---|---|---|---|---|---|---|---|---|
| R3 7 %, 4/3 mm (Startfeld) | 21,7 g | 0,297 | 0,411 | 3,86 | – | 5,0 | ca. 10 GB | – |
| 1: R3 7 %, unverändert | 22,3 g | 0,170 | 0,293 | 2,99 | 21 % | 92,2 | 26,3 GB | 21,4 GB |
| 2: R3 7 %, Mindestbreite 2,5 mm | 22,3 g | 0,185 | 0,358 | 4,44 | 15 % | 92,5 | 26,3 GB | 21,1 GB |
| R3 5 %, 4/3 mm (Startfeld) | 13,9 g | 0,944 | 1,27 | 14,9 | – | 5,0 | ca. 10 GB | – |
| 3: R3 5 %, unverändert | 14,4 g | 0,561 | 0,847 | 12,3 | – | 92,7 | 26,3 GB | 21,4 GB |
| 4: Runde 4, weich, f1 ≥ 300 Hz, 2 mm, 6 % | 17,1 g | 0,483 | 0,891 | 12,7 | 33 % | 179,1 | 26,3 GB | 22,9 GB |

s/Iteration ist der Mittelwert ab Iteration 2. Iteration 1 enthält Domänenaufbau und Anpassen des Netzes an das Startfeld: ca. 790 s, davon 675 s Fit, RMSE 0,19 → 0,02. GPU-Speicher ist das Maximum der Gerätebelegung nach jeder cuDSS-Phase; cuDSS schätzt selbst 24 GB pro System.

Lauf 4 endet mit f1 = 326 Hz (Ersatzrechnung, symmetrisch 326/370 Hz, antisymmetrisch 358/416 Hz). Die Bedingung ist ab Iteration 11 erfüllt. Der Fit an das 7-%-Feld erreicht nur RMSE 0,060 statt 0,02: Das Feld hat mehr Material (Mittel 0,052) als das gewichtete 6-%-Budget erlaubt (0,042), und die Lernrate ist kleiner. Iteration 1 dauert 1795 s, davon 551 s Fit, der Rest sind Domänenaufbau und der Modal-Start (30 Unterraum-Iterationen).

Die Fourier-Grenze ist keine harte Mindestbreite. In Lauf 4 liegen 33 % der Skelettpunkte unter 2 mm (p10 = 0,67 mm), in Lauf 2 trotz 2,5-mm-Grenze noch 15 %.

Die Compliance-Werte von 4/3 mm und 0,75 mm stammen aus verschiedenen Rastern. Ein Teil des Unterschieds ist Diskretisierung, nicht Optimierung. Ein direkter Vergleich braucht eine gemeinsame FEA.

## Festlegungen

- **Raster:** `shape` [182, 170, 48], also 0,747 × 0,753 × 0,667 mm; halbe Domäne 91 × 170 × 48. Genau 0,75 mm geht in der Domäne 136 × 128 × 32 mm nicht. In z liegt mit 42 Zellen die Deck-Ebene 28 mm nicht auf einem Knoten, der Selektor ist dann leer. 48 Zellen treffen 28, 28/3 und 20/3 mm exakt.
- **Warmstart:** `neural.initial_density` = `density_half.npz` des 4/3-mm-Laufs, halb → halb linear resampelt. Erhaltene Zellen werden auf 1 gesetzt, gesperrte auf 0. Das Netz wird 300 Adam-Schritte per MSE an das Feld angepasst. Danach gilt die Schärfe 8 ab Iteration 1, `max_iterations` = `minimum_iterations` = 40, `max_runtime_s` = None.
- **Mindestbreite:** Sie wird über die Fourier-Grenze `max_frequency_per_mm` gesetzt. Lauf 2 nutzt 0,16 nach der Regel aus `main` (0,2 · 2 / 2,5 mm), passend zu R3s eigenen 0,2 bei 2 mm. Lauf 4 nutzt 0,125 nach der Runde-4-Regel 1 / (2 · 2 · 2 mm). Die Runde-4-Regel ergäbe für Lauf 2 den Wert 0,10.
- **Speicher:** `gpu_solver_residency` = `transient`. Resident passen nur drei Systeme à ca. 25 GB auf die Karte; das vierte bricht mit `CUDSS_STATUS_ALLOC_FAILED` ab.
- **Lauf 4:** gerechnet im Worktree von `dgx/round4-075` (`origin/feature/round4` plus Warmstart- und Logging-Commits). Gestartet aus dem 7-%-Feld. Der erste Versuch mit Lernrate 0,01 ist in Iteration 2 entgleist: Compliance × 154, f1 35 Hz. Neu gestartet mit `learning_rate` 0,002; der Wert gilt auch für den Fit. Der Verlauf des abgebrochenen Versuchs liegt in `_aborted_run4_lr001/iterations.jsonl`.
- **ManaFly-Ausrichtung:** Die Ausrichtung folgt dem dokumentierten Rezept (180° um z). In der Draufsicht steht ManaFly trotzdem anders als die Optimierungsergebnisse. Das ist eine Darstellungsfrage, Messwerte betrifft sie nicht.

## Startfelder

`exports/handoff/neural_r3_v07` und `neural_r3_v05` wurden auf der DGX neu erzeugt, mit Commit `a644edd` und R3-Einstellungen aus `STUDY` (`shape` [102, 96, 24]). Laufzeiten: 102 bzw. 109 Iterationen, Abbruch bei Stagnation, 513 bzw. 543 s. Das sind neue Felder mit eigener Provenienz, nicht die Workstation-Felder. Die Masse von 21,7 g (STL) ist nicht direkt mit den 29,8 g der Rekonstruktion in `r4recon_r3_v07` vergleichbar.

## Dateien

Konfigurationen liegen unter `exports/handoff/configs/`. Je Lauf: `info.json`, `iterations.jsonl`, `density_half.npz`, `density_fine.npz` (2/3 mm, voll), `iso.png`, `top.png` und `side.png`. Die STLs sind per `.gitignore` ausgeschlossen. Die ManaFly-Referenz unter `exports/handoff/manafly/` stammt aus `Examples/Frames/ManaFly3/Manafly+3inch+BETA+V4+Frame.stl`, um 180° um z gedreht, auf z = 0 gesetzt und mit `tools/reconstruction_study.py render` gerendert.
