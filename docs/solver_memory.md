# Solver-Speicher

Ziel: weniger GPU-Speicher und Zeit pro Iteration im SIMP-MMA-Frameproblem (Lastmodell, `simp_mma_cov3`), Ergebnis identisch innerhalb 1e-6.

## Benchmark

`tools/formulation_study.py solver_memory [override.json]`, Konfiguration `FORMULATION["solver_memory"]`:

- Feld: feste Entwürfe `coarse/design.npz` (68x64x24) und `fine/design.npz` (102x96x24) aus `C:/clones/Deep_Frame-cov/exports/runs/simp_mma_cov3_opt`, letzte Schärfestufe β64, 4/3 mm, Halbdomäne.
- Je Raster: Aufbau, 3 Auswertungen (Ziel, alle Nebenbedingungen, alle Sensitivitäten, f1), davon 0 und 1 kalt (Modalvektoren zurückgesetzt, 30 Unterraumiterationen) und 2 warm (3 Iterationen), danach 5 MMA-Iterationen.
- Referenz: Auswertung 0 als `exports/solver_memory/<lauf>/<raster>.npz` (Ziel, Zielgradient, g, Werte, alle Sensitivitätsvektoren, f1). Mit `"reference": "<ordner>"` vergleicht der Befehl gegen diese Datei; Rauschboden = Auswertung 1 gegen 0.
- Messung: eigener GPU-Speicher des Prozesses (Windows-Zähler `\GPU Process Memory(pid_*)\Dedicated Usage` und `Shared Usage`, 1 s), gesamte GPU (nvidia-smi, 0,5 s), CuPy-Pool, Host-RSS, cuDSS-Schätzung je Zerlegung, Faktorisierungen je Auswertung.
- Start nur über Ray: `compute.py density_simp --cwd <worktree> -- <gpu-venv-python> tools/formulation_study.py solver_memory`.

## Zerlegungen im aktuellen Code

Halbdomäne mit Symmetrieebene x = 0: jede Lagerung zerfällt in einen symmetrischen (+) und einen antisymmetrischen (−) Teil, also zwei Matrizen.

| Zerlegung | Lagerung | Lastfälle (rechte Seiten) | freie DOF fein |
|---|---|---|---|
| statisch + | Trägheitsentlastung, 3 Preserve-Knoten, statisch bestimmt | thrust_all, alle 9 Crash-Fälle, twist, torsion_yaw (12) | 277 912 |
| statisch − | Trägheitsentlastung | 6 Crash-Fälle mit Querlast, twist, torsion_yaw (8) | 276 209 |
| statisch + | fest: Stack-Unterseiten (96 DOF) | stiffness_arm_tip + 26 Kovarianzrichtungen sigma_k (27) | 277 819 |
| statisch − | fest: Stack-Unterseiten | stiffness_arm_tip + 22 sigma_k (23) | 276 116 |
| modal + | fest: Lagerung des Falls `modes` (336 DOF) | 6 Startvektoren, Unterraumiteration | 277 579 |
| modal − | fest: `modes` | 6 Startvektoren | 275 876 |

- Crash läuft bereits mit Trägheitsentlastung; fest gelagert sind nur die Kovarianz-/Armspitzenfälle (Stack) und das Eigenproblem (`modes`).
- Pro Auswertung: 6 numerische Cholesky-Zerlegungen (4 statisch, 2 modal), danach Vorwärts-/Rückwärtssubstitution für alle rechten Seiten und bei f1 weitere Substitutionen je Unterraumiteration.
- Alle 6 cuDSS-Löser bleiben resident (`gpu_solver_residency="resident"`): Der Speicherbedarf ist die Summe aller sechs Faktoren plus Matrix- und RHS-Puffer, nicht der größte einzelne Faktor.

## Baseline (dcc6108, RTX 4080 16 GB, allein auf der Karte)

`exports/solver_memory/baseline/` (`coarse.npz`, `fine.npz`, `result.json`), Lauf über Ray `density_simp`.

| Raster | DOF aktiv | Faktorisierungen / Auswertung | s / Auswertung (kalt, warm) | s / MMA-Iteration | dediziert Spitze GB | geteilt Spitze GB | nvidia-smi gesamt GB | CuPy-Pool GB | Host-RSS GB |
|---|---|---|---|---|---|---|---|---|---|
| grob 68x64x24 | 125 535 | 6 | 6,4, 4,5 | 7,9 | 6,28 | 0,32 | 7,4 | 0,80 | 2,55 |
| fein 102x96x24 | 279 618 | 6 | 16,8, 17,8 | 14,9 | 15,20 | 0,83 | 15,65 | 1,81 | 5,04 |

- Grob, Auswertung 0 (14,5 s) überlappte noch mit dem auslaufenden Neural-Prozess (nvidia-smi 15,5 GB); kalt deshalb Auswertung 1.
- Fein lagert aus: 15,2 GB dediziert, 0,83 GB geteilt (0,3 GB davon schon nach dem Aufbau, CUDA-Kontext).
- cuDSS-Schätzung je Faktor fein 2,17–2,25 GB, Summe der 6 residenten Faktoren 13,2 GB (grob 5,2 GB); CuPy-Pool 1,8 GB = Matrixwerte/Indizes (je 20 Mio. nnz) und RHS-Puffer der 6 Löser. Rest ≈ CUDA-Kontext.
- Analysephase je Faktor 1,1–1,5 s, nur beim Aufbau.
- Ziel fein 0,446017, f1 310,62 Hz, Masse 18,22 g; grob 0,486275, f1 305,50 Hz.
- Rauschboden (Auswertung 1 gegen 0): Sensitivitäten relativ ≤ 2,4e-11, g ≤ 3,6e-12. Wiederholungslauf (Auswertungen, Ray-Kopf danach weg): Abweichung zur Referenz ≤ 2,1e-11.
- Lastfälle thrust_all, twist, torsion_yaw und stiffness_arm_tip werden mitgelöst (RHS-Spalten), stehen aber nicht im Nebenbedingungsvektor; g = load_mean, load_worst (aus sigma_k), 9 Crash, f1, Volumen, Schatten.
