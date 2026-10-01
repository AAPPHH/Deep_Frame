# Feiner robuster Dichtelauf (S10)

Dichtequelle: `C:/clones/Deep_Frame-robust/exports/topology/robust_fine/grid4over3_robust` (manifest sha256 `d54b8e46…`, fields.npz sha256 `74c6a697…`, alle Artefakt-Hashes geprüft). Maschinenlesbar: `robust_fine_run.json`. Code-Stand: 186ecc5.

## Aufbau

- Gitter 4/3 mm, 102×96×24, Ursprung (−68, −64, 0) mm; 177 280 aktive Elemente, 611 145 aktive DOFs.
- Physik identisch zu gpu708 (`grid8over3_gpu1000`): Regionen, Lastfälle, Vergleichslastfälle, Material, Punktmassen, Fertigung, Volumenanteil 0,1. Einzige weitere Abweichung außer Projektion und Abbruch: `gpu_solver_residency: transient`.
- Robuste Heaviside-Projektion η_e 0,75 / η_i 0,5 / η_d 0,25; Ziel = normierte Mehrlast-Compliance des erodierten Designs, OC-Volumenrestriktion auf dem dilatierten Feld.
- Filterradius 3,4 mm (gpu708: 6,0 mm bei 8/3 mm).
- β 1→2→4→8→16, Wechsel nach 40 Iterationen oder bei Stagnation < 0,01 (min. 20 je Stufe); Move-Limit 0,12, ab β 8 0,05; max. 360 Iterationen, `max_runtime_s` 9900, Prozess-Timeout 10 800 s.
- FP64-cuDSS-Cholesky (cuDSS 0.8.0, RTX 4080), Faktoren transient (234 Freigaben), max. relatives Residuum 3,8e−9.

## Smoke (S10b)

- Residente FP64-Faktoren (je ~6,9 GB): Speicherüberlauf, WDDM-Paging bei 15 768 MiB, keine Iteration in 823 s.
- Transient: 20 Iterationen, Median 25,9 s/Iteration; Projektion 500 Iterationen 12 938 s > 2,7 h, daher laut Plan beta_interval 40 und max. 360 Iterationen.

## Ergebnis

| Größe | Wert |
|---|---|
| Status | ok, Exit 0 |
| Iterationen | 116 (+ Endauswertung) |
| Laufzeit | 2771,8 s optimiert, 2775,8 s Wandzeit (Budget 10 800 s) |
| s/Iteration | Median 24,7, Mittel 23,6 (ab Iteration 3), erste 30,1 |
| GPU-Speicher | nvidia-smi-Spitze 11 962 MiB geräteweit inkl. anderer Prozesse (vorher 4163, minimal 1695 MiB); im Prozess nach Solve 8463,5 MiB; also Prozess-Spitze ≤ ~10,3 GB von 16 GB |
| β-Stufen | β 1 ab It. 1 (40), β 2 ab 41 (20), β 4 ab 61 (22), β 8 ab 83 (20), β 16 ab 103 (14) |
| Volumenanteil intermediär | 0,1002 (Ziel 0,1) |
| Masse | 45,88 g intermediär, 60,26 g dilatiert (Obergrenze) |
| Volumenanteil erodiert / dilatiert | 0,0745 / 0,1316 |
| Graufraktion frei | 1,5 % (gpu708: 15,5 %) |
| Compliance erodiert [N·mm] | arm_tip 0,00590, battery_impact 0,00613, camera_side 0,00500 |
| arm_tip-Steifigkeit (Voxel-Surrogat) | 169,5 N/mm |

## Konvergenz

Gestoppt durch `objective_stall` bei β 16 nach 14 Iterationen: relative Spanne der Zielfunktion über 10 Iterationen 0,49 % < 0,5 %. Das ist das in S9 eingebaute Abbruchkriterium; `converged: true` im Record bedeutet nur diesen Stagnationsstopp. **Im Sinne des Plans (Abschnitt 11) ist der Lauf nicht konvergiert.** Der Plan erlaubt den Stopp nur über `change_tolerance` 0,005 bei β 16 und den Stufenwechsel über Änderung < 0,01 nach mindestens 20 Iterationen oder nach dem Intervall. β 16 lief nur 14 Iterationen (`minimum_iterations` 10). `max_iterations` war 360 statt der im Plan genannten 400. Der Lauf (46 min GPU) wurde nicht wiederholt; die Dichte gilt als gestoppte, nicht als konvergierte Blaupause. Das Designänderungs-Kriterium (`change_tolerance` 0,005) wurde nicht erreicht: die maximale Designänderung lag in jeder Iteration genau am Move-Limit (0,12 bzw. 0,05), d. h. einzelne Zellen pendeln weiter. Auch die Wechsel 2→4, 4→8 und 8→16 kamen über Stagnation, nicht über kleine Änderung; 1→2 über das Intervall von 40. Mit 116 Iterationen deutlich unter den im Plan erwarteten 300–500.

## Minimale Längenskala

Methode: gefiltertes Feld trilinear 4-fach auf h/4 = 1/3 mm hochgetastet (verboten 0, preserve 1), Iso bei η; dünn = Phasenvoxel, die eine morphologische Öffnung mit Kugel des Durchmessers d nicht abdeckt (EDT); freie Zone = > 2 mm Abstand zu verbotenen, preserve- und Hüllflächen; lokale Dicke = 2× EDT auf dem Kamm.

| Feld | Phase | Dicke p1 / p5 / p50 [mm] | Anteil < 2 mm (frei) | Anteil < 3 mm (frei) |
|---|---|---|---|---|
| erodiert (0,75) | fest | 1,33 / 1,33 / 2,0 | 4,4 % | 12,1 % |
| intermediär (0,5) | fest | 4,0 / 4,0 / 4,67 | 0,27 % | 1,35 % |
| dilatiert (0,25) | fest | 4,22 / 4,67 / 5,33 | 0,12 % | 0,26 % |
| intermediär (0,5) | leer | 5,33 / 5,33 / 8,0 | 0,004 % | 0,012 % |

- Das Blueprint-Feld (intermediär) hat in der freien Zone Stege ≥ 4 mm und Spalte ≥ 5,3 mm, also deutlich über 2 mm, ohne nachträgliche Öffnung. gpu708 hatte 0,89 mm minimale Strahldicke.
- Das erodierte Feld ist erwartungsgemäß dünner (Median 2,0 mm); es ist der Worst-Case-Entwurf, nicht die Druckgeometrie.
- Schachtelung erodiert ≤ intermediär ≤ dilatiert gilt punktweise.
- Zusammenhang (ρ ≥ 0,5): intermediär und dilatiert je 1 Komponente. Erodiert 3 Komponenten, davon 2 Einzelvoxel (je 2,4 mm³); alle Preserve-Voxel (16 zusammenhängende Bereiche) liegen in der einen Hauptkomponente (13 317 Voxel). Damit hängen alle Montagepunkte auch über das erodierte Feld zusammen (R2).
