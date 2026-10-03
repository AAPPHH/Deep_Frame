# Frame-Lauf mit Lastmodell als Verteilung (SIMP+MMA, 4/3 mm)

Stand: Optimierung, Rohkörper und 1:1-Vergleich fertig. **Rekonstruktion v3 steht noch aus**: `feature/recon-v3` (/c/clones/Deep_Frame-recon3) hat ihren Nachweis noch nicht committet (HEAD 216db36, Nachweis läuft; SIMP+MMA v3 liegt dort aktuell bei f1 -54 %, Armspitze -31 % gegen roh). Nach Regel wird gewartet; der 1:1-Körper ist **nicht** das Ergebnis, nur Vergleich.

Stand 03.10. 22:20: v3 weiter nicht bereit. recon3 arbeitet (neuer Commit fa0e475 um 22:07, Knoten/Splines in Hülle und Keep-outs gesetzt), Nachweis (Glieder als Splines, SIMP+MMA und neural_v06_f1 innerhalb 10 % vom Rohkörper) aber noch nicht committet; letzte Nachweisdateien dort: `within_10_percent` false für beide. Polls 21:53 / 22:03 / 22:13. **Infrastruktur-Blocker:** Der Ray-Head wurde um 21:49 neu gestartet; sein Dashboard-Agent konnte Port 52365 nicht binden (liegt im Windows-Ausschlussbereich 52292-52391), daher scheitert jede Job-Einreichung mit "No available agent to submit job". Damit kann weder recon3 seinen Nachweis noch dieser Lauf v3/Evaluator rechnen. Abhilfe (nicht durch mich, kein `ray stop`): Head mit Agent-Port außerhalb der Ausschlussbereiche neu starten. Ein Neulauf von Evaluator/Datenblatt für raw und 1:1 (Statuszeile "evaluation: failed" wegen des cp1252-Drucks) scheiterte daran ebenfalls; datasheet.md und evaluation.json sind unverändert, Hinweis im manifest.

## Lauf

- Genau ein Lauf, Start vom heutigen SIMP+MMA-Feld (`Deep_Frame-mma/exports/runs/simp_mma_opt/fine/density_half.npz`, 17,2 g), grob 68x64x24 (152 It., 5,1 s/It.) -> fein 102x96x24 (91 It., 13,2 s/It.), beide `converged`, 34 min gesamt, Ray-Typ `density_simp` (4 Kerne, 8 GB, 10 GPU-GB).
- Neuer Standard-Satz: tr(ΣF) und λmax(Σ^1/2 F Σ^1/2) (KS) statt Armspitze und Richtungsfällen; behalten: f1 >= 300 Hz, Radialabschattung <= ManaFly, Volumen <= 10 %, robuste Mindestbreite 2,5 mm, Spiegelsymmetrie, Crash-Nachgiebigkeiten.
- Kalibrierung (fcfb875): Grenzen stammen aus dem Evaluator-/Gap-Maß (diagonal). Optimierer-Maß / Evaluator-Maß am selben 17,2-g-Entwurf: Mittel 0,8551/1,4366 = 0,5952, Worst 0,1967/0,5007 = 0,3928 -> Optimierergrenzen 0,232 / 0,066 N mm. Gegenprobe mit dem vollen 42x42-Maß (Aether4 voll skaliert 0,296 / 0,107; Faktoren 0,735 / 0,686) ergibt 0,218 / 0,073 N mm, also dieselbe Größenordnung.

## Masse

| | alt (SIMP+MMA heute) | neu (Lastmodell) |
|---|---|---|
| Optimierer (Feld) | 17,18 g | 29,36 g |
| Rohkörper (STL) | 14,56 g | 26,56 g |
| 1:1 (nur Vergleich) | 16,39 g | 26,34 g |
| recon v3 | ausstehend (recon3) | ausstehend |

Erwartung 20-25 g: Rohkörper liegt mit 26,6 g knapp darüber.

## Nebenbedingungen (Optimierer, fein, Endstand)

| constraint | value | limit | unit | status | margin |
|---|---|---|---|---|---|
| load_mean | 0.2322 | <= 0.2322 | N mm | active | -0.0 % |
| load_worst | 0.05851 | <= 0.06607 | N mm | satisfied | +11.4 % |
| crash_front | 3.364 | <= 9.268 | N mm | satisfied | +63.7 % |
| crash_side_left | 6.938 | <= 20.44 | N mm | satisfied | +66.1 % |
| crash_side_right | 6.938 | <= 20.44 | N mm | satisfied | +66.1 % |
| crash_arm_front_left | 41.32 | <= 211.6 | N mm | satisfied | +80.5 % |
| crash_arm_front_right | 19.62 | <= 33.81 | N mm | satisfied | +42.0 % |
| crash_arm_rear_left | 19.51 | <= 65.64 | N mm | satisfied | +70.3 % |
| crash_arm_rear_right | 39.79 | <= 108.2 | N mm | satisfied | +63.2 % |
| crash_below | 13.82 | <= 41.28 | N mm | satisfied | +66.5 % |
| crash_back | 1.726 | <= 10.41 | N mm | satisfied | +83.4 % |
| f1 | 304.1 | >= 300 | Hz | satisfied | +2.7 % |
| volume | 0.07426 | <= 0.1 | - | satisfied | +25.7 % |
| shadow | 0.2807 | <= 0.3137 | mm | satisfied | +10.5 % |

Aktiv ist nur die mittlere Nachgiebigkeit. Der Worst-Case-Modus ist jetzt das alternierende Pad-Mz (Gegenmoment, Diagonalpaare +/-) mit Pad-My-Anteil; beim alten Entwurf war es alternierendes Pad-Mx.

Überwacht (alte Lastfälle, nur Information), alt -> neu: thrust_all 2,77 -> 0,96 N mm, torsion_yaw 0,130 -> 0,039 N mm, twist 8,49 -> 1,64 N mm, f1 intermediär 332 -> 337 Hz.

## Lastmodell-Maße im Evaluator (gleicher Evaluator, Gruppe stack_fixed)

| Körper | tr voll | λmax voll | tr diagonal | λmax diagonal | Status gegen Grenze diag. 0,390 / 0,168 | Status gegen Grenze voll 0,296 / 0,107 |
|---|---|---|---|---|---|---|
| alt roh | 1,163 | 0,287 | 1,437 | 0,501 | verfehlt (3,7x / 3,0x) | verfehlt (3,9x / 2,7x) |
| alt 1:1 | 1,191 | 0,300 | 1,446 | 0,520 | verfehlt | verfehlt |
| **neu roh** | **0,186** | **0,047** | **0,232** | **0,086** | erfüllt (-41 % / -49 %) | erfüllt (-37 % / -56 %) |
| neu 1:1 (Vergleich) | 0,255 | 0,069 | 0,312 | 0,119 | erfüllt (-20 % / -29 %) | erfüllt (-14 % / -35 %) |
| neu recon v3 | ausstehend | | | | | |
| ManaFly 3 | 0,859 | 0,286 | 1,248 | 0,579 | Referenz | |
| Aether4 (unskaliert) | 0,391 | 0,143 | 0,528 | 0,242 | Referenz | |

Der Rohkörper erfüllt beide Grenzen im Evaluator deutlich, obwohl tr im Optimierer aktiv ist: das Optimierer-Maß (erodiertes Feld, Voxel) ist über die Kalibrierung konservativ. Hinweis: Primärwert der Neun-Kriterien-Zeile ist das volle Maß, die Zielgrenze im Evaluator (0,390 / 0,168) stammt aus dem diagonalen Maß; die volle Kopplung senkt tr um 18-31 % und λmax um 41-51 % (limits.md).

## Steifigkeit je Schnittstelle und Richtung

Quelle: Evaluator-Teil `sigma` (Einheitslasten je Schnittstelle, 6 Freiheitsgrade, Lagerung wie Gap-Finder). Er liefert bitgleich die Gap-Finder-Werte (simp_mma_raw_1, motor_front_left Fz: 8.466052669697811 N/mm in beiden), deshalb wurde der Gap-Finder nicht zusätzlich gestartet (RAM-Priorität). F in N/mm, M in N mm/rad.

| Schnittstelle/Richtung | alt roh | alt 1:1 | neu roh | neu 1:1 | ManaFly | Aether4 | neu roh / alt roh |
|---|---|---|---|---|---|---|---|
| motor_front_left Fx | 44.5 | 46.1 | 137.8 | 86.9 | 64.9 | 88.8 | 3.10 |
| motor_front_left Fy | 3.59 | 3.03 | 20.0 | 12.9 | 54.8 | 62.9 | 5.57 |
| motor_front_left Fz | 8.47 | 8.09 | 51.8 | 34.3 | 6.56 | 12.4 | 6.12 |
| motor_front_left Mx | 1503 | 1414 | 11730 | 7577 | 2917 | 20040 | 7.80 |
| motor_front_left My | 2953 | 3198 | 17770 | 15450 | 4710 | 21050 | 6.02 |
| motor_front_left Mz | 2307 | 1980 | 13540 | 8700 | 21470 | 126700 | 5.87 |
| motor_rear_left Fx | 32.2 | 32.6 | 76.0 | 64.4 | 31.8 | 125.6 | 2.36 |
| motor_rear_left Fy | 10.3 | 9.41 | 19.1 | 16.5 | 14.7 | 135.0 | 1.85 |
| motor_rear_left Fz | 7.43 | 7.53 | 44.7 | 35.3 | 6.30 | 21.8 | 6.01 |
| motor_rear_left Mx | 2310 | 2305 | 12660 | 9054 | 5673 | 35130 | 5.48 |
| motor_rear_left My | 2920 | 2755 | 19780 | 16550 | 8700 | 29150 | 6.77 |
| motor_rear_left Mz | 5087 | 4454 | 10890 | 9100 | 13870 | 69600 | 2.14 |
| stack Fx | 309 | 352 | 2581 | 1665 | 434 | 2606 | 8.36 |
| stack Fy | 208 | 219 | 1526 | 1084 | 684 | 1874 | 7.32 |
| stack Fz | 113 | 134 | 459 | 313 | 94.3 | 310 | 4.05 |
| stack Mx | 21370 | 38340 | 193900 | 144700 | 34340 | 92730 | 9.07 |
| stack My | 24390 | 33850 | 275900 | 166500 | 18980 | 113100 | 11.31 |
| stack Mz | 104100 | 171600 | 500800 | 364700 | 405100 | 1943000 | 4.81 |
| battery Fx | 50.1 | 61.7 | 446 | 341 | 49.7 | 124 | 8.90 |
| battery Fy | 89.8 | 97.8 | 692 | 686 | 88.2 | 132 | 7.71 |
| battery Fz | 230 | 214 | 1405 | 1153 | 407 | 246 | 6.12 |
| battery Mx | 21490 | 14810 | 37640 | 36260 | 21920 | 8273 | 1.75 |
| battery My | 49490 | 43660 | 280900 | 299700 | 43910 | 74550 | 5.68 |
| battery Mz | 40480 | 43340 | 301900 | 301700 | 140900 | 130700 | 7.46 |
| camera Fx | 88.8 | 101.5 | 19.8 | 26.4 | 27.6 | 46.9 | 0.22 |
| camera Fy | 113 | 123 | 387 | 423 | 60.2 | 42.0 | 3.42 |
| camera Fz | 92.7 | 112.4 | 78.7 | 87.2 | 42.0 | 41.1 | 0.85 |
| camera Mx | 29310 | 46950 | 13080 | 17890 | 25290 | 30570 | 0.45 |
| camera My | 27300 | 31440 | 10460 | 11410 | 19130 | 58110 | 0.38 |
| camera Mz | 42320 | 46710 | 99440 | 109600 | 55870 | 86930 | 2.35 |

Rechte Motoren spiegelgleich (vollständige Tabelle mit allen Pads in `comparison.json`, Feld `stiffness_table`). Motor-Fy bleibt weit unter ManaFly/Aether4 (Σ gewichtet Fy an den Pads kaum), die Kamera wird weicher (Σ enthält dort nur Masse x Fluglast).

## Verwindung (Diagonalpaare +/-)

Lastmuster Fz = +1 N vorne links und hinten rechts, -1 N vorne rechts und hinten links (Stack fest), aus der vollen Evaluator-Flexibilität: k = |w|^2 / (w^T F w).

| Körper | Nachgiebigkeit N mm | Steifigkeit N/mm |
|---|---|---|
| alt roh | 0,589 | 6,8 |
| alt 1:1 | 0,575 | 7,0 |
| **neu roh** | **0,086** | **46,3** |
| neu 1:1 (Vergleich) | 0,128 | 31,3 |
| neu recon v3 | ausstehend | |
| ManaFly 3 | 0,236 | 16,9 |
| Aether4 (unskaliert) | 0,120 | 33,2 |

Faktor 6,8 gegenüber heute, 2,7x ManaFly.

## Neun Kriterien (Rohkörper; 1:1 nur zum Vergleich)

- neu roh: 26,6 g; Armspitze 57,7 N/mm (alt 8,6); Σ tr 0,186 / λmax 0,047 N mm; f1 332 Hz; verfehlt: bolt_patterns, keep_outs_free, wall_deep_fraction, wall_deep_component, wall_motor_zones, target:symmetry; Warnung: loops. Wandregel: tiefer Anteil 0,86 % (alt 3,09 %), 20 Komponenten, größte 19,3 mm³ (alt 230 mm³). FEA-Oberfläche: Warnung 0,25 mm > 0,20 mm.
- neu 1:1 (Vergleich): 26,3 g; 36,0 N/mm; Σ 0,255 / 0,069; f1 209 Hz; verfehlt: tools_reachable, wall_deep_component, wall_motor_zones, target:f1; FEA-Oberfläche: Warnung 0,34 mm.
- Datenblätter (10 Felder + Neun-Kriterien-Zeile + Oberflächenwarnung): `exports/runs/simp_mma_cov_raw_1/datasheet.md`, `exports/runs/simp_mma_cov_recon_1/datasheet.md`. Die Stufe `evaluation` steht dort als `failed`, weil der Evaluator nach dem Schreiben von evaluation.json beim Ausgeben des Σ-Zeichens auf der cp1252-Konsole abbrach; evaluation.json ist vollständig, behoben in 70921dc.

## Form

Sichtprüfung der Renders: Die Arme sind jetzt zweigurtige Fachwerkträger (Ober- und Untergurt mit Fenstern im Steg), in der Seitenansicht schließt sich ein Rahmen aus Untergurt, Obergurt (Deckschienen) und Pfosten mit Schrägstreben zum Bügel. Datenblatt Feld 2: "Raumfachwerk: Untergurt und Obergurt in der Rumpfmitte". Geschlossene Querschnitte im Sinne von Kastenträgern/Ringen: ja. Diagonalen: in der Seitenebene ja, in der Draufsicht kein durchgehendes X über die Rumpfmitte; die Arme laufen in einen Ring um den Akkubereich.

## Bilder und STLs

- `exports/cov/cov_4views.png` (roh | recon v3 | ManaFly): ausstehend bis v3 steht.
- `exports/cov/recon_1to1_vs_v3.png`: ausstehend.
- STL roh: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov/geometry.stl`; v3: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov_recon/geometry.stl` (ausstehend); 1:1 nur als Vergleich unter `simp_mma_cov_1to1`.

## Abschluss, sobald v3 committet ist

1. `frame_runs` mit `"bodies": ["raw", "recon", "v3"]` (nutzt die fertigen raw/1:1-Läufe, baut v3 aus einem git-archive des recon3-Commits, Evaluator inkl. Σ, Datenblatt, Renders, STL nach `simp_mma_cov_recon`).
2. `compose` (beide Bilder), `cov_compare`, diese Datei ergänzen.
