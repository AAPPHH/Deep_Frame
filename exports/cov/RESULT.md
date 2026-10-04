# Frame mit Lastmodell als Verteilung, Kalibrierschleife (SIMP+MMA, 4/3 mm)

Stand 04.10. 09:05. Kalibrierschleife abgeschlossen und stabil (3 Läufe). Ergebnis ist jetzt **recon v3b des Endlaufs, 17,6 g** (`simp_mma_cov3_v3_2`), aus dem Rohkörper 15,7 g (`simp_mma_cov3_raw_1`). v3b = feature/recon-v3b b74dcf2 (af19376 + Querschnittsorientierung aus Schnittebenen, Gabelstege aus dem Rohkörper, Rohkörper-Hülle). **v3b erfüllt λmax (0,207 N mm, 90 % der Grenze) und alle neun Kriterien; Armspitze −4 % und λmax −6 % liegen innerhalb 10 % vom Rohkörper, Masse +12 %, f1 +16 % und tr −12 % (steifer) nicht.** Das wird berichtet, nicht ersetzt. Der frühere v3 (af19376, `simp_mma_cov3_v3_1`, λmax 108 %) steht weiter unten zum Vergleich; 1:1 bleibt nur Vergleich.

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
| recon v3 (af19376) | – | – | – | 17,63 g |
| **recon v3b (b74dcf2)** | – | – | – | **17,62 g** |

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
| Endlauf recon v3 af19376 (Ersatz-FEA) | 17,6 g | 0,646 | 0,247 | 0,747 | 0,297 | tr erfüllt (94 %), λmax verfehlt (108 %) |
| **Endlauf recon v3b b74dcf2** (Standard-FEA) | **17,6 g** | **0,602** | **0,207** | 0,705 | 0,264 | **erfüllt (88 % / 90 %)** |
| Endlauf recon v3b (Ersatz-FEA) | 17,6 g | 0,600 | 0,206 | 0,703 | 0,263 | erfüllt (87 % / 90 %) |
| Endlauf roh (Ersatz-FEA, Vergleichsbasis) | 15,7 g | 0,686 | 0,220 | 0,807 | 0,304 | erfüllt (99,8 % / 96 %) |
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
| Endlauf recon v3 af19376 | 0,307 | 13,0 |
| **Endlauf recon v3b** | **0,287** | **13,9** |
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
  - Die Fehlpunkte sind dieselben wie beim 26,6-g-Rohkörper. Bohrbilder und Keep-outs verfehlt der Rohkörper grundsätzlich; v3 erfüllt sie (siehe oben).
- **Endlauf 1:1** (`simp_mma_cov3_recon_1`, nur Vergleich):
  - 16,3 g; Bohrbilder 20/20.
  - FEA nicht lösbar: gmsh scheiterte in allen 6 Vernetzungsversuchen. Deshalb fehlen fea_solved, die Crash-Festigkeiten und die Zielwerte (Armspitze, f1, Σ).
- **Datenblätter** (10 Felder + Neun-Kriterien-Zeile + Oberflächenwarnung): `exports/runs/simp_mma_cov3_raw_1/datasheet.md`, `exports/runs/simp_mma_cov3_recon_1/datasheet.md`, `exports/runs/simp_mma_cov3_v3_1/datasheet.md`.

## Rekonstruktion v3b des Endlaufs (Armwurzeln, Gabelstege)

**Ursache beim alten v3 (af19376).** Lastfall Armspitze: Kraft am vorderen linken Motor, Lagerung an den vier AIO-Standoffs unten in der Mitte. Der Lastpfad läuft über die Armwurzel in eine senkrechte Wand des Rohkörpers (x ≈ ±9…14, y ≈ 19, z 4…17) zwischen oberem Querriegel, Pfosten und unterem Querriegel. Das Skelett macht daraus einzelne Stäbe. Schnitte normal zur Gliedachse, roh gegen af19376:

- Glied 18 (Knoten → AIO-Standoff vorne links): roh 99 mm², 18,4 mm hoch; v3 36 mm², 6,2 mm.
- Glied 16 (oberer Querriegel) bei x = −9: roh Iv 1967 mm⁴, 17,3 mm hoch; v3 Iv 64 mm⁴, 5,3 mm.
- Glied 20 (unterer Querriegel) bei x = +8,3: roh 109 mm²; v3 29 mm².
- Arm-Mitte vorn: gleiche Fläche, aber roh hochkant (h 7,5 / b 5,8 mm, Iv 105–129 mm⁴), v3 flach (5,9 / 7,3 mm, Iv 72–82 mm⁴). Grund: ein Seitenverhältnis (Median) und eine mittlere Hauptachse je Glied.

**Änderungen (feature/recon-v3b, je ein Commit, Tests dazu):**

1. 48c517a: Seitenverhältnis und Hauptachse je Spline-Stützstelle aus Schnittebenen normal zur Kurve (zentrierte Flächenmomente des Rohkörpers, 2 mm geglättet). Die Flächenkurve bleibt unverändert. Der Arm stimmt jetzt mit roh überein (Iv 116–119 gegen 105–129 mm⁴). Allein bringt das aber nichts: Armspitze 9,17 N/mm, f1 515 Hz.
2. b74dcf2: Gabelstege. Zwei Glieder, deren Enden ≤ 3 mm auseinander liegen und die sich um ≤ 120° spreizen, bekommen einen Steg. Er läuft zwischen den beiden Splines vom Gabelpunkt so weit, wie der Rohkörper die Verbindungslinie zu ≥ 70 % füllt. Die Form kommt aus dem Rohkörper, mit einem Kern von 2,5 mm. Gefunden werden 24 Stege, 2–5 mm über die Gliedradien hinaus. Rohfeld ist jetzt die Rohkörper-Belegung statt der Dichte: die Dichte-Isofläche ist 17,1 cm³ groß, der Rohkörper 14,4 cm³. Splines dürfen höchstens 0,5 mm über den Rohkörper hinausragen, Schalen und Knoten nicht. Mindestkerne 2,5 mm bleiben, ebenso Spline-Glieder und der feste Übergangsradius 1,5 mm. Keine Skalierung.

**Variantenreihe (Ersatz-FEA, gleiche Einstellungen; roh 15,72 g / 10,82 N/mm / 407 Hz):**

| Variante | Masse | Armspitze | f1 |
|---|---|---|---|
| af19376 (alt) | 17,63 g (+12,2 %) | 9,40 (−13,2 %) | 521 (+28 %) |
| A: nur Schnittorientierung | 17,62 g (+12,1 %) | 9,17 (−15,3 %) | 515 (+27 %) |
| C: + Stege aus der Dichte | 19,53 g (+24 %) | 11,36 (+5,0 %) | 540 (+33 %) |
| F: + Rohkörper als Feld, Hülle 0,5 mm überall | 18,04 g (+14,7 %) | 10,26 (−5,2 %) | 484 (+19 %) |
| G: Hülle 0 mm überall | 16,66 g (+6,0 %) | 7,81 (−27,8 %) | 436 (+7 %) |
| H: Hülle 0,25 mm überall | 17,39 g (+10,6 %) | 9,04 (−16,5 %) | 464 (+14 %) |
| I: Stäbe frei, Schalen/Knoten 0 mm | 18,17 g (+15,6 %) | – | – |
| **J = v3b: Stäbe 0,5 mm, Schalen/Knoten 0 mm** | **17,70 g (+12,6 %)** | **9,84 (−9,1 %)** | **472 (+16 %)** |

Wird v3 enger an den Rohkörper gelegt, sinkt die Steifigkeit schneller als die Masse. Bei G fehlen die Teile des Rohkörpers außerhalb der Spline-Ellipse, vor allem am Übergang Arm–Pad. Auf Rohmasse verliert v3 also etwa 25 % Steifigkeit, auf Rohsteifigkeit kostet es etwa 15 % Masse.

Die Mehrmasse liegt nicht in den festen Bereichen: innerhalb der Preserve-Maske roh 4,24 / v3b 4,87 cm³, außerhalb roh 10,17 / v3b 11,30 cm³ (+11 %). Ein Teil kommt aus der 2,5-mm-Mindestbreite, denn der Rohkörper unterschreitet sie: Akkuschienen-Stützen 2,1 mm (Glieder 8, 31), oberer Querriegel hinten 2,1 × 2,9 mm (Glied 17), unterer Hinterarmgurt 1,5 mm, AIO-Steg unten 0,5–0,9 mm (Glied 14).

**f1 ist ein Zielkonflikt.** Der Modalfall spannt das vordere linke Motorpad ein, der Akku (37 g) sitzt auf den Schienen. Die Schienenstützen des Rohkörpers sind dünner als 2,5 mm. Mit 2,5 mm werden sie steifer, f1 steigt. Jede Variante liegt bei +14 bis +28 %, außer G, und G bezahlt das mit −28 % Armspitze. Das ist kein Fehler, den ich weiter tune. Es wird berichtet.

**Ergebnis v3b** (`simp_mma_cov3_v3_2`, b74dcf2, 1 Körper, 15 Knoten, 30 Glieder, 7 Schalen, 24 Stege, Kontinuität und exakte Booleans bestanden). Die Standard-FEA hat v3b diesmal vernetzt. Daher steht unten Standard gegen Standard und Ersatz-FEA gegen Ersatz-FEA (`evaluation_fallback/`, identische Einstellungen wie beim Rohkörper):

| Größe | roh Std | v3b Std | Abw. | roh Ersatz | v3b Ersatz | Abw. | innerhalb 10 %? |
|---|---|---|---|---|---|---|---|
| Masse | 15,72 g | 17,62 g | +12,1 % | 15,72 g | 17,62 g | +12,1 % | nein |
| Armspitze | 10,88 N/mm | 10,46 N/mm | −3,9 % | 10,82 | 10,40 | −3,9 % | ja |
| f1 | 407,0 Hz | 472,8 Hz | +16,2 % | 407,0 | 472,8 | +16,2 % | nein |
| Σ tr voll | 0,687 N mm | 0,602 N mm | −12,4 % | 0,686 | 0,600 | −12,5 % | nein (steifer) |
| Σ λmax voll | 0,220 N mm | 0,207 N mm | −5,7 % | 0,220 | 0,206 | −6,4 % | ja |
| Verwindung | 12,2 N/mm | 13,9 N/mm | +14 % | | | | |

Gegen die ManaFly-×-0,8-Grenzen: tr 0,602 ≤ 0,687 (88 %) und **λmax 0,207 ≤ 0,229 (90 %), beide erfüllt**. Alt (af19376) war λmax 108 %.

**Neun Kriterien v3b** (`exports/runs/simp_mma_cov3_v3_2/datasheet.md`, Standard-FEA):

- Verfehlt: keine. Warnung: cog_offset.
- Bohrbilder 20/20, Passung ok, Keep-outs frei.
- Wandregel bestanden: 0,03 % / 2,8 mm³, Motorzonen 0.
- Spannungen front 6,3 / arm 13,5 / back 2,2 MPa (SF 2).
- Überhang 20 %.
- FEA-Oberfläche: Warnung, 0,31 mm > 0,20 mm (nur Rechenmodell).

**Neural-Gegenprobe** (recon3 `neural_v06_f1`, v3b b74dcf2 gegen 879713d; roh 19,60 g, Ersatz-FEA 4,56 N/mm / 330 Hz):

| | Masse | Armspitze | f1 |
|---|---|---|---|
| 879713d | +9,5 % | −7,9 % | +2,0 % |
| v3b | **+4,2 %** | **+30,6 %** (5,96 N/mm) | +7,9 % |

Die Masse ist besser. Die Armspitze liegt nicht mehr innerhalb 10 %, und zwar auf der steifen Seite. Gegen den 10-%-Maßstab ist das eine Regression, gegen das Bauteil keine. Vermutete Ursache: Stege und 2,5-mm-Kerne an dünnen Rohgliedern. Nicht weiter untersucht. Ein Basislauf von af19376 auf dieser Dichte (ohne Sitze) wurde aus Zeitgründen nicht gerechnet; der 879713d-Wert stammt noch aus dem Lauf mit Sitzen.

## Symmetrie-Kriterium (target:symmetry)

**Was gemessen wird:** 20 000 Oberflächenpunkte von `frame.stl` werden an x = mittleres Motor-x (= 0) gespiegelt. Gemessen wird der RMS-Abstand zur Originaloberfläche. Grenze: RMS / Radstand ≤ 0,0025, also 0,331 mm bei 132,5 mm. Der Evaluator ist korrekt: Ebene, Metrik und Netz passen, und die Grenze liegt bei einer halben feinen Zelle (0,667 mm). Er bleibt unverändert.

**Ursache (Extraktion, nicht Evaluator):** `tools/neural_study.py surface()` setzt die Marching-Cubes-Ecken nach dem 1-Zellen-Pad auf `origin − spacing` statt `origin − spacing/2`. Dadurch liegt der Rohkörper in x, y und z um eine halbe feine Zelle (0,333 mm) zu tief: x −61,52 … +60,85, z_min −0,28. Die Dichte `density_fine.npz` ist exakt spiegelsymmetrisch (max |d − d[::−1]| = 0). Allein diese Verschiebung ergibt 0,33 mm RMS. Dieselbe Zeile steht in `tools/multi_crash_study.py`.

**Fix 544334c:** `origin − spacing/2` in beiden Tools, dazu ein Test (Zellflächen auf dem Gitter, Spiegelgrenzen). Den Rohkörper habe ich mit derselben Dichte neu extrahiert: `exports/cov/symmetry/raw_fixed/`, x ±61,19, z_min 0,06, 15,65 g. Geometrie und Wandregel liefen neu. FEA, Σ und Slicer stammen aus `simp_mma_cov3_raw_1`, denn das Netz ist nur um 0,33 mm verschoben und wurde nicht neu vernetzt. Der Ray-Head nahm keine Jobs an („No available agent“), deshalb liefen beide CPU-Schritte direkt.

| Körper | RMS mm | p95 mm | max mm | RMS/Radstand | ≤ 0,0025 | Verfehlt |
|---|---|---|---|---|---|---|
| roh vorher (`simp_mma_cov3_raw_1`) | 0,334 | 0,649 | 0,722 | 0,00252 | nein | bolt_patterns, keep_outs_free, wall_deep_fraction, wall_deep_component, target:symmetry |
| **roh nach Fix** (`symmetry/raw_fixed`) | **0,052** | 0,112 | 0,202 | **0,00040** | **ja** | bolt_patterns, keep_outs_free, wall_deep_fraction, wall_deep_component (Warnung neu: section_ratio H/B 0,98) |
| recon 1:1 | 0,156 | 0,384 | 1,119 | 0,00118 | ja | (FEA-Kriterien, siehe oben) |
| recon v3 (af19376) | 0,364 | 0,648 | 2,457 | 0,00275 | nein | target:symmetry, target:sigma_worst |
| **recon v3b** (vorher = nachher, STL unverändert) | 0,225 | 0,508 | 1,208 | 0,00170 | ja | keine |
| ManaFly 3 (`docs/validation`, 5000 Punkte, Radstand 160,1 mm) | 0,038 | 0,048 | 0,601 | 0,00024 | ja | keine |

v3b liest den Rohkörper `geometry.stl` als `section_body` (Querschnittsorientierung, Gabelstege, Rohkörper-Hülle). Ein Teil der alten Verschiebung steckt also noch in v3b (Schwerpunkt x +0,14 mm). Das verschwindet erst, wenn v3b aus dem neu extrahierten Rohkörper gebaut wird. Diesen Neubau habe ich nicht gestartet. Zahlen: `exports/cov/symmetry/symmetry.json`.

## Rekonstruktion v3 des Endlaufs (af19376, abgelöst durch v3b)

**Quelle.** v3 wird per `git archive` gebaut. recon3 (879713d) zweigt vor dem Merge 2bafe8f ab und hat deshalb noch XT30-, Balancer- und Antennensitze als Preserve. Der erste v3-Bau enthielt sie: 4 Körper, drei lose Sitzblöcke hinten in der Mitte (264 / 226 / 160 mm³), exakte Booleans nicht bestanden. Dieser Bau ist verworfen (`exports/runs/simp_mma_cov3_v3_seats_discarded`).

Neu gebaut auf Commit af19376 (Branch `recon-v3-noseats`): 879713d plus genau die Änderung von 2bafe8f, konfliktfrei. Sonst keine Änderung an v3. Ergebnis: 1 Körper, 15 Knoten, 30 Glieder, 7 Schalen, Kontinuität bestanden, exakte Booleans bestanden. Override: `exports/cov/post_v3.json`.

**FEA.** Die Standard-Evaluator-FEA konnte v3 nicht vernetzen (remesh 2,0 mm: Volumenabweichung −1,015 % > 1 %; refine/direct: Elementbudget 9728 MB überschritten; classify: Topologiefehler). Daher läuft die recon3-Ersatz-FEA (`fea_memory_budget_mb` 16384, remesh 2,0/1,5/1,2/1,0 mm) **identisch für roh und v3**, inkl. Σ (`evaluation_fallback/` je Lauf). Bei beiden greift `refine_hxt` bei 2,0 mm. Roh mit Ersatz-FEA weicht ≤ 0,6 % vom Standardwert ab.

| Größe | roh (Ersatz-FEA) | recon v3 (Ersatz-FEA) | Abweichung | innerhalb 10 %? |
|---|---|---|---|---|
| Masse | 15,72 g | 17,63 g | +12,2 % | nein |
| Armspitze | 10,82 N/mm | 9,40 N/mm | −13,2 % | nein |
| f1 | 407,0 Hz | 521,3 Hz | +28,1 % | nein |
| Σ tr voll | 0,686 N mm | 0,646 N mm | −5,7 % | ja |
| Σ λmax voll | 0,220 N mm | 0,247 N mm | +12,5 % | nein |
| Σ tr / λmax diag. | 0,807 / 0,304 | 0,747 / 0,297 | −7,3 % / −2,1 % | ja / ja |
| Verwindung | 12,2 N/mm | 13,0 N/mm | +6,7 % | ja |

Gegen die ManaFly-×-0,8-Grenzen: tr 0,646 ≤ 0,687 (94 %) erfüllt, **λmax 0,247 > 0,229 (108 %) verfehlt**. Der Rohkörper lag schon bei 96 %; v3 verschiebt Steifigkeit (tr und Verwindung besser, schwächste Richtung schlechter). f1 steigt stark (die ersten Moden 407/408 Hz des Rohkörpers sind ein Paar, bei v3 liegt das Paar bei 521/536 Hz). Das Muster ähnelt dem alten 14,6-g-Nachweis (+12,4 % Masse, −19 % Armspitze), nur ohne f1-Verlust.

**Neun Kriterien v3** (`exports/runs/simp_mma_cov3_v3_1/datasheet.md`, Bewertung aus der Ersatz-FEA):

- Werte: 17,6 g; Armspitze 9,4 N/mm; Σ tr 0,646 / λmax 0,247 N mm; f1 521 Hz; Spannungen front 6,3 / arm 12,1 / back 2,0 MPa (SF 2).
- Verfehlt: target:symmetry, target:sigma_worst.
- Gegenüber roh neu erfüllt: bolt_patterns (20/20, Passung ok), keep_outs_free, wall_deep_fraction, wall_deep_component (Wandregel: tiefer Anteil 0,00 %).
- Warnungen: loops, cog_offset. FEA-Oberfläche: Warnung, 0,28 mm > 0,20 mm (nur Rechenmodell).
- Die Manifest-Stufe `evaluation` steht auf failed (Standard-FEA); `evaluation.json` und Datenblatt enthalten die Ersatz-FEA.

**Sichtprüfung v3** (`cov_4views.png`, `recon_1to1_vs_v3.png`):

- Topologie wie roh: vier Arme, zwei Akkuschienen, Querriegel, Bügel. Kein Glied fehlt; Schlaufen 15 statt 16.
- Seitenansicht: der Fachwerkträger mit Pfosten und Schrägstreben vorn und hinten bleibt erhalten, ebenso die Diagonalen.
- Draufsicht: wie roh kein X über die Rumpfmitte. Der hintere Querriegel ist bei v3 eine flache Schale zwischen den Schienen.
- Querschnitte: Glieder als gefüllte, runde bis ovale Splines (geschlossen, keine C-Profile). Arme glatter und gleichmäßiger dick als roh. Die Motorpads haben die vollen Bohrbilder.
- Gegen 1:1: gleiche Gliederführung, v3 glatter, dickere Knoten.

## Form (Sichtprüfung der Renders, Rohkörper)

- **Seitenansicht:** geschlossener Rahmen aus Untergurt, Deckschienen als Obergurt, Pfosten und Schrägstreben vorn und hinten. Ein geschlossener Querschnitt in der Seitenebene, also ein Fachwerkträger mit Diagonalen.
- **Draufsicht:** Die Arme laufen in einen Leiterrahmen aus den zwei Akkuschienen mit drei Querriegeln (vorn, Mitte, hinten). Es gibt kein durchgehendes X über die Rumpfmitte; Torsion übernimmt der geschlossene Ring um den Akkuschacht.
- **Rückseite:** keine Sitze mehr; das Antennenauge und die Steckersitze sind weg.
- **Arme:** einzelne, im Querschnitt gefüllte Holme; die Fenster im Steg des 26,6-g-Laufs sind verschwunden.

## Bilder und STLs

- Rohkörper-STL: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov3/geometry.stl` (= `simp_mma_cov3_raw_1/frame.stl`).
- v3b-STL (Ergebnis): `/c/clones/Deep_Frame-neural/exports/simp_mma_cov3_v3/geometry.stl` (= `simp_mma_cov3_v3_2/frame.stl`; der alte v3 liegt in `simp_mma_cov3_v3_1/frame.stl`).
- 1:1 nur als Vergleich: `/c/clones/Deep_Frame-neural/exports/simp_mma_cov3_1to1/geometry.stl` (= `simp_mma_cov3_recon_1/frame.stl`).
- `exports/cov/cov_4views.png` (roh | v3b | ManaFly; iso/top/side/front) und `exports/cov/recon_1to1_vs_v3.png` (1:1 | v3b).
- Renders je Lauf: `exports/runs/simp_mma_cov3_{raw,v3}_1/renders/{iso,top,side,front}.png`.

## Reproduktion

1. v3b: `python tools/formulation_study.py frame_runs exports/cov/post_v3b.json` (`v3.ref` = `feature/recon-v3b`, b74dcf2; vorher `v3` aus `runs.json` entfernt, alte Fassung `runs_v3_af19376.json`), dann `compose` und `cov_compare` mit derselben Datei (alte `comparison.json` als `comparison_v3_af19376.json`). Alt: dasselbe mit `post_v3.json` (`recon-v3-noseats`).
2. Ersatz-FEA für roh und v3: `evaluation/frame.json` mit `fea_memory_budget_mb` 16384 nach `evaluation_fallback/`, Teile `fea` und `sigma` über Ray (Typ `reconstruction`), dann `report`; v3-`evaluation.json` in den Lauf kopiert und `run.py datasheet` neu.
