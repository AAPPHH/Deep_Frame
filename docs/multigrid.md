# Geometrisches Mehrgitter (GMG-PCG)

Modul `deep_frame/topology_multigrid.py`, Studien `tools/multigrid_study.py {operator,cantilever,truss}`, Belege `docs/validation/multigrid_*.json`.
Verfahren nach Wu, Dick, Westermann 2016: matrixfreier Operator auf dem feinen Gitter, Mehrgitter als Vorkonditionierer für CG.

## Altlast `wip-multigrid` (stash@{2})

Übernommen: 8×24×24-Kindinterpolation und elementweise Galerkin-Komposition `Σ_c P_cᵀ K_c P_c`, Auffüllen auf 2^(L-1), Maske toter Freiheitsgrade über die Diagonale, spaltenweise Abbruchlogik.
Nicht wiederholt:
- dichte Inverse `cp.linalg.inv` am gröbsten Gitter mit 20k Freiheitsgraden (3,2 GB FP64) → jetzt cuDSS-Cholesky auf CSR;
- CuPy-Sparse-Matrizen für P/R → jetzt Faltungen mit Stride 2;
- Toleranz 1e-6 → jetzt 1e-8 auf dem neu berechneten wahren Residuum;
- keine Starrkörperbehandlung → Projektion auf allen Ebenen inkl. gröbster;
- keine Prüfung der Galerkin-Operatoren → Test `R A P` gegen gespeicherte Blöcke.

## Schritt 1: Operator als Faltung

`K(ρ) u = conv_transpose3d(E(ρ) · conv3d(u, K_e·S), S)`, wobei S die Auswahl-Faltung (24 Ausgänge, 2×2×2) ist.
K_e ist die orthotrope Hex8-Matrix der Formulierung, E(ρ) = SIMP p=3, E_min = 1e-6·E; inaktive Zellen haben E = 0.
Randbedingungen werden als Maske auf Freiheitsgraden umgesetzt, die Symmetrieebene über die Fixierung pro Teilproblem.

Nachweis (FP64, je 3 Zufallsvektoren, Felder uniform, binär 30 %, glatt):

| Gitter | Zellen | DOF | max. rel. Fehler voll / frei | Faltung FP64 / FP32 (1 RHS) | CSR SpMV FP64 | Speicher Operator / CSR |
|---|---|---|---|---|---|---|
| Kragträger | 32×6×12 | 9 009 | 3,7e-16 / 3,7e-16 | 0,08 / 0,05 ms | 0,09 ms | 0,02 / 7 MiB |
| Kragträger fein | 64×12×24 | 63 375 | 3,5e-16 / 3,5e-16 | 0,34 / 0,22 ms | 0,27 ms | 0,14 / 54 MiB |
| Frame grob (halb) | 34×64×24 | 170 625 | 3,5e-16 / 3,5e-16 | 7,1 / 0,16 ms | 0,34 ms | 0,42 / 103 MiB |
| Frame 4/3 mm (halb) | 51×96×24 | 378 300 | 3,5e-16 / 3,5e-16 | 11,4 / 0,19 ms | 1,03 ms | 0,91 / 234 MiB |

Kriterium 1e-12 erfüllt. Die FP64-Faltung ist auf der RTX 4080 langsamer als CSR, weil die Karte FP64 nur mit 1/64 Durchsatz rechnet.
Deshalb läuft der Operator im äußeren CG zwar in FP64, die Vorkonditionierung aber in FP32/BF16.
Versionen: torch 2.9.0+cu129, cuDNN 9.10.02, CuPy 14.2.0, cuDSS 0.8.0.10, Python 3.13.5, Treiber 610.88.

## Schritt 2: MG-PCG am Kragträger

V-Zyklus mit Galerkin-Grobgittern (`Σ_c P_cᵀ K_c P_c` je Grobelement, identisch mit `R A P`, Test `test_galerkin_coarse_operator_equals_rap`), Chebyshev-Glätter Grad 3 auf dem feinsten Gitter, ×2 je Ebene, Jacobi-skaliert, λ_max per Potenziteration.
Gröbstes Gitter: cuDSS-Cholesky. Äußeres CG in FP64 mit Polak-Ribière-β (flexibel), Abbruch auf dem neu berechneten wahren Residuum 1e-8.
Arbeitsgenauigkeit des Vorkonditionierers: `float32` (Standard), `float16`/`bfloat16` = FP32-Tensoren mit `torch.autocast` nur im V-Zyklus.
Glattes Dichtefeld einmal bei s1 erzeugt und auf s2–s4 hochgetastet (gleiches physikalisches Feld, SIMP p=3, E_min=1e-6), damit Iterationen gegen Gittergröße vergleichbar sind.

| Gitter | DOF | Ebenen | Iter. FP64 / FP32 / FP16 / BF16 / Jacobi-BF16 | Iter. tief (gröbstes ≤1,5k DOF) | Zeit MG FP32 / cuDSS | Speicher MG / cuDSS |
|---|---|---|---|---|---|---|
| s1 voll | 9 009 | 2 | 8 / 8 / 8 / 9 / 17 | 8 (2) | 0,18 / 0,20 s | 7 / 168 MiB |
| s1 glatt | 9 009 | 2 | 12 / 12 / 13 / 40 / 40 | 12 (2) | 0,17 / 0,16 s | 7 / 142 MiB |
| s2 voll | 63 375 | 2 | 8 / 8 / 9 / 10 / 18 | 8 (3) | 0,45 / 1,19 s | 52 / 1 040 MiB |
| s2 glatt | 63 375 | 2 | 10 / 10 / 18 / 48 / 48 | 11 (3) | 0,38 / 1,17 s | 52 / 1 040 MiB |
| s3 voll | 204 573 | 3 | 8 / 8 / 9 / 15 / 19 | 8 (4) | 0,40 / 5,44 s | 161 / 4 466 MiB |
| s3 glatt | 204 573 | 3 | 12 / 12 / 25 / 89 / 84 | 14 (4) | 0,38 / 5,46 s | 161 / 4 466 MiB |
| s4 voll | 474 075 | 3 | 8 / 8 / 9 / 17 / 20 | 8 (4) | 0,69 / 19,6 s | 342 / 14 371 MiB |
| s4 glatt | 474 075 | 3 | 9 / 9 / 36 / 710 / 157 | 10 (4) | 0,50 / 605 s* | 343 / 14 118 MiB |

Genauigkeit gegen cuDSS (alle Varianten konvergiert, Residuum < 1e-8): Compliance rel. ≤ 3,8e-10, Sensitivitäten ≤ 7,8e-10, Verschiebungen ≤ 2e-8 (FP32), ≤ 1,7e-7 (BF16, s4 glatt) – Verstärkung des Residuums durch die Kondition.
Iterationen bleiben mit FP32 über 53-fache Gittergröße bei 8 (voll) bzw. 9–12 (glatt) und auch bei tieferer Hierarchie konstant.
Autocast lohnt auf der RTX 4080 nicht: FP16 kostet bis 4×, BF16 bis 80× Iterationen (8-Bit-Mantisse reicht für die Glättung bei E-Kontrast 1e6 nicht), FP32 ist zugleich am schnellsten. Standard deshalb `float32`.
Zeit MG = Aufbau + Lösen beider Lastrichtungen; Speicher MG = PyTorch-Spitze, cuDSS = Gerätespeicher (memGetInfo-Differenz; für MG unbrauchbar, da Pool-Freigaben negative Werte liefern).
\* cuDSS s4 glatt: 597 s Faktorisierung bei 14 GB auf der geteilten 16-GB-Karte (Speicherdruck); derselbe Fall im Lauf davor 15,0 s.
Beleg `docs/validation/multigrid_cantilever.json`.

## Schritt 3: dünnes Fachwerk und reales SIMP-Feld

Fachwerk: 128×16×32 Zellen à 1,25 mm (Halbmodell, 217 107 DOF), Stäbe 2,5 mm (2 Zellen), sonst Leerraum mit E_min=1e-6 (SIMP p=3), eingespannt bzw. frei schwebend mit Trägheitsentlastung.
Frame: Lastmodell-Feld `simp_mma_cov3_opt/fine/density_half.npz`, 4/3 mm, 51×96×24 (279 618 DOF), Fälle thrust_all, crash_front, crash_side_left (frei schwebend) und stiffness_arm_tip (eingespannt).

Ursache der Iterationszunahme (Fachwerk eingespannt, FP32, Iterationen bis 1e-8):

| E_min | 2 Ebenen | 3 | 4 | 5 | Rediskretisierung 3 / 4 / 5 |
|---|---|---|---|---|---|
| 1e-6 | 34 | 74 | 104 | 105 | 103 / 113 / 116 |
| 1e-4 | 30 | 63 | 85 | 85 | 87 / 94 / 95 |
| 1e-2 | 18 | 23 | 24 | 25 | 26 / 26 / 26 |

Bei E_min=1e-2 ist die Tiefe egal, bei 1e-6 wachsen die Iterationen bis zur Ebene, deren Zellen breiter als der Stab sind (5 mm > 2,5 mm), danach nicht mehr.
Ursache ist die Grobgitterkorrektur: die feste trilineare Interpolation kann die Verformung eines Stabs im Leerraum nicht abbilden, sobald eine Grobzelle Stab und Leerraum überspannt; Galerkin mildert das (74 statt 103 bei 3 Ebenen), ersetzt die Interpolation aber nicht.
Ausgeschlossen: λ_max-Schätzung (100 statt 20 Potenziterationen: 104 = 104), Glätter (Chebyshev Grad 6: −15 bis −25 %, Jacobi +25 %, Wachstum ×4: −10 %, alle ohne Zeitgewinn), W-Zyklus (−35 % Iterationen, 1,7× Zeit).
Abhilfe innerhalb der Vorgabe: Tiefe über die Größe des gröbsten Gitters begrenzen. Standard `coarsest_dofs` jetzt 80 000 (Schritt 2 lief mit 20 000; Ebenen stehen je Zeile im Beleg).

Fachwerk (FP32, Standard = 2 Ebenen):

| Variante | eingespannt Iter. / Zeit | schwebend Iter. / Zeit | Kernprüfung je Ebene |
|---|---|---|---|
| Galerkin, Projektion | 34 / 1,65 s | 33 / 1,73 s | 5e-10, 6e-10 |
| Galerkin FP64 / FP16 / BF16 | 34 / 39 / 97 | 33 / 46 / 124 | FP64 7e-18, 2e-17 |
| Rediskretisierung | 37 | 35 | |
| Galerkin 3 Ebenen / Rediskr. 3 Ebenen | 74 / 102 | 70 / 99 | |
| Galerkin 4 Ebenen / Rediskr. / W-Zyklus | 104 / 113 / 66 | 99 / 115 / 65 | 5e-10 … 2e-9 |
| statisch bestimmte Lager statt Projektion | 34 | 33 | – |
| Projektion nur auf feinen Ebenen | 34 | cuDSS-Fehler 29833 (gröbstes Gitter singulär) | |
| cuDSS | 5,5 s, 4,5 GB | 5,5 s, 4,5 GB | |

Frame 4/3 mm (vier Fälle, 6 rechte Seiten, Zeit inkl. Aufbau):

| Variante | Ebenen / gröbstes DOF | Iter. schwebend / eingespannt | Zeit | PyTorch-Spitze | rel. Diff. Compliance / Sensitivität / u (Material) |
|---|---|---|---|---|---|
| Galerkin FP32 (Standard) | 2 / 42 501 | 18 / 19 | 4,5 s | 580 MiB | 6e-11 / 2e-10 / 3e-10 |
| Galerkin FP64 | 2 / 42 501 | 18 / 19 | 5,2 s | 326 MiB | 6e-11 / 2e-10 / 3e-10 |
| Galerkin FP16-Autocast | 2 / 42 501 | 76 / 19 | 7,1 s | 580 MiB | 7e-11 / 2e-10 / 3e-10 |
| statisch bestimmte Lager | 2 / 42 504 | 19 / 19 | 4,3 s | 560 MiB | 6e-12 / 1e-10 / 5e-11 |
| Galerkin 3 Ebenen | 3 / 6 768 | 63 / 63 | 4,3 s | 423 MiB | 1e-10 / 2e-9 / 2e-9 |
| Galerkin 5 Ebenen | 5 / 312 | 113 / 107 | 12,4 s | 646 MiB | 2e-10 / 6e-10 / 1e-9 |
| Rediskretisierung 2 / 3 Ebenen | 2 / 3 | 2000 n. konv. / 302 bzw. 362 | 70 / 95 s | | 6e-5 / 2e-4 |
| Projektion nur feine Ebenen | 2 | cuDSS-Fehler 42495 | | | |
| cuDSS | direkt | – | 12,3 s (5,1 s Faktor.+Lösen) | 13,0 GB Gerät | Referenz |

Warum Galerkin: rediskretisierte Grobgitter (gemittelte Dichte, neue K_e) haben die Starrkörpermoden nicht mehr im Kern (Kernprüfung 2e-3/8e-3 statt 1e-9), die Projektion wird inkonsistent, schwebende Fälle konvergieren nicht; auch eingespannt 300–360 statt 19–63 Iterationen.
Starrkörperprojektion: auf jeder Ebene orthonormierte Starrkörpermoden (3 im Halbmodell), am gröbsten Gitter zusätzlich gepinnte Freiheitsgrade per pivotisierter QR; ohne Projektion am gröbsten Gitter scheitert die Cholesky-Zerlegung.
Gefundener Fehler: die Starrkörperverschiebung nach dem Lösen wurde auch auf tote Knoten (nur inaktive Elemente) angewandt → u-Abweichung 0,59 bei exakter Compliance; jetzt maskiert, Abweichung 1,4e-8 (Material 2,5e-10).
FP16-Autocast verdoppelt bis vervierfacht die Iterationen auf dem Frame, BF16 ist unbrauchbar; Standard bleibt FP32 im Vorkonditionierer, FP64 im äußeren CG.
Offen: bei 2 Ebenen kostet eine Iteration auf dem Frame ≈ 55 ms (3 Ebenen: 11 ms); vermutete Ursache ist die Host-Rundreise zum gröbsten cuDSS-Gitter je V-Zyklus plus die Substitution mit 42k DOF (nicht gemessen). GPU-residente Übergabe per DLPack ist der nächste Schritt.
Beleg `docs/validation/multigrid_truss.json` (Befehl `tools/multigrid_study.py truss`).

## Schritt 5: f1 per LOBPCG mit MG-Vorkonditionierer

`GeometricMultigrid.eigenpairs` ist ein eigenes Block-LOBPCG (torch.lobpcg nimmt nur Tensoren, keine Operatoren) für K φ = λ M φ, alles in FP64. K ist der matrixfreie Operator, M ist matrixfrei als Faltung mit der konsistenten Hex8-Massenmatrix plus diagonalen Punktmassen. Vorkonditionierer ist der V-Zyklus aus Schritt 2–3 (FP32, Galerkin, eingespannt, ohne Projektion).
Basis [X, W=T(R), P], B-Orthonormierung je Block per SVQB mit Verwerfen kleiner Gram-Eigenwerte, Rayleigh-Ritz über die Gram-Matrizen, weiches Locking (W nur aus nicht konvergierten Spalten; der V-Zyklus läuft immer auf dem ganzen Block, damit der cuDSS-Faktor des gröbsten Gitters nicht je Blockgröße neu entsteht), Produkte alle 10 Schritte neu berechnet. Blockgröße = `modes` + 2 Wächter, Abbruch, wenn alle `modes` das relative Residuum ‖Kx−λMx‖/‖Kx‖ < 1e-8 erreichen.
Formulierung unverändert übernommen (`MultigridModal(ModalConstraint)`): getrennte Interpolation Steifigkeit SIMP p=3 / Masse linear mit ρ⁶/c⁵ unter c=0,1, Punktmassen aus `ModalConstraint.lumped`, je Symmetrieteil (symmetrisch/antisymmetrisch) eigene Hierarchie, KS (s=40) über die `tracked` Moden beider Teile und Sensitivität φᵀ(dK−λ dM)φ mit M-normierten Vektoren über die gemeinsame Methode `ModalConstraint.aggregate`.
Beide Studienwege nutzen dasselbe Dichtefeld für Steifigkeit und Masse.
Referenz: Shift-Invert-Lanczos (`eigsh`, tol 0) mit dem cuDSS-Faktor (Residuum ≤ 3,6e-9); zusätzlich der bisherige cuDSS-Pfad der Formulierung (Unterraumiteration, 30 = Standard bzw. 300 Iterationen).
Kragträger: Lastfall `modes` an der Einspannung, 2 g Spitzenmasse, glattes Feld wie Schritt 2 (s1–s3). Frame: Lastmodell-Feld `simp_mma_cov3_opt` 4/3 mm, Fall `modes` (Motorsitz-Unterseiten fest), 8 Punktmassen (Akku, AIO, Kamera), 6 Moden, 3 verfolgt.

| Fall | DOF | f1 | LOBPCG-Iter. sym/anti | Eigenwerte (alle 6×2) rel. | f1 / KS rel. | Sensitivität rel. | Zeit MG / Lanczos / Unterraum 30 | Speicher MG Spitze / Gerät; cuDSS Gerät |
|---|---|---|---|---|---|---|---|---|
| s1 | 9 009 | 39,98 Hz | 14 / 13 | 3,4e-11 | 1e-13 / 1e-12 | 5,9e-11 | 0,52 / 0,17 / 0,28 s | 36 / 160 MiB; 168 MiB |
| s2 | 63 375 | 41,12 Hz | 30 / 27 | 6,2e-11 | 1e-12 | 1,1e-10 | 1,29 / 1,58 / 2,43 s | 207 / 412 MiB; 1 040 MiB |
| s3 | 204 573 | 39,54 Hz | 21 / 29 | 2,4e-10 | 4e-12 | 2,2e-10 | 3,04 / 6,07 / 9,49 s | 663 / 1 322 MiB; 4 468 MiB |
| Frame 4/3 mm | 378 300 (277 579 / 275 876 frei) | 374,96 Hz | 21 / 21 | 2,1e-11 | 1e-12 | 1,1e-11 | 5,18 / 6,38 / 11,17 s | 1 258 / 2 308 MiB; 5 010 MiB |

Ziel 1e-6 für f1 und die tiefsten Moden um mehr als vier Größenordnungen erfüllt; FP64-Vorkonditionierer liefert dieselben Iterationen (21/21 am Frame) bei 14 % mehr Zeit.
Fast gleiche Moden: Kragträger s3 sym. 59,51 / 60,74 / 61,72 Hz, Frame 534,12 Hz (sym.) gegen 534,73 Hz (anti.) – die KS-Sensitivität stimmt trotzdem auf 1e-11 bis 2e-10.
Nebenbefund: der Standardpfad der Formulierung (30 Unterraumiterationen) trifft die Eigenwerte, die Sensitivität aber nur auf 5e-6 (s2) bzw. 1,5e-5 (s3), weil die Eigenvektoren noch nicht konvergiert sind; mit 300 Iterationen 1e-11, dann aber 54 s (s3) bzw. 64 s (Frame).
Grenzen: kleine Gitter sind mit MG langsamer (s1); Zeit inkl. Hierarchieaufbau und cuDSS-Faktor des gröbsten Gitters je Teil (Frame 0,5 s); Warmstart über `part["vectors"]` vorhanden, aber nicht gemessen (jede Zeile kalt, Zufallsstart mit Seed 0).
Beleg `docs/validation/multigrid_modal.json` (Befehl `tools/multigrid_study.py modal`), Test `test_lobpcg_modes_match_shift_invert`.

## API

`mg = GeometricMultigrid(shape, spacing, ke, settings)`; `mg.update(moduli)` je Optimierungsschritt (baut Hierarchie und gröbstes cuDSS-Gitter neu, Frame 1,0 s bei 2 Ebenen);
`u, report = mg.solve(key, fixed, forces[ndof, nrhs], support=None)` – gebündeltes PCG: alle rechten Seiten teilen sich einen V-Zyklus-Aufruf, α/β je Spalte, konvergierte Spalten fallen heraus (kein Block-Krylov).
`solve_elasticity(system, mg, density, p, e_min)` liefert pro Lastfall Compliance und Ableitung wie `HexElasticity.solve`.
`GeometricMultigrid(..., settings, me)` aktiviert den Massenoperator; `MultigridModal(system, modal_settings, mg).measure(density, p, e_min)` liefert (Verletzung, Gradient, Info) wie `ModalConstraint.measure`.
