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
