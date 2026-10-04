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
