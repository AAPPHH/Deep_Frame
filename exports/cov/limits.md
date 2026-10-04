| Rahmen | Rolle | Arm mm | tr roh | λmax roh | tr skaliert | λmax skaliert | tr gekoppelt | λmax gekoppelt | Top-3 λ roh | tr-Anteil Motoren/Akku/Stack | Evaluator tr / λmax (voll) | Abw. voll/diag tr | Abw. voll/diag λmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| manafly3 | references | 80.04 | 1.25 | 0.579 | 1.08 | 0.48 | 1.29 | 0.581 | 0.579, 0.157, 0.141 | 77.9 % / 22.1 % / 0.1 % | 0.859 / 0.286 | -31.1 % / +0.0 % | -50.6 % / +0.0 % |
| aether4 | references | 95.34 | 0.528 | 0.242 | 0.39 | 0.168 | 0.537 | 0.244 | 0.242, 0.0845, 0.064 | 64.6 % / 35.3 % / 0.1 % | 0.391 / 0.143 | -26.0 % / -0.0 % | -40.9 % / +0.0 % |
| simp_mma_raw_1 | candidates | 66.25 | 1.44 | 0.501 | 1.44 | 0.501 | 1.41 | 0.48 | 0.501, 0.3, 0.186 | 79.8 % / 20.2 % / 0.1 % | 1.16 / 0.287 | -19.0 % / -0.0 % | -42.6 % / -0.0 % |
| simp_mma_recon_1 | candidates | 66.25 | 1.45 | 0.52 | 1.45 | 0.52 | 1.4 | 0.491 | 0.52, 0.311, 0.182 | 81.8 % / 18.2 % / 0.1 % | 1.19 / 0.3 | -17.6 % / +0.0 % | -42.2 % / +0.0 % |
| simp_mma_cov_raw_1 | evaluator only | 66.25 | – | – | – | – | – | – | – | – | 0.186 / 0.0474 | – | – |
| simp_mma_cov_recon_1 | evaluator only | 66.25 | – | – | – | – | – | – | – | – | 0.255 / 0.0694 | – | – |
| simp_mma_cov2_raw_1 | evaluator only | 66.25 | – | – | – | – | – | – | – | – | 0.619 / 0.187 | – | – |
| simp_mma_cov3_raw_1 | evaluator only | 66.25 | – | – | – | – | – | – | – | – | 0.687 / 0.22 | – | – |

Grenzen (massgeblich evaluator_full = Referenz x Reserve aus LOAD_COVARIANCE_LIMITS; scaled/raw = strengere Gap-Referenz, nur Vergleich):
- scaled: tr(ΣF) ≤ 0.3901 N mm (aether4), λmax ≤ 0.1682 N mm (aether4)
- raw: tr(ΣF) ≤ 0.5278 N mm (aether4), λmax ≤ 0.2416 N mm (aether4)
- evaluator_full: tr(ΣF) ≤ 0.6872 N mm (manafly3 x 0.8), λmax ≤ 0.2287 N mm (manafly3 x 0.8)

- Definition: F = interface flexibility (42 x 42, LOAD_COVARIANCE order): unit force split equally over the selector nodes, unit moment as minimum-norm couple field about the selector node centroid (Sigma's reference point is taken at that centroid); F_ij = a_i . u_j (work-conjugate). Support per group as the evaluator arm-tip/crash fixtures: stack mount undersides fixed for motor pads, battery, camera; four motor seat undersides fixed for the stack; F is block-diagonal across the two groups (stack <-> rest coupling zero by construction). Mean compliance tr(Sigma F) = sum_k l_k^T F l_k, worst-case compliance lambda_max(Sigma^1/2 F Sigma^1/2), both in N mm. The limits and the evaluator targets use limit_group (36 x 36: motors, battery, camera with the stack fixed, as the optimizer's COVARIANCE support); the stack block is reported under groups and all_dofs (about 0.1 % of tr for ManaFly and Aether4)
- Skalierung auf Arm 66.25 mm: force DOFs of a reference scaled to our arm like the arm-tip slope rule (k * arm_ref / arm_ours): F_scaled = S F S, S = sqrt(arm_ours / arm_ref) on force DOFs, 1 on moment DOFs
- roh/skaliert/gekoppelt: Gap-Finder-Flexibilitaeten (diagonal; gekoppelt = zusaetzlich F-F und F-M innerhalb einer Schnittstelle aus mean_displacement); Evaluator voll = 42 x 42 inkl. Kopplung zwischen Schnittstellen; Abw. = Evaluator voll bzw. diagonal gegen Gap roh.
