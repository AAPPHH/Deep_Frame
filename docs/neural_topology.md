# Neural topology optimization (TOuNN-style proof of concept)

The historical TOuNN volume-constrained prototype below remains available. The neural route of the shared pipeline is `NeuralOptimizer` in `deep_frame/topology_optimizers.py`, an Adam/augmented-Lagrangian optimizer in the network parameters on the same `TopologyProblem` formulation as SIMP/MMA (free camera contact, ideally pressed battery, covariance, crash and modal constraints, mass-only objective, lens clearance at 15° nose down as a hard forbidden halfspace). The design is x = clip((sigmoid(z) − sigmoid(−4)) / (1 − 2 sigmoid(−4)), 0, 1) of a Fourier-feature MLP on the free cells, so the box is [0, 1] as in SIMP; a logit-bound penalty (weight 100, mean over cells) keeps |z| near 4. Each iteration takes one Adam step on L = f0 + Σ AL terms + logit penalty, with the rate 0.01 × 0.7^level; the step is scaled in parameter space until the largest cell change is at most the MMA move limit of the level. Per-constraint penalties start at 1 and double up to 10 when a violation does not halve; multipliers are updated every 5 iterations once the design change has settled (at the latest every 50). Multipliers, penalties and Adam moments are kept across β levels; only the progress baseline is cleared at a level change. An intermediate level additionally needs an active constraint (max g ≥ −0.05) to advance. Stationarity: the NNLS residual min_{λ≥0} ‖∇θf0 + Σλ∇θg‖/‖∇θf0‖ over the near-active constraints is folded into the design change (change = max(actual, 10⁻³ × residual / 0.05)), so `converged` needs a parameter-space KKT point and a quiet design. The start fit (800 monotone Adam steps, rate 0.01) to the shared seed fails loudly above RMSE 0.05. Benchmark ([neural_al_cantilever.json](validation/neural_al_cantilever.json)): on the (16, 4, 6) min-mass cantilever SIMP-MMA converges at 0.567 g in 106 iterations; the neural route with 192 frequencies, [96, 96], 1.0/mm reaches 0.585 g (+3.2 %) with the stiffness constraint at g = +0.0014 and ends `not_converged_stalled` after 569 iterations: Adam keeps cell changes of 0.01–0.06 and the residual at 0.2–0.4, so the shared design-change criterion is not reached. The neural route is therefore not expected to report `converged` on the frame either; replacing its design-change criterion by the parameter-space residual would be a specification change. Wavenumbers reach 0.25 cycles/mm by default; the ground15 frame configuration uses 256 frequencies with [96, 96] (800-step fit to the failed ground15 SIMP design: RMSE 0.015). Both routes start from one shared seed written by `tools/formulation_study.py frame_seed` (uniform `initial_density` 0.5 with `body_start` on the fine grid, deterministic from the configuration; `start_design` regenerates it when missing and checks the pinned sha256 and fine problem fingerprint). Network-only resume is not supported. The cluster workflow is documented in [dgx_handoff.md](dgx_handoff.md).

## Method

- `deep_frame/topology_neural.py` replaces the OC update of the GPU SIMP with a neural reparameterization of the density.
- **Network:** `FourierField` maps cell centres (x, y, z) in mm to a logit.
  - Fourier features: `frequencies` random directions with wavenumbers uniform in `[0, max_frequency_per_mm]`, encoded as sin/cos.
  - Hidden layers: two layers of 48 units with leaky ReLU (about 14k parameters).
  - Plain numpy with hand-written backprop. Neither venv has torch, and the FE solve dominates runtime anyway.
- **Physics:** unchanged `HexElasticity`.
  - Hex8 SIMP with p = 3 and the cuDSS solver.
  - Same normalized multi-load compliance as c01 (`_case_scaling`, the same `case_weights`), `preserve_adjacent` interface nodes.
  - dJ/dρ from `solve()` is chained through the sigmoid and the network to the weights. Optimizer: Adam, lr 0.01.
- **Symmetry by construction:**
  - The frame's mirror plane is x = 0. The longitudinal axis is y: camera at +y, connectors at −y.
  - Flip checks on the c01 masks: the x-flip differs in 140 preserve cells, the y-flip in 1924. So the net is fed |x| (`mirror_axis: 0`).
  - The raw network output is exactly symmetric; this is tested bitwise. The masked density is only as symmetric as the masks, which are slightly asymmetric (AIO side access, rasterization).
- **Masks are hard:** preserve cells are 1, forbidden cells are 0, and only free cells carry the network output, in every iteration and in the fine sample (tested).
- **Volume:**
  - Exact, through a scalar logit shift per evaluation: ρ = σ(s·(z + c)), with c found by bisection so that Σρ hits the budget, as in the OC multiplier search.
  - The gradient uses the implicit derivative dc/dz, so dJ/dz = ρ'(g − ⟨g⟩ρ').
  - First attempt: an augmented Lagrangian with Adam. It oscillated at ±25 % volume on the cantilever test, so it was replaced.
- **Sharpness:** a ramp s = 1 → 4 over 80 iterations gives a near-binary field without a density filter. Grey fraction is about 0.5 %.
- **Stop criterion:** objective stall below 0.3 % over 10 iterations, or 150 iterations.
- **Grid independence:**
  - The optimization runs on a 2 mm FE grid (68×64×16).
  - The trained field is then sampled on c01's 4/3 mm grid (102×96×24) with the fine-grid masks, its own volume shift and the same fraction.
  - The implicit route then runs exactly as for c01: subdivisions 5, threshold 0.5.
- **Driver:**
  - `tools/topology_study.py neural <json>`: sweeps `volume_fractions` into `<directory>/fNN` in the density-study format, so `tools/implicit_study.py run` reads it unchanged.
  - New flag `diagnostic_fea_always` in `implicit_study run` forces diagnostic FEA when any gate fails. The PoC needs FEA numbers even when the 2 mm wall gate or the v0 mass screen fails.

## Results

Run date: 2026-10-02. FE grid 2 mm on cuDSS, 1 GPU slot. Route as for c01: subdivisions 5, threshold 0.5. FEA is the diagnostic CalculiX run with the same load cases and modal analysis.

- **c01 reference** (re-read from `Deep_Frame-int/exports/implicit/fine_robust`): 10 % volume, robust SIMP on the 4/3 mm grid.
- **Neural rows:** `fmaxXXXX` is `max_frequency_per_mm`; `fNN` is the volume fraction in % of the allowed cells.
- The route was run with the extensions `preserve_forbidden`, `preserve` and `none`. The table shows the first candidate that has FEA results, falling back to the first candidate that has a mesh.

| run | frame mass g | f1 Hz | arm-tip N/mm | optimization s | route s | gate violations | member width p10/p25/p50 mm (share < 2 mm) |
|---|---|---|---|---|---|---|---|
| c01 fine robust SIMP (10 %) | 45.7 | 748 | 279 | 2771 (116 it, 4/3 mm) | 452 | features | - |
| neural fmax00625 f12 (preserve) | 51.6 | 1064 | 167 | 378 (150 it, 2 mm) | 391 | features | 3.3/4.0/4.9 (5 %) |
| neural fmax0125 f12 (preserve) | 48.9 | 142 | 107 | 386 (150 it, 2 mm) | 532 | features | 1.3/2.4/4.0 (23 %) |
| neural fmax0125 f18 (none) | 75.0 | 182 | 340 | 271 (108 it, 2 mm) | 1020 | features, mass screen | 1.3/2.4/4.6 (16 %) |
| neural fmax0125 f25 (preserve) | 106.6 | no FEA (tet meshing) | no FEA | 248 (97 it, 2 mm) | 522 | features, mass screen | 2.4/4.0/6.7 (9 %) |
| neural fmax025 f12 | no mesh | - | - | 327 (116 it, 2 mm) | 78 | marching cubes not one closed body | 1.3/1.3/2.4 (40 %) |

Renders (iso/top/side): `exports/neural/renders/<run>_{isometric,top,side}.png`.

### Minimum feature size vs. Fourier cutoff

- **Measurement:** member width is 2 × EDT on the 3D skeleton of ρ ≥ 0.5, sampled on the 4/3 mm grid, minus one cell. Skeleton cells within two cells of the masks are excluded. 1.33 mm is the resolution floor.
- **Results, all at 12 %:**

  | f_max (1/mm) | p10 (mm) | median (mm) | share below 2 mm | route |
  |---|---|---|---|---|
  | 0.0625 | 3.3 | 4.9 | 5 % | fine |
  | 0.125 | 1.3 | 4.0 | 23 % | fine |
  | 0.25 | 1.3 | 2.4 | 40 % | breaks: the extracted surface is no longer one closed body |

- **Rule of thumb:** halving f_max moves mostly the thin tail. p10 goes from 1.3 mm to 3.3 mm, while the median grows only weakly (2.4 → 4.0 → 4.9 mm).
  - For the 2 mm printing wall: f_max ≤ 0.0625/mm, i.e. a wavelength ≥ 16 mm, keeps 95 % of the members above 2 mm.
  - The rule fixes the typical size, not a hard minimum. A hard minimum still needs the route's opening step or a robust projection.

## What worked

- **Speed:** a full 3D frame run on the GPU took 4–6.5 min (97–150 iterations, about 2.5 s per iteration on the 2 mm grid), against 46 min for c01's robust SIMP on the 4/3 mm grid.
- **Field quality:**
  - Grey fraction 0.3–0.9 %, without a density filter or projection.
  - Exact volume and exact symmetry.
  - Masks are never violated.
  - At ρ ≥ 0.5 all preserves lie in one face-connected component in every run.
  - Loose islands: 1–9 components for f_max ≤ 0.125, but 91 at f_max 0.25.
- **Resolution-free output:** the field is trained on 2 mm and resampled directly on the 4/3 mm route grid. The route takes it unchanged through `load_source`.
- **Best variant, fmax 0.0625 / 12 %:** compact, symmetric and recognisably ManaFly-like (X arms, battery cradle, closed centre ring).
  - f1 is 1064 Hz, above c01's 748 Hz, at +13 % mass.
- **Volume handling:** the logit-shift volume constraint (OC-like bisection plus implicit gradient) converges monotonically. The augmented Lagrangian oscillated.

## What did not work

- **Arm-tip stiffness falls short of c01** at comparable mass: 107–167 N/mm against 279 N/mm.
  - The Hex8 surrogates of the optimized fields are nearly equal: c01 170 N/mm on its eroded field, neural fmax 0.0625 167 N/mm.
  - The neural FEA reproduces its surrogate (167). c01 gains in the FEA because it optimizes the eroded design but builds the intermediate one.
  - So the gap comes from the robust formulation, which gives c01 a stiffness reserve, not from the route.
  - fmax 0.125 also spreads load paths over thin struts (surrogate 123 N/mm).
- **Low first modes at fmax 0.125:** 142/153 Hz at 12 % and 182 Hz at 18 %. The battery-impact stiffness of 1154 N/mm (c01: 2863) points to a soft battery-deck support. Not diagnosed further (coarse first).
- **`preserve_forbidden` fails** (`extension_changed_topology`) on 3 of 4 neural fields (only 18 % built). Neural fields carry material right up to the keep-outs, so the extension changes the topology. The `preserve` and `none` extensions build.
- **The 2 mm wall gate (`features`) fails everywhere**, as with c01 (known open issue).
- **18 % and 25 % fail the v0 mass screen** (2 × 32.2 g), which is expected at that mass.
- **No FEA at 25 %** because tet meshing failed:
  - at 2.0 mm, 250k elements exceed the 9.7 GB element budget;
  - coarser 3.0/2.5 mm targets fail SICN < 0.01;
  - classify times out.
- **At 18 %, tet meshing failed for the `preserve_forbidden` and `preserve` candidates.** The FEA row comes from the `none` extension.
- **Visible terracing on the arm flanks** (top/iso renders). Its cause is unclear; it could come from the 4/3 mm cell sampling with PCHIP, or from the network itself.

## Hardening needed before integration

1. **Objective:**
   - robust (eroded) evaluation;
   - a frequency or battery-support term;
   - optionally a 4/3 mm FE grid for a final fine-tuning phase. Warm-starting the network is cheap.
2. **Hard minimum width:** f_max alone only shifts the distribution, so add a minimum-length-scale term or a robust projection on the network output.
3. **Make `preserve_forbidden` compatible:** keep a small void margin at the keep-outs in the network output, or soften the extension guard for neural fields.
4. **Tet meshing for heavy designs:** element budget and SICN handling.
5. **Smoke test** for `topology_study neural`.

## Reproduce

```
jobslot.py gpu -- <gpu venv>/python.exe tools/topology_study.py neural cfg.json
# cfg.json: {"directory": "exports/neural/density/fmax0125", "volume_fractions": [0.12, 0.18, 0.25], "max_frequency_per_mm": 0.125, "max_iterations": 150}
jobslot.py cpu -- <main venv>/python.exe tools/implicit_study.py run route.json
# route.json: {"source": ".../f12", "output": "...", "subdivisions": 5, "thresholds": [0.5], "extensions": ["preserve_forbidden", "preserve", "none"], "diagnostic_fea": true, "diagnostic_fea_always": true}
```
