# Neural topology optimization (TOuNN-style proof of concept)

Status: prototype on branch `feature/neural-topo`, not integrated into the main pipeline.

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
  - The raw network output is exactly symmetric; this is tested bitwise. The masked density is only as symmetric as the masks, which are slightly asymmetric (XT30/balancer keep-outs, rasterization).
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

RESULTS_PLACEHOLDER
