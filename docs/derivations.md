# Derivations behind the code

Claude drafted the code in `src/gdr4/model/` and `src/gdr4/eval/`. CLAUDE.md asks that the student
re-derive every step here **on paper** and tick it off in `logs/ai_use_log.md`. Each section
names the function that implements it and the test that checks it.

---

## 1. Kepler's equation (`model/kepler.py`, `tests/test_kepler.py`)

Mean anomaly `M = 2π (t − T_p) / P`. The eccentric anomaly `E` solves

    f(E) = E − e sin E − M = 0.

Derivatives: `f' = 1 − e cos E`, `f'' = e sin E`, `f''' = e cos E`.

- Reduce `M` to `[−π, π)` first. `f` is odd and 2π-periodic up to the shift, so the solution for
  `M + 2πk` is `E + 2πk`; the code adds the `2πk` back at the end.
- Starting guess (Danby 1988): `E₀ = M + 0.85 e sign(sin M)`.
- Update (Danby's quartic step): with `δ₁ = −f/f'`, `δ₂ = −f/(f' + ½δ₁f'')`,
  `δ₃ = −f/(f' + ½δ₂f'' + δ₂²f'''/6)`, set `E ← E + δ₃`. This is a Taylor expansion of `f` to
  third order, solved for the step iteratively.
- **Check:** the test requires `|E − e sin E − M| < 1e-12` for `e ∈ [0, 0.95]` and agrees with
  gaiamock's C solver.

Then `X = cos E − e` and `Y = √(1−e²) sin E` are the star's position in the orbital plane in
units of the semi-major axis, with the origin at the centre of mass.
(Exercise: show that `X² + Y²/(1−e²)` traces an ellipse offset by `e` from its centre.)

---

## 2. Along-scan forward model (`model/astrometry.py`, `tests/test_astrometry.py`)

Gaia measures one coordinate per transit: the position along the scan direction. If the scan
direction makes position angle `ψ` (from north through east), the unit vector along the scan in
the local tangent plane has components (east, north) = (sin ψ, cos ψ). A sky offset
(Δα*, Δδ) projects to `Δα* sin ψ + Δδ cos ψ` along the scan.

The offset of the photocentre at time `t` is the sum of:

1. a reference-position offset (Δα*, Δδ);
2. proper motion (μα*, μδ) × `t` (here `t` is in **years**, since μ is in mas/yr);
3. parallax: `ϖ f_ϖ`, where `f_ϖ` is the along-scan parallax factor (projection of the
   observer's barycentric position, in AU, onto the scan direction, with a minus sign);
4. the orbit, in Thiele-Innes form: east offset `x = B X + G Y`, north offset `y = A X + F Y`.

Projecting gives

    w = (Δα* + μα* t) sin ψ + (Δδ + μδ t) cos ψ + ϖ f_ϖ + (B X + G Y) sin ψ + (A X + F Y) cos ψ.

Given `(P, e, T_p)`, `X` and `Y` are fixed numbers, so `w` is **linear** in the nine parameters
`θ_lin = (Δα*, Δδ, ϖ, μα*, μδ, A, B, F, G)`: `w = M(θ_nl) θ_lin`, with design-matrix columns

    sin ψ, cos ψ, f_ϖ, t sin ψ, t cos ψ, X cos ψ, X sin ψ, Y cos ψ, Y sin ψ.

- **Check:** `test_reproduces_gaiamock_abscissae` evaluates gaiamock's own C model
  (`get_residual_array_12par_solution`) on the same inputs and agrees to 1e-8 mas.
- Phase convention: the code samples `M₀`, the mean anomaly at `t = 0` (J2017.5). Then
  `M(t) = M₀ + 2πt/P`, so `T_p = −M₀P/(2π)`. gaiamock's `phi_p` equals `−M₀`.
- Thiele-Innes from Campbell elements (a₀, Ω, ω, i) uses gaiamock's convention; the test
  `test_campbell_conversion_matches_gaiamock_c` checks it against gaiamock's Campbell C model.

**Still to verify (CLAUDE.md):** the sign conventions against the DR4 data model once it is
public (in particular the sign of `ψ` and of `f_ϖ` in the real `epoch_astrometry` table).

---

## 3. Analytic marginalisation of θ_lin (`model/linear.py`, `tests/test_linear.py`)

Model `w = Mθ + ε`, `ε ~ N(0, C)`, `C = diag(σ²)`; prior `θ ~ N(μ₀, Λ₀)`, `Λ₀ = diag(s₀²)`.

**Posterior of θ.** Up to constants,

    −2 log p(θ | w) = (w − Mθ)ᵀ C⁻¹ (w − Mθ) + (θ − μ₀)ᵀ Λ₀⁻¹ (θ − μ₀)
                    = θᵀ H θ − 2 θᵀ b + const,

with `H = Λ₀⁻¹ + MᵀC⁻¹M` and `b = Λ₀⁻¹μ₀ + MᵀC⁻¹w`. Completing the square gives
`θ | w ~ N(H⁻¹b, H⁻¹)`.

**Marginal likelihood.** `w = Mθ + ε` is a sum of Gaussians, so
`w ~ N(Mμ₀, S)` with `S = C + MΛ₀Mᵀ`. Computing `S⁻¹` (n × n) directly is slow, so use two
identities:

- *Determinant lemma:* `|S| = |C| |Λ₀| |H|`.
- *Quadratic form:* `(w − Mμ₀)ᵀ S⁻¹ (w − Mμ₀) = wᵀC⁻¹w + μ₀ᵀΛ₀⁻¹μ₀ − bᵀH⁻¹b`.
  (Derive this by integrating θ out of the joint density: the exponent left over after
  completing the square is exactly this expression.)

So

    log p(w | θ_nl) = −½ [wᵀC⁻¹w + μ₀ᵀΛ₀⁻¹μ₀ − bᵀH⁻¹b] − ½ [log|C| + log|Λ₀| + log|H|] − (n/2) log 2π.

The code uses the Cholesky factor `H = LLᵀ`: `bᵀH⁻¹b = |L⁻¹b|²` and `log|H| = 2 Σ log Lᵢᵢ`.
Only 9 × 9 matrices are factorised, which is what makes evaluating 10⁶ trial orbits feasible.

- **Check:** `test_log_marginal_matches_direct_gaussian` compares against
  `scipy.stats.multivariate_normal` built from the full `S`.

**Consistency rule.** `p(θ_nl | w) ∝ p(w | θ_nl) p(θ_nl)` is the correct posterior only if the
simulator draws θ_lin from the *same* `N(μ₀, Λ₀)`. The prior lives in one place,
`sim/priors.py`, and both the simulator and the reference sampler read it from there.

---

## 4. Amplitude hyperparameter (proposed design choice, needs student approval)

A single fixed Gaussian prior on (A, B, F, G) gives a narrow range of signal-to-noise. To cover
S/N from ≪ 1 to ≫ 10, the drafted prior uses a **scale mixture**: `s_a ~ LogUniform(s_min, s_max)`
and then `A, B, F, G ~ N(0, s_a²)` independently.

- Given `s_a`, the prior on θ_lin is still Gaussian, so section 3 applies unchanged.
- The flow learns `q(P, e, M₀ | w)` only; `s_a` is marginalised implicitly, because NPE trained
  on joint samples `(θ_nl, s_a, w)` and asked only for `θ_nl` learns the `θ_nl` marginal.
- The brute-force reference samples `(P, e, M₀, s_a)` from the prior and weights by
  `p(w | P, e, M₀, s_a)`; dropping `s_a` from the weighted samples gives the same marginal.
- To get θ_lin for one posterior sample of θ_nl, sample `s_a` from the 1-D conditional
  `p(s_a | θ_nl, w) ∝ p(w | θ_nl, s_a) p(s_a)` on a grid, then θ_lin from section 3.

**Physical prior implied:** `scripts/figures/ti_prior_check.py` draws i.i.d. Gaussian (A, B, F, G)
and converts to Campbell elements (`results/prior_checks/ti_prior.png`). Result: ω and Ω come
out uniform, but **cos i does not**: it peaks at cos i = 0 (edge-on). So the implied orientation
prior is not isotropic. This does not affect the (P, e, M₀) posterior in the kill test, but
inclinations and companion masses must be importance-reweighted to an isotropic prior before
they are reported (CLAUDE.md). Exercise: derive the Jacobian of (A, B, F, G) → (a₀, i, ω, Ω)
and explain the shape you see.

---

## 5. Posterior model for (P, e, M₀) (`nn/flow.py`)

Unconstrained coordinates:

- `u_P = Φ⁻¹((log P − log P_min) / (log P_max − log P_min))`
- `u_e = Φ⁻¹(e / e_max)`

where Φ⁻¹ is the inverse standard-normal CDF (probit). Because log P and e are uniform under the
prior, u_P and u_e are exactly N(0, 1) under the prior: the same as the flow's base
distribution, so an untrained flow starts at the prior. (A first draft used a logit; its
exponential tails made the spline flow's loss spike, see the research log for 2026-10-20.)
- `M₀ ∈ [0, 2π)` is circular, so it is **not** given to a flow on ℝ.

The joint density factorises exactly as

    q(u_P, u_e, M₀ | x) = q_flow(u_P, u_e | x) · q_vM(M₀ | u_P, u_e, x),

where `q_flow` is a conditional neural spline flow and `q_vM` is a mixture of K von Mises
densities whose weights, means and concentrations come from a small network. A von Mises
density `exp(κ cos(M₀ − μ)) / (2π I₀(κ))` is periodic, so the wrap-around at 0 = 2π is handled
exactly. Change of variables back to (P, e) multiplies by the Jacobian, but coverage is computed
in the unconstrained space, where it does not matter.

---

## 5b. Periodogram input feature (`nn/periodogram.py`, `tests/test_periodogram.py`)

For a circular orbit, the four orbital columns at trial frequency f are
`cos(2πft) cos ψ, cos(2πft) sin ψ, sin(2πft) cos ψ, sin(2πft) sin ψ`; any phase and orientation
is a linear combination of them, so the 9-parameter model is linear and its best fit is exact.
The feature is `Δχ²(f) = χ²₅ − χ²₉(f)` on 512 frequencies, computed as `bᵀG⁻¹b` with
`b = OᵀWr`, `G = OᵀWO − OᵀWM₅(M₅ᵀWM₅)⁻¹M₅ᵀWO` (the part of the orbital columns not already
explained by the 5-parameter model). Exercise: derive this from section 3 with a flat prior.

Why: without it, the transformer had to discover periodicity from irregular samples and barely
learned (400 steps on S/N > 20 stars: loss 4.68 → 4.50; with it: → 2.89; research log 2026-10-20).
`model.pgram_k: 0` turns it off, which is a natural ablation for Phase 4.

**Peak features** (`peak_features`, `model.pgram_peaks`). The 8 highest local maxima of Δχ²(f)
are also given to the network directly. Each is refined below the grid spacing: evaluate Δχ²
at 9 points spanning ±1 grid step around the peak, take the highest point `y₀` and its
neighbours `y₋, y₊` (spacing `h`), and fit a parabola. Its vertex is at
`f* = f₀ + δh` with `δ = ½(y₋ − y₊)/(y₋ − 2y₀ + y₊)`, height `y₀ − ¼(y₋ − y₊)δ`.
(Exercise: derive δ by fitting `y = a(f − f*)² + c` through the three points.)
Per peak the network gets: log-period scaled to [0, 1] over the grid's period range,
log(1 + Δχ²), that minus the highest peak's value, and a validity flag.
Why: on the local check the periodogram peaked at the true period for 10 of the 14 high-S/N
failures, but the network did not read the position out precisely (research log 2026-10-27).
Giving it several peaks also lets it compare aliases and harmonics (P, P/2, ...).

## 6. Calibration metrics (`eval/calibration.py`)

- **HPD coverage (expected coverage).** For each test star, the credibility level of the true
  value is the fraction of posterior samples with *higher* density `q` than the true value.
  For a calibrated posterior this level is Uniform(0, 1), so the fraction of stars with level
  `< α` should equal `α`. Coverage error at `α` = |observed fraction − α|.
- **TARP** (Lemos et al. 2023): pick a random reference point per star; the statistic is the
  fraction of samples closer to it than the true value. Lemos et al. show that, with suitably
  random reference points, this being Uniform(0, 1) is necessary and sufficient for calibration
  (read their Theorem before quoting this in the paper).
- **SBC ranks** (Talts et al. 2018) for P and e separately.
