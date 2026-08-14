# RAWCLIC Stock-and-Flow Model — Technical Description

**Converted from `RAWCLIC_StockFlow_Model_Description.docx` (13 July 2026) and
brought up to date on 14 August 2026.** Markdown from here on, so updates travel
with the code instead of sitting in a binary nobody diffs.

Sections 1–4 are the original description of the cohort-survival mechanics and the
Monte Carlo methodology; they remain accurate. **Section 5 is new** and covers
everything added since the original was written. If the two ever disagree, section 5
is the current behaviour.

---

## 1. Overview: two paradigms, one fleet

The pipeline models the same EU vehicle fleet through two different lenses, cross-validated against each other:

Stock-driven (stage 02, 02_stockdriven.py): a target STOCK trajectory is prescribed (from REMIND scenario output); the model works out what INFLOW would have been required, year by year, to hit that target given Weibull attrition. Inflow is the unknown, solved as a residual.

Flow-driven (stage 03_01/03_02): the mirror image. A prescribed INFLOW trajectory (derived from stage 02's own solved-for inflow, then disaggregated into 12 vehicle segments) is simulated forward; stock and outflow emerge as outputs, not inputs.

Both paradigms share the same underlying cohort-survival mechanics (a Weibull hazard applied to an age-tracked population); they differ only in which quantity is prescribed and which is solved for. Comparing the two (“03_01_stockdriven_vs_flowdriven.png”) is a standing sanity check — any persistent mismatch signals a real modeling inconsistency, not just numerical noise.

**Stage**

**File**

**Paradigm**

**Solves for**

02

02_stockdriven.py

Stock-driven

Inflow (given stock target + lifetime)

03_01

03_01_flowdriven.py

Disaggregation + flow-driven validation

Segment/sub-drivetrain split of stage 02's results; a separate flow-driven re-derivation as a sanity check

03_02

03_02_adjustedflows.py

Flow-driven, 11 scenarios

Stock & outflow, given a perturbed inflow assumption per scenario


## 2. The stock-driven model (stage 02)


### 2.1 Cohort-survival recurrence

The fleet is tracked as a set of annual cohorts (one per vintage/birth year), each aging forward under a Weibull hazard. Every simulated year t:

S(age) = exp( -(age / scale_lambda) ^ shape_k )        # survival function

h(age) = 1 - S(age+1) / S(age)                          # hazard: P(retire between age and age+1)

out_survival[cohort] = stock[cohort] * h(age of cohort at t)

remaining_total(t)   = sum over all cohorts of (stock - out_survival)

The core recurrence lives in stockflow_model.py's _run_cohort_recurrence — a single vectorized function that serves BOTH the deterministic single run (run_cohort_survival_model, n_draws=1) and the Monte Carlo run (run_cohort_survival_monte_carlo, n_draws up to 200,000) through the same code path. This is deliberate: a Monte Carlo bug fixed once is fixed for the deterministic reference too, and the two can never silently diverge in their core math.


### 2.2 Backcasting: reconstructing pre-t0 history

The model needs a plausible age structure for the fleet AT t0 (the first simulated year), not just its total size — vehicles of different ages retire at different rates going forward. prepare_backcasting_state assumes CONSTANT historical inflow for init_max_age years before t0: the initial stock is split across ages 0..init_max_age proportional to the Weibull survival curve itself — exactly the age distribution that a constant registration rate over that window would produce.

This is a standard, defensible default but a weak assumption for a genuinely young technology (e.g. BEV) whose real historical inflow was near zero for most of that window, not constant. A stage-03_01-specific, INDEPENDENT segment-level backcast (1975–2005, a different 30-year window) coexists with this one — two separately-designed backcasting methodologies feeding nominally one coherent pipeline; see §5 for the practical consequence.


### 2.3 The residual-inflow formula

inflow_raw(t) = target(t) − remaining_total(t)

inflow_raw is NEGATIVE exactly when natural attrition alone removes stock FASTER than the prescribed target is falling — i.e. the target's decline outpaces what Weibull retirement alone explains. What happens next is entirely determined by negative_inflow_policy (§2.4).

Conversely, a POSITIVE inflow_raw in a declining-target period is not necessarily benign either: it can mean the residual formula is manufacturing “phantom” new registrations purely to keep pace with a target that's falling faster than the survivor fleet would otherwise decline — with zero basis in real registrations. This is exactly the situation hard_zero_inflow_from_year_by_drv exists to correct (§2.5).


### 2.4 Negative-inflow policy

**Policy**

**Mechanics**

**Direction of bias**

report_only (default)

Add zero inflow that year. Modeled stock KEEPS remaining_total(t), which is HIGHER than target from that point on.

OVERSTATES fleet size and everything downstream that scales with it (material demand, etc.) — a growing surplus, not a shortfall.

clip_to_target

Add zero inflow, but ALSO force additional pro-rata outflow across every surviving cohort so modeled stock hits the target exactly that year.

Removes the surplus — modeled and target stock overlap exactly. Not the default; switching changes real numbers, and correcting one year's surplus changes the starting stock for every subsequent year (the two policies genuinely diverge over time, not just a diagnostic-vs-corrected view of one sequence).

check_negative_inflows()'s diagnostic report is unaffected by which policy is active — it always reports the RAW residual value, so a negative-inflow year is never hidden just because clip_to_target absorbed its effect on the cohort matrix.


### 2.5 Hard-zero inflow overrides

A deterministic if/else tied to an external, documented fact — NOT a statistical correction of the residual formula. Applied AFTER negative_inflow_policy, overriding whatever that policy computed:

hard_zero_inflow_from_year_by_drv — forces inflow to exactly 0 from a given year onward (e.g. Liquids/Hybrid from 2050: models a real or assumed ICE-sales phase-out).

hard_zero_inflow_until_year_by_drv — the mirror image, forces 0 before a given year (e.g. BEV before 2011: the drivetrain simply didn't exist yet).

Verified directly against real data: for “Liquids,” the residual formula manufactures a genuine, non-trivial POSITIVE phantom inflow from ~2047 onward (peaking ~0.45M/year around 2054), purely because natural attrition on the aging survivor fleet removes stock faster than REMIND's own smooth, monotonically-declining target is falling. The hard-zero override is the correction: past a real sales ban, modeled stock should be ALLOWED to fall below the target — the target simply stops being achievable/binding from that point.

Nothing is hidden by this override: the raw, pre-override residual is ALWAYS tracked separately (inflow_pre_hard_zero_override_by_year), whether or not the override is active for that year/drivetrain.

Scope: applied once, at stage 02, on the aggregate drivetrain, BEFORE the Diesel/Petrol and HEV/PHEV splits happen downstream. Stages 03_01/03_02 only DISAGGREGATE stage 02's own resulting inflow — they never recompute it — so a zero here becomes a genuine zero for both post-split drivetrains automatically, with no override logic duplicated in those files.


### 2.6 inflow_raw vs. inflow_applied: diagnostic vs. simulated

A genuine, non-obvious distinction that caused a real bug (fixed during this project) before it was made explicit:

**Field**

**Value**

**Purpose**

inflow_by_year

inflow_applied where a hard-zero override is active; inflow_raw (can be NEGATIVE) otherwise

DIAGNOSTIC: deliberately preserves the raw residual so a negative-inflow year is never silently hidden. Feeds check_negative_inflows() and the deterministic 02_flows_by_drivetrain_check.png plot, which explicitly wants to show this mechanism.

inflow_applied_by_year

max(inflow_raw, 0), always ≥ 0

SIMULATED: the actual per-draw quantity added to the cohort. What cumulative_inflow sums. The correct series for anything that should never show a negative value — e.g. a Monte Carlo uncertainty BAND over time.

The bug this distinction fixed: a Monte Carlo inflow-over-time plot was built from inflow_by_year (the raw diagnostic series), so its P2.5–P97.5 band could show negative inflow for Liquids/Hybrid in years approaching their hard-zero cutoff — even though no individual simulated draw's inflow was ever actually negative. Flooring AFTER computing a percentile of already-signed values also gives a mathematically different (misleadingly narrow) result than flooring BEFORE the percentile — percentile and “clip at 0” do not commute. The fix: track both series explicitly (nothing discarded, matching this codebase's “never silently drop a diagnostic value” convention), and point anything that represents “what the model actually did” at inflow_applied_by_year specifically.


## 3. The flow-driven model (stage 03_01 / 03_02)


### 3.1 Mirror-image mechanics

Given a PRESCRIBED inflow series (stage 02's own solved-for inflow, disaggregated into 12 segments), the model simulates forward: each year, apply the Weibull hazard to existing cohorts, add the prescribed new inflow, and read off the resulting STOCK and OUTFLOW as OUTPUTS — the exact inverse of stage 02's own solve-for-inflow direction.

03_01_flowdriven.py's own flow-driven pass exists as a validation/sanity check (“does the flow-driven view roughly agree with the stock-driven view, given the same inflow?”), producing 03_01_stockdriven_vs_flowdriven.png. Its result — flows_03 — is not a side artifact: it's what actually feeds stage 03_02's 11 scenarios as their shared inflow starting point.


### 3.2 outflow_timing: pre_inflow vs. post_inflow

Controls exactly what “this year's hazard” is applied to — a genuine, documented methodological difference between the two stage-03 scripts, not an oversight:

**Mode**

**Mechanics**

**Used by**

post_inflow

This year's new inflow is added to the newborn cohort FIRST, then the age-0 hazard is applied to it in the SAME year — a brand-new vehicle has a small but nonzero chance of retiring in its own registration year.

03_02_adjustedflows.py

pre_inflow

Existing cohorts get hazard applied using their age AS OF THE START of the year (one less than current-year age). The newborn cohort is excluded from hazard entirely, added only AFTER everyone else's hazard is computed — no same-year retirement risk for new inflow.

03_01_flowdriven.py


### 3.3 Segment disaggregation and drivetrain splits

Stage 02 tracks 9 aggregate drivetrains. Stage 03_01 splits these further:

Hybrid → HEV / PHEV, using a time-varying EEA-derived ratio (build_hev_phev_split).

Liquids → Diesel / Petrol, same mechanism.

Aggregate stock → 12 vehicle segments (A–F, JA–JF), via a base-year segment-share profile (seg_share_by_drv).

**IMPORTANT scope limitation: segment-level (and therefore materials-stage) accounting is restricted to exactly 5 drivetrains — BEV, HEV, PHEV, Diesel, Petrol. FCEV and Gases exist in stage 02's own matrices_by_key but are silently excluded from every downstream segment-level or materials computation from stage 03_01 onward.**


### 3.4 The collected / export / unknown-whereabouts split

Every year, a drivetrain's total outflow (out_survival) splits three ways: collected (recycling/take-back statistics), export (trade statistics, vehicles leaving the EU), and unknown_whereabouts (neither of the above).


#### 3.4.1 Original design (superseded)

collected_share was never an independent quantity — only export_share and unknown_whereabouts_share had point estimates; collected was always computed as the remainder, 1 − export − unknown. Deterministically this is exact and harmless (the three necessarily sum to 1). Under the ORIGINAL Monte Carlo treatment it was not: export and unknown were sampled as two INDEPENDENT Normal(point, std, clip 0..1) draws, and collected inherited whatever was left over — meaning collected had NO uncertainty of its own, silently absorbing both other shares' sampling noise, and a bad tail draw of export+unknown could exceed 1 (making collected negative for that draw, without complaint).


#### 3.4.2 Current design

Corrected after re-examining the actual epistemic structure of the three quantities: total outflow itself is a MODEL OUTPUT of the lifetime survival curve, not an independent measurement; collected and export ARE independently measured (with their own real uncertainty); unknown_whereabouts is itself an ESTIMATE, not a clean residual of a known total. None of the three has a privileged “remainder” status. Current mechanics:

All three (collected, export, unknown_whereabouts) have their own explicit point estimate AND their own independent Triangular spread.

All three are sampled independently, then NORMALIZED per draw — each divided by that draw's own (a+b+c) sum — so they land on exactly 1 together (disaggregation.compute_collected_export_unknown_shares / cohort_flow_mc.normalize_three_shares).

unknown_whereabouts_share additionally has its center SHIFTED per draw based on that same draw's own lifetime outcome — see §4.7.

Verified numerically: for the deterministic (scalar) case, where the point estimates already sum to 1 by construction, normalizing is a mathematical no-op — byte-identical to the old remainder-based output. The behavior change is entirely in the Monte Carlo path.


### 3.5 The inflow-flooring fix

A subtle bug found by tracing a spurious dip in a Monte Carlo outflow-over-time chart back to its source: matrices_by_key's flows_df["inflow"] column — which feeds stage 03's flow-driven model as the prescribed driving inflow — is stage 02's RAW residual (§2.6), which is DELIBERATELY allowed to go negative in diagnostics near a hard-zero cutoff.

run_flow_driven_model_with_outflow_disaggregation (and the generic Monte Carlo engine, cohort_flow_mc.py) used to add this value DIRECTLY to a cohort's stock, unfloored. A negative raw residual therefore SUBTRACTED from a single vintage cohort's stock at creation — producing an artificially depleted cohort whose effect showed up as a sharp, spurious dip in total outflow the following year, once that cohort's age-based hazard was applied (self-correcting once other cohorts dominated the sum again).

Fix: the value actually added to any cohort is now max(inflow, 0) — mirroring stage 02's OWN recurrence, which always floors this internally. The raw (possibly negative) value is still what gets recorded for diagnostics (flows_rows["inflow"]), matching stage 02's own “never hide the raw residual” convention — only what gets ADDED to a cohort's stock changed.


## 4. Monte Carlo methodology

This is the most extensively developed part of the model. Every stage's deterministic output is completely unaffected unless params.monte_carlo.enabled = True — Monte Carlo is strictly an additional pass, never a silent change to default behavior.


### 4.1 Why Triangular, not Normal

Every uncertainty in this model — lifetime, and (as of the current design) all three outflow shares — is sampled from a Triangular distribution, built fresh at sampling time from whatever the current point estimate is:

Triangular( point×(1−lower), point, point×(1+upper) )

Triangular was chosen over Normal for two reasons: (1) it is naturally BOUNDED — a Normal needs an explicit clip (as the old export/unknown sampling did, clip 0..1), which distorts the tails and can silently pile probability mass at the boundary; Triangular's bounds are the actual, intended range. (2) it naturally supports ASYMMETRY (AsymmetricSpread, §4.2) without any special-casing — an asymmetric belief (“more likely to run longer than shorter”) is simply a Triangular with unequal lower/upper half-widths, expressed in the same object either way.


### 4.2 AsymmetricSpread and directional beliefs

A point estimate paired with independently-specifiable lower and upper half-widths. A plain float means symmetric (lower = upper). Consequence worth internalizing: an asymmetric Triangular's MEAN is (low+mode+high)/3, not the point estimate itself — it shifts toward whichever side has the wider spread. For example, the current lifetime spread (lower=0.25, upper=0.40 for every drivetrain) means the SAMPLED MEAN lifetime sits meaningfully above the point estimate, not centered on it — correct behavior for a genuinely asymmetric belief, not a bug to correct back to the point estimate.


### 4.3 Per-entity draw caching and the seed hierarchy

Draws are cached per ENTITY (e.g. per drivetrain), not per group — every (Region, Drivetrain, Segment) group sharing the same drivetrain gets the IDENTICAL sampled lifetime/share draws, not independently resampled per group. This matches “one entity, one set of uncertain parameters, applied identically wherever that entity appears.”

Reproducibility and independence are both achieved through numpy's SeedSequence spawning hierarchy: a single root seed (params.monte_carlo.seed, default 42) spawns independent child seeds per stage, and within a stage, per entity. Different stages deliberately use different spawn_key offsets:

**Stage**

**spawn_key**

**Why**

02 (lifetime)

implicit / 0

The root draw stream.

03_01 (shares)

(1,)

Deliberately differs from stage 02's, so this stage's shares are NOT correlated with stage 02's lifetime by accident of sharing the same raw seed stream (any INTENTIONAL correlation, e.g. §4.7, is built explicitly via a formula, never by seed-sharing).

03_02 (scenarios)

(2,)

Differs from both — scenario-level draws are independent of stages 02/03_01's own runs.

Same-draw-index propagation is deliberate and load-bearing: draw #12,345's stage-02 lifetime outcome is the SAME draw #12,345 used for that entity's share sampling in stage 03_01/03_02 — genuine propagation of uncertainty through the pipeline, not independent per-stage resampling that would understate compounding uncertainty.


### 4.4 Chunked vectorization

The core Monte Carlo engine (cohort_flow_mc.py) processes draws in batches of monte_carlo.chunk_size (default 20,000), not all n_draws at once — peak memory is bounded by chunk_size × n_cohorts rather than n_draws × n_cohorts, which matters at the 200,000-draw, 12-segment scale this model runs at. Every metric is accumulated incrementally across chunks (e.g. cum_inflow_by_period[p] += ... inside the chunk loop), so no full (n_draws, n_cohorts, n_years) history is ever materialized in memory.


### 4.5 Sensitivity analysis

Spearman rank correlation between each sampled input (scale_lambda, collected_share, export_share, unknown_share — per drivetrain) and a chosen output metric (typically cumulative_out_survival or cumulative_collected over the full horizon), reported as a tornado chart (monte_carlo.sensitivity_correlations / plot_tornado). Spearman rather than Pearson because the relationship between an input like scale_lambda and an output like cumulative outflow is monotonic but not necessarily linear.

A direct consequence of the lifetime↔unknown coupling (§4.7): unknown_share now shows up as CORRELATED with scale_lambda in this analysis by construction — expected and correct given the coupling exists, not a surprise finding or an artifact to investigate.


### 4.6 Three-way share normalization under Monte Carlo

a_draws = Triangular(collected_point, collected_lower, collected_upper).sample(n_draws)

b_draws = <lifetime-coupled Triangular, see §4.7>

c_draws = Triangular(export_point, export_lower, export_upper).sample(n_draws)

total   = a_draws + b_draws + c_draws

collected_share, unknown_share, export_share = a_draws/total, b_draws/total, c_draws/total

Each input is clipped to ≥ 0 before normalizing (guards a stray negative Triangular-tail draw). If all three happen to be ≤ 0 for a given draw, that draw is forced to (0, 0, 0) rather than dividing by zero, with a count printed — the same defensive pattern used throughout this codebase for rare tail-draw edge cases (never silently propagate a NaN, never abort a 200,000-draw run over a handful of extreme draws).

This machinery exists in TWO places with the same logic, deliberately not shared code: disaggregation.py's compute_collected_export_unknown_shares (vehicle-pipeline-specific, used by the deterministic path and stage 03_01's own MC block) and cohort_flow_mc.py's normalize_three_shares (product-agnostic, used by the generic vectorized engine that stage 03_02 and stage 03_01's optional MC block both call through). The two were kept independent because cohort_flow_mc.py is explicitly designed to be reusable by a different product/pipeline in the future, and importing from a vehicle-specific module would break that.


### 4.7 The lifetime ↔ unknown_share coupling

The single most model-specific piece of the Monte Carlo design. Motivation: a shorter-than-point-estimate scale_lambda draw implies MORE total outflow than collected+export alone can explain for that draw — so the inferred “unknown” gap should plausibly be LARGER for that same draw, not sampled as if lifetime and “how much of the outflow we can't account for” were unrelated.

rel_dev = (scale_lambda_draw − scale_lambda_point) / scale_lambda_point

unknown_share_center = unknown_whereabouts_share_point × (1 − k × rel_dev)

unknown_share_draw   = TriangularVaryingCenter(unknown_share_center, residual_lower, residual_upper)

rel_dev < 0 (a shorter-than-point-estimate lifetime draw) with k > 0 pushes unknown_share_center ABOVE the flat point estimate for that specific draw — matching the intended direction. k = 0 recovers the fully independent, pre-coupling behavior exactly (unknown_share centered on its own flat point estimate, uncorrelated with scale_lambda).


#### 4.7.1 Per-draw-varying-center Triangular sampling

Unlike every other Triangular draw in this model (one fixed point shared by every draw), unknown_share's center genuinely varies PER DRAW — draw #500's center depends on draw #500's own scale_lambda outcome. numpy's rng.triangular(left, mode, right) broadcasts array arguments natively, so this needed one new function (sample_relative_triangular_scale_varying_center), not a per-draw Python loop: left/mode/right are each full (n_draws,) arrays, and one call produces one correctly-centered sample per draw.


#### 4.7.2 Verified correlation strength

Standalone numeric test, before this went into the pipeline: with k=1.0 and a realistic asymmetric lifetime spread (lower=0.15, upper=0.30), the resulting Pearson correlation between scale_lambda and unknown_share draws was approximately −0.85 — a strong, reliable negative relationship, all resulting unknown_share values remaining strictly positive, and k=0 exactly reproducing the pre-coupling independent behavior (confirming the change is a strict generalization, not a breaking one).


#### 4.7.3 What is deliberately NOT modeled

**These three shares were historically part of how the LIFETIME parameters themselves were back-calculated — meaning a real correlation between lifetime_scale_lambda_relative_spread draws and ALL THREE shares (not just unknown_whereabouts) likely exists at some level beyond the one coupling implemented here. This is not modeled further, for lack of the historical calibration data that would be needed to derive a real coupling strength rather than an assumed placeholder (k=1.0 for the mature drivetrain group, k=1.5 for FCEV/Gases — both currently unverified assumptions, not fitted values).**


### 4.8 Negative-inflow handling under Monte Carlo

The generic Monte Carlo engine floors every year's driving inflow at 0 (np.maximum(inflow_t_draws, 0.0)) before it is used for EITHER the cohort-stock update or the reported cumulative_inflow metric — applied once, consistently, for every draw. This mirrors the deterministic inflow-flooring fix (§3.5) at the vectorized level, and was fixed in the same investigation, for the identical underlying reason: the inflow values this engine consumes are ultimately sourced from stage 02's raw (diagnostically-negative-permitted) residual.


### 4.9 Scenario-level Monte Carlo (stage 03_02)


#### 4.9.1 Two separate re-simulations per scenario

For each active scenario, monte_carlo_enabled=True triggers TWO independent runs of the vectorized engine, not one:

Segment-level (12 segments × drivetrain): the main result, matching the deterministic run's own resolution.

By-drivetrain (independent re-simulation): the SAME entity draws (via a cloned, spawn-aligned seed — mc_seed_by_drivetrain), but run directly at (Region, Drivetrain) granularity, bypassing the base-year segment-share split entirely. This is a genuine second simulation, not a post-hoc sum of the 12-segment result — comparing the two isolates exactly how much of any discrepancy is attributable to the base-year segment-mix assumption versus genuine simulation randomness.


#### 4.9.2 collected_share re-derivation for override scenarios

losses_zero and losses_high override export_share/unknown_whereabouts_share directly (ScenarioSpec has no collected_share override field of its own). Left unhandled, this would silently break the sum-to-1 invariant for those two scenarios' own point estimates. Fixed by re-deriving, per scenario:

collected_share_scenario = max(0, 1 − export_share_scenario − unknown_whereabouts_share_scenario)

— byte-identical to the base collected_share for every scenario that doesn't override either input (the base values already sum to 1 by construction), and correctly adjusted for the two that do. The Monte Carlo layer still gives this its own independent Triangular uncertainty around whatever this re-derived point estimate turns out to be for that scenario — this only fixes the CENTER, not collected's own uncertainty.


## 5. Special cases — quick reference

**Case**

**What happens**

**Where**

Negative residual inflow

report_only (default): zero added, stock silently exceeds target. clip_to_target: forced pro-rata extra outflow removes the surplus.

stockflow_model.py, §2.4

Hard-zero phase-out / introduction

Inflow forced to exactly 0 for a window, overriding whatever the residual formula computed — represents a real external fact, not a statistical correction.

params_schema.py hard_zero_*, §2.5

Diagnostic vs. simulated inflow

inflow_by_year (raw, can be negative) vs. inflow_applied_by_year (floored, ≥ 0) — using the wrong one for a Monte Carlo band produces a mathematically wrong, misleadingly narrow result.

§2.6

outflow_timing difference

03_01 uses pre_inflow (no same-year retirement risk for new vehicles); 03_02 uses post_inflow (small same-year risk) — an intentional, documented divergence, not an inconsistency to fix.

§3.2

FCEV / Gases excluded downstream

Present in stage 02's own matrices, but silently absent from all segment-level and materials-stage accounting from stage 03_01 onward.

§3.3

Collected/export/unknown normalization

All three independently sampled and normalized to sum to 1 — no privileged remainder share, unlike the pre-fix design.

§3.4, §4.6

Negative inflow bleeding into a cohort

Fixed: the engine now floors the value actually added to any cohort at 0, both deterministically and under Monte Carlo, while still recording the raw value for diagnostics.

§3.5, §4.8

Lifetime ↔ unknown_share coupling

unknown_share's Monte Carlo center shifts with that SAME draw's own scale_lambda outcome — shorter lifetime draw implies larger inferred unknown share.

§4.7

Scenario share overrides

losses_zero/losses_high override two of three shares; collected_share is re-derived per scenario to preserve the sum-to-1 invariant.

§4.9.2

Two independent backcasts

Stage 02's aggregate 50-year constant-inflow backcast and stage 03_01's independent 30-year segment-level backcast are two separately-designed methodologies feeding one pipeline.

§2.2


## 6. Known limitations and open items

Lifetime↔share correlation beyond the one implemented coupling (§4.7.3) is not modeled — no historical calibration data currently available to derive it properly.

Every Monte Carlo spread value (lifetime, all three shares, the coupling coefficient k) is currently a PLACEHOLDER — differentiated by drivetrain maturity/data quality as a matter of judgment, not fitted to real confidence intervals or historical calibration records.

FCEV and Gases point estimates and spreads are explicitly flagged “not a verified real value” throughout params_schema.py — both drivetrains are also excluded from segment-level/materials accounting entirely (§3.3, §5).

Only BAU currently has future segment-mix Monte Carlo uncertainty wired in (inflow_segment_share_spread) — a deliberate pilot, not yet extended to the other four segment-mix scenarios (BEV_A_F, BEV_JA_JF, BEV_large, BEV_small).

The stage-02 aggregate backcast and the stage-03_01 segment-level backcast use two different historical-inflow assumptions over two different windows — not reconciled into one methodology.

Generated to accompany the RAWCLICStockAndFlow codebase (stockflow_model.py, flowdriven_model.py, cohort_flow_mc.py, disaggregation.py, and the three pipeline scripts 02_stockdriven.py / 03_01_flowdriven.py / 03_02_adjustedflows.py). Cross-reference: “Parameter Reference” for every parameter named in this document.

---

## 7. Added since the original description (August 2026)

Written in the order the changes matter, not chronologically.

### 7.1 The correlated drivetrain mix

**What changed.** Stage 02 used to sample each drivetrain's stock target
independently. A Monte Carlo draw could therefore have more BEVs *and* more petrol
cars at once, which is not how buying a car works: a person buys one drivetrain and
by doing so does not buy the others.

**How it works now.** After the per-drivetrain targets are drawn they are
renormalised across drivetrains onto the simplex, so the shares sum to one, and one
shared total-fleet multiplier is applied on top. Substitution therefore emerges
from the arithmetic rather than being imposed.

Two parameters, both in `StockFlowParams`:

| parameter | default | meaning |
|---|---|---|
| `stock_target_correlated_mix` | `True` | the master switch; `False` reproduces the old independent behaviour byte-identically |
| `total_fleet_relative_spread` | ±2% | how uncertain the total fleet is, applied to every drivetrain together |

**Measured substitution**, correlation between BEV and Liquids stock targets:

| year | correlation |
|---|---|
| 2030 | −0.33 |
| 2035 | −0.80 |
| 2040 | −0.83 |
| 2050 | −0.16 |

**Why the fleet spread is 2% and not more.** The two axes fight each other. The
total-fleet axis is perfectly *positively* correlated — every drivetrain moves the
same way — while the mix axis is *negatively* correlated. Raise the fleet spread
much above 3% and it overwhelms the mix in the years where one drivetrain dominates
the fleet, and the drivetrains start appearing to rise and fall together. Measured:

| fleet spread | 2030 | 2035 | 2040 | 2050 |
|---|---|---|---|---|
| 0% | −0.92 | −0.96 | −0.96 | −0.91 |
| **2%** | **−0.33** | **−0.80** | **−0.83** | **−0.16** |
| 5% | **+0.23** | −0.27 | −0.38 | **+0.13** |

At 5% the sign flips in the dominated years, hiding the substitution the model
exists to show. The positive correlation is not wrong in itself — a bigger fleet
really does mean more of everything — it simply masks the effect of interest.

Verified: with the switch off, stage 02 is byte-identical to the previous
behaviour across 45,121 compared values; sum conservation holds to 6.6e-16; and
years before `stock_target_uncertainty_start_year` are exactly unchanged.

### 7.2 One shared orchestration for stage 02's Monte Carlo

`run_stage02_cohort_monte_carlo` in `src/stockflow_model.py` now owns the whole
sequence — target sampling, the natural run, and the phase-out cap — together with
the random-seed spawn order.

**Why this exists.** Stage 03_01 needs stage 02's per-year, per-draw results, which
stage 02 cannot persist (hundreds of MB per drivetrain at production draw counts).
It used to rebuild stage 02's call by hand, and drifted three separate times, each
time silently:

1. the hard-zero overrides were omitted, so Liquids and Hybrid after 2050 and BEV
   before 2011 were wrong;
2. `stock_target_draws` was omitted, so none of the stock-target uncertainty was
   present — measured at up to **20.8%** error on per-year inflow, with 0 of 15
   arrays matching stage 02;
3. the inflow-mode resolution and the phase-out cap pass were never replicated at
   all, latent only because no drivetrain was configured to use them.

Each was invisible because a re-run missing an argument still returns
plausible-looking numbers. After the fix, 15 of 15 arrays match exactly.

**The rule this establishes, and the reason it is written down:** a stage reads
another stage's actual draws, or calls the same function that produced them. It
never re-derives them.

### 7.3 Stage 02's inflow uncertainty now reaches stage 03_02

Until August 2026 it did not. `flows_03`, the artifact between the stages, holds one
number per row and has no draw dimension, so total BEV inflow arrived varying by
0.000001% where stage 02 had it varying by 9.6%. Every inflow band downstream was
too narrow.

Stage 02's **absolute** per-draw inflow is now transferred, split to the fine
drivetrains by stage 03_01's own shares and composed with stage 03_02's segment mix.
Three effects compose and none replaces another.

Governed by `propagate_stage02_inflow_uncertainty` (default `True`).

Two points of methodology are worth stating here rather than leaving in the design
document:

**Absolute values, not a multiplier.** Inflow is not sampled anywhere — it is the
residual `stock target − survivors`, which crosses zero during a phase-out. Its
relative spread then diverges: Liquids reaches a CV of **1775%** in 2040 on an
absolute spread of 0.005 million vehicles, and `flows_03` carries genuinely negative
values. A ratio is meaningless there; an absolute value is not.

**Flooring happens per draw.** Flooring is nonlinear, so flooring each draw is not
the same as flooring one average trajectory. It lifts the Liquids mean by about

0.228 million vehicles around 2035. That is the correct Monte Carlo answer: in a
draw where the fleet target lands higher, liquid-fuel inflow really is still
positive that year, and a single-trajectory pipeline had no way to express it.

Full record, including the two rejected designs and the measurements that killed
them: `DESIGN_inflow_uncertainty_propagation.md`.

### 7.4 Stage 04_02 — BEV electronics

New stage linking this model to the separate BEV-electronics study. It multiplies
the two Monte Carlos together, draw by draw, from the persisted draws of both sides
rather than from summaries.

**The segment problem and how it is solved without inventing anything.** The
electronics study groups cars into AB, CD and EF; this model uses twelve segments.
Each pair is split by *re-weighting* the published distribution — the smaller
segment draws preferentially from its lower values, the larger from its upper — with
the weights constructed so the two halves recombine to the original. Measured error
at 200,000 draws: **0.007%**.

The tilt strength is `bev_electronics_segment_tilt` (default 0.2, giving a 3.1%
separation between the pair's means). It has a hard ceiling of
`2 × min(share, 1−share)`; beyond that the recombination property fails and the code
raises rather than clipping.

**One assumption, stated with its direction.** The fleet and electronics Monte
Carlos are treated as independent — draw *i* paired with draw *i*. Both respond to
technology adoption, so this makes the reported band slightly **narrower** than the
truth, never wider.

**What it found.** Almost all the uncertainty comes from the electronics
composition, not the fleet: the fleet contributes roughly 45 kt of a 210–340 kt
band. Narrowing the fleet model would barely move the result.

### 7.5 Where uncertainty enters and stops

Maintained separately in `UNCERTAINTY_MAP.md`, because it is the document that makes
a missing propagation visible before someone notices a flat line on a chart.