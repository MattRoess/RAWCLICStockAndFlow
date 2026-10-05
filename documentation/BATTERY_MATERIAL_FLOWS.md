# Battery material flows (stage 04_04)

How much of each element enters the European BEV fleet in a battery, how much
comes back out, and how much is actually **collected** from it, year by year,
under three assumptions about which chemistry the cars carry.

Only the collected part reaches a recycler. Of the BEVs that leave the fleet,
88 % are collected, 2 % are exported second-hand and 10 % are never traced — so a
secondary-supply number built on the outflow is an upper bound, not a supply.
03_02 draws those shares, so the collected series carries that uncertainty rather
than being 0.88 × the outflow: measured at 87.9 % in 2070 with a 86.8–89.0 %
band.

Reported **every year**, 2020 to 2070.

```bash
.venv/bin/python code/04_04_batteries.py     # an hour or more at 200,000 draws (47 min with five chemistries), writes the figures too
.venv/bin/python code/test_04_04_figures.py  # ~30 s, redraws them from the saved draws
```

The annual grid costs one thing worth knowing: the battery project writes its
mass-improvement factor every **five** years, so four years in five have none on
file. `src/battery_composition.py` interpolates them **per draw** — draw *i*'s
factor in 2022 lies between draw *i*'s own 2020 and 2025 factors. Before that
change those years silently carried no improvement at all, which by 2070 would
have been about 20 % too much mass in most years.

## The three scenarios

They are **assumptions, not data** — the observed record ends in 2026. Shares are
stated at 2025, 2035, 2050 and 2070, interpolated between, and set per segment
group (small / medium / large), in `src/params_schema.py`.

| | what it says | whose cell is not described |
|---|---|---|
| **S1** incumbents hold | LFP carries the volume, NMC the premium cars, LMFP grows into both. Nothing new ever arrives. | nobody — 0 % throughout |
| **S2** sodium enters | Sodium-ion takes the small segments (68 % by 2070, as Prussian white and layered oxide), NMC shrinks to a niche. | nobody — 0 % throughout |
| **S3** sodium and solid-state | S2, plus bipolar solid-state from 2040, large segments first (70 % of them by 2070). | solid-state: about 48 % of new cars by 2070 |

![chemistry scenarios](../data/processed/figures/04_04_1_chemistry_scenarios.png)

## The one thing to know before reading any number

**Solid-state has no cell composition.** Nobody has published one that survives
scrutiny, so its cars carry no cathode, anode or electrolyte in this model. Its share
is reported as an explicit gap instead of being dropped, because a total that quietly
fell would read as falling demand rather than as a hole in the data. Its packaging is
carried, so its steel, aluminium and copper reach the totals.

The two sodium cells are built from the literature by the battery project — a scenario,
not a bill of materials — and are counted like any other chemistry. Every comparison
figure carries the uncovered share next to the curves for that reason, and says on
its face that it is an open item rather than an oversight. Full account:
[DESIGN_chemistries_without_composition.md](DESIGN_chemistries_without_composition.md).

![uncovered share](../data/processed/figures/04_04_4_uncovered_share.png)

## Two levels, because the elements do not add up to the pack

The cell casing and the separator have no element rows at all, and the
electrolyte's cover 1 % of its mass. Measured against the component totals, the
element arrays miss **8.3 % of a 25 kWh LFP pack, 9.8 % at 60 kWh and 10.9 % at
100** — 7.2 to 7.7 % for NMC. Over the whole S1 inflow that is 601 kt of 5 806 kt
in 2070.

That missing tenth is plastics, polymer separator and organic electrolyte:
precisely what a recycler has to deal with rather than sell. So the stage carries
both levels, from the same drawn cars — same capacity, same voltage, same
extrapolation seed — and keeps them in two frames rather than one, because they
do not add up to each other and a single frame would invite summing them.

| | |
|---|---|
| `battery_material_flows` | 14 elements. Fe, C, Al, O, Cu, P, Mn, Ni, Li, Si, Co, and the sodium cells' Na, N and F |
| `battery_component_flows` | 13 components. Cathode and anode active material, the two current collectors, support frame, thermal conductor, module enclosure, cables, cell terminals, electrolyte, casing, separator, and the sodium cells' unitemised cell mass |

![components](../data/processed/figures/04_04_7_components_collected.png)

Oxygen is in the data and out of the element figures: it is bound in the cathode
oxides and the phosphate, never leaves as oxygen, and nothing recovers it.

## What is drawn and what is not

| | |
|---|---|
| vehicles | per-draw arrays from 03_02, one array per segment and flow |
| pack size | drawn per segment per draw from a five-level discrete mixture, held for the car's life |
| voltage | drawn per draw, 400 or 800 V, never blended |
| composition | per-draw element masses from `RAWCLICVehicleBattery`, interpolated in capacity |
| vintages | per draw, from that draw's own build history and the lifetime curve |
| chemistry shares | drawn — see below |

Draw *i* is one coherent world on every side, because both projects run at
200,000 draws. Nothing is averaged before the end.

**The result is the draws**, in
`data/processed/battery_draws/<flow>/<scenario>/<chemistry>.npy`, shaped
(draws, years, names) in tonnes, one pair of arrays per level — three flows,
about 48 GB at 200,000 draws on the annual grid, six chemistries (40 GB with five). The table in
`04_04_battery_material_flows.pkl` — mean, median, 2.5 % and 97.5 % — is computed
from them and is for reading, never an input to further arithmetic. Recovery is a
ratio of two of these numbers, and a ratio of percentiles is not the percentile of
a ratio.

## The scenarios are assumptions; how wrong they might be is not

A scenario is a stated assumption and the model does not argue with it. But a
share stated for 2070 is a guess made forty-five years early, and a share stated
for 2021 is nearly a measurement. That distance is drawn: a **triangular
multiplier on each chemistry's own share**, 0.70 / 1.00 / 1.30 at full width,
ramping from nothing in 2020 to the full width in 2070. Relative, so a stated
40 % runs ±12 pp and a stated 4 % runs ±1.2 pp.

**Nothing is a residual.** Every chemistry is perturbed and the group is then
renormalised to one. The renormalisation IS the correlation — in a draw where
sodium runs ahead, the others give way — rather than a bookkeeping trick that
dumps the imbalance on whichever chemistry happened to be listed last. One draw
per chemistry, shared across the segment groups and held across every year:
sodium beating expectations is one event, not twelve.

Measured on the drawn shares themselves, 200,000 draws: the group sums to
**1.000000** in every draw and every year; the 2020 band is exactly zero; and
renormalising moves each chemistry's mean **+0.01 % to +0.78 %** off its stated
value, which is small but is not nothing.

**Where this lands, and where it does not.** Band as a share of the median, S1,
inflow 2070 — measured on 2026-09-15, and S1 is untouched by the sodium cells. S2 and
S3 are measured again on the first run with them, because the layered cell carries
nickel, manganese and copper:

| | S1 | carried by |
|---|---|---|
| Li | 59 % | all three chemistries |
| Cu | 51 % | all three |
| **Ni, Co** | **73 %** | NMC only |
| Mn | 70 % | LMFP and NMC |

Lithium's total band is unchanged by drawing the shares, and that is the
mechanism rather than a bug. Measured on the same draws: each chemistry's OWN
lithium carries a 70–73 % band while their sum carries 59 %. The drawn mix moves
lithium between chemistries, and all three contain it, so it largely cancels in
the total. In S1 nickel and cobalt come only from NMC and cannot cancel — there the
share uncertainty arrives in full.

**The two sodium cells draw independently**, one multiplier each, as every chemistry
does. Measured on the drawn shares at 40,000 draws, the 95 % band of sodium's total
share in 2070 is ±8.3 % of its median in small cars and ±18.0 % in medium ones, where one
multiplier on all of sodium gave ±9.3 % and ±21.8 %: narrower, by a point or four, and
the medians are the same.

## Inflow is built this year, what leaves was built long ago

A car scrapped in 2050 was built around 2036 — measured, not assumed; the lag
runs 9 years in 2030 and settles at 16.5 by 2070 as the fleet stops growing — and
carries the chemistry and the pack of 2036. The outflow and the collected series
are therefore spread back over the build years that could have produced them
before any composition is applied (`src/battery_vintage.py`):

    weight(build year b → scrap year t) = inflow(b) × f(t − b)

with *f* the Weibull density of the age at scrapping (k = 3, λ = 18) that stage 02
and 03_02 already use, and the inflow the draw's own.

This matters more than it sounds. Without it the chemistry mix multiplies both
flows by the same factor, cancels out of every outflow-over-inflow ratio, and all
three scenarios produce one identical curve. With it, the collected material of a
scenario is a different multiple of its own inflow for each element, because the cars
being scrapped were built when lithium chemistries still dominated, while the new ones
are not. The figure below shows the result, and the difference between its two lines is
what never arrives.

![secondary supply](../data/processed/figures/04_04_5_secondary_supply.png)

**What the reconstruction cannot do.** 03_02 reports a year's outflow as one
number, not as a matrix by build year, so the vintage mixture is rebuilt from the
inflow history and the lifetime curve rather than read off. Two consequences,
measured rather than assumed:

- the lifetime scale is drawn per draw in 03_02, but only its central value enters
  here, so the vintage weights carry less spread than the flows they weight;
- the composition files start in 2020, so earlier build years get 2020's
  composition. That touches 100 % of the 2020 outflow, 87 % of 2024, 40 % of
  2030, 8 % of 2036 and under 1 % after 2042 — the years where the outflow is
  small anyway. The pack size is read at the true build year, so only the
  composition is clamped.

## Reading the comparison

![scenario comparison](../data/processed/figures/04_04_2_scenario_comparison.png)

Lithium demand peaks around 2040 in every scenario (S1: 111 kt, copper 561 kt);
what separates them is what happens after. The 95 % band widens with distance —
lithium from 21 % of the median in 2020 to 59 % in 2070, copper from 8 % to 51 %
— because capacity growth, plateau year, voltage, the mass-improvement factor and
the extrapolation above 100 kWh all compound.

![chemistry contribution](../data/processed/figures/04_04_3_chemistry_contribution.png)

### Every element

Fourteen elements carry mass: Fe, C, Al, O, Cu, P, Mn, Ni, Li, Si and Co from the
lithium chemistries, and Na, N and F, which exist only in the sodium cells and so only in
S2 and S3. Oxygen is excluded from the figures. One overview per flow — but only for the
inflow and the collected: measured, the collected flow is **87.9 % of the outflow for
every element to three decimals**, because the collection share is drawn on
vehicles and not on materials. An outflow panel would be the collected one times
a constant. Sulphur and vanadium sit in the arrays as columns of zeros — the
element axis is the union over the chemistry files, and none of the six contains
them.

![all elements](../data/processed/figures/04_04_6_all_elements_inflow.png)

Cobalt comes only from NMC, so its return is NMC's own, which is why it is left out of
the secondary-supply figure. Lithium and copper moved within 6 pp of each other in the
earlier runs because both scale with the pack rather than with the chemistry; manganese,
which LMFP, NMC and the layered sodium cell carry, moved apart from them. Those
distances were measured before the sodium cells had a composition and are measured again
on the first run with them.

## Where things live

| | |
|---|---|
| `code/04_04_batteries.py` | the stage, and the figures it draws |
| `code/test_04_04_figures.py` | bench tool: redraws those figures alone, without the stage |
| `src/battery_capacity.py` | the pack-size mixture |
| `src/battery_voltage.py` | 400 or 800 V |
| `src/battery_composition.py` | masses at a drawn capacity, from the battery project |
| `src/battery_chemistry.py` | scenario shares and the uncovered share |
| `src/battery_vintage.py` | where the scrapped cars were built |
| `DESIGN_chemistries_without_composition.md` | the gap, why it stays open, and how to halve it |

Figures and draws are written under `data/`, which is not tracked in git — run the
stage to produce them.
