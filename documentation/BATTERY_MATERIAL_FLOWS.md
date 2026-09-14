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

Reported **every two years**, 2020 to 2070.

```bash
.venv/bin/python code/04_04_batteries.py     # ~6.5 min at 200,000 draws, writes the figures too
.venv/bin/python code/04_04_figures.py       # ~30 s, redraws them from the saved draws
```

The two-year grid costs one thing worth knowing: the battery project writes its
mass-improvement factor every **five** years, so 2022, 2024, 2026 … have none on
file. `src/battery_composition.py` interpolates them **per draw** — draw *i*'s
factor in 2022 lies between draw *i*'s own 2020 and 2025 factors. Before that
change those years silently carried no improvement at all, which by 2070 would
have been about 20 % too much mass in every second year.

## The three scenarios

They are **assumptions, not data** — the observed record ends in 2026. Shares are
stated at 2025, 2035, 2050 and 2070, interpolated between, and set per segment
group (small / medium / large), in `src/params_schema.py`.

| | what it says | who has no composition |
|---|---|---|
| **S1** incumbents hold | LFP carries the volume, NMC the premium cars, LMFP grows into both. Nothing new ever arrives. | nobody — 0 % throughout |
| **S2** sodium enters | Sodium-ion takes the small segments (68 % by 2070), NMC shrinks to a niche. | 34 % of new cars by 2070 |
| **S3** sodium and solid-state | S2, plus bipolar solid-state from 2040, large segments first (70 % of them by 2070). | 69 % of new cars by 2070 |

![chemistry scenarios](../data/processed/figures/04_04_1_chemistry_scenarios.png)

## The one thing to know before reading any number

**Sodium-ion and solid-state have no composition.** Nobody has published one that
survives scrutiny, so their cars carry no material in this model at all. Their
share is reported as an explicit gap instead of being dropped, because a total
that quietly fell would read as falling demand rather than as a hole in the data.

So S2 and S3 curves that sink are mostly cars leaving the picture. Every
comparison figure carries the uncovered share next to the curves for that reason.

![uncovered share](../data/processed/figures/04_04_4_uncovered_share.png)

## What is drawn and what is not

| | |
|---|---|
| vehicles | per-draw arrays from 03_02, one array per segment and flow |
| pack size | drawn per segment per draw from a five-level discrete mixture, held for the car's life |
| voltage | drawn per draw, 400 or 800 V, never blended |
| composition | per-draw element masses from `RAWCLICVehicleBattery`, interpolated in capacity |
| vintages | per draw, from that draw's own build history and the lifetime curve |
| **chemistry shares** | **not drawn.** A scenario saying 30 % LFP is an assumption about a real fleet mix; drawing it would turn a stated input into a spread it never had |

Draw *i* is one coherent world on every side, because both projects run at
200,000 draws. Nothing is averaged before the end.

**The result is the draws**, in
`data/processed/battery_draws/<flow>/<scenario>/<chemistry>.npy`, shaped
(draws, years, elements) in tonnes — three flows, 6.5 GB at 200,000 draws on
the two-year grid. The table in
`04_04_battery_material_flows.pkl` — mean, median, 2.5 % and 97.5 % — is computed
from them and is for reading, never an input to further arithmetic. Recovery is a
ratio of two of these numbers, and a ratio of percentiles is not the percentile of
a ratio.

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
three scenarios produce one identical curve. With it, S3's **collected** material
reaches **153 %** of its own lithium demand by 2070, **154 %** of its copper and
**232 %** of its nickel (peaking at 369 % in 2054). The outflow behind those
numbers is 174 %, 175 % and 264 % — the difference is what never arrives: the cars being scrapped
were built when lithium chemistries still dominated, while the new ones are not.

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

Eleven elements carry mass: Fe, C, Al, O, Cu, P, Mn, Ni, Li, Si, Co, in that
order of size. One overview figure per flow — inflow, outflow and collected. Sulphur and vanadium sit in the arrays as columns of zeros — the
element axis is the union over the chemistry files, and none of these three
contains them.

![all elements](../data/processed/figures/04_04_6_all_elements_inflow.png)

Two pairs move together for structural reasons, not by accident. **Nickel and
cobalt come only from NMC**, so their return ratios are identical to within
0.03 pp — cobalt is therefore left out of the secondary-supply figure. Lithium
and copper sit within 6 pp of each other because both scale with the pack rather
than with the chemistry; manganese, which only LMFP and NMC carry, is 18 pp
away.

## Where things live

| | |
|---|---|
| `code/04_04_batteries.py` | the stage, and the figures it draws |
| `code/04_04_figures.py` | redraws those figures alone, without the stage |
| `src/battery_capacity.py` | the pack-size mixture |
| `src/battery_voltage.py` | 400 or 800 V |
| `src/battery_composition.py` | masses at a drawn capacity, from the battery project |
| `src/battery_chemistry.py` | scenario shares and the uncovered share |
| `src/battery_vintage.py` | where the scrapped cars were built |

Figures and draws are written under `data/`, which is not tracked in git — run the
stage to produce them.
