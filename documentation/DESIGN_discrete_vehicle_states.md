# Discrete vehicle states: voltage and capacity

A car has one voltage and one pack size, never a blend. This note is the
argument for how that is kept true, and the one decision still open.

---

## The rule

Share-weighting a discrete state replaces the mixture with its mean. If 42% of
EF cars are 800 V, a share-weighted car carries 42% of the copper saving — and
no car does. So: **draw the state per iteration, hold it across years.**

---

## Voltage — settled

Binary in every draw and every year, verified on 50,000 draws. What is drawn is
a percentile held for life: the car is 800 V in any year whose adoption share
exceeds it. Early adopters stay early; nobody flickers.

---

## Capacity — discrete inside a draw, smeared across draws

Inside one draw, JC has four sizes and nothing between them:

```
draw 0, 2040:   77.3   83.2   95.1  101.1 kWh
draw 1, 2040:   83.9   90.3  103.2  109.7 kWh
```

Across draws there are ~50,000 values at 2040 against 4 at the anchor. **Those
are not 50,000 pack sizes in one market — they are 50,000 guesses about what
the four sizes will be.** The continuity is in the uncertainty, not in any car.

---

## Rounding to a 5 kWh grid — rejected

Proposed on the reasoning that pack sizes are round numbers. They are not:

| grid | share of 1,068 models on it |
|---|---|
| 5 kWh | 25.6% |
| 10 kWh | 16.8% |

Commonest sizes: 82, 100, 84, 54, 105, 91 kWh. Capacity is cell count × cell
capacity. Rounding moves 82 → 80 and 84 → 85, which nobody builds.

**The real structure is better and already holds**: 17 values cover half the
fleet because platforms share pack designs. That is the small discrete set a
draw already produces. No re-rounding.

*(Our levels are themselves 5 kWh bin centres, so JC's "80" is the 82 kWh
Tesla/VW pack binned down. That approximation sits at the measurement step,
where it is visible — it is not applied twice.)*

---

## OPEN: how to use the voltage band

The source gives Min/Mode/Max and **selects one**, running three times. This
implementation **draws** the band per iteration.

Consequence, where the band is clipped at zero:

| | mode | drawn mean |
|---|---|---|
| AB 2020 | 0.7% | **4.5%** |
| CD 2020 | 0.5% | **4.3%** |

Drawing gives a `Max` meant as an upper bound real weight. In 2020, 800 V was
essentially the Taycan — an EF car. 4.5% of A-segment cars at 800 V is wrong.

**Mode only** — matches the source, kills the artefact, loses the spread.
**Draw the band** — keeps spread, wrong in the early years.
**Draw, but clipped below the mode** — keeps spread where the band is real,
removes the upside-only inflation at the zero end.

Recommendation: the third. The band is genuine from ~2030 on, and the
distortion is confined to years when almost no BEVs existed.

---

## Reading the bands

Three different things, never to be added up:

1. **voltage band** — share of cars, each at one voltage
2. **capacity band** — the mixture of sizes, plus ignorance of where it lands
3. **composition band** — how much material a given battery holds

Only the second grows with the horizon.
