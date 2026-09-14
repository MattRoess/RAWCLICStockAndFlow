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

## The voltage band: the upper arm may not exceed the lower one

The source's band is a symmetric ±0.12 belief, clipped where it would cross
zero. Drawing the clipped shape hands the surviving upper arm all the weight
the lower one lost — AB 2020 is `(0, 0.007, 0.127)`, and a triangular through
that puts 4.5% of A-segment cars at 800 V against a mode of 0.7%. In 2020 that
was the Taycan, an EF car.

**Decided 2026-09-14:** cut the upper arm back to the length of the lower one.

It bites only where the floor truncated the band:

| | before | after | mode |
|---|---|---|---|
| AB 2020 | 4.5% | **0.7%** | 0.7% |
| CD 2030 | 40.0% | 39.5% | 40.0% |
| EF 2020 | 12.3% | 12.4% | 12.3% |

A symmetric band is untouched. A band clipped at the **top** keeps its downside
— EF 2070 stays `(0.88, 1.0, 1.0)` and draws 96%, because a ceiling at 100% is
real rather than an artefact, and the band genuinely allows 88%.

---

## Reading the bands

Three different things, never to be added up:

1. **voltage band** — share of cars, each at one voltage
2. **capacity band** — the mixture of sizes, plus ignorance of where it lands
3. **composition band** — how much material a given battery holds

Only the second grows with the horizon.
