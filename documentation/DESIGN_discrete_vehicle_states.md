# Discrete vehicle states: voltage and capacity

A car has one pack voltage and one pack size. It is never a blend of two. This
note records how that is kept true through the Monte Carlo, what "discrete"
does and does not mean once a draw runs forty years into the future, and one
argument that was proposed and then refuted by the data.

Written 2026-09-14. Companion to `DESIGN_bev_capacity_for_04_04.md`.

---

## 1. Why blending is the failure to avoid

Share-weighting a discrete state replaces a mixture with its mean. If 42% of
EF cars are 800 V in 2025, a share-weighted car carries 42% of the copper
saving — and no car does that. The band collapses to a point that describes
nothing in the fleet.

RAWCLICVehicleElectronics hit this and wrote the rule into `drivers.py`:

> *"THE STATES ARE DISCRETE. A vehicle is one architecture, never a blend. A
> caller must draw a state per iteration and hold it across years —
> share-weighting instead collapses a bimodal mixture into its mean and
> destroys the band."*

The same rule governs both states here.

---

## 2. Voltage: binary, and it stays binary

`src/battery_voltage.py`. Every draw in every year is **400 or 800**, never
between. Verified on 50,000 draws.

What is drawn is a **percentile, held for life**: `u` in [0, 1], and the car is
800 V in any year whose adoption share exceeds `u`. A low `u` adopts early, a
high one late, and no car flickers between voltages from one year to the next.
This is the `u_volt` mechanism from the electronics project.

The share itself comes from that project's penetration workbook, resolved at
its own AB / CD / EF grain — EF leads hard, 0.42 in 2025 against CD 0.07 and
AB 0.02, which is where 800 V actually appeared first.

> **Open question, not decided.** The source gives Min/Mode/Max and its own
> callers *select* one of the three, running the model three times. This
> implementation *draws* the band per iteration instead, which is consistent
> with everything else here but is not what the source does — and where the
> band is clipped at zero the mean sits well above the mode: AB in 2020 comes
> out at 4.5% against a mode of 0.7%, because a `Max` of 0.127 meant as an
> upper bound gets real weight. 800 V in 2020 was essentially the Taycan.
> See the parameter comment.

---

## 3. Capacity: discrete inside a draw, continuous across draws

`src/battery_capacity.py`. This is the subtle one, and both halves matter.

**Inside one draw, a segment has exactly its levels and nothing between them.**
One growth rate and one plateau year are drawn per world and shared by every
segment, so the four JC levels stay four numbers:

```
draw 0, 2040:   77.3   83.2   95.1  101.1 kWh
draw 1, 2040:   83.9   90.3  103.2  109.7 kWh
```

A car in draw 0 is one of those four. It is never 88 kWh.

**Across draws the values smear** — 4 distinct capacities at the 2024 anchor,
about 50,000 at 2040. Those are not fifty thousand pack sizes in one market.
They are fifty thousand guesses about *what the four sizes will be*, each guess
internally discrete. The continuity is in the uncertainty about the future
market, not inside any car.

---

## 4. The argument that was refuted

It was proposed that each draw's grown levels be **rounded to a 5 kWh grid**,
on the reasoning that pack sizes in practice come in round numbers, so the
output would both stay discrete and read like a real market.

**The data refutes it.** Of 1,068 models introduced since 2022:

| grid | share of models landing on it |
|---|---|
| multiples of 1 kWh | 76.7% |
| multiples of 5 kWh | **25.6%** |
| multiples of 10 kWh | 16.8% |

The twelve commonest sizes are 82, 100, 84, 54, 105, 91, 75, 63, 50, 61, 68 and
66 kWh. Pack capacity is an engineering outcome — cell count times cell
capacity — not a marketing round number. Rounding to 5 would move 82 → 80 and
84 → 85, neither of which anybody builds: tidier output, less true.

**What is true is a better version of the same idea.** There are 149 distinct
values across those 1,068 models, and **17 of them cover half the fleet**;
82 kWh alone appears in 115 models, 11% of everything. Packs repeat because
platforms share them — one pack design serves many cars.

So the market genuinely is a small set of specific sizes, which is exactly what
a draw already produces. **No re-rounding.** It would add error without adding
truth.

> One consequence worth stating: the levels in
> `materials.battery_capacity_levels` are themselves 5 kWh bin centres, so JC's
> dominant "80 kWh" is really the 82 kWh pack of the Teslas and VWs that
> dominate that segment, binned down. The approximation is at the measurement
> step, where it is visible, rather than applied again to the future.

---

## 5. What this means for reading the output

- **A band on voltage** is a share of cars, and every car in it is at one
  voltage or the other.
- **A band on capacity** is two things at once: the mixture of sizes a segment
  offers, and our ignorance of where that mixture lands in a given future year.
  The second grows with the horizon; the first does not.
- **Neither band is composition uncertainty.** How much material a battery of a
  given size and chemistry contains is a separate distribution, carried in the
  composition files.

Anything reporting these bands should say which of the three it is showing, or
a reader will add them up wrongly.
