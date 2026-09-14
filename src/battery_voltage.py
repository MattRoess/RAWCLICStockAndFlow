"""
src/battery_voltage.py
======================

Whether a BEV carries a 400 V or an 800 V pack, per Monte Carlo draw and year.

    from src.battery_voltage import voltage_draws
    voltage_draws(params, "JC", years, n_draws=200_000, seed=11)   # -> 400 / 800

It decides a real material, not a label: the composition files carry a row per
voltage, and the 800 V one has a third less copper in the cables and the cell
terminals. The anode current collector is untouched -- it is sized by the cell,
not by the pack bus.

A PERCENTILE PER DRAW, NOT A COIN FLIP PER YEAR
------------------------------------------------
The share of new sales that are 800 V rises over time. A draw is one car's
place in that adoption order, held for life: `u` is drawn once in [0, 1] and the
car is 800 V in any year whose share exceeds it. A low `u` adopts early, a high
one late, and nobody flickers between voltages from year to year.

This is the mechanism RAWCLICVehicleElectronics uses for the same driver
(`u_volt` in its `vehicle_state.py`), and its reasoning applies unchanged:

    "THE STATES ARE DISCRETE. A vehicle is one architecture, never a blend. A
     caller must draw a state per iteration and hold it across years --
     share-weighting instead collapses a bimodal mixture into its mean and
     destroys the band."

Share-weighting here would give every car two-thirds of the copper saving at
once, which is not a thing any car is.

⚠️ THE SHARE ITSELF IS UNCERTAIN. The source gives Min/Mode/Max, about +/-0.12,
and that band is drawn too -- a draw sits at the same point of it in every year,
so a world where 800 V arrives early stays early.
"""

from __future__ import annotations

import numpy as np

LOW_VOLTAGE = 400
HIGH_VOLTAGE = 800


def _share_curve(params, group: str, years: np.ndarray,
                 position: np.ndarray) -> np.ndarray:
    """
    800 V share per year, shape (n_draws, n_years).

    `position` is each draw's place in the Min-Mode-Max band, in [0, 1], held
    across years. Interpolation between the source's anchor years is linear:
    the anchors are five and ten years apart and a smoother would invent turns
    between them that nobody chose.
    """
    table = params.materials.pack_voltage_800v_share[group]
    anchors = np.array(sorted(table), dtype=float)
    lo = np.array([table[int(y)][0] for y in anchors], dtype=float)
    mode = np.array([table[int(y)][1] for y in anchors], dtype=float)
    hi = np.array([table[int(y)][2] for y in anchors], dtype=float)

    lo_y = np.interp(years, anchors, lo)
    mode_y = np.interp(years, anchors, mode)
    hi_y = np.interp(years, anchors, hi)

    # THE UPPER ARM MAY NOT EXCEED THE LOWER ARM.
    #
    # The source's band is a symmetric +/-0.12 belief that has been clipped
    # where it would cross zero. Drawing the clipped shape gives the surviving
    # upper arm the whole of the weight the lower one lost: AB in 2020 has
    # (0, 0.007, 0.127) and a triangular through it means 4.5% of A-segment
    # cars at 800 V against a mode of 0.7%. In 2020 that was the Taycan, an EF
    # car.
    #
    # So the upper arm is cut back to the length of the lower one. This bites
    # ONLY where the floor truncated the band: a symmetric band is untouched
    # (CD 2030 stays 0.28/0.40/0.52), and a band clipped at the TOP keeps its
    # downside (EF 2070 stays 0.88/1.0/1.0) because a ceiling at 100% is real
    # rather than an artefact.
    hi_y = np.minimum(hi_y, mode_y + (mode_y - lo_y))

    # A triangular's inverse CDF, evaluated at each draw's fixed position.
    span = np.where(hi_y > lo_y, hi_y - lo_y, 1.0)
    split = np.where(hi_y > lo_y, (mode_y - lo_y) / span, 0.0)
    u = position[:, None]
    left = lo_y + np.sqrt(np.clip(u * span * (mode_y - lo_y), 0.0, None))
    right = hi_y - np.sqrt(np.clip((1.0 - u) * span * (hi_y - mode_y), 0.0, None))
    share = np.where(u <= split, left, right)
    return np.clip(share, 0.0, 1.0)


def voltage_draws(params, segment: str, years, *, n_draws: int,
                  seed: int) -> np.ndarray:
    """Pack voltage in volts, shape (n_draws, n_years), values 400 or 800."""
    materials = params.materials
    group = materials.pack_voltage_segment_groups.get(segment)
    if group is None:
        raise KeyError(
            f"no materials.pack_voltage_segment_groups entry for {segment!r}. "
            f"Known: {sorted(materials.pack_voltage_segment_groups)}")

    years = np.asarray(years, dtype=float)
    # Two independent streams: where this car sits in the adoption order, and
    # where this world sits in the share's own band.
    rng = np.random.default_rng([seed, abs(hash(segment)) % (2**32)])
    adoption = rng.random(n_draws)
    band_rng = np.random.default_rng(seed)          # the band is market-wide
    position = band_rng.random(n_draws)

    share = _share_curve(params, group, years, position)
    return np.where(adoption[:, None] < share, HIGH_VOLTAGE, LOW_VOLTAGE)


def main() -> int:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import pandas as pd
    from src.params_schema import Params

    params = Params()
    years = [2020, 2025, 2030, 2040, 2050, 2070]
    segments = list(params.materials.pack_voltage_segment_groups)
    rows = []
    for segment in segments:
        drawn = voltage_draws(params, segment, years, n_draws=20_000, seed=11)
        rows.append({"segment": segment,
                     **{int(y): float((drawn[:, i] == HIGH_VOLTAGE).mean())
                        for i, y in enumerate(years)}})
    table = pd.DataFrame(rows).set_index("segment")
    print("share of draws at 800 V, by segment and year\n")
    print(table.round(3).to_string())
    print("\nsource mode, for comparison:")
    for group in ("AB", "CD", "EF"):
        share = params.materials.pack_voltage_800v_share[group]
        print(f"  {group}: " + ", ".join(f"{y}: {share[y][1]:.3f}"
                                         for y in years if y in share))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
