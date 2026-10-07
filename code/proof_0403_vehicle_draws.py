"""
PROOF ONLY -- writes nothing, modifies nothing, runs no stage.

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Question: what happens to 04_03's uncertainty bands when the per-draw vehicle
counts in data/processed/bev_draws/ are used, instead of being collapsed to a
deterministic scalar at src/traction_draws.py:222?

Both paths below use the SAME composition machinery -- DrawLibrary, draw_matrix,
joint_shares, scale -- imported from src/traction_draws.py and src/traction.py.
The ONLY difference is where the vehicle count comes from.

  CURRENT   mass(draw) = SUM_seg [ vehicles_seg(year) x compdraws ]
                         vehicles is one number      <-- the shortcut

  PROPOSED  mass(draw) = SUM_seg [ V_seg(draw, year) x mix_seg x compdraws ]
                         V is (draws x years) from disk

mix_seg is the cohort mix, normalised to 1 per (flow, year, segment). It stays
deterministic -- agreed 2026-09-22. Percentiles are taken only at the very end,
of the finished sum. Nothing is added or multiplied percentile-wise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/rm/Documents/GitHub/RAWCLICStockAndFlow")
sys.path.insert(0, str(ROOT))

from src.artifacts import load_many          # noqa: E402
from src import traction                     # noqa: E402
from src import traction_draws as td         # noqa: E402

N_DRAWS = 2000
DRIVES = ("BEV",)
FLOWS = ("inflow", "collected")              # both exist in tracker AND on disk
REPORT_YEARS = (2040, 2060)

params = load_many("params", root=ROOT)["params"]
tracker_keyed = load_many("tracker_keyed", root=ROOT)["tracker_keyed"]
library = td.DrawLibrary(params)

# -- segment torques, exactly as 04_03 builds them (line 374) -----------------
p04 = params.materials
composition = pd.read_excel(
    Path(p04.traction_composition_dir) / p04.traction_composition_file_name,
    sheet_name="Consolidated data")
composition = composition.rename(columns={c: str(c).strip()
                                          for c in composition.columns})
composition = composition[
    composition["parameterCode"].astype(str).str.strip()
    == str(p04.composition_parameter_code)].copy()
segment_torque = (composition.groupby("productKeyLevel3")["torque_nm"]
                  .mean().to_dict())
segment_torque = {str(k).strip().upper(): float(v)
                  for k, v in segment_torque.items()}

# -- the per-draw vehicle counts ---------------------------------------------
BEV = ROOT / "data" / "processed" / "bev_draws" / "BAU"
YEARS = np.load(BEV / "years.npy").astype(int)
YPOS = {int(y): i for i, y in enumerate(YEARS)}


def vehicles_draws(segment: str, flow: str) -> np.ndarray | None:
    """(N_DRAWS, years) vehicles, in units -- the file holds millions."""
    path = BEV / f"BEV_{segment}_{flow}.npy"
    if not path.exists():
        return None
    return np.asarray(np.load(path, mmap_mode="r")[:N_DRAWS],
                      dtype=np.float64) * 1e6


# ============================================================================
# PER-VEHICLE COEFFICIENTS: src/traction_draws.py coefficients(), with the
# vehicle count replaced by the normalised cohort mix. Everything else -- the
# segment group, joint_shares, the per-file year x voltage scale -- is the
# module's own, called here, not reimplemented.
# ============================================================================
def per_vehicle_coefficients(material: str):
    files = library.files_for(material)
    if not files:
        return None, [], {}
    segments = sorted(segment_torque)
    columns = [(name, seg) for name in files for seg in segments]
    position = {key: i for i, key in enumerate(columns)}
    motor_of = {name: library.motor_of(name) for name in files}

    share_cache: dict[tuple, dict] = {}
    scale_cache: dict[tuple, float] = {}
    rows: dict[tuple, np.ndarray] = {}
    seen_seg: dict[tuple, set] = {}

    for (region, drive), frame in tracker_keyed.items():
        if drive not in DRIVES:
            continue
        block = frame[["flow", "scrap_year", "cohort_year", "Segment",
                       "amount"]].copy()
        block["Segment"] = (block["Segment"].astype(str).str.strip()
                            .str.upper())
        block = block[block["Segment"].isin(segments)
                      & block["flow"].isin(FLOWS)]
        if block.empty:
            continue
        block = block.groupby(["flow", "scrap_year", "cohort_year", "Segment"],
                              as_index=False)["amount"].sum()

        # the cohort mix: amount / total over cohorts, per (flow, year, segment)
        total = block.groupby(["flow", "scrap_year", "Segment"])["amount"] \
                     .transform("sum")
        block["mix"] = np.where(total != 0, block["amount"] / total, 0.0)

        for flow, scrap_year, cohort, segment, amount, mix in \
                block.itertuples(index=False):
            if mix == 0.0:
                continue
            group = traction.segment_group(segment, params)
            key = (group, int(cohort))
            if key not in share_cache:
                share_cache[key] = traction.joint_shares(params, group,
                                                         int(cohort))
            joint = share_cache[key]
            row_key = (flow, int(scrap_year))
            row = rows.setdefault(row_key, np.zeros(len(columns)))
            seen_seg.setdefault(row_key, set()).add(segment)
            for name in files:
                motor = motor_of[name]
                slot = position[(name, segment)]
                for voltage in traction.VOLTAGES:
                    share = joint.get((motor, voltage), 0.0)
                    if share <= 0.0:
                        continue
                    sk = (name, int(cohort), voltage)
                    if sk not in scale_cache:
                        scale_cache[sk] = library.scale(name, int(cohort),
                                                        voltage)
                    # kg per vehicle -- no vehicle count here
                    row[slot] += mix * share * scale_cache[sk]

    if not rows:
        return None, columns, {}
    order = sorted(rows)
    matrix = np.vstack([rows[k] for k in order])
    index = pd.MultiIndex.from_tuples(order, names=["flow", "scrap_year"])
    return pd.DataFrame(matrix, index=index), columns, seen_seg


def proposed(material: str) -> dict[tuple, np.ndarray]:
    """{(flow, year): (N_DRAWS,)} kg, vehicles drawn."""
    frame, columns, _ = per_vehicle_coefficients(material)
    if frame is None or frame.empty:
        return {}
    V = td.draw_matrix(library, columns, segment_torque)[:, :N_DRAWS]
    V = np.asarray(V, dtype=np.float64)

    segments = sorted({seg for _, seg in columns})
    col_of_seg = {seg: [i for i, (_, s) in enumerate(columns) if s == seg]
                  for seg in segments}

    out: dict[tuple, np.ndarray] = {}
    C = frame.to_numpy(dtype=np.float64)
    flows = [f for f, _ in frame.index]
    years = [y for _, y in frame.index]

    veh_cache: dict[tuple, np.ndarray] = {}
    for seg in segments:
        idx = col_of_seg[seg]
        if not idx:
            continue
        # per-vehicle kg for this segment: (rows x draws)
        per_vehicle = C[:, idx] @ V[idx, :]
        for flow in FLOWS:
            if (seg, flow) not in veh_cache:
                veh_cache[(seg, flow)] = vehicles_draws(seg, flow)
            Vd = veh_cache[(seg, flow)]
            if Vd is None:
                continue
            for r, (f, y) in enumerate(zip(flows, years)):
                if f != flow or int(y) not in YPOS:
                    continue
                contrib = per_vehicle[r, :] * Vd[:, YPOS[int(y)]]
                key = (flow, int(y))
                out[key] = out.get(key, 0.0) + contrib
    return out


def current(material: str) -> dict[tuple, np.ndarray]:
    """{(flow, year): (N_DRAWS,)} kg, vehicles deterministic -- today's engine."""
    frame, columns = td.coefficients(tracker_keyed, segment_torque, params,
                                     library, material, DRIVES)
    if frame.empty:
        return {}
    V = np.asarray(td.draw_matrix(library, columns, segment_torque)[:, :N_DRAWS],
                   dtype=np.float64)
    product = frame.to_numpy(dtype=np.float64) @ V
    out = {}
    for r, key in enumerate(frame.index):
        region, flow, year = key
        if flow in FLOWS:
            out[(flow, int(year))] = product[r, :]
    return out


def band(v: np.ndarray) -> tuple[float, float, float, float]:
    lo, mid, hi = np.percentile(v, [2.5, 50, 97.5])
    return mid, lo, hi, (100.0 * (hi - lo) / mid if mid else float("nan"))


# ============================================================================
materials = sorted(library.manifest["materialClass"].dropna().unique())
print(f"\n  draws {N_DRAWS:,}   materials {materials}")
print(f"  segments {sorted(segment_torque)}\n")

print(f"  {'material':<10}{'flow':<11}{'year':>6}"
      f"{'median kt':>12}{'CURRENT band':>16}{'PROPOSED band':>16}")
print("  " + "-" * 71)

store = {}
for material in materials:
    cur = current(material)
    pro = proposed(material)
    store[material] = (cur, pro)
    for flow in FLOWS:
        for year in REPORT_YEARS:
            a, b = cur.get((flow, year)), pro.get((flow, year))
            if a is None or b is None:
                continue
            ma, _, _, wa = band(a / 1e6)
            mb, _, _, wb = band(b / 1e6)
            print(f"  {material:<10}{flow:<11}{year:>6}{mb:>12.3f}"
                  f"{wa:>15.1f}%{wb:>15.1f}%")

print("\n  RATIO collected/inflow, formed PER DRAW (never percentile/percentile):")
print(f"  {'material':<10}{'year':>6}{'median':>10}{'95% band':>22}{'width':>9}")
print("  " + "-" * 57)
for material in materials:
    cur, pro = store[material]
    for year in REPORT_YEARS:
        inf, col = pro.get(("inflow", year)), pro.get(("collected", year))
        if inf is None or col is None:
            continue
        r = np.where(inf > 0, col / inf, np.nan) * 100.0
        lo, mid, hi = np.nanpercentile(r, [2.5, 50, 97.5])
        print(f"  {material:<10}{year:>6}{mid:>9.2f}%"
              f"   [{lo:7.2f} - {hi:7.2f}]{hi - lo:>8.2f} pp")

print("\n  SANITY -- medians must stay close; only the band should open up:")
for material in materials:
    cur, pro = store[material]
    for flow in FLOWS:
        for year in REPORT_YEARS:
            a, b = cur.get((flow, year)), pro.get((flow, year))
            if a is None or b is None:
                continue
            ma, mb = np.median(a), np.median(b)
            rel = 100.0 * (mb - ma) / ma if ma else float("nan")
            print(f"    {material:<10}{flow:<11}{year}  "
                  f"median shift {rel:+7.2f}%")
