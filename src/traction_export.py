"""
src/traction_export.py
======================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Writes the traction-motor draws in the layout `RAWCLICRecoveryModel` reads.

⚠️ THE DRAWS, NOT THE PERCENTILES. 04_03 computes a 200 000-long distribution
for every flow and year and then reduces it to mean, p025 and p975. The
recovery model runs its own Monte Carlo through a transfer-coefficient chain,
so it needs the distribution itself -- handed percentiles it could only
multiply them, which is the arithmetic that model forbids.

THE LAYOUT, which is `src/upstream.py`'s contract over there:

    <root>/<grade>/<flow>/
        years.npy                        the years exported
        __component____<group>.npy       (draws, years)  the group itself
        <element>__<group>.npy           (draws, years)  an element in it

HOW THE FOUR LAYERS MAP HERE
    Layer 1  product     BEV
    Layer 2  component   the material class: magnet, copper, aluminium,
                         lamination, steel -- these are what the recycling
                         routes actually act on, and they line up with the
                         report's steps 9-11 (winding separation, housing
                         recovery, lamination recovery)
    Layer 3  material    not resolved here; the recovery case supplies a
                         placeholder
    Layer 4  element     Nd, Pr, Dy, Tb -- MAGNET ONLY. Copper, aluminium and
                         both steels stay materials, decided 2026-09-24: that
                         is where the supply question is, and what a recovery
                         route returns from a lamination stack is electrical
                         steel, not iron and silicon separately.

THE GRADE FOLDERS
    mix   the grade drawn per draw across the classes on disk -- a fleet is a
          mixture, not uniformly one class
    SH    pinned, and likewise UH and EH

    Pinning is one setting in the recovery model (`--scenario EH`) and needs no
    case edit. It answers "what if only EH is feasible" directly.

    ⚠️ ONLY Dy, Tb AND Fe DIFFER BETWEEN THEM. Nd and Pr are byte-identical in
    every class, because the workbook's 0.29-0.32 didymium applies to all of
    them. So `mix` and `EH` differ in two of the four exported element arrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ⚠️ KILOGRAMS IN, KILOTONNES OUT. 04_03 works in kg -- vehicle counts in
# millions times 1e6, times kg per vehicle -- and the recovery model's upstream
# contract is kilotonnes, as the battery's recovery export is written. Exporting kg
# was silently a factor of a million: the traction total at 2050 read
# 9.43e8 against the battery's 4281, where the real ratio is a ~100 kg motor
# against a ~400 kg pack. Caught by comparing magnitudes with the battery, not
# by anything failing.
KG_PER_KT = 1e6

MIX = 'mix'
ELEMENTS = ('Nd', 'Pr', 'Dy', 'Tb')
GROUP_MARKER = '__component__'

# ⚠️ THE TRACKER HAS NO `outflow` ROW. It keys every end-of-life vehicle by
# WHERE IT WENT -- `collected`, `export`, `unknown_whereabouts` -- and those
# three PARTITION the outflow: a vehicle leaving the fleet is collected,
# exported, or unaccounted for, and there is no fourth destination
# (documentation/DESIGN_collected_flow_definition.md, which exists because
# 03_02 once dropped the export share and overstated collected by 2.3% for BEV
# and 16.3% for the rest). So the outflow is the three added together, and
# `inflow` is what entered the fleet.
#
# This export wrote `collected` and `inflow` and stopped, and the recovery
# model's whole-account figures need the third: `outflow - collected` is the
# mass that left the fleet and never reached a recycler at all, which is the
# band those figures draw as "never collected" and the reason they exist. With
# `outflow` missing, `account()` over there returns None and FOUR figures --
# account, losses, trapped, fate -- are silently not drawn. The battery export
# has carried all three since it was written (`04_04_batteries.FLOWS`), and
# the same definition holds here: of the BEVs that leave the fleet, most are
# collected, a few are exported and the rest are never traced.
#
# Added 2026-09-28, after the recovery model went looking for copper's account
# and found it could not be computed.
COMPOSED = {'outflow': ('collected', 'export', 'unknown_whereabouts')}
EXPORTED_FLOWS = ('collected', 'inflow', 'outflow')


def wanted_years(index_years: np.ndarray, first: int, last: int,
                 step: int) -> np.ndarray:
    """The exported years: those asked for that the flows actually contain."""
    asked = set(range(first, last + 1, step))
    return np.array(sorted(y for y in index_years if int(y) in asked), dtype=int)


def _rows_for(frame: pd.DataFrame, flow: str, years: np.ndarray):
    """Row positions of one flow at the exported years, in year order."""
    index = frame.index
    wanted = {(f, int(y)): None for f, y in zip(
        index.get_level_values('flow'), index.get_level_values('scrap_year'))}
    positions = []
    for year in years:
        match = [i for i, (_r, f, y) in enumerate(index)
                 if f == flow and int(y) == int(year)]
        if len(match) != 1:
            return None
        positions.append(match[0])
    return np.array(positions)


def _mixed_chemistry(library, motor: str, grades: list[str], element: str,
                     rng: np.random.Generator) -> np.ndarray | None:
    """
    One chemistry draw per draw, with the grade drawn too.

    ⚠️ THE GRADE IS DRAWN, NOT AVERAGED. Averaging the three classes would
    produce a magnet that exists nowhere and would understate the spread: the
    SH and EH dysprosium ranges do not overlap, so their mean sits between two
    populated regions. Drawing the class per draw keeps the result a mixture of
    real magnets.
    """
    columns = []
    for grade in grades:
        names, array = library.chemistry(motor, grade)
        if element not in names:
            return None
        columns.append(np.asarray(array)[:, names.index(element)])
    stack = np.stack(columns, axis=1)                    # (draws, grades)
    pick = rng.integers(0, len(grades), size=stack.shape[0])
    return stack[np.arange(stack.shape[0]), pick]


def _draw_matrix_mixed(library, columns, segment_torque, element, grades,
                       rng) -> np.ndarray:
    """`draw_matrix` with the grade drawn per draw instead of pinned."""
    stack = []
    for name, segment in columns:
        vector = library.at_torque(name, segment_torque[segment])
        fractions = _mixed_chemistry(library, library.motor_of(name), grades,
                                     element, rng)
        if fractions is None:
            return np.empty((0, 0), dtype=np.float32)
        stack.append((vector * fractions).astype(np.float32))
    return np.vstack(stack)



# ⚠️ THE VEHICLE COUNT IS SAMPLED, AND UNTIL 2026-09-30 IT WAS NOT.
#
# `traction_draws.coefficients()` folds the tracker's vehicle count into its
# deterministic matrix -- `vehicles = float(amount) * 1e6`, a scalar -- so the
# only random thing in `matrix @ draw_matrix` was the composition. Mass is
# count x composition, and a product of two uncertain quantities cannot be more
# certain than either of them: the count alone has a CV of 7% to 21% depending
# on the year, the composition 2.5% to 3.3%, and the exported arrays carried
# the composition figure alone. Every interval downstream was 2.9x to 6.6x too
# narrow, and nothing failed, because an interval that is too small does not
# misdraw anything.
#
# `03_02` already writes the count distribution -- `bev_draws/<scenario>/
# BEV_<segment>_<flow>.npy`, (draws, years), millions of vehicles -- and
# `04_02` has used it all along. This stage simply never opened it.
#
# PER SEGMENT, WHICH IS THE RESOLUTION THE DISTRIBUTION EXISTS AT. Scaling the
# finished total by one aggregate ratio would inject the right overall spread
# through the wrong mix: segments carry different motors, so one number for all
# twelve is a different distribution. The sum factorises by segment, so this
# costs twelve narrower matrix products instead of one wide one.
#
# ⚠️ THE RATIO IS AGAINST THE TRACKER'S TOTAL, SO THE DRAWN COUNT REPLACES IT
# AND THE MEANS MOVE. That is the point, and it was settled in this project on
# 22 September 2026 (`documentation/HANDOVER.md`, "THE BLOCKER IS RESOLVED"):
#
#   Outflow is a nonlinear function of the drawn lifetime, so
#   E[outflow(lambda)] != outflow(E[lambda]). The tracker is one deterministic
#   run at the point lifetime and therefore CANNOT equal the Monte Carlo mean.
#   The draws are the correct quantity; the tracker's collected is biased high
#   by 2-6%. So the fix may adopt bev_draws -- that is the right answer, not a
#   silent substitution.
#
# A first version of this divided by the DRAWS' own mean so that no mean would
# move. That looks safer and is not: it preserves a bias that had already been
# measured and diagnosed here, and calls the result an uncertainty fix.
#
# Algebraically this is the agreed form. `matrix` holds
# `SUM_cohort vehicles * share * scale`, so dividing by the tracker's total for
# that (segment, flow, year) turns `vehicles` into the cohort mix normalised to
# 1 -- `mix_seg` in `code/proof_0403_vehicle_draws.py` -- and the drawn count
# multiplies it. The cohort mix stays deterministic, agreed the same day.
#
# WHAT IS STILL DETERMINISTIC AFTER THIS, and it is not nothing: the
# motor-type share, the voltage share, their product (assumed independent --
# see `traction.joint_shares`), the year x voltage scale factor, the torque per
# segment, and the split of a segment-year across its 60 cohorts. The count is
# the largest missing term, not the only one, and the intervals remain too
# narrow by an amount nobody has measured.
BEV_DRAWS = 'bev_draws'
COUNT_SCENARIO = 'BAU'          # the tracker this stage loads is tracker_keyed_BAU


def tracker_totals(tracker_keyed: dict, drive_trains) -> dict:
    """
    `{(segment, flow, scrap_year): vehicles}` as the deterministic matrix uses.

    The denominator that turns `vehicles` in `coefficients()` into a cohort mix
    normalised to 1. Summed over regions and cohorts, exactly as that function
    sums them, and over the same drive trains.
    """
    totals: dict = {}
    for (_region, drive), frame in tracker_keyed.items():
        if drive not in drive_trains:
            continue
        block = frame[['flow', 'scrap_year', 'Segment', 'amount']].copy()
        block['Segment'] = block['Segment'].astype(str).str.strip().str.upper()
        for flow, year, segment, amount in block.itertuples(index=False):
            key = (str(segment), str(flow), int(year))
            totals[key] = totals.get(key, 0.0) + float(amount) * 1e6

    # ⚠️ AND THE COMPOSED FLOWS, or they go unsampled. The tracker has no
    # `outflow` row -- it keys every vehicle by where it went, and outflow is
    # the sum of those destinations (see COMPOSED). Without this the ratios
    # covered two flows of three and `outflow` kept its deterministic count
    # silently, which is the defect this whole section is about, surviving in
    # a third of the arrays.
    for flow, parts in COMPOSED.items():
        for (segment, part, year), value in list(totals.items()):
            if part not in parts:
                continue
            key = (segment, flow, year)
            totals[key] = totals.get(key, 0.0) + value
    return totals


def count_ratios(root: Path, years: np.ndarray, flows, totals: dict,
                 scenario: str = COUNT_SCENARIO):
    """
    `{(segment, flow, year): (draws,)}` -- the drawn count over the tracker's.

    Multiplying a segment's deterministic contribution by this replaces the
    tracker's count with the drawn one, so the spread arrives AND the mean
    moves to the draws' mean, which is the unbiased quantity. See the note
    above.

    Returns `{}` when the draws are not on disk, and the caller then exports as
    before and says so -- a missing artefact must not silently produce narrow
    intervals again.
    """
    folder = Path(root) / BEV_DRAWS / scenario
    axis = folder / 'years.npy'
    if not axis.exists():
        return {}
    available = np.load(axis)
    where = {int(y): int(np.where(available == y)[0][0])
             for y in years if y in available}
    if len(where) != len(years):
        missing = sorted(set(int(y) for y in years) - set(where))
        raise FileNotFoundError(
            f'{folder} has no vehicle counts for {missing}. The export years '
            f'and the count draws must cover the same span, or the arrays '
            f'written would mix sampled and unsampled years.')

    out = {}
    for path in sorted(folder.glob('BEV_*_*.npy')):
        stem = path.stem[len('BEV_'):]
        segment, _, flow = stem.rpartition('_')
        if flow not in set(flows) | set(sum(COMPOSED.values(), ())):
            continue
        block = np.load(path, mmap_mode='r')
        for year, column in where.items():
            key = (segment.upper(), flow, year)
            deterministic = totals.get(key, 0.0)
            if deterministic <= 0:
                continue          # the tracker has no vehicles here; nothing to scale
            drawn = np.asarray(block[:, column], dtype=np.float64) * 1e6
            out[key] = (drawn / deterministic).astype(np.float32)
    return out


def sampled_product(matrix, columns, vectors, frame, ratios, years) -> np.ndarray:
    """
    `matrix @ vectors`, with each segment scaled by its own count draws.

    The deterministic sum factorises by segment, so

        result[row, draw] = SUM_segment  ratio[segment, flow, year, draw]
                            * ( matrix[row, segment columns] @ vectors[segment columns] )

    With `ratios` empty this is exactly `matrix @ vectors`, which is what the
    caller falls back to when the count draws are absent.
    """
    if not ratios:
        return matrix @ vectors

    by_segment: dict[str, list[int]] = {}
    for position, (_file, segment) in enumerate(columns):
        by_segment.setdefault(str(segment).upper(), []).append(position)

    flow_of = frame.index.get_level_values('flow')
    year_of = frame.index.get_level_values('scrap_year')
    wanted = {int(y) for y in years}

    total = np.zeros((matrix.shape[0], vectors.shape[1]), dtype=np.float32)
    for segment, slots in by_segment.items():
        part = matrix[:, slots] @ vectors[slots, :]
        for row in range(part.shape[0]):
            year = int(year_of[row])
            if year not in wanted:
                continue                      # not exported; left unscaled
            ratio = ratios.get((segment, str(flow_of[row]), year))
            if ratio is not None:
                part[row, :] *= ratio
        total += part
    return total


def export(tracker_keyed: dict, composition: pd.DataFrame, params,
           out_root: Path, flows=EXPORTED_FLOWS,
           first: int = 2020, last: int = 2070, step: int = 5,
           counts_root: Path | None = None) -> dict:
    """
    Write every grade folder the recovery model may ask for.

    Returns a manifest of what was written, which 04_03 prints. A run that
    writes nothing says so rather than leaving an empty tree that looks
    finished.

    `counts_root` is the `data/processed` holding `bev_draws/`; the vehicle
    count is sampled from it per segment -- see `count_ratios`. Left None it
    is taken to be `out_root`'s parent, which is where 04_03 puts both.
    """
    from src import traction_draws as td

    out_root = Path(out_root)
    library = td.DrawLibrary(params)
    segment_torque = (composition.groupby('productKeyLevel3')
                      ['torque_nm'].first().to_dict())
    drive_trains = tuple(params.materials.traction_drive_trains)
    grades = library.grade_classes()
    materials = sorted(composition['materialClass'].dropna().unique())

    # ⚠️ ONE SEED FOR THE WHOLE EXPORT, so the `mix` folder's grade draw is
    # reproducible and the same draw index means the same grade in every
    # element array. Drawing Dy and Tb from different grade picks would build a
    # magnet that is SH in one element and EH in another.
    rng = np.random.default_rng(20260924)
    manifest = []

    # The count draws, once. Read before the loop so a missing artefact is
    # reported at the top rather than silently producing narrow intervals for
    # twenty minutes -- which is the defect this whole section exists for.
    every_year = wanted_years(
        np.arange(first, last + 1), first, last, step)
    ratios = count_ratios(Path(counts_root) if counts_root is not None
                          else out_root.parent, every_year, flows,
                          tracker_totals(tracker_keyed, drive_trains))
    if ratios:
        print(f'  vehicle count SAMPLED from {BEV_DRAWS}/{COUNT_SCENARIO}: '
              f'{len({k[0] for k in ratios})} segments x '
              f'{len({k[1] for k in ratios})} flows x '
              f'{len({k[2] for k in ratios})} years')
    else:
        print(f'  ⚠️ NO VEHICLE-COUNT DRAWS at '
              f'{Path(counts_root or out_root.parent) / BEV_DRAWS / COUNT_SCENARIO}. '
              f'Exporting with a DETERMINISTIC count: every interval '
              f'downstream will be several times too narrow. Run 03_02 with '
              f'bev_electronics_export_draws = True.')

    for grade in [MIX] + list(grades):
        for material in materials:
            frame, columns = td.coefficients(
                tracker_keyed, segment_torque, params, library, material,
                drive_trains)
            if frame.empty:
                continue
            years = wanted_years(
                np.unique(frame.index.get_level_values('scrap_year')),
                first, last, step)
            matrix = frame.to_numpy(dtype=np.float32)

            # the group total
            product = sampled_product(
                matrix, columns,
                td.draw_matrix(library, columns, segment_torque),
                frame, ratios, years)
            manifest += _write(out_root, grade, flows, frame, years, product,
                               f'{GROUP_MARKER}__{material}')

            # its elements -- magnet only, by the 2026-09-24 decision
            if material != 'magnet':
                continue
            for element in ELEMENTS:
                if grade == MIX:
                    vectors = _draw_matrix_mixed(library, columns,
                                                 segment_torque, element,
                                                 grades, rng)
                else:
                    names, _ = library.chemistry(
                        library.motor_of(columns[0][0]), grade)
                    if element not in names:
                        continue
                    vectors = td.draw_matrix(library, columns, segment_torque,
                                             element=element, grade=grade)
                if vectors.size == 0:
                    continue
                manifest += _write(out_root, grade, flows, frame, years,
                                   sampled_product(matrix, columns, vectors,
                                                   frame, ratios, years),
                                   f'{element}__{material}')
    return pd.DataFrame(manifest)


def _parts_of(frame, flow: str, years: np.ndarray):
    """
    Row positions per tracker flow making up one exported flow, or None.

    A composed flow is all of its parts or none of them. Some present and
    some missing would write an outflow that is quietly too small, and too
    small on the outflow means too small on "never collected" -- a recovery
    figure that flatters collection without anything failing. If the tracker's
    vocabulary ever changes, this says so instead.
    """
    parts = COMPOSED.get(flow, (flow,))
    found = {part: _rows_for(frame, part, years) for part in parts}
    present = [part for part, rows in found.items() if rows is not None]
    if not present:
        return None
    if len(present) != len(parts):
        missing = sorted(set(parts) - set(present))
        raise ValueError(
            f"the tracker has {sorted(present)} but not {missing}, so "
            f"'{flow}' cannot be composed. It is the sum of {list(parts)}; "
            f"writing it from the parts that happen to be there would "
            f"understate what left the fleet.")
    return [found[part] for part in parts]


def _write(out_root: Path, grade: str, flows, frame, years, product,
           stem: str) -> list[dict]:
    """One array per flow, transposed to (draws, years) as the reader wants."""
    written = []
    for flow in flows:
        parts = _parts_of(frame, flow, years)
        if parts is None:
            continue
        # Summed over the parts BEFORE the transpose, so a composed flow is
        # one array of exactly the shape a single one has.
        total = sum(product[positions] for positions in parts)
        folder = out_root / grade / flow
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'years.npy', years)
        array = np.ascontiguousarray((total.T / KG_PER_KT).astype(np.float32))
        np.save(folder / f'{stem}.npy', array)
        written.append({'grade': grade, 'flow': flow, 'array': stem,
                        'draws': array.shape[0], 'years': array.shape[1],
                        'MB': round(array.nbytes / 1e6, 1)})
    return written
