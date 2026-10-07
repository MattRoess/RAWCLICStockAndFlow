"""
src/traction_draws.py
=====================

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

Traction-motor material and element flows from the 200,000 draws themselves.

    from src.traction_draws import DrawLibrary
    library = DrawLibrary(params)
    library.flow_percentiles(coefficients, "copper")     # -> mean, p025, p975

⚠️ WHY THIS EXISTS. Everything before it multiplied a mean by a vehicle count
and carried p025/p975 alongside as if they were quantities you may do arithmetic
on. They are not, and the traction project says so in as many words: "a
percentile is a property of a distribution, and arithmetic on two percentiles is
not the percentile of the result". That project persists its arrays precisely so
the consumer does not have to do that -- 31 mass arrays and 12 chemistry arrays,
200,000 draws each -- and this module uses them.

THE ARITHMETIC, and why it is a matrix multiply rather than a loop.

A flow in one scrap year is a sum over segments, cohort years, motor types and
voltage classes. Written naively that is millions of length-200,000 vector
operations. But the draws depend only on (file, torque), and the torque depends
only on the segment -- everything else in the term is a DETERMINISTIC scalar:
the vehicle count, the motor-type share, the voltage share, the year factor.
So:

    mass_draws[row] = SUM over (file, segment) of
                      coefficient[row, (file, segment)] x draws[(file, segment)]

which is one matrix product, `C @ V`, with `C` the deterministic coefficients
and `V` the draws evaluated at each segment's torque. Rows are (scrap year,
flow).

⚠️ AND THE SHARES STAY DETERMINISTIC ON PURPOSE. A share is the probability one
CAR is of a type, and a fleet-year holds millions of cars, so the realised
fraction is the share to many decimal places. Drawing a motor type per Monte
Carlo draw would add sampling noise that does not exist in a fleet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class DrawLibrary:
    """The traction project's draw arrays, addressable by torque."""

    def __init__(self, params) -> None:
        materials = params.materials
        self.directory = (Path(materials.traction_composition_dir)
                          / materials.traction_draws_dir)
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"no draw arrays at {self.directory}. They are written by "
                f"RAWCLICVehicleTractionMotor's 01_composition.py -- run that "
                f"project, or set params.materials.traction_draws_dir."
            )
        self.grid = np.array(
            [float(line) for line in
             (self.directory / "torque_grid.txt").read_text().split()],
            dtype=float)
        self.manifest = pd.read_csv(self.directory / "draws_manifest.csv")
        self.scales = pd.read_csv(self.directory / "draw_scales.csv")
        self._cache: dict[str, np.ndarray] = {}

    # -- the arrays ----------------------------------------------------------

    def array(self, name: str) -> np.ndarray:
        """One draw array, memory-mapped and cached."""
        if name not in self._cache:
            self._cache[name] = np.load(self.directory / name, mmap_mode="r")
        return self._cache[name]

    def at_torque(self, name: str, torque: float) -> np.ndarray:
        """
        A file's draws at one torque: (draws,) float64.

        Linear between the two grid points that bracket it, clamped outside --
        the same rule the traction project uses when it interpolates its own
        anchor, so the two projects cannot disagree about what 166 Nm means.
        """
        block = self.array(name)
        grid = self.grid
        if torque <= grid[0]:
            return np.asarray(block[:, 0], dtype=np.float64)
        if torque >= grid[-1]:
            return np.asarray(block[:, -1], dtype=np.float64)
        upper = int(np.searchsorted(grid, torque))
        lower = upper - 1
        weight = (torque - grid[lower]) / (grid[upper] - grid[lower])
        return ((1.0 - weight) * np.asarray(block[:, lower], dtype=np.float64)
                + weight * np.asarray(block[:, upper], dtype=np.float64))

    def grade_classes(self) -> list[str]:
        """
        Which magnet grade classes have chemistry arrays on disk.

        ⚠️ READ FROM THE FILES, NOT FROM A PARAMETER. The grade scenarios are
        the TRACTION project's setting -- `run.magnet_grade_scenarios` -- and
        this model's params have no `run` namespace at all. Asking for one here
        returned nothing and silently left the consumer with a single scenario,
        which is how UH and EH vanished from a figure that had been drawing
        them. The arrays are named `<motor>__<CLASS>__magnet_element_fractions`,
        so the files are the authority and cannot go out of step with them.
        """
        found = set()
        for path in self.directory.glob("*__magnet_element_fractions.npy"):
            parts = path.name.split("__")
            if len(parts) == 3:          # motor, CLASS, fractions
                found.add(parts[1])
        return sorted(found)

    def chemistry(self, motor: str, grade: str) -> tuple[list[str], np.ndarray]:
        """The magnet's element fractions for one motor and grade class."""
        stem = f"{motor}__{grade}__magnet_element"
        names = (self.directory / f"{stem}s.txt").read_text().split()
        return names, self.array(f"{stem}_fractions.npy")

    # -- what belongs to what ------------------------------------------------

    def files_for(self, material: str, motor: str | None = None) -> list[str]:
        """Every array holding that material, optionally for one motor type."""
        rows = self.manifest[self.manifest["materialClass"] == material]
        if motor is not None:
            rows = rows[rows["componentKeyLevel1"] == motor]
        return list(rows["file"])

    def motor_of(self, name: str) -> str:
        row = self.manifest[self.manifest["file"] == name]
        return str(row["componentKeyLevel1"].iloc[0])

    def scale(self, name: str, year: int, voltage: int) -> float:
        """
        The deterministic year x voltage factor for one array.

        ⚠️ PER FILE, NOT PER MATERIAL, and that is the whole reason the files are
        not collapsed before this point. Voltage touches the copper of a STATOR
        and not the copper of an EESM rotor winding, so a material-level scale
        would apply the 800 V saving to copper that never sees it.
        """
        rows = self.scales
        match = rows[(rows["file"] == name)
                     & (rows["productionYear"] == year)
                     & (rows["voltageClass"] == voltage)]
        if match.empty:
            raise KeyError(f"no scale for {name} at {year}, {voltage} V")
        return float(match["scale"].iloc[0])


def percentiles(draws: np.ndarray) -> tuple[float, float, float]:
    """Mean, p025 and p975 of one draw vector -- percentiles OF a distribution."""
    return (float(np.mean(draws)),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


# ======================================================================
# THE DETERMINISTIC SIDE
# ======================================================================


def coefficients(tracker_keyed: dict, segment_torque: dict, params,
                 library: DrawLibrary, material: str,
                 drive_trains: tuple[str, ...]) -> tuple[pd.DataFrame, list]:
    """
    The scalar in front of every draw vector, as a matrix.

    Returns `(frame, columns)`: `frame` has one row per (region, flow, scrap
    year) and one column per (file, segment); `columns` names them in order.

    ⚠️ EVERYTHING DETERMINISTIC LIVES HERE AND NOTHING ELSE DOES. The vehicle
    count, the motor-type share, the voltage share and the year factor are all
    scalars; the only random thing in the product is the draw vector this
    multiplies. Keeping them apart is what turns millions of vector operations
    into one matrix product.
    """
    from src import traction

    files = library.files_for(material)
    if not files:
        return pd.DataFrame(), []
    segments = sorted(segment_torque)
    columns = [(name, segment) for name in files for segment in segments]
    position = {key: index for index, key in enumerate(columns)}

    rows: dict[tuple, np.ndarray] = {}
    motor_of = {name: library.motor_of(name) for name in files}
    voltages = traction.VOLTAGES

    # Shares and scales are asked for once per (group, cohort) and per
    # (file, cohort, voltage); both are pure functions of their arguments.
    share_cache: dict[tuple, dict] = {}
    scale_cache: dict[tuple, float] = {}

    for (region, drive), frame in tracker_keyed.items():
        if drive not in drive_trains:
            continue
        block = frame[["flow", "scrap_year", "cohort_year", "Segment",
                       "amount"]].copy()
        block["Segment"] = block["Segment"].astype(str).str.strip().str.upper()
        block = block[block["Segment"].isin(segments)]
        if block.empty:
            continue
        block = block.groupby(["flow", "scrap_year", "cohort_year", "Segment"],
                              as_index=False)["amount"].sum()

        for flow, scrap_year, cohort, segment, amount in block.itertuples(
                index=False):
            if amount == 0:
                continue
            group = traction.segment_group(segment, params)
            key = (group, int(cohort))
            if key not in share_cache:
                share_cache[key] = traction.joint_shares(params, group,
                                                         int(cohort))
            joint = share_cache[key]
            row_key = (region, flow, int(scrap_year))
            row = rows.setdefault(row_key, np.zeros(len(columns)))
            vehicles = float(amount) * 1e6
            for name in files:
                motor = motor_of[name]
                slot = position[(name, segment)]
                for voltage in voltages:
                    share = joint.get((motor, voltage), 0.0)
                    if share <= 0.0:
                        continue
                    scale_key = (name, int(cohort), voltage)
                    if scale_key not in scale_cache:
                        scale_cache[scale_key] = library.scale(
                            name, int(cohort), voltage)
                    row[slot] += vehicles * share * scale_cache[scale_key]

    if not rows:
        return pd.DataFrame(), columns
    index = pd.MultiIndex.from_tuples(sorted(rows),
                                      names=["Region", "flow", "scrap_year"])
    matrix = np.vstack([rows[key] for key in sorted(rows)])
    return pd.DataFrame(matrix, index=index), columns


def draw_matrix(library: DrawLibrary, columns: list, segment_torque: dict,
                element: str | None = None, grade: str | None = None,
                ) -> np.ndarray:
    """
    The draw vector for every (file, segment) column: (columns, draws) float32.

    With `element` and `grade`, each column is additionally multiplied by that
    motor's chemistry draws for that element -- draw i of the magnet mass times
    draw i of the chemistry, which is the whole point of holding both.
    """
    stack = []
    for name, segment in columns:
        vector = library.at_torque(name, segment_torque[segment])
        if element is not None:
            names, fractions = library.chemistry(library.motor_of(name), grade)
            vector = vector * np.asarray(fractions[:, names.index(element)],
                                         dtype=np.float64)
        stack.append(vector.astype(np.float32))
    return np.vstack(stack)


def flow_distribution(frame: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    """
    `C @ V`, then the percentiles of each row's 200,000 draws.

    ⚠️ THE PERCENTILE IS TAKEN OF THE SUM, NOT SUMMED FROM PERCENTILES. That is
    the entire difference between this and what it replaces: the sum over
    segments, motor types, voltages and cohort years happens draw by draw, and
    only the finished distribution is reduced.
    """
    if frame.empty:
        return pd.DataFrame()
    product = frame.to_numpy(dtype=np.float32) @ draws
    out = pd.DataFrame({
        "mass": product.mean(axis=1),
        "mass_low": np.percentile(product, 2.5, axis=1),
        "mass_high": np.percentile(product, 97.5, axis=1),
    }, index=frame.index).reset_index()
    return out
