"""
test_battery_seeds.py -- the random draws of 04_04 are the same in every run, and no two of them are one stream.

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

    .venv/bin/python code/test_battery_seeds.py

RUN THIS AFTER ANY CHANGE to `src/battery_capacity.py`, `src/battery_voltage.py`,
`src/battery_composition.py` or `src/battery_chemistry.py`, or to how 04_04 calls them.
It needs only the code -- no data, no pipeline -- and takes seconds. Any other way of
asking its question costs a six-hour run of 04_04: will the next run draw the world the
last one did, and are independent uncertainties independent?

WHY THIS EXISTS
---------------
Two faults, neither of which ever failed, both found by comparing two runs of 04_04
(2026-10-08 and 2026-10-09).

1. REPRODUCIBILITY. `battery_capacity.py` and `battery_voltage.py` seeded each segment's
   stream with `abs(hash(segment))`. Python salts the hash of a `str` per process, so
   every run drew a different pack size and voltage -- `[33.7, 35.8, 34.1, ...]` kWh in
   one process and `[33.7, 29.9, 34.1, ...]` in the next, with the same seed -- and an
   export could not be reproduced. The sweep of 2026-09-28 that fixed the one `hash()` in
   04_02 had not found these two.

2. INDEPENDENCE. Streams seeded alike are ONE stream, and four pairs of them were. The
   pack size and the voltage of a segment were both `[seed, segment]`, so the uniform that
   picked a pack size also placed the car in the 800 V adoption order: a small pack was
   almost always 800 V and a large one almost never (segment A, 2050: 100 %, 63 %, 0.6 %
   for 25, 30 and 35 kWh), which put up to +0.5 % on the mean cable copper and +1.0 % on
   the terminal copper per pack. And the capacity growth, the voltage band and the
   composition's extrapolation factor were all `default_rng(404)`: the growth rate and the
   band had rank correlation +1.0000, so the world in which capacity grows fastest was,
   exactly, the one in which 800 V arrives earliest.

What keeps both fixed, in the order it is asked below:

  1  every stream is seeded with a tag -- `[seed, crc32(<its name>), ...]` -- and never
     with the bare seed;
  2  each purpose has one tag and no two purposes share it; what is per segment, per
     chemistry or market-wide is the way the design says;
  3  replayed, the uniforms of every pair of streams are uncorrelated;
  4  pack size and voltage are independent through the real functions;
  5  the draws are byte-identical in three processes with different hash salts.

It records the integers numpy is ASKED for rather than reading the source, so a stream added
later with a bare seed, or with a tag that is already taken, fails here and not in a run.
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.battery_capacity import capacity_draws, growth_draws, level_draws  # noqa: E402
from src.battery_chemistry import chemistry_share_draws  # noqa: E402
from src.battery_composition import CompositionAtCapacity  # noqa: E402
from src.battery_voltage import voltage_draws  # noqa: E402
from src.params_schema import Params  # noqa: E402

SEED = 404              # what 04_04 passes to every one of them
DRAWS = 200_000         # what a real run has; the correlations below mean nothing much narrower
EXTRAPOLATION = 10.0    # the extrapolation factor draws only when the parameter is above zero

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"\n          {detail}"
                                                     if not ok and detail else ""))


def asked_for(call) -> list[tuple[int, ...]]:
    """The integers numpy is given for every generator `call()` makes, in the order it makes them."""
    real = np.random.default_rng
    seen: list[tuple[int, ...]] = []

    def recording(seed=None):
        entropy = getattr(seed, "entropy", seed)          # a SeedSequence keeps its list here
        seen.append(tuple(int(x) for x in np.atleast_1d(entropy)) if entropy is not None else ())
        return real(seed)

    np.random.default_rng = recording
    try:
        call()
    finally:
        np.random.default_rng = real
    return seen


def rank(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values))


def the_streams(params: Params) -> dict[str, list[tuple[int, ...]]]:
    """Every purpose a battery module draws for, with the seed list of each generator it makes."""
    materials = params.materials
    segments = sorted(materials.battery_capacity_levels)[:2]
    scenario = next(iter(materials.battery_chemistry_scenarios))
    group = sorted(set(materials.battery_chemistry_segment_groups.values()))[0]
    stub = types.SimpleNamespace(params=params)           # `_extrapolation_factor` reads only `params`

    streams: dict[str, list[tuple[int, ...]]] = {}
    streams["growth"] = asked_for(lambda: growth_draws(params, 10, seed=SEED))
    streams["level"] = [s for seg in segments
                        for s in asked_for(lambda seg=seg: level_draws(params, seg, 10, seed=SEED))]
    made = [asked_for(lambda seg=seg: voltage_draws(params, seg, [2050], n_draws=10, seed=SEED))
            for seg in segments]
    assert all(len(m) == 2 for m in made), f"voltage_draws should make two generators: {made}"
    streams["adoption"] = [m[0] for m in made]            # the car's place in the adoption order
    streams["band"] = [m[1] for m in made]                # where the world sits in the share's band
    streams["extrapolation"] = asked_for(lambda: CompositionAtCapacity._extrapolation_factor(
        stub, np.full(10, 150.0), 100.0, SEED))
    streams["share"] = asked_for(lambda: chemistry_share_draws(
        params, scenario, group, list(range(2004, 2071)), n_draws=10, seed=SEED))
    return streams


# What a fresh process computes, as one line of digest per draw. Printed, not returned, so that
# the processes can be compared as text.
PROGRAM = '''
import dataclasses, hashlib, sys, types, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, {root!r})
import numpy as np
from src.params_schema import Params
from src.battery_capacity import capacity_draws
from src.battery_voltage import voltage_draws
from src.battery_chemistry import chemistry_share_draws
from src.battery_composition import CompositionAtCapacity

p = Params()
p = dataclasses.replace(p, materials=dataclasses.replace(
    p.materials, battery_extrapolation_uncertainty_per_100kwh={extrapolation!r}))
m = p.materials
digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
years = list(range(2004, 2071))
for s in sorted(m.battery_capacity_levels):
    print("capacity", s, digest(capacity_draws(p, s, years, n_draws=5000, seed={seed})))
for s in sorted(m.pack_voltage_segment_groups):
    print("voltage", s, digest(voltage_draws(p, s, years, n_draws=5000, seed={seed})))
for scenario in m.battery_chemistry_scenarios:
    for group in sorted(set(m.battery_chemistry_segment_groups.values())):
        shares = chemistry_share_draws(p, scenario, group, years, n_draws=2000, seed={seed})
        print("shares", scenario, group, digest(np.stack([shares[c] for c in sorted(shares)])))
stub = types.SimpleNamespace(params=p)
print("extrapolation", digest(CompositionAtCapacity._extrapolation_factor(stub, np.full(5000, 150.0), 100.0, {seed})))
'''


def in_a_fresh_process(hash_seed: str | None) -> tuple[int, str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    if hash_seed is not None:
        env["PYTHONHASHSEED"] = hash_seed                 # None: leave it to Python, which salts at random
    done = subprocess.run(
        [sys.executable, "-W", "ignore", "-c",
         PROGRAM.format(root=str(ROOT), seed=SEED, extrapolation=EXTRAPOLATION)],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=900)
    return done.returncode, done.stdout.strip(), done.stderr.strip()


def main() -> int:
    params = Params()
    params = dataclasses.replace(params, materials=dataclasses.replace(
        params.materials, battery_extrapolation_uncertainty_per_100kwh=EXTRAPOLATION))
    segments = sorted(params.materials.battery_capacity_levels)[:2]
    streams = the_streams(params)

    # 1. A tag, never the bare seed.
    bare = {purpose: seeds for purpose, seeds in streams.items() if any(len(s) < 2 for s in seeds)}
    check("every stream is seeded with a tag, never the bare seed", not bare,
          f"seeded with the bare seed, so the same stream as anything else that is: {sorted(bare)}")

    # 2. One tag per purpose, no two purposes sharing it -- and per segment, per chemistry,
    #    or market-wide, the way the design says.
    tags = {purpose: {s[1] for s in seeds if len(s) > 1} for purpose, seeds in streams.items()}
    check("each purpose has one tag, and no two purposes share it",
          all(len(t) == 1 for t in tags.values())
          and len({next(iter(t)) for t in tags.values() if t}) == len(tags),
          f"tags: {tags}")
    check("the pack size and the adoption order are per segment",
          len(set(streams["level"])) == len(streams["level"])
          and len(set(streams["adoption"])) == len(streams["adoption"]),
          f"level {streams['level']}, adoption {streams['adoption']}")
    check("the voltage band is market-wide: the same stream for every segment",
          len(set(streams["band"])) == 1, f"{streams['band']}")
    made = [asked_for(lambda seg=seg: capacity_draws(params, seg, [2050], n_draws=10, seed=SEED))
            for seg in segments]
    check("the capacity growth is shared by every segment, and the pack size is not",
          len(made[0]) == 2 and made[0][1] == made[1][1] and made[0][0] != made[1][0],
          f"growth {made[0][1]} vs {made[1][1]}; level {made[0][0]} vs {made[1][0]}")
    check("each chemistry's share multiplier is its own stream",
          len(streams["share"]) >= 2 and len(set(streams["share"])) == len(streams["share"]),
          f"{streams['share']}")

    # 3. Replayed, the first DRAWS uniforms of every pair of purposes are uncorrelated.
    #    Independent streams this wide scatter by about 0.002, so 0.01 is four and a half
    #    standard deviations -- and the seeds are fixed, so this never flakes either way.
    ranks = {purpose: rank(np.random.default_rng(list(seeds[0])).random(DRAWS))
             for purpose, seeds in streams.items()}
    worst, pair = 0.0, ("", "")
    names = list(ranks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = abs(float(np.corrcoef(ranks[a], ranks[b])[0, 1]))
            if rho > worst:
                worst, pair = rho, (a, b)
    check("replayed, the uniforms of every pair of streams are uncorrelated", worst < 0.01,
          f"worst pair {pair[0]} and {pair[1]}: |rank correlation| {worst:.4f}")

    # 4. Through the real functions, where the fault was seen: a small pack was 800 V.
    segment = segments[0]
    levels = level_draws(params, segment, DRAWS, seed=SEED)
    volts = voltage_draws(params, segment, [2050], n_draws=DRAWS, seed=SEED)[:, 0]
    high = float((volts == 800).mean())
    noise = np.random.default_rng(0)                      # breaks the ties of two discrete variables
    rho = float(np.corrcoef(rank(levels + noise.random(DRAWS) * 1e-9),
                            rank((volts == 800) + noise.random(DRAWS) * 1e-9))[0, 1])
    check(f"segment {segment}'s pack size and its 800 V are independent",
          abs(rho) < 0.01 and 0.05 < high < 0.95,
          f"rank correlation {rho:+.3f} (it was -0.563); the 800 V share is {high:.2f}, "
          f"which has to be neither 0 nor 1 for this to prove anything")

    # 5. The same draws, byte for byte, in three processes: two with different hash salts
    #    pinned, one left to Python's own random one.
    outs = [in_a_fresh_process(h) for h in ("1", "2", None)]
    failed = [o for o in outs if o[0] != 0]
    lines = [o[1] for o in outs]
    check("the draws are identical in three processes with different hash salts",
          not failed and len(set(lines)) == 1 and len(lines[0].splitlines()) >= 30,
          (f"a process failed: {failed[0][2][-400:]}" if failed else
           f"{len(lines[0].splitlines())} digests; differing lines: "
           + "; ".join(sorted(set(a for l in lines[1:] for a in l.splitlines()) ^ set(lines[0].splitlines()))[:4])))

    failed = [name for ok, name, _ in _results if not ok]
    print(f"\n  {len(_results) - len(failed)}/{len(_results)} passed")
    if failed:
        print("\n  Do not commit a change that leaves these failing: each one costs a six-hour\n"
              "  run of 04_04 to discover any other way.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
