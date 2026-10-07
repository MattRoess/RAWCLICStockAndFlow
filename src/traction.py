"""
src/traction.py
===============

**Copyright notice:** Copyright © 2026 Empa, Matthias Roesslein

How the fleet splits across traction motor types and voltage classes.

    from src.traction import type_shares, voltage_shares
    type_shares(params, "EF", 2070)     # -> {'PMElectricMotors': 0.25, ...}
    voltage_shares(params, "EF", 2070)  # -> {400: 0.0, 800: 0.67, 1000: 0.33}

The traction project reports a composition for every combination of motor type,
voltage class, torque and year, and takes no view on how many cars are of each.
That view lives here, because it is a fleet question and this is the fleet
model.

⚠️ A SHARE IS A PROBABILITY, NOT A FRACTION OF EVERY CAR. A vehicle is one motor
type at one voltage, for life. This is the rule `battery_voltage.py` already
states for the pack -- "a vehicle is one architecture, never a blend" -- and it
matters because the composition of a PM car is not the average of a PM car and
an induction car.

⚠️ AND THE CATEGORIES ARE DRIVE CONFIGURATIONS, NOT MACHINES. In the source data
`IMandPMElectricMotors` is a two-motor car with one permanent-magnet and one
induction machine; `PMElectricMotors` is everything permanent-magnet, one motor
or two. So a twin-PM car is a PM car with more torque, and needs no category of
its own -- the composition is a function of torque, and two machines at half the
torque carry about the same material as one at full torque. That is the
traction project's own measured finding, not an approximation made here.
"""
from __future__ import annotations

# The five this project models, spelled as the composition file spells them.
PM = 'PMElectricMotors'
EESM = 'EESMElectricMotors'
IM_PM = 'IMandPMElectricMotors'
AXIAL = 'axialFluxPMElectricMotors'
DUAL_ROTOR = 'dualRotorRadialPMElectricMotors'

MOTOR_TYPES = (PM, EESM, IM_PM, AXIAL, DUAL_ROTOR)

# The report's categories, and where each one goes. 'SynRM' is resolved from
# `materials.traction_synrm_goes_to` rather than fixed here, because it is a
# decision and not a translation.
REPORT_TO_MODEL = {'PMSM': PM, 'EESM': EESM, 'axial': AXIAL}

VOLTAGES = (400, 800, 1000)


def segment_group(segment: str, params) -> str:
    """Which of AB / CD / EF a segment belongs to."""
    groups = params.materials.pack_voltage_segment_groups
    key = str(segment).strip().upper()
    if key not in groups:
        raise KeyError(f'segment {segment!r} has no group in '
                       f'pack_voltage_segment_groups')
    return groups[key]


def _at_year(table: dict, year: int):
    """
    A horizon table read at any year, linearly between the horizons it states.

    The report gives 2025, 2030, 2040, 2050, 2060 and 2070. The fleet model asks
    for every year from 2010, so the years between are interpolated and the ends
    are held flat -- extrapolating a share trajectory past its last horizon
    invents adoption nobody argued for.
    """
    years = sorted(table)
    if year <= years[0]:
        return table[years[0]]
    if year >= years[-1]:
        return table[years[-1]]
    upper = next(y for y in years if y >= year)
    lower = max(y for y in years if y <= year)
    if upper == lower:
        return table[lower]
    weight = (year - lower) / (upper - lower)
    low, high = table[lower], table[upper]
    if isinstance(low, dict):
        return {k: low[k] + weight * (high[k] - low[k]) for k in low}
    return tuple(a + weight * (b - a) for a, b in zip(low, high))


def type_shares(params, group: str, year: int) -> dict[str, float]:
    """
    The probability a car of this group and year has each motor type.

    THE ORDER OF OPERATIONS, and each step is a decision recorded in
    `params_schema`:

    1. SynRM/PMa is not modelled, so its share goes to `traction_synrm_goes_to`
       (PMSM). That keeps small cars magnet-heavy where the report had them
       going magnet-light, which overstates rare earth rather than understating
       it.
    2. The report's ASM share is NOT used for `IMandPMElectricMotors`. The two
       count different populations: the report's ASM is a car whose dominant
       machine is induction, while the source category is the two-motor
       PM-plus-induction configuration. Using the report's 4-9% would have put
       almost every heavy AWD car in the PM category by accident rather than by
       argument.
    3. Instead `IMandPM` is the measured AWD share less the part that carries
       two permanent-magnet machines:

           IM+PM = awd_share x (1 - twin_pm_share_of_awd)

       and the twin-PM part stays in `PMElectricMotors`, where a twin-PM car
       belongs. In EF that is the largest single lever on magnet demand in this
       model: the two categories differ by a constant 1.20 kg of magnet.
    4. What is left is split across PM, EESM and axial on the report's own
       proportions with ASM removed, so the report's trajectory is preserved in
       shape even though its ASM row is not used as a level.
    5. The dual rotor takes `traction_dual_rotor_share`, zero by default,
       because no source gives it one.
    """
    materials = params.materials
    report = _at_year(materials.traction_type_shares[group], int(year))

    # 1. SynRM into whichever category was chosen for it.
    moved = dict(report)
    synrm = moved.pop('SynRM', 0.0)
    destination = materials.traction_synrm_goes_to
    moved[destination] = moved.get(destination, 0.0) + synrm

    # 3. The two-motor induction configuration, from the fleet not the report.
    awd = materials.traction_awd_share[group]
    twin_pm = materials.traction_twin_pm_share_of_awd[group]
    im_pm = awd * (1.0 - twin_pm)

    # 5. The dual rotor, out of the same total.
    dual = float(materials.traction_dual_rotor_share)

    # 4. The rest, in the report's proportions, with ASM dropped.
    rest = {REPORT_TO_MODEL[name]: value for name, value in moved.items()
            if name in REPORT_TO_MODEL}
    total = sum(rest.values())
    remaining = max(0.0, 1.0 - im_pm - dual)
    shares = {motor: (remaining * value / total if total else 0.0)
              for motor, value in rest.items()}
    shares[IM_PM] = im_pm
    shares[DUAL_ROTOR] = dual
    return {motor: shares.get(motor, 0.0) for motor in MOTOR_TYPES}


def voltage_shares(params, group: str, year: int) -> dict[int, float]:
    """
    The probability a car of this group and year is 400, 800 or 1000 V.

    ⚠️ 1000 V COMES OUT OF THE 800 V POPULATION, NOT ON TOP OF IT. Both source
    tables are shares of all cars, and by 2050 the 800 V share reaches 1.0 in CD
    and EF -- adding 1000 V to that would put more than every car above 400 V.
    A car takes 1000 V first, then 800 V from what is left of the 800 V share,
    then 400 V.
    """
    materials = params.materials
    high = _at_year(materials.pack_voltage_1000v_share[group], int(year))[1]
    eight = _at_year(materials.pack_voltage_800v_share[group], int(year))[1]
    high = min(high, eight)
    return {1000: high, 800: eight - high, 400: max(0.0, 1.0 - eight)}


def joint_shares(params, group: str, year: int) -> dict[tuple[str, int], float]:
    """
    Motor type and voltage class together, as one distribution over 15 states.

    ⚠️ TREATED AS INDEPENDENT, AND THAT IS AN ASSUMPTION. Nothing says an
    800 V car is more or less likely to be axial flux, so the two are multiplied
    -- but they plausibly correlate, since both follow the premium end of the
    market. If evidence for the correlation ever arrives it belongs here.
    """
    types = type_shares(params, group, year)
    volts = voltage_shares(params, group, year)
    return {(motor, volt): type_share * volt_share
            for motor, type_share in types.items()
            for volt, volt_share in volts.items()}
