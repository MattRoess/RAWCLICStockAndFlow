# Documentation

Start here.

| I want to… | read |
|---|---|
| set it up on a new machine | **[SETUP.md](SETUP.md)** — Python, the environment, and the inputs a clone does not have |
| run the model | **[RUNNING.md](RUNNING.md)** |
| change a parameter | **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** — generated from the code, and it states its own count |
| understand how the model works | **[MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md)** |
| read the battery results | **[BATTERY_MATERIAL_FLOWS.md](BATTERY_MATERIAL_FLOWS.md)** — stage 04_04: three chemistry scenarios, what the figures do and do not say |
| know whether a band can be trusted | **[UNCERTAINTY_MAP.md](UNCERTAINTY_MAP.md)** |
| pick up where the last session stopped | **[HANDOVER.md](HANDOVER.md)** |

Design records, for decisions that need their reasoning kept:

- [DESIGN_hev_carve_from_liquids.md](DESIGN_hev_carve_from_liquids.md)
  — why HEV is carved out of Liquids rather than split off Hybrid (REMIND has no
  non-plug-in hybrid category), with the ACEA/EEA validation.
- [DESIGN_inflow_parent_split.md](DESIGN_inflow_parent_split.md)
  — how stage 02's inflow uncertainty is carried into 03_02, the factor-of-two
  defect it replaced, and the validation against ACEA/EEA registrations.
  **Protected: run `code/test_stage03_inflow.py` after any change.**
- [DESIGN_collected_flow_definition.md](DESIGN_collected_flow_definition.md)
  — what "collected" means, the four implementations that disagreed, and the
  invariant now asserted so the same class of defect fails loudly.
- [DESIGN_chemistries_without_composition.md](DESIGN_chemistries_without_composition.md)
  — solid-state batteries have no described cell: how big that hole is, why it is not
  filled, and what would close it. Rewritten 2026-10-05, when sodium-ion stopped being
  part of it.
- [DESIGN_bev_capacity_for_04_04.md](DESIGN_bev_capacity_for_04_04.md)
  — how many kWh a BEV of a given segment carries in a given year, and how that enters
  the Monte Carlo. Settled 2026-09-14; implemented in `src/battery_capacity.py`.
- [DESIGN_discrete_vehicle_states.md](DESIGN_discrete_vehicle_states.md)
  — a car has one voltage and one pack size, never a blend: draw the state per
  iteration and hold it across years.
- [DESIGN_element_resolution.md](DESIGN_element_resolution.md)
  — how 04_02 resolves elements across the four domains, and the two defects that
  had to be fixed to make it correct (motor denominator, sensor mode-vs-mean).
- [DESIGN_inflow_uncertainty_propagation.md](DESIGN_inflow_uncertainty_propagation.md)
  — why stage 02's inflow uncertainty is carried the way it is, including two
  rejected designs and the measurements that ruled them out.

## Which document covers which stage

| stage | described in |
|---|---|
| 01, 02, 03_01, 03_02 | [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md), and [UNCERTAINTY_MAP.md](UNCERTAINTY_MAP.md) for where the uncertainty goes |
| 04_01 | the header of `code/04_01_carcomposition.py`, and [UNCERTAINTY_MAP.md](UNCERTAINTY_MAP.md) |
| 04_02 | [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md) §7.4 and [DESIGN_element_resolution.md](DESIGN_element_resolution.md) |
| 04_03 | the headers of `code/04_03_tractionmotors.py` and `src/traction*.py` |
| 04_04 | [BATTERY_MATERIAL_FLOWS.md](BATTERY_MATERIAL_FLOWS.md) and the three design records above |

04_01 and 04_03 have no description of their own yet beyond their script headers, and
[MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md) was last revised on 2026-08-14, before
04_03 and 04_04 existed. [HANDOVER.md](HANDOVER.md) is the current state of every stage.

[superseded/](superseded/) holds the original Word documents, kept for reference and
no longer maintained.

---

## How these documents are kept honest

**`PARAMETER_REFERENCE.md` is generated, never hand-written.**

```bash
.venv/bin/python code/generate_parameter_reference.py
```

The previous hand-written reference described 46 of 129 parameters and omitted two
entire pipeline stages, and nothing flagged it. Generating it from
`src/params_schema.py` means the document cannot drift from the code: the comments
in that file are the reference, and the script only presents them. It also reports
how many parameters lack an explanation, so a gap is visible rather than silent.
That count is currently **zero**.

**Everything else is written by hand and carries measurements, not assertions.**
Where a document states a number — a correlation, an error, a memory figure — that
number was measured on the real data and the measurement is described alongside it.

**When something is wrong, it says so.** `UNCERTAINTY_MAP.md` existed for a day
describing a defect the pipeline had, in detail, before it was fixed. Documentation
that only records successes is not much use for finding the next problem.
