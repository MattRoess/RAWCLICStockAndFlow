# Documentation

Start here.

| I want to… | read |
|---|---|
| run the model | **[RUNNING.md](RUNNING.md)** |
| change a parameter | **[PARAMETER_REFERENCE.md](PARAMETER_REFERENCE.md)** — all 140, generated from the code |
| understand how the model works | **[MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md)** |
| know whether a band can be trusted | **[UNCERTAINTY_MAP.md](UNCERTAINTY_MAP.md)** |
| pick up where the last session stopped | **[HANDOVER.md](HANDOVER.md)** |

Design records, for decisions that need their reasoning kept:

- [DESIGN_collected_flow_definition.md](DESIGN_collected_flow_definition.md)
  — what "collected" means, the four implementations that disagreed, and the
  invariant now asserted so the same class of defect fails loudly.
- [DESIGN_element_resolution.md](DESIGN_element_resolution.md)
  — how 04_02 resolves elements across the four domains, and the two defects that
  had to be fixed to make it correct (motor denominator, sensor mode-vs-mean).
- [DESIGN_inflow_uncertainty_propagation.md](DESIGN_inflow_uncertainty_propagation.md)
  — why stage 02's inflow uncertainty is carried the way it is, including two
  rejected designs and the measurements that ruled them out.

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
