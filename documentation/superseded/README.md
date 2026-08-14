# Superseded documents

These are the original Word documents, dated **13 July 2026**. They are kept for
reference only and are **out of date**.

| file | replaced by |
|---|---|
| `RAWCLIC_StockFlow_Model_Description.docx` | `../MODEL_DESCRIPTION.md` |
| `RAWCLIC_StockFlow_Parameter_Reference.docx` | `../PARAMETER_REFERENCE.md` |

## Why they were replaced

**They had gone stale without anyone noticing.** By August 2026 they predated the
correlated drivetrain mix, the stage 03_01 Monte Carlo fix, the reorganisation of
`params_schema.py`, stage 04_02, and the inflow-uncertainty propagation.

The parameter reference described **46 of the 129** parameters that existed at the
time. The 83 it omitted included every parameter of stage 01 and stage 04.

**Markdown, and generated where possible.** A `.docx` cannot be diffed, cannot be
reviewed in a pull request, and gives no signal when the code moves underneath it.
`PARAMETER_REFERENCE.md` is now generated directly from `src/params_schema.py`, so
it cannot drift again: change a parameter, regenerate, and the document is correct
by construction.

The content of both originals was carried over, not discarded — the model
description keeps all its original sections and adds section 7 for everything since.
