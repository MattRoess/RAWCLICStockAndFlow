"""
generate_parameter_reference.py -- build documentation/PARAMETER_REFERENCE.md from
src/params_schema.py.

WHY THIS IS GENERATED AND NOT WRITTEN BY HAND. A hand-written parameter reference
drifts the moment a parameter is added, and drifts silently, because nothing checks
it. The previous one did exactly that: it described 46 of the 129 parameters that
existed, and the 83 it omitted included two entire pipeline stages. Every comment in
`params_schema.py` is written for a reader rather than a compiler, so the code is
already the reference -- this script simply presents it.

Run it after changing any parameter:

    .venv/bin/python code/generate_parameter_reference.py

The conceptual material that does NOT belong next to a single parameter -- how the
uncertainty axes trade off against each other, why a default is what it is -- lives
in MODEL_DESCRIPTION.md and UNCERTAINTY_MAP.md, which are written by hand and
reviewed. This file never invents explanation; it only relays what the code says.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir():
            return candidate
    return start


ROOT = _find_project_root(Path(__file__).resolve().parent)
SCHEMA = ROOT / "src" / "params_schema.py"
OUT = ROOT / "documentation" / "PARAMETER_REFERENCE.md"

# Which stage reads each section, and what the section is for. The only thing in
# this script that is not taken from the code, because the code cannot know it.
SECTIONS = {
    "DataPrepParams": ("Stage 01 — Data preparation", "code/01_data_prep.py"),
    "StockFlowParams": ("Stage 02 — Stock-driven flows", "code/02_stockdriven.py"),
    "DisaggregationParams": ("Stage 03_01 — Disaggregation", "code/03_01_flowdriven.py"),
    "AdjustedFlowsParams": ("Stage 03_02 — Adjusted flows / scenarios", "code/03_02_adjustedflows.py"),
    "MaterialsParams": ("Stage 04 — Materials, car composition, BEV electronics",
                        "code/04_01_carcomposition.py, code/04_02_BEVelectronics.py"),
    "MonteCarloParams": ("Monte Carlo — cross-cutting", "every stage that runs an uncertainty pass"),
    "AsymmetricSpread": ("Shared building block", "used by several stages"),
    "WeibullLifetime": ("Shared building block", "used by several stages"),
    "LifetimeOverride": ("Shared building block", "used by several stages"),
    "OpenEndedLifetimeChange": ("Shared building block", "used by stage 03_02 scenarios"),
    "ScenarioSpec": ("Shared building block", "one stage 03_02 scenario"),
    "Params": ("The container", "holds every section above"),
}
ORDER = list(SECTIONS)


def comment_block(lines: list[str], start_line: int) -> tuple[list[str], bool]:
    """
    The run of `#` comments immediately above a parameter.

    Related parameters are deliberately written as a group under one comment --
    `start_year` and `end_year`, `lower` and `upper`, `shape_k` and `scale_lambda`.
    The second of such a pair has no comment of its own but is not undocumented, so
    this walks back over any adjacent parameter definitions to find the comment that
    covers the group, and reports that it was shared. Treating those as missing was
    wrong: it would have sent someone off to write comments that already exist.

    Returns (comment lines, shared) where `shared` means the comment sits above an
    earlier parameter in the same group.
    """
    i = start_line - 2                       # 0-based, line above the definition
    shared = False
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            out: list[str] = []
            while i >= 0 and lines[i].strip().startswith("#"):
                out.append(lines[i].strip().lstrip("#").strip())
                i -= 1
            return list(reversed(out)), shared
        if not stripped:
            return [], shared
        # another parameter in the same group: keep walking up
        if ":" in stripped and "=" in stripped:
            shared = True
            i -= 1
            continue
        return [], shared
    return [], shared


def default_of(node: ast.AnnAssign, src: str) -> str:
    if node.value is None:
        return "*(required)*"
    text = ast.get_source_segment(src, node.value) or ""
    text = " ".join(text.split())
    if text.startswith("field(default_factory=lambda:"):
        text = text[len("field(default_factory=lambda:"):].rstrip(")").strip()
    elif text.startswith("field("):
        text = text[len("field("):].rstrip(")").strip()
    if len(text) > 90:
        text = text[:87] + "..."
    return f"`{text}`"


def main() -> Path:
    src = SCHEMA.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}

    total = sum(1 for c in classes.values() for x in c.body if isinstance(x, ast.AnnAssign))

    md: list[str] = []
    md.append("# RAWCLIC Stock-and-Flow — Parameter Reference\n")
    md.append(
        "**Generated from `src/params_schema.py`. Do not edit this file by hand.**\n\n"
        "Regenerate with:\n\n"
        "```bash\n.venv/bin/python code/generate_parameter_reference.py\n```\n\n"
        f"Covers all **{total}** parameters. It is generated precisely because the "
        "previous hand-written reference described 46 of them and silently omitted "
        "two entire pipeline stages.\n"
    )
    md.append(
        "## How to change a parameter\n\n"
        "Everything the model uses lives in `src/params_schema.py`. Nothing is "
        "hardcoded elsewhere in the pipeline.\n\n"
        "1. Edit only what is to the **right** of the `=`. Renaming a parameter breaks the code.\n"
        "2. Keep the **type**: a number stays a number, text stays quoted, `True`/`False` stay capitalised.\n"
        "3. Keep the punctuation inside `{ }` and `( )` — a missing comma is the most common breakage.\n\n"
        "Then run:\n\n"
        "```bash\n.venv/bin/python code/00_parameters.py\n```\n\n"
        "That regenerates the parameter file the stages read **and validates your edit**, "
        "so a mistake surfaces there rather than hours into a Monte Carlo run.\n\n"
        "For how the model actually works, and why defaults are what they are, see "
        "`MODEL_DESCRIPTION.md`. For where uncertainty enters and stops, see "
        "`UNCERTAINTY_MAP.md`.\n"
    )

    # index
    md.append("## Sections\n")
    for name in ORDER:
        if name not in classes:
            continue
        n_params = sum(1 for x in classes[name].body if isinstance(x, ast.AnnAssign))
        title, used_by = SECTIONS[name]
        md.append(f"- [{title}](#{title.lower().replace(' ', '-').replace('—', '').replace('/', '').replace(',', '').replace('--', '-')}) "
                  f"— `{name}`, {n_params} parameters")
    md.append("")

    for name in ORDER:
        node = classes.get(name)
        if node is None:
            continue
        title, used_by = SECTIONS[name]
        params = [x for x in node.body if isinstance(x, ast.AnnAssign)]
        md.append(f"\n---\n\n## {title}\n")
        md.append(f"`{name}` in `src/params_schema.py` — read by {used_by}.\n")

        doc = ast.get_docstring(node)
        if doc:
            md.append("> " + " ".join(doc.split()) + "\n")

        md.append(f"**{len(params)} parameters.**\n")
        md.append("| parameter | default |")
        md.append("|---|---|")
        for x in params:
            md.append(f"| `{x.target.id}` | {default_of(x, src)} |")
        md.append("")

        for x in params:
            notes, shared = comment_block(lines, x.lineno)
            md.append(f"\n### `{x.target.id}`\n")
            md.append(f"Default: {default_of(x, src)}\n")
            if notes and shared:
                md.append("*Documented together with the parameter(s) above it:*\n")
                md.append("\n".join(notes) + "\n")
            elif notes:
                md.append("\n".join(notes) + "\n")
            elif doc:
                # Small shared types (AsymmetricSpread, WeibullLifetime and the
                # like) explain all their fields together in the class docstring
                # rather than one comment per field, because the fields only mean
                # anything as a set. That is documentation, so say where it is
                # instead of reporting the field as undocumented.
                md.append("*Explained in the description of "
                          f"`{name}` at the top of this section.*\n")
            else:
                md.append("*No explanation in the code. Add a comment above it in "
                          "`params_schema.py` and regenerate.*\n")

    OUT.write_text("\n".join(md))
    undocumented = sum(
        1 for cname, c in classes.items() for x in c.body
        if isinstance(x, ast.AnnAssign)
        and not comment_block(lines, x.lineno)[0]
        and not ast.get_docstring(c)
    )
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {total} parameters, {len(ORDER)} sections")
    print(f"  {undocumented} parameters have no explanatory comment in the code")
    return OUT


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
