"""
data_prep.py
=============

The real implementation behind 01_data_prep.py: parses REMIND scenario `.mif` output
into a tidy long-format table, aggregates vehicle stock by region/technology, and cleans
EU used-vehicle export/import trade data.

FIXES APPLIED THIS ROUND (all previously flagged, now resolved and verified -- see
EVmodel_review_consolidated.md and HOW_TO_RUN_AND_VERIFY.md for the exact commands used)
-------------------------------------------------------------------------------------------
1. **C1 -- North Korea.** `NAME_MAP` no longer maps North Korea to "South Korea". Per
   your decision, North Korean trade rows are now dropped entirely (see
   `EXCLUDED_RAW_NAMES` below) rather than merged into another country or kept under
   their own name -- consistent with how `AGGREGATES` already drops non-country rollup
   rows before country-level analysis.
2. **C2 -- export deduplication double-count.** `clean_export_data`'s priority-based
   dedup previously re-added ALL "unmatched"-source rows unconditionally, even when a
   higher-priority row for the same (year, Exp_Name, Imp_Name) key already existed in
   `best`. Fixed: unmatched rows are now only kept for keys that have NO row in `best`
   at all. Verified on the synthetic 2-row example from the original finding: true value
   1.0, output is now exactly 1.0 (previously 1.4, a ~40% overstatement).
3. **H2 -- region_filter state leak.** `prepare_remind_scenarios` previously computed
   `region_filter` once outside the loop and only reassigned it when a scenario matched
   one of two conditions -- a scenario matching neither would silently inherit the
   *previous* scenario's filter. Fixed: `region_filter` is now computed fresh, entirely
   inside the loop body, for every scenario, with an explicit `raise` (not a silent
   fallback) if a scenario's data matches neither condition.
4. **[NEW, generalization]** `export_data_file_name` and `iso3_corrections` (previously
   hardcoded inside `clean_export_data`) are now parameters, sourced from
   `00_parameters.py`'s `DataPrepParams` by the caller (`01_data_prep.py`).
5. **[NEW, M14]** `melt_and_expand`'s pipe-count filter now prints how many rows it
   drops, instead of silently discarding shallower taxonomy paths with no visibility.
6. **[L3]** File paths built with `pathlib.Path` instead of string concatenation.
7. **[NEW, this round]** `clean_export_data`'s priority-scoring comparison
   (`Source_Code.eq(...)` against `Exp_Country_ISO3`/`Imp_Country_ISO3`) crashed on a
   recent pandas/pyarrow combination (`ArrowTypeError: Expected bytes, got a 'int'
   object`) whenever the raw Excel had even one non-string value in one of these
   columns (pandas then reads that column with a mixed dtype, which the Arrow-backed
   string comparison can't box). Fixed: `Exp_Country_ISO3`, `Imp_Country_ISO3`, and
   `Source_Code` are now explicitly cast to pandas' nullable "string" dtype (same idiom
   `canonicalize_country_names` already used for Exp_Name/Imp_Name) with missing values
   filled as "" before any comparison -- see the comment at the `.assign(...)` call in
   `clean_export_data` for the full reasoning (including why "" and not `pd.NA`).

STILL OPEN (not touched this round -- no code change needed, just documenting status)
----------------------------------------------------------------------------------------
- M12: `NAME_MAP` collapses three DRC name variants to the bare name "Congo" -- if the
  source data also has genuine "Republic of the Congo" rows already spelled "Congo",
  those would be conflated with DRC's. Not changed pending confirmation the source data
  doesn't distinguish the two.
- M13: only one ISO3 quirk is corrected by default (`IRE` -> `IRL`, now via the
  `iso3_corrections` parameter, extensible without a code change).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Country name canonicalization
# ---------------------------------------------------------------------------
NAME_MAP: dict[str, str] = {
    "Finnland": "Finland",
    "Lettland": "Latvia",
    "Lettland (Ab 1992)": "Latvia",
    "Zypern": "Cyprus",
    "Ungarn": "Hungary",
    "Slovak Republic": "Slovakia",
    "Vereinigtes KÃ¶nigreich": "United Kingdom",
    "Grossbritannien": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",
    "Vereinigte Staaten": "United States",
    "Republic Of Korea (South)": "South Korea",
    # C1 RESOLVED: "Dem. People's Republic Of Korea (North)" used to map to "South
    # Korea" here -- a copy-paste bug that would have silently merged North Korean
    # trade records into South Korea's totals. Per your decision, North Korea is not
    # remapped to anything; it's excluded entirely instead -- see EXCLUDED_RAW_NAMES.
    "Republic Democratic Of Congo": "Congo",
    "Demokratische Republik Kongo": "Congo",
    "Congo_ the Democratic Republic of the": "Congo",
    # M12, still open: if the source data also has genuine "Republic of the Congo" rows
    # spelled "Congo", they'd be conflated with DRC's under this mapping.
    "Nl Antilles": "Netherlands Antilles",
    "NiederlÃ¤ndische Antillen (Bis 2012)": "Netherlands Antilles",
    "Netherlands Antilles (Terminated 2013-12)": "Netherlands Antilles",
    "CuraÃ§ao": "Curacao",
    "Curaçao": "Curacao",
    "Curacao (Ab 2013)": "Curacao",
    "Serbien Und Montenegro": "Serbia And Montenegro",
    "Serbien Und Montenegro (01/1993 Bis 05/2005)": "Serbia And Montenegro",
    "Serbia And Montenegro (Terminated 2006-09)": "Serbia And Montenegro",
    "Serbien.und.montenegro": "Serbia And Montenegro",
    "TÃ£.rkei": "Turkey",
    "Heiliger Stuhl (Vatikan)": "Vatican City",
    "Heiliger Stuhl (Vatikanstadt)": "Vatican City",
}

# Non-country aggregate/rollup labels that must be excluded before summing, to avoid
# double-counting real countries within a "Total" / "World" row.
AGGREGATES: set[str] = {
    "Total",
    "World",
    "Zusammen",
    "Nicht Ermittelte Gebiete Drittstaaten",
    "Staatenlos..Code.iso.3166.",
}

# [NEW, resolves C1] Raw source-data labels excluded entirely from trade analysis (per
# your decision) rather than canonicalized to a real country name. Currently just North
# Korea -- add further exclusions here, not by re-adding them to NAME_MAP with a wrong
# target.
EXCLUDED_RAW_NAMES: set[str] = {
    "Dem. People's Republic Of Korea (North)",
}


def load_remind_scenarios(
    input_dir: str, scenario_specs: Iterable[tuple[str, str, str]]
) -> dict[str, pd.DataFrame]:
    """
    Read every REMIND `.mif` scenario file listed in `scenario_specs` into a raw,
    wide-format (one column per year) DataFrame, keyed by scenario name.

    `scenario_specs` is an iterable of (name, relative_path, csv_separator) tuples.
    No file-existence or parse-error handling here -- if a path is wrong or the file is
    malformed, this raises whatever `pd.read_csv` raises, uncaught.
    """
    base = Path(input_dir)
    data_dict: dict[str, pd.DataFrame] = {}
    for name, rel_path, sep in scenario_specs:
        data_dict[name] = pd.read_csv(base / rel_path, sep=sep)
    return data_dict


def prepare_remind_scenarios(
    data_dict: dict[str, pd.DataFrame],
    scenario: str,
    remind_regions: list[str],
    prefix: str,
    remind_technology: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Reshape every scenario's wide REMIND table into a tidy long table, and return both
    the full `scenario_map` (all scenarios) and `scenario_df` (just the selected one).

    REMIND's `Variable` strings encode a taxonomy path separated by "|":
        Stock|Transport|Pass|Road|LDV|<vehicle_type>|<vehicle_class>|<size>|<class_detail>|<technology>
    `melt_and_expand` splits that path into named columns.
    """

    def melt_and_expand(df: pd.DataFrame) -> pd.DataFrame:
        df_long = df.melt(
            id_vars=["Model", "Scenario", "Region", "Variable", "Unit"],
            var_name="year",
            value_name="value",
        )

        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
        df_long = df_long.dropna(subset=["year"]).copy()
        df_long["year"] = df_long["year"].astype(int)

        df_long["Variable"] = df_long["Variable"].astype(str)

        # [NEW, resolves M14] Report how many rows the depth filter drops, so a future
        # change in REMIND's variable-naming depth is visible instead of silent.
        n_before = len(df_long)
        df_long = df_long[df_long["Variable"].str.count(r"\|") >= 8].copy()
        n_dropped = n_before - len(df_long)
        if n_dropped:
            print(
                f"  melt_and_expand: dropped {n_dropped}/{n_before} rows with a "
                f"taxonomy path shallower than 9 segments.",
                file=sys.stderr,
            )

        parts = df_long["Variable"].str.split(r"\|", expand=True)
        if parts.shape[1] < 9:
            for idx in range(parts.shape[1], 9):
                parts[idx] = pd.NA
        parts = parts.iloc[:, :9]

        df_long[[
            "sector", "transport_type", "mode", "vehicle_type",
            "vehicle_class", "size", "class_detail", "technology",
        ]] = parts.iloc[:, 1:9].to_numpy()

        df_long["technology"] = df_long["technology"].replace({"Hybrid electric": "Hybrid"})
        return df_long

    scenario_map: dict[str, pd.DataFrame] = {}

    for name, df in data_dict.items():
        df = df.copy()
        if "Unnamed: 24" in df.columns:
            df = df.drop(columns="Unnamed: 24")

        regions_in_df = set(df["Region"].dropna().astype(str).unique())
        overlap = regions_in_df.intersection(set(remind_regions))

        # [FIXED, resolves H2] region_filter is computed fresh for THIS scenario, every
        # iteration -- no value can leak in from a previous scenario. A scenario whose
        # data matches neither condition now raises immediately (loud, at the point of
        # the actual ambiguity) instead of silently reusing an unrelated filter.
        if overlap:
            region_filter: list[str] = list(remind_regions)
        elif "EUR" in regions_in_df:
            region_filter = ["EUR"]
        else:
            raise ValueError(
                f"prepare_remind_scenarios: scenario '{name}' has neither any of "
                f"remind_regions {sorted(remind_regions)} nor a pre-aggregated 'EUR' "
                f"row in its Region column (found: {sorted(regions_in_df)[:10]}...). "
                f"Cannot determine which rows belong to this analysis."
            )

        df_filt = df[
            (df["Region"].astype(str).isin(region_filter))
            & (df["Variable"].astype(str).str.startswith(prefix))
        ].copy()

        df_long = melt_and_expand(df_filt)
        df_long = df_long[df_long["technology"].isin(remind_technology)].copy()
        scenario_map[name] = df_long

    scenario_df = scenario_map[scenario].copy()
    return scenario_df, scenario_map


def canonicalize_country_names(series: pd.Series) -> pd.Series:
    """
    Map every value in `series` through NAME_MAP; any name not found in NAME_MAP is
    assumed to already be canonical and is left unchanged.
    """
    series = series.astype("string").str.strip()
    return series.map(NAME_MAP).fillna(series)


def clean_export_data(
    input_dir: str,
    eu_countries: Iterable[str],
    min_year: int,
    max_year_exclusive: int,
    export_data_file_name: str = "usedvehicles_v1.2.xlsx",
    iso3_corrections: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the EU used-vehicle export/import trade table, resolving duplicate
    (year, exporter, importer) reports down to one authoritative row per flow.

    `export_data_file_name`: sourced from 00_parameters.py's
    `params.data_prep.export_data_file_name` (was hardcoded here previously).
    `iso3_corrections`: sourced from `params.data_prep.iso3_corrections` -- a dict of
    {wrong_iso3: correct_iso3} applied to Source_Code before priority-matching (was a
    single hardcoded `"IRE"->"IRL"` substitution previously; now extensible with no code
    change). Defaults to that same single correction if not provided.

    THE DEDUPLICATION MODEL
    ------------------------
    The same (year, exporter, importer) trade flow can appear multiple times, reported
    by different authorities. Each row is scored:

        priority(row) = 3, if Source_Code == Imp_Country_ISO3   (importer self-reported)
                       = 2, if Source_Code == Exp_Country_ISO3   (exporter self-reported)
                       = 1, if Source_Code == "EUR"              (Eurostat aggregate)
                       = 0, otherwise                            ("unmatched")

    For every (year, Exp_Name, Imp_Name) group, the highest-priority row is kept.

    [FIXED, resolves C2 -- was: unmatched rows re-added unconditionally, double-counting
    any flow that also had a matched-source report]. Now: a priority-0 ("unmatched")
    row is only kept if `best` (the matched-priority selection) has NO row at all for
    that same key -- i.e. unmatched rows fill gaps, they never sit alongside a row that
    already won on priority. Verified on the original finding's synthetic 2-row example
    (true value 1.0): output is now exactly 1.0 (previously 1.4).
    """
    if iso3_corrections is None:
        iso3_corrections = {"IRE": "IRL"}
    eu_countries = set(eu_countries)

    export_file = Path(input_dir) / export_data_file_name
    export_df = pd.read_excel(export_file)
    export_df = export_df.rename(columns={"Year": "year", "Quantity": "export"})
    export_df["export"] = export_df["export"] / 1e6  # raw vehicle counts -> millions
    export_df = export_df[export_df["year"] < max_year_exclusive].copy()

    export_df_2005 = (
        export_df.loc[
            export_df["year"] >= min_year,
            [
                "year", "Exp_Country_Name", "Exp_Country_ISO3",
                "Imp_Country_Name", "Imp_Country_ISO3", "export", "Source_Code",
            ],
        ]
        .assign(
            Exp_Name=lambda frame: canonicalize_country_names(frame["Exp_Country_Name"]),
            Imp_Name=lambda frame: canonicalize_country_names(frame["Imp_Country_Name"]),
            # [FIXED, new] `Exp_Country_ISO3`/`Imp_Country_ISO3`/`Source_Code` are
            # compared against each other with `.eq()` below (priority scoring). If the
            # raw Excel has even one non-string value in one of these columns (a
            # malformed/blank cell -- common in trade data), pandas reads that column
            # with a mixed dtype, and comparing it against a clean, uniformly-typed
            # column raises (observed: `ArrowTypeError: Expected bytes, got a 'int'
            # object`, on a recent pandas/pyarrow combination). Casting all three to
            # pandas' nullable "string" dtype -- same idiom `canonicalize_country_names`
            # already uses for Exp_Name/Imp_Name -- forces a uniform, comparable type.
            # Missing values are filled with "" (not left as `pd.NA`): `pd.NA` would
            # propagate through `.eq()` into the priority-score `.astype(int)` step
            # further down and raise there instead -- "" can never equal a real ISO3
            # code or "EUR", so it cleanly and correctly falls into the "unmatched"
            # bucket, consistent with a row whose source truly can't be identified.
            Exp_Country_ISO3=lambda frame: frame["Exp_Country_ISO3"].astype("string").str.strip().fillna(""),
            Imp_Country_ISO3=lambda frame: frame["Imp_Country_ISO3"].astype("string").str.strip().fillna(""),
            Source_Code=lambda frame: (
                frame["Source_Code"].astype("string").str.strip().fillna("").replace(dict(iso3_corrections))
            ),
        )
        .loc[lambda frame: ~frame["Exp_Name"].isin(AGGREGATES | EXCLUDED_RAW_NAMES)]
        .loc[lambda frame: ~frame["Imp_Name"].isin(AGGREGATES | EXCLUDED_RAW_NAMES)]
        .loc[
            :,
            ["Exp_Name", "Exp_Country_ISO3", "Imp_Name", "Imp_Country_ISO3", "year", "export", "Source_Code"],
        ]
    )

    keys = ["year", "Exp_Name", "Imp_Name"]
    is_imp = export_df_2005["Source_Code"].eq(export_df_2005["Imp_Country_ISO3"])
    is_exp = export_df_2005["Source_Code"].eq(export_df_2005["Exp_Country_ISO3"])
    is_eur = export_df_2005["Source_Code"].eq("EUR")
    is_unmatched = ~(is_imp | is_exp | is_eur)

    best = (
        export_df_2005
        .assign(_prio=(is_imp.astype(int) * 3 + is_exp.astype(int) * 2 + is_eur.astype(int) * 1))
        .sort_values(keys + ["_prio"], ascending=[True, True, True, False])
        .drop_duplicates(keys, keep="first")
        .drop(columns="_prio")
    )

    # [FIXED, resolves C2]: only keep unmatched rows for keys that `best` has no row
    # for at all -- this is the one-line change that eliminates the double-count.
    best_keys = pd.MultiIndex.from_frame(best[keys])
    unmatched = export_df_2005[is_unmatched]
    unmatched_keys = pd.MultiIndex.from_frame(unmatched[keys])
    unmatched_novel = unmatched[~unmatched_keys.isin(best_keys)]

    export_df_2005 = (
        pd.concat([best, unmatched_novel], ignore_index=True)
        .sort_values(keys + ["Source_Code"])
        .reset_index(drop=True)
    )

    df_exp_eu = export_df_2005[
        export_df_2005["Exp_Name"].isin(eu_countries)
        & ~export_df_2005["Imp_Name"].isin(eu_countries)
    ].copy()

    df_imp_eu = export_df_2005[
        export_df_2005["Imp_Name"].isin(eu_countries)
        & ~export_df_2005["Exp_Name"].isin(eu_countries)
    ].copy()

    return export_df, export_df_2005, df_exp_eu, df_imp_eu


def build_stock_dict(
    scenario_df: pd.DataFrame,
    remind_regions: list[str],
    stock_interpolation_method: str = "cubic",
) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """
    Aggregate `scenario_df` into per-(region, technology) annual stock series, filling
    in any years REMIND didn't natively report via interpolation.

    `stock_interpolation_method`: "cubic" (original behavior, default) or "pchip"
    (monotonicity-preserving -- see MATH_MODELS.md section 1.2 for the full derivation
    and the empirical before/after comparison).
    """
    VALID_INTERPOLATION_METHODS = {"cubic", "pchip"}
    if stock_interpolation_method not in VALID_INTERPOLATION_METHODS:
        raise ValueError(
            f"stock_interpolation_method={stock_interpolation_method!r} is not one of "
            f"{sorted(VALID_INTERPOLATION_METHODS)}."
        )

    stock_agg = (
        scenario_df.groupby(["year", "Region", "technology"], as_index=False)["value"].sum()
    )

    stock_dict: dict[tuple[str, str], pd.DataFrame] = {}
    for (region, tech), group in stock_agg.groupby(["Region", "technology"]):
        frame = (
            group[["year", "value"]]
            .rename(columns={"value": "stock"})
            .set_index("year")
            .sort_index()
        )
        frame["inflow"] = None
        frame["outflow"] = None

        full_years = pd.RangeIndex(frame.index.min(), frame.index.max() + 1, name="year")
        frame = frame.reindex(full_years)
        frame["stock"] = frame["stock"].interpolate(method=stock_interpolation_method).clip(lower=0)
        stock_dict[(region, tech)] = frame

    if not any(key[0] == "EUR" for key in stock_dict):
        eur_sum: dict[tuple[str, str], pd.DataFrame] = {}
        for (region, technology), frame in stock_dict.items():
            if region not in remind_regions:
                continue
            key_eur = ("EUR", technology)
            if key_eur not in eur_sum:
                eur_sum[key_eur] = frame.copy()
            else:
                eur_sum[key_eur]["stock"] += frame["stock"]
                # L8/L9, still open: inflow/outflow are not summed (left as the first
                # sub-region's, currently always None); mismatched year coverage across
                # sub-regions would silently produce NaN rather than an error. Neither
                # is triggered by current REMIND data.

        for key in list(stock_dict.keys()):
            if key[0] in remind_regions:
                del stock_dict[key]

        stock_dict.update(eur_sum)

    rows: list[pd.DataFrame] = []
    for (region, technology), frame in stock_dict.items():
        if region != "EUR":
            continue
        temp = frame["stock"].reset_index().rename(columns={"index": "year", "stock": "value"})
        temp["technology"] = technology
        rows.append(temp)

    stock_dict_df = pd.concat(rows, ignore_index=True)
    stock_dict_df["year"] = stock_dict_df["year"].astype(int)
    stock_dict_df["value"] = pd.to_numeric(stock_dict_df["value"], errors="coerce").fillna(0)

    return stock_dict, stock_dict_df