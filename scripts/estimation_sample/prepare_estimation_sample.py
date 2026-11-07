"""Prepare aligned FF3 and FF5 estimation samples for July 1963-June 2026.

Inputs (created by the three separate cleaning scripts):
    datasets/clean/size_bm_portfolios_monthly.csv
    datasets/clean/ff3_factors_monthly.csv
    datasets/clean/ff5_factors_monthly.csv

Outputs:
    datasets/ff3_vs_ff5_estimation_sample/ff3_estimation_sample.csv
    datasets/ff3_vs_ff5_estimation_sample/ff5_estimation_sample.csv
    datasets/ff3_vs_ff5_estimation_sample/estimation_sample_report.txt
    datasets/ff3_vs_ff5_estimation_sample/estimation_sample_manifest.json

IMPORTANT:
    The official FF3 and FF5 files are treated as separate published factor
    specifications. Each model retains ALL of its factors from its own file,
    including its own Mkt-RF, SMB, HML, and RF series. Do not replace one
    model's market or SMB series with the other model's series.

    SMB is expected to differ by construction. In current French Data Library
    files, Mkt-RF can also differ slightly between the separately published
    FF3 and FF5 files even when the files report the same CRSP database vintage.
    Therefore Mkt-RF differences are documented, not treated as a fatal error.
    HML and RF are checked strictly because they should agree in the same
    release; any discrepancy stops the script for investigation.

All returns/factors remain in percentage points. Portfolio excess returns
are calculated in the regression scripts as portfolio return minus each
model's own RF series.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean"
ESTIMATION_DIR = PROJECT_ROOT / "datasets" / "ff3_vs_ff5_estimation_sample"

PORTFOLIO_FILE = CLEAN_DIR / "size_bm_portfolios_monthly.csv"
FF3_FILE = CLEAN_DIR / "ff3_factors_monthly.csv"
FF5_FILE = CLEAN_DIR / "ff5_factors_monthly.csv"

FF3_OUTPUT = ESTIMATION_DIR / "ff3_estimation_sample.csv"
FF5_OUTPUT = ESTIMATION_DIR / "ff5_estimation_sample.csv"
REPORT_OUTPUT = ESTIMATION_DIR / "estimation_sample_report.txt"
MANIFEST_OUTPUT = ESTIMATION_DIR / "estimation_sample_manifest.json"

START_DATE = "1963-07-01"
END_DATE = "2026-06-01"
FACTOR_MATCH_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def find_date_column(frame: pd.DataFrame, path: Path) -> str:
    lookup = {normalize_name(column): column for column in frame.columns}
    for candidate in ("date", "month", "yyyymm"):
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not identify the date column in {path.name}. "
        f"Columns found: {list(frame.columns)}"
    )


def parse_month(value: object) -> pd.Timestamp:
    """Parse YYYYMM, YYYY-MM, or date-like values to first of month."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if re.fullmatch(r"\d{6}", text):
        parsed = pd.to_datetime(text, format="%Y%m", errors="coerce")
    elif re.fullmatch(r"\d{4}-\d{2}", text):
        parsed = pd.to_datetime(text + "-01", format="%Y-%m-%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).to_period("M").to_timestamp()


def read_monthly_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required cleaned dataset not found: {path}\n"
            "Run the corresponding dataset-cleaning script first."
        )
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    date_column = find_date_column(frame, path)
    frame = frame.rename(columns={date_column: "Date"})
    frame["Date"] = frame["Date"].map(parse_month)
    if frame["Date"].isna().any():
        bad = frame.loc[frame["Date"].isna()].head(5)
        raise ValueError(f"Unparseable dates in {path.name}:\n{bad}")
    if frame["Date"].duplicated().any():
        duplicates = (
            frame.loc[frame["Date"].duplicated(), "Date"]
            .dt.strftime("%Y-%m")
            .tolist()
        )
        raise ValueError(f"Duplicate monthly dates in {path.name}: {duplicates[:10]}")
    return frame.sort_values("Date").reset_index(drop=True)


def select_factor_columns(
    frame: pd.DataFrame, required: list[str], path: Path
) -> pd.DataFrame:
    normalized_lookup = {normalize_name(column): column for column in frame.columns}
    selected = {"Date": frame["Date"]}
    for factor in required:
        key = normalize_name(factor)
        if key not in normalized_lookup:
            raise ValueError(
                f"Missing factor {factor!r} in {path.name}. "
                f"Columns found: {list(frame.columns)}"
            )
        original_column = normalized_lookup[key]
        selected[factor] = pd.to_numeric(frame[original_column], errors="coerce")
    return pd.DataFrame(selected)


def assert_complete(frame: pd.DataFrame, label: str) -> None:
    missing = frame.isna()
    if missing.any().any():
        locations = []
        for column in frame.columns:
            if column == "Date":
                continue
            bad_dates = (
                frame.loc[missing[column], "Date"]
                .dt.strftime("%Y-%m")
                .tolist()
            )
            if bad_dates:
                locations.append(f"{column}: {bad_dates[:8]}")
        raise ValueError(
            f"Missing/non-numeric values found in {label}. No rows were silently dropped.\n"
            + "\n".join(locations[:20])
        )


def align_to_window(frame: pd.DataFrame, label: str, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    indexed = frame.set_index("Date").sort_index()
    in_window = indexed.loc[
        (indexed.index >= expected_dates[0]) & (indexed.index <= expected_dates[-1])
    ]
    aligned = in_window.reindex(expected_dates)
    absent = aligned.index[aligned.isna().all(axis=1)]
    if len(absent):
        missing_months = [date.strftime("%Y-%m") for date in absent[:12]]
        raise ValueError(
            f"{label} is missing {len(absent)} month(s) in the required sample. "
            f"First missing months: {missing_months}. Expected "
            f"{expected_dates[0]:%Y-%m} to {expected_dates[-1]:%Y-%m}."
        )
    aligned.index.name = "Date"
    return aligned.reset_index()


def compare_series(
    ff3_aligned: pd.DataFrame, ff5_aligned: pd.DataFrame, factor: str
) -> dict[str, object]:
    difference = (ff3_aligned[factor] - ff5_aligned[factor]).abs()
    changed = difference > FACTOR_MATCH_TOLERANCE
    if changed.any():
        examples = []
        for index in difference[changed].index[:5]:
            examples.append({
                "month": ff3_aligned.loc[index, "Date"].strftime("%Y-%m"),
                "ff3_value": float(ff3_aligned.loc[index, factor]),
                "ff5_value": float(ff5_aligned.loc[index, factor]),
                "absolute_difference": float(difference.loc[index]),
            })
    else:
        examples = []
    return {
        "factor": factor,
        "months_different": int(changed.sum()),
        "months_compared": int(len(difference)),
        "max_abs_difference": float(difference.max()),
        "tolerance": FACTOR_MATCH_TOLERANCE,
        "examples": examples,
    }


def main() -> None:
    portfolios_raw = read_monthly_csv(PORTFOLIO_FILE)
    ff3_raw = read_monthly_csv(FF3_FILE)
    ff5_raw = read_monthly_csv(FF5_FILE)

    portfolio_columns = [column for column in portfolios_raw.columns if column != "Date"]
    if len(portfolio_columns) != 25:
        raise ValueError(
            f"Expected exactly 25 Size-B/M portfolio return columns; found {len(portfolio_columns)}.\n"
            f"Detected columns: {portfolio_columns}"
        )

    portfolios = portfolios_raw[["Date"] + portfolio_columns].copy()
    for column in portfolio_columns:
        portfolios[column] = pd.to_numeric(portfolios[column], errors="coerce")

    ff3 = select_factor_columns(ff3_raw, ["Mkt_RF", "SMB", "HML", "RF"], FF3_FILE)
    ff5 = select_factor_columns(
        ff5_raw, ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"], FF5_FILE
    )

    expected_dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    expected_count = len(expected_dates)
    if expected_count != 756:
        raise RuntimeError(
            f"Internal date-window error: expected 756 months, got {expected_count}."
        )

    portfolios_aligned = align_to_window(portfolios, "25 Size-B/M portfolios", expected_dates)
    ff3_aligned = align_to_window(ff3, "FF3 factors", expected_dates)
    ff5_aligned = align_to_window(ff5, "FF5 factors", expected_dates)

    assert_complete(portfolios_aligned, "25 Size-B/M portfolios")
    assert_complete(ff3_aligned, "FF3 factors")
    assert_complete(ff5_aligned, "FF5 factors")

    # Diagnostics only for the market series: the independently published
    # FF3 and FF5 files can have small Mkt-RF differences even in the same
    # CRSP release. Preserve each model's own official Mkt-RF series.
    market_check = compare_series(ff3_aligned, ff5_aligned, "Mkt_RF")

    # HML and RF are common to the official definitions. A discrepancy here
    # is more likely to indicate a cleaning/file-selection problem, so stop.
    strict_checks = []
    for factor in ("HML", "RF"):
        result = compare_series(ff3_aligned, ff5_aligned, factor)
        strict_checks.append(result)
        if result["max_abs_difference"] > FACTOR_MATCH_TOLERANCE:
            raise ValueError(
                f"{factor} differs between the FF3 and FF5 cleaned files. "
                f"Maximum absolute difference: {result['max_abs_difference']}. "
                "Check that the four-column file is FF3, the six-column file is FF5, "
                "and that both cleaning scripts parsed the monthly section correctly.\n"
                f"Examples: {result['examples']}"
            )

    # The official SMB definitions differ. Never require them to match.
    ff3_sample = pd.concat(
        [
            ff3_aligned[["Date", "Mkt_RF", "SMB", "HML", "RF"]],
            portfolios_aligned[portfolio_columns],
        ],
        axis=1,
    )
    ff5_sample = pd.concat(
        [
            ff5_aligned[["Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]],
            portfolios_aligned[portfolio_columns],
        ],
        axis=1,
    )

    for label, sample in (("FF3", ff3_sample), ("FF5", ff5_sample)):
        if len(sample) != 756 or not sample["Date"].equals(
            pd.Series(expected_dates, name="Date")
        ):
            raise RuntimeError(
                f"{label} sample does not contain the exact 756-month date sequence."
            )

    ESTIMATION_DIR.mkdir(parents=True, exist_ok=True)
    ff3_sample.to_csv(
        FF3_OUTPUT, index=False, date_format="%Y-%m-%d", float_format="%.8f"
    )
    ff5_sample.to_csv(
        FF5_OUTPUT, index=False, date_format="%Y-%m-%d", float_format="%.8f"
    )

    report_lines = [
        "ECO1400 COMMON FF3/FF5 ESTIMATION SAMPLE",
        "=" * 48,
        f"Required period: {START_DATE[:7]} to {END_DATE[:7]}",
        f"Expected monthly observations: {expected_count}",
        f"Actual monthly observations: {len(ff3_sample)} in FF3 and {len(ff5_sample)} in FF5",
        f"Test assets: {len(portfolio_columns)} value-weighted Size-B/M portfolios",
        "Frequency: Monthly",
        "Return units: Percentage points",
        "Dependent variable: each portfolio return minus that model file's RF",
        "Official FF3 and FF5 factor series are preserved separately.",
        "Official FF3 SMB and official FF5 SMB are retained separately by construction.",
        "Mkt-RF diagnostic (difference is recorded, not treated as an error):",
        f"  Months different: {market_check['months_different']} / {market_check['months_compared']}",
        f"  Maximum absolute difference: {market_check['max_abs_difference']:.10g} percentage points",
        f"  Example differences: {market_check['examples']}",
        "Strict common-factor checks:",
    ]
    for result in strict_checks:
        report_lines.append(
            f"  {result['factor']}: {result['months_different']} months different; "
            f"maximum absolute difference = {result['max_abs_difference']:.10g}"
        )
    report_lines.extend([
        "",
        f"FF3 estimation sample: {FF3_OUTPUT.relative_to(PROJECT_ROOT)}",
        f"FF5 estimation sample: {FF5_OUTPUT.relative_to(PROJECT_ROOT)}",
        "No missing months or missing values were silently dropped.",
    ])
    REPORT_OUTPUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "project": "ECO1400_complete",
        "sample_start": START_DATE[:7],
        "sample_end": END_DATE[:7],
        "frequency": "monthly",
        "expected_observations": expected_count,
        "actual_observations": int(len(ff3_sample)),
        "test_assets": portfolio_columns,
        "return_units": "percentage points",
        "excess_return_definition": "portfolio return minus the corresponding model's RF",
        "ff3_factor_file": FF3_FILE.name,
        "ff5_factor_file": FF5_FILE.name,
        "ff3_smb_source": "official FF3 factor file",
        "ff5_smb_source": "official FF5 factor file",
        "market_factor_comparison": market_check,
        "strict_common_factor_checks": strict_checks,
        "source_files_sha256": {
            PORTFOLIO_FILE.name: sha256_file(PORTFOLIO_FILE),
            FF3_FILE.name: sha256_file(FF3_FILE),
            FF5_FILE.name: sha256_file(FF5_FILE),
        },
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("COMMON ESTIMATION SAMPLES CREATED")
    print(f"Period: {START_DATE[:7]} to {END_DATE[:7]}")
    print(f"Monthly observations: {len(ff3_sample)}")
    print(f"Test assets: {len(portfolio_columns)}")
    print(f"Mkt-RF differences between official files: {market_check['months_different']}")
    print(f"Maximum Mkt-RF difference: {market_check['max_abs_difference']:.8f} percentage points")
    print(f"FF3 sample: {FF3_OUTPUT}")
    print(f"FF5 sample: {FF5_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
