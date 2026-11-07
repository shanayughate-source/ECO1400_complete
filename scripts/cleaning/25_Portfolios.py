"""Clean the monthly value-weighted 25 Size-B/M portfolio returns.

Expected input:
    datasets/raw/25_Portfolios_5x5 2.csv

Outputs:
    datasets/clean/size_bm_portfolios_monthly.csv
    datasets/clean/size_bm_portfolios_cleaning_report.txt

The cleaned returns remain in percentage points, as published by the
Kenneth French Data Library. The first column is a monthly date in YYYY-MM-01
format. Only the Average Value Weighted Returns -- Monthly section is used.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from datetime import datetime
from pathlib import Path


# Project root is ECO1400_complete/, two levels above this script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "datasets" / "raw" / "25_Portfolios_5x5 2.csv"
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean"
OUTPUT_FILE = CLEAN_DIR / "size_bm_portfolios_monthly.csv"
REPORT_FILE = CLEAN_DIR / "size_bm_portfolios_cleaning_report.txt"

MISSING_CODES = {-99.99, -999.0, -999.99}
DATE_PATTERN = re.compile(r"^\d{6}$")


def read_csv_rows(path: Path) -> list[list[str]]:
    """Read the source CSV while handling a UTF-8 BOM and quoted fields."""
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            return [[cell.strip() for cell in row] for row in csv.reader(file)]
    except OSError as exc:
        raise RuntimeError(f"Could not read input file: {path}\n{exc}") from exc


def to_month(value: str) -> str | None:
    """Convert YYYYMM or YYYY-MM into YYYY-MM-01; return None otherwise."""
    value = value.strip()
    if DATE_PATTERN.fullmatch(value):
        year = int(value[:4])
        month = int(value[4:6])
    elif re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(part) for part in value.split("-"))
    else:
        return None

    if month < 1 or month > 12:
        return None
    return f"{year:04d}-{month:02d}-01"


def clean_number(value: str, row_number: int, column_name: str) -> float | None:
    """Convert a source value to float and handle French missing-value codes."""
    value = value.strip().replace(",", "")
    if value == "" or value.lower() in {"na", "n/a", "nan"}:
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value {value!r} in source row {row_number}, "
            f"column {column_name!r}. Check that the correct monthly CSV was used."
        ) from exc

    if not math.isfinite(number):
        return None
    if any(math.isclose(number, code, abs_tol=1e-8) for code in MISSING_CODES):
        return None
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}\n\n"
            "Check the filename in datasets/raw. The expected filename is "
            "'25_Portfolios_5x5 2.csv'."
        )

    rows = read_csv_rows(RAW_FILE)

    # Locate the value-weighted monthly section. This prevents accidentally
    # extracting equal-weighted returns or annual observations.
    section_index = None
    for index, row in enumerate(rows):
        description = " ".join(row).casefold()
        if "value weighted" in description and "monthly" in description:
            section_index = index
            break

    if section_index is None:
        raise ValueError(
            "Could not find the 'Average Value Weighted Returns -- Monthly' "
            "section. Confirm that this is the monthly 25 Size-B/M portfolio file."
        )

    # Find the first YYYYMM observation after the monthly section heading.
    first_data_index = None
    for index in range(section_index + 1, len(rows)):
        if rows[index] and to_month(rows[index][0]) is not None:
            first_data_index = index
            break

    if first_data_index is None:
        raise ValueError("Found the monthly section heading but no monthly observations.")

    # The portfolio-label row is the nearest preceding non-empty row that
    # contains 25 labels, excluding its leading date/blank column.
    portfolio_names = None
    for index in range(first_data_index - 1, section_index, -1):
        candidate = [cell.strip() for cell in rows[index]]
        while candidate and candidate[-1] == "":
            candidate.pop()
        if not candidate:
            continue

        first_cell = candidate[0].strip().casefold()
        if first_cell in {"", "date", "yyyymm"}:
            labels = candidate[1:]
            if len(labels) == 25 and all(labels):
                portfolio_names = labels
                break

    if portfolio_names is None:
        raise ValueError(
            "Could not identify exactly 25 portfolio names above the monthly "
            "observations. The source CSV may have a different layout."
        )

    cleaned_rows: list[dict[str, str | float]] = []
    seen_dates: set[str] = set()
    special_missing_count = 0

    # Extract only the contiguous monthly block. The next text row normally
    # starts the equal-weighted or annual section, so extraction stops there.
    for source_index in range(first_data_index, len(rows)):
        row = rows[source_index]
        if not row:
            break

        month = to_month(row[0])
        if month is None:
            break

        if month in seen_dates:
            raise ValueError(f"Duplicate monthly observation found: {month}")
        seen_dates.add(month)

        if len(row) < 26:
            raise ValueError(
                f"Monthly observation {month} has {len(row) - 1} portfolio values; "
                "25 are required. Check the raw CSV layout."
            )

        output_row: dict[str, str | float] = {"date": month}
        for column_index, portfolio_name in enumerate(portfolio_names, start=1):
            raw_value = row[column_index].strip()
            parsed = clean_number(raw_value, source_index + 1, portfolio_name)
            if raw_value and parsed is None:
                special_missing_count += 1
            output_row[portfolio_name] = "" if parsed is None else parsed
        cleaned_rows.append(output_row)

    if not cleaned_rows:
        raise ValueError("No monthly portfolio observations were extracted.")

    cleaned_rows.sort(key=lambda item: str(item["date"]))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = ["date"] + portfolio_names
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    missing_by_portfolio = {}
    for name in portfolio_names:
        missing_by_portfolio[name] = sum(
            1 for row in cleaned_rows if row[name] == ""
        )

    first_month = str(cleaned_rows[0]["date"])[:7]
    last_month = str(cleaned_rows[-1]["date"])[:7]
    missing_total = sum(missing_by_portfolio.values())

    report_lines = [
        "KENNETH FRENCH 25 SIZE-B/M PORTFOLIOS: CLEANING REPORT",
        "=" * 62,
        f"Source file: {RAW_FILE.name}",
        f"Source SHA-256: {sha256_file(RAW_FILE)}",
        "Extracted section: Average Value Weighted Returns -- Monthly",
        "Return weighting: Value-weighted",
        f"Observations: {len(cleaned_rows)}",
        f"Portfolios: {len(portfolio_names)}",
        f"First month: {first_month}",
        f"Last month: {last_month}",
        "Date format: YYYY-MM-01",
        "Units: Percentage points (not decimals); no division by 100 performed.",
        f"Missing/special-code cells: {missing_total}",
        f"Special missing-code values handled: {sorted(MISSING_CODES)}",
        "",
        "Missing observations by portfolio:",
    ]
    for name, count in missing_by_portfolio.items():
        report_lines.append(f"  {name}: {count}")
    report_lines.extend(
        [
            "",
            f"Cleaned CSV: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}",
            "No missing values were interpolated or forward-filled.",
        ]
    )
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("25-portfolio cleaning completed.")
    print(f"Rows: {len(cleaned_rows)} | Portfolios: {len(portfolio_names)}")
    print(f"Sample: {first_month} to {last_month}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
