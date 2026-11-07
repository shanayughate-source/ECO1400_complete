"""Clean the official monthly Fama/French 5-factor dataset.

Expected input:
    datasets/raw/F-F_Research_Data_5_Factors_2x3 3.csv

Outputs:
    datasets/clean/ff5_factors_monthly.csv
    datasets/clean/ff5_factors_cleaning_report.txt

The published factor values remain in percentage points. The output date
column is formatted as YYYY-MM-01.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "F-F_Research_Data_5_Factors_2x3 3.csv"
)
CLEAN_DIR = PROJECT_ROOT / "datasets" / "clean"
OUTPUT_FILE = CLEAN_DIR / "ff5_factors_monthly.csv"
REPORT_FILE = CLEAN_DIR / "ff5_factors_cleaning_report.txt"

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
MISSING_CODES = {-99.99, -999.0, -999.99}
DATE_PATTERN = re.compile(r"^\d{6}$")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            return [[cell.strip() for cell in row] for row in csv.reader(file)]
    except OSError as exc:
        raise RuntimeError(f"Could not read input file: {path}\n{exc}") from exc


def to_month(value: str) -> str | None:
    value = value.strip()
    if DATE_PATTERN.fullmatch(value):
        year, month = int(value[:4]), int(value[4:6])
    elif re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = (int(part) for part in value.split("-"))
    else:
        return None
    if month < 1 or month > 12:
        return None
    return f"{year:04d}-{month:02d}-01"


def clean_number(value: str, row_number: int, column_name: str) -> float | None:
    value = value.strip().replace(",", "")
    if value == "" or value.casefold() in {"na", "n/a", "nan"}:
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value {value!r} in source row {row_number}, "
            f"column {column_name!r}."
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


def find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    required = {normalize_name(name) for name in FACTOR_COLUMNS}
    for row_index, row in enumerate(rows):
        positions: dict[str, int] = {}
        for column_index, value in enumerate(row):
            normalized = normalize_name(value)
            if normalized:
                positions[normalized] = column_index
        if required.issubset(positions):
            return row_index, positions
    raise ValueError(
        "Could not locate the monthly FF5 factor header. Expected columns: "
        "Mkt-RF, SMB, HML, RMW, CMA and RF. Confirm that the raw file is the "
        "official monthly Fama/French 5-factor CSV."
    )


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}\n\n"
            "Expected filename: 'F-F_Research_Data_5_Factors_2x3 3.csv'."
        )

    rows = read_csv_rows(RAW_FILE)
    header_index, positions = find_header(rows)
    factor_positions = {
        name: positions[normalize_name(name)] for name in FACTOR_COLUMNS
    }

    first_data_index = None
    for index in range(header_index + 1, len(rows)):
        if rows[index] and to_month(rows[index][0]) is not None:
            first_data_index = index
            break
    if first_data_index is None:
        raise ValueError("The FF5 header was found, but no monthly observations followed it.")

    cleaned_rows: list[dict[str, str | float]] = []
    seen_dates: set[str] = set()
    missing_by_factor = {name: 0 for name in FACTOR_COLUMNS}

    # Stop at the first non-date row after the monthly block, excluding annual data.
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

        output_row: dict[str, str | float] = {"date": month}
        for factor in FACTOR_COLUMNS:
            source_column = factor_positions[factor]
            if source_column >= len(row):
                raise ValueError(
                    f"Row for {month} does not contain the {factor} column."
                )
            raw_value = row[source_column].strip()
            parsed = clean_number(raw_value, source_index + 1, factor)
            if raw_value and parsed is None:
                missing_by_factor[factor] += 1
            output_row[factor] = "" if parsed is None else parsed
        cleaned_rows.append(output_row)

    if not cleaned_rows:
        raise ValueError("No monthly FF5 observations were extracted.")

    cleaned_rows.sort(key=lambda item: str(item["date"]))
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["date"] + FACTOR_COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    first_month = str(cleaned_rows[0]["date"])[:7]
    last_month = str(cleaned_rows[-1]["date"])[:7]
    missing_total = sum(missing_by_factor.values())
    report = [
        "FAMA/FRENCH 5 FACTORS: CLEANING REPORT",
        "=" * 48,
        f"Source file: {RAW_FILE.name}",
        f"Source SHA-256: {sha256_file(RAW_FILE)}",
        f"Observations: {len(cleaned_rows)}",
        f"First month: {first_month}",
        f"Last month: {last_month}",
        f"Factors: {', '.join(FACTOR_COLUMNS)}",
        "Date format: YYYY-MM-01",
        "Units: Percentage points (not decimals); no division by 100 performed.",
        f"Missing/special-code cells: {missing_total}",
        f"Special missing-code values handled: {sorted(MISSING_CODES)}",
        "",
        "Missing observations by factor:",
    ]
    for name, count in missing_by_factor.items():
        report.append(f"  {name}: {count}")
    report.extend(
        [
            "",
            f"Cleaned CSV: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}",
            "Only monthly observations were extracted; annual observations were excluded.",
            "No missing values were interpolated or forward-filled.",
        ]
    )
    REPORT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("FF5 factor cleaning completed.")
    print(f"Rows: {len(cleaned_rows)} | Factors: {len(FACTOR_COLUMNS)}")
    print(f"Sample: {first_month} to {last_month}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
