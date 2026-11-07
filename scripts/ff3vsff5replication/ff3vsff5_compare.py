from __future__ import annotations

"""Compare official FF3, official FF5, and nested FF3+RMW+CMA.

Reads existing outputs from:
    results/ff3/
    results/ff5/
    results/ff3_plus_rmw_cma/

Writes to:
    results/ff3vsff5/

The portfolio-level absolute-alpha reduction is defined as:
    abs(alpha of baseline) - abs(alpha of comparison model)

Positive values mean the comparison model has a smaller absolute alpha.
"""

from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_OBSERVATIONS = 756
EXPECTED_PORTFOLIOS = 25
START_EXPECTED = "1963-07"
END_EXPECTED = "2026-06"

SUMMARY_METRICS = [
    ("mean_r_squared", "Mean R-squared", "higher"),
    ("median_r_squared", "Median R-squared", "higher"),
    ("mean_abs_alpha_pct_month", "Mean absolute alpha (% per month)", "lower"),
    ("mean_abs_alpha_pct_year_simple", "Mean absolute alpha (% simple annualized)", "lower"),
    ("rmse_alpha_pct_month", "RMSE alpha (% per month)", "lower"),
    ("max_abs_alpha_pct_month", "Maximum absolute alpha (% per month)", "lower"),
    ("grs_f_statistic", "GRS F-statistic", "interpret"),
    ("grs_p_value", "GRS p-value", "interpret"),
    ("hac6_joint_alpha_wald_chi2", "HAC(6) joint-alpha Wald chi-square", "interpret"),
    ("hac6_joint_alpha_wald_p", "HAC(6) joint-alpha Wald p-value", "interpret"),
    ("hac12_joint_alpha_wald_chi2", "HAC(12) joint-alpha Wald chi-square", "interpret"),
    ("hac12_joint_alpha_wald_p", "HAC(12) joint-alpha Wald p-value", "interpret"),
]

MODEL_DIRS = {
    "FF3": "ff3",
    "FF5": "ff5",
    "FF3+RMW+CMA": "ff3_plus_rmw_cma",
}


def read_summary(path: Path, model: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing model summary: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected one summary row in {path}; found {len(frame)}")
    row = frame.iloc[0]
    required = ["sample_start", "sample_end", "observations", "test_assets"]
    missing = [c for c in required if c not in row.index]
    if missing:
        raise ValueError(f"{model} summary is missing metadata columns: {missing}")
    return row


def read_portfolio_results(path: Path, model: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing portfolio regression file: {path}")
    frame = pd.read_csv(path)
    required = ["portfolio", "alpha_pct_month", "alpha_hac6_se", "alpha_hac6_t", "alpha_hac6_p",
                "alpha_hac12_p", "r_squared", "adjusted_r_squared", "observations"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{model} portfolio results are missing columns: {missing}")
    if frame["portfolio"].duplicated().any():
        raise ValueError(f"Duplicate portfolio labels in {path}")
    if len(frame) != EXPECTED_PORTFOLIOS:
        raise ValueError(f"Expected 25 portfolios in {path}; found {len(frame)}")
    return frame.copy()


def validate_common_sample(summaries: dict[str, pd.Series]) -> None:
    expected = {
        "sample_start": START_EXPECTED,
        "sample_end": END_EXPECTED,
        "observations": EXPECTED_OBSERVATIONS,
        "test_assets": EXPECTED_PORTFOLIOS,
    }
    for model, row in summaries.items():
        for key, target in expected.items():
            value = row[key]
            if str(value) != str(target):
                raise ValueError(f"{model} has {key}={value}; expected {target}.")


def make_portfolio_absolute_comparison(regressions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for model, frame in regressions.items():
        selected = frame[[
            "portfolio", "alpha_pct_month", "alpha_hac6_se", "alpha_hac6_t", "alpha_hac6_p",
            "alpha_hac12_p", "r_squared", "adjusted_r_squared"
        ]].copy()
        selected = selected.rename(columns={
            "alpha_pct_month": f"alpha_{model}_pct_month",
            "alpha_hac6_se": f"alpha_{model}_hac6_se",
            "alpha_hac6_t": f"alpha_{model}_hac6_t",
            "alpha_hac6_p": f"alpha_{model}_hac6_p",
            "alpha_hac12_p": f"alpha_{model}_hac12_p",
            "r_squared": f"r_squared_{model}",
            "adjusted_r_squared": f"adjusted_r_squared_{model}",
        })
        merged = selected if merged is None else merged.merge(selected, on="portfolio", how="inner", validate="one_to_one")

    if merged is None or len(merged) != EXPECTED_PORTFOLIOS:
        raise ValueError("Could not align all 25 portfolios across all three models.")

    for model in MODEL_DIRS:
        merged[f"absolute_alpha_{model}_pct_month"] = merged[f"alpha_{model}_pct_month"].abs()

    comparisons = [
        ("FF3", "FF5"),
        ("FF3", "FF3+RMW+CMA"),
        ("FF5", "FF3+RMW+CMA"),
    ]
    for baseline, comparison in comparisons:
        # Positive means the comparison model reduces the absolute pricing error.
        merged[f"absolute_alpha_reduction_{baseline}_to_{comparison}"] = (
            merged[f"absolute_alpha_{baseline}_pct_month"]
            - merged[f"absolute_alpha_{comparison}_pct_month"]
        )
        merged[f"{comparison}_improves_absolute_alpha_vs_{baseline}"] = (
            merged[f"absolute_alpha_reduction_{baseline}_to_{comparison}"] > 0
        )
        merged[f"r_squared_difference_{comparison}_minus_{baseline}"] = (
            merged[f"r_squared_{comparison}"] - merged[f"r_squared_{baseline}"]
        )

    return merged.sort_values("portfolio").reset_index(drop=True)


def make_summary_tables(summaries: dict[str, pd.Series]):
    rows = []
    for key, label, direction in SUMMARY_METRICS:
        row = {"metric": label, "preferred_direction": direction}
        for model, summary in summaries.items():
            row[model] = float(summary[key]) if key in summary.index and pd.notna(summary[key]) else np.nan
        rows.append(row)
    long = pd.DataFrame(rows)

    wide_rows = []
    for key, label, direction in SUMMARY_METRICS:
        wide_rows.append({
            "metric": label,
            "preferred_direction": direction,
            "FF3": float(summaries["FF3"][key]),
            "FF5": float(summaries["FF5"][key]),
            "FF3+RMW+CMA": float(summaries["FF3+RMW+CMA"][key]),
        })
    wide = pd.DataFrame(wide_rows)

    for model_a, model_b in [("FF3", "FF5"), ("FF3", "FF3+RMW+CMA"), ("FF5", "FF3+RMW+CMA")]:
        wide[f"{model_b}_minus_{model_a}"] = wide[model_b] - wide[model_a]
    return long, wide


def factor_input_diagnostics(project_root: Path) -> pd.DataFrame:
    sample_dir = project_root / "datasets" / "ff3_vs_ff5_estimation_sample"
    ff3_path = sample_dir / "ff3_estimation_sample.csv"
    ff5_path = sample_dir / "ff5_estimation_sample.csv"
    for path in [ff3_path, ff5_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing estimation sample: {path}")

    ff3 = pd.read_csv(ff3_path)
    ff5 = pd.read_csv(ff5_path)
    ff3.columns = [str(c).strip() for c in ff3.columns]
    ff5.columns = [str(c).strip() for c in ff5.columns]
    ff3["Date"] = pd.to_datetime(ff3["Date"], errors="coerce")
    ff5["Date"] = pd.to_datetime(ff5["Date"], errors="coerce")
    if ff3["Date"].isna().any() or ff5["Date"].isna().any():
        raise ValueError("Invalid dates in estimation sample.")
    ff3 = ff3.sort_values("Date").reset_index(drop=True)
    ff5 = ff5.sort_values("Date").reset_index(drop=True)

    if len(ff3) != EXPECTED_OBSERVATIONS or len(ff5) != EXPECTED_OBSERVATIONS:
        raise ValueError("Estimation samples must both contain 756 observations.")
    if not ff3["Date"].equals(ff5["Date"]):
        raise ValueError("FF3 and FF5 estimation dates do not match.")

    exclude3 = {"Date", "Mkt_RF", "SMB", "HML", "RF"}
    exclude5 = {"Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"}
    p3 = [c for c in ff3.columns if c not in exclude3]
    p5 = [c for c in ff5.columns if c not in exclude5]
    if set(p3) != set(p5) or len(p3) != EXPECTED_PORTFOLIOS:
        raise ValueError("FF3 and FF5 estimation samples do not have identical portfolio columns.")

    rows = []
    for factor in ["Mkt_RF", "HML", "RF"]:
        x = pd.to_numeric(ff3[factor], errors="coerce").to_numpy(float)
        y = pd.to_numeric(ff5[factor], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Missing or invalid values in {factor}.")
        diff = x - y
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        max_difference = float(np.max(np.abs(diff)))
        if factor == "RF" and max_difference > 1e-10:
            raise ValueError(
                "FF3 and FF5 RF series differ. Since the model comparison should use the same "
                "excess-return dependent variables, standardize RF across the two estimation "
                f"samples and rerun both regressions. Maximum absolute RF difference={max_difference:.12g}."
            )
        rows.append({
            "series": factor,
            "max_abs_difference_percentage_points": max_difference,
            "mean_abs_difference_percentage_points": float(np.mean(np.abs(diff))),
            "months_differing_at_1e-10": int(np.sum(np.abs(diff) > 1e-10)),
            "correlation": corr,
            "interpretation": "Descriptive diagnostic; each official model retains its own factor series.",
        })

    x = ff3[p3].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = ff5[p3].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Missing or invalid raw portfolio returns in estimation samples.")
    diff = x - y
    max_portfolio_difference = float(np.max(np.abs(diff)))
    if max_portfolio_difference > 1e-10:
        raise ValueError(
            "The 25 raw portfolio return series differ between the FF3 and FF5 estimation samples. "
            "All model comparisons must use identical test-asset returns. "
            f"Maximum absolute difference={max_portfolio_difference:.12g}."
        )
    rows.append({
        "series": "25 raw portfolio returns",
        "max_abs_difference_percentage_points": max_portfolio_difference,
        "mean_abs_difference_percentage_points": float(np.mean(np.abs(diff))),
        "months_differing_at_1e-10": int(np.sum(np.abs(diff) > 1e-10)),
        "correlation": np.nan,
        "interpretation": "These should match exactly because all models use the same 25 test assets.",
    })
    return pd.DataFrame(rows)


def build_report(summaries: dict[str, pd.Series], portfolio_table: pd.DataFrame) -> str:
    models = list(MODEL_DIRS.keys())
    lines = [
        "THREE-MODEL FAMA-FRENCH COMPARISON",
        "=" * 76,
        "Sample: July 1963 through June 2026 (756 monthly observations)",
        "Test assets: the same 25 value-weighted Size-Book-to-Market portfolios",
        "Models: official FF3; official FF5; nested FF3 + RMW + CMA",
        "",
        "AGGREGATE RESULTS",
    ]
    for model in models:
        s = summaries[model]
        lines.extend([
            f"{model}:",
            f"  Mean R-squared: {float(s['mean_r_squared']):.8f}",
            f"  Mean absolute alpha: {float(s['mean_abs_alpha_pct_month']):.8f}% per month",
            f"  RMSE alpha: {float(s['rmse_alpha_pct_month']):.8f}% per month",
            f"  Maximum absolute alpha: {float(s['max_abs_alpha_pct_month']):.8f}% per month",
            f"  GRS F-statistic: {float(s['grs_f_statistic']):.8f}; p={float(s['grs_p_value']):.8g}",
            f"  HAC(6) joint-alpha Wald p-value: {float(s['hac6_joint_alpha_wald_p']):.8g}",
            f"  HAC(12) joint-alpha Wald p-value: {float(s['hac12_joint_alpha_wald_p']):.8g}",
        ])
        if model == "FF3+RMW+CMA":
            if "hac6_added_rmw_cma_wald_p" in s.index:
                lines.append(
                    f"  HAC(6) joint test: all 50 added RMW/CMA loadings = 0; "
                    f"p={float(s['hac6_added_rmw_cma_wald_p']):.8g}"
                )
        lines.append("")

    lines.extend([
        "PORTFOLIO-LEVEL ABSOLUTE ALPHA COMPARISON",
        f"Portfolios where FF5 has lower |alpha| than FF3: "
        f"{int(portfolio_table['FF5_improves_absolute_alpha_vs_FF3'].sum())} of 25",
        f"Portfolios where nested model has lower |alpha| than FF3: "
        f"{int(portfolio_table['FF3+RMW+CMA_improves_absolute_alpha_vs_FF3'].sum())} of 25",
        f"Portfolios where nested model has lower |alpha| than FF5: "
        f"{int(portfolio_table['FF3+RMW+CMA_improves_absolute_alpha_vs_FF5'].sum())} of 25",
        "",
        "Absolute-alpha reduction is defined as |alpha_baseline| - |alpha_comparison|.",
        "Positive values mean the comparison model has a smaller absolute alpha.",
        "Negative values mean the comparison model has a larger absolute alpha.",
        "",
        "INTERPRETATION",
        "The official FF3 and official FF5 are not strictly nested because their SMB factors differ.",
        "The nested FF3+RMW+CMA specification keeps the official FF3 Mkt-RF, SMB and HML series",
        "and adds RMW and CMA from the official FF5 file. It is a diagnostic extension, not official FF5.",
        "The nested model's joint HAC Wald test of the 50 added loadings assesses whether RMW and CMA",
        "loadings are jointly zero across the 25 portfolios, conditional on the FF3 factors.",
        "Do not select a model solely on R-squared or by mechanically ranking GRS p-values.",
        "GRS is a separate classical joint-alpha test for each model; it is not a direct test of",
        "whether one model is statistically better than another.",
        "The table compares in-sample pricing errors. It does not establish out-of-sample superiority.",
    ])
    return "\n".join(lines) + "\n"


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "results" / "ff3vsff5"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    regressions = {}
    for model, folder in MODEL_DIRS.items():
        model_dir = project_root / "results" / folder
        summaries[model] = read_summary(model_dir / "model_summary.csv", model)
        regressions[model] = read_portfolio_results(model_dir / "portfolio_regressions.csv", model)

    for model, summary in summaries.items():
        if str(summary["sample_start"]) != START_EXPECTED:
            raise ValueError(f"{model} sample starts at {summary['sample_start']}, expected {START_EXPECTED}.")
        if str(summary["sample_end"]) != END_EXPECTED:
            raise ValueError(f"{model} sample ends at {summary['sample_end']}, expected {END_EXPECTED}.")
        if int(float(summary["observations"])) != EXPECTED_OBSERVATIONS:
            raise ValueError(f"{model} does not have 756 observations.")
        if int(float(summary["test_assets"])) != EXPECTED_PORTFOLIOS:
            raise ValueError(f"{model} does not have 25 test assets.")

    # Compare portfolio labels explicitly before joining results.
    labels = {model: set(frame["portfolio"]) for model, frame in regressions.items()}
    if not (labels["FF3"] == labels["FF5"] == labels["FF3+RMW+CMA"]):
        raise ValueError("Portfolio labels differ across the three models.")

    portfolio_table = make_portfolio_absolute_comparison(regressions)
    summary_long, summary_wide = make_summary_tables(summaries)
    diagnostics = factor_input_diagnostics(project_root)

    summary_long.to_csv(output_dir / "three_model_comparison_long.csv", index=False)
    summary_wide.to_csv(output_dir / "three_model_comparison_wide.csv", index=False)
    portfolio_table.to_csv(output_dir / "portfolio_absolute_alpha_comparison.csv", index=False, float_format="%.10f")
    # Keep the previous filename as a compatibility copy.
    portfolio_table.to_csv(output_dir / "portfolio_alpha_comparison.csv", index=False, float_format="%.10f")
    diagnostics.to_csv(output_dir / "factor_input_diagnostics.csv", index=False, float_format="%.12g")
    report = build_report(summaries, portfolio_table)
    (output_dir / "three_model_comparison_report.txt").write_text(report, encoding="utf-8")

    print("\nTHREE-MODEL COMPARISON COMPLETE")
    print("Sample: July 1963-June 2026 | 756 observations | 25 portfolios")
    print(f"Results saved to: {output_dir}")
    print("\nSummary:")
    print(summary_wide.to_string(index=False))
    print("\nAbsolute-alpha improvements by portfolio:")
    print("FF5 vs FF3:", int(portfolio_table["FF5_improves_absolute_alpha_vs_FF3"].sum()), "/ 25")
    print("Nested vs FF3:", int(portfolio_table["FF3+RMW+CMA_improves_absolute_alpha_vs_FF3"].sum()), "/ 25")
    print("Nested vs FF5:", int(portfolio_table["FF3+RMW+CMA_improves_absolute_alpha_vs_FF5"].sum()), "/ 25")


if __name__ == "__main__":
    main()
