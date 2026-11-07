from __future__ import annotations

"""Nested FF3 + RMW + CMA model for ECO1400.

Uses official FF3 Mkt-RF, SMB, HML and RF, then adds RMW and CMA
from the official FF5 factor file. This isolates the additions of RMW
and CMA while retaining the original FF3 factor series.

Inputs:
    datasets/ff3_vs_ff5_estimation_sample/ff3_estimation_sample.csv
    datasets/ff3_vs_ff5_estimation_sample/ff5_estimation_sample.csv

Outputs:
    results/ff3_plus_rmw_cma/

Sample: July 1963 - June 2026 (756 monthly observations).
Portfolio returns in the input files must be raw returns in percentage
points. This script subtracts the FF3 RF exactly once.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


START_DATE = "1963-07-01"
END_DATE = "2026-06-01"
EXPECTED_OBSERVATIONS = 756
EXPECTED_PORTFOLIOS = 25
HAC_PRIMARY_LAGS = 6
HAC_ROBUSTNESS_LAGS = 12
SAMPLE_DIRECTORY = "ff3_vs_ff5_estimation_sample"


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------
def _read_sample(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} estimation sample not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]

    if "Date" not in frame.columns:
        raise ValueError(f"{label} sample must contain a Date column.")

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    if frame["Date"].isna().any():
        raise ValueError(f"{label} sample contains invalid dates.")

    # No deprecated inplace=True usage.
    frame = frame.sort_values("Date").reset_index(drop=True)

    if frame["Date"].duplicated().any():
        raise ValueError(f"{label} sample contains duplicate dates.")

    expected_dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    if len(frame) != EXPECTED_OBSERVATIONS or not pd.DatetimeIndex(frame["Date"]).equals(expected_dates):
        first = frame["Date"].min().strftime("%Y-%m") if len(frame) else "none"
        last = frame["Date"].max().strftime("%Y-%m") if len(frame) else "none"
        raise ValueError(
            f"{label} sample must contain exactly {EXPECTED_OBSERVATIONS} consecutive "
            f"months from 1963-07 through 2026-06. Found {len(frame)} rows ({first} to {last})."
        )
    return frame


def load_nested_data(project_root: Path):
    sample_dir = project_root / "datasets" / SAMPLE_DIRECTORY
    ff3 = _read_sample(sample_dir / "ff3_estimation_sample.csv", "FF3")
    ff5 = _read_sample(sample_dir / "ff5_estimation_sample.csv", "FF5")

    ff3_factors = ["Mkt_RF", "SMB", "HML", "RF"]
    ff5_additions = ["RMW", "CMA"]

    missing_ff3 = [c for c in ff3_factors if c not in ff3.columns]
    missing_ff5 = [c for c in ff5_additions if c not in ff5.columns]
    if missing_ff3:
        raise ValueError(f"FF3 sample is missing columns: {missing_ff3}")
    if missing_ff5:
        raise ValueError(f"FF5 sample is missing added factors: {missing_ff5}")

    if not ff3["Date"].equals(ff5["Date"]):
        raise ValueError("FF3 and FF5 sample dates do not match exactly.")

    excluded_ff3 = {"Date", *ff3_factors}
    excluded_ff5 = {"Date", "Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"}
    portfolios_ff3 = [c for c in ff3.columns if c not in excluded_ff3]
    portfolios_ff5 = [c for c in ff5.columns if c not in excluded_ff5]

    if len(portfolios_ff3) != EXPECTED_PORTFOLIOS:
        raise ValueError(f"Expected 25 FF3 portfolio columns; found {len(portfolios_ff3)}.")
    if set(portfolios_ff3) != set(portfolios_ff5):
        raise ValueError("The FF3 and FF5 samples do not contain the same 25 portfolio columns.")

    # Reorder FF5 portfolio columns to match FF3 exactly.
    portfolios = portfolios_ff3

    # The raw portfolio return series should be the same in both model samples.
    p3 = ff3[portfolios].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    p5 = ff5[portfolios].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p3).all() or not np.isfinite(p5).all():
        raise ValueError("Portfolio return columns contain missing or non-numeric values.")
    max_portfolio_difference = float(np.max(np.abs(p3 - p5)))
    if max_portfolio_difference > 1e-10:
        raise ValueError(
            "The raw 25 portfolio return series differ between the FF3 and FF5 samples. "
            f"Maximum absolute difference = {diagnostics['max_abs_difference_raw_portfolio_returns']:.12g}. "
            "Resolve this before estimating the nested model."
        )

    # Numeric checks. Use FF3 factor series and RF to anchor the nested model to FF3.
    for col in ff3_factors:
        ff3[col] = pd.to_numeric(ff3[col], errors="coerce")
    for col in ff5_additions + ["RF"]:
        ff5[col] = pd.to_numeric(ff5[col], errors="coerce")

    max_rf_difference = float(
        np.max(np.abs(ff3["RF"].to_numpy(dtype=float) - ff5["RF"].to_numpy(dtype=float)))
    )
    if max_rf_difference > 1e-10:
        raise ValueError(
            "The FF3 and FF5 RF series differ. The nested model uses FF3 RF, so the official "
            "FF5 regression would not use the same excess-return dependent variables. "
            f"Maximum absolute RF difference = {max_rf_difference:.12g}. "
            "Standardize the risk-free rate across estimation samples and rerun FF5 before comparison."
        )

    factor_frame = pd.DataFrame({
        "Mkt_RF": ff3["Mkt_RF"],
        "SMB": ff3["SMB"],
        "HML": ff3["HML"],
        "RMW": ff5["RMW"],
        "CMA": ff5["CMA"],
        "RF": ff3["RF"],
    })

    raw_returns = ff3[portfolios].apply(pd.to_numeric, errors="coerce")
    all_values = np.column_stack([
        factor_frame.to_numpy(dtype=float),
        raw_returns.to_numpy(dtype=float),
    ])
    if not np.isfinite(all_values).all():
        raise ValueError("Missing, infinite or non-numeric values found in nested-model inputs.")

    diagnostics = {
        "observations": len(ff3),
        "portfolios": len(portfolios),
        "first_month": ff3["Date"].min().strftime("%Y-%m"),
        "last_month": ff3["Date"].max().strftime("%Y-%m"),
        "max_abs_difference_raw_portfolio_returns": max_portfolio_difference,
        "risk_free_source": "Official FF3 sample RF (validated equal to FF5 RF)",
        "max_abs_difference_RF_FF3_vs_FF5": max_rf_difference,
        "Mkt_RF_SMB_HML_source": "Official FF3 sample",
        "RMW_CMA_source": "Official FF5 sample",
    }

    return ff3["Date"].copy(), factor_frame, raw_returns, portfolios, diagnostics


# ---------------------------------------------------------------------
# Classical GRS test
# ---------------------------------------------------------------------
def calculate_grs(alphas: np.ndarray, residuals: np.ndarray, factors: np.ndarray):
    t_obs, n_assets = residuals.shape
    n_factors = factors.shape[1]
    denominator_df = t_obs - n_assets - n_factors
    if denominator_df <= 0:
        raise ValueError("Insufficient observations for the GRS test.")

    residual_cov = (residuals.T @ residuals) / t_obs
    centered_factors = factors - factors.mean(axis=0)
    factor_cov = (centered_factors.T @ centered_factors) / t_obs
    mean_factors = factors.mean(axis=0)

    try:
        alpha_term = float(alphas.T @ np.linalg.solve(residual_cov, alphas))
        factor_term = float(mean_factors.T @ np.linalg.solve(factor_cov, mean_factors))
    except np.linalg.LinAlgError as exc:
        raise ValueError("A covariance matrix is singular in the GRS calculation.") from exc

    grs = (denominator_df / n_assets) * alpha_term / (1.0 + factor_term)
    p_value = float(stats.f.sf(grs, n_assets, denominator_df))
    return float(grs), p_value, n_assets, denominator_df


# ---------------------------------------------------------------------
# Stacked cross-equation HAC covariance and Wald tests
# ---------------------------------------------------------------------
def system_hac_covariance(design: np.ndarray, residuals: np.ndarray, max_lags: int):
    t_obs, n_assets = residuals.shape
    n_regressors = design.shape[1]
    bread_one = np.linalg.inv(design.T @ design)

    # Stack score vectors asset by asset: vec(X' * epsilon_i).
    score_blocks = [design * residuals[:, [i]] for i in range(n_assets)]
    scores = np.concatenate(score_blocks, axis=1)
    meat = scores.T @ scores

    used_lags = min(max_lags, t_obs - 1)
    for lag in range(1, used_lags + 1):
        weight = 1.0 - lag / (max_lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)

    # Small-sample correction consistent with the equation-level OLS correction.
    if t_obs > n_regressors:
        meat *= t_obs / (t_obs - n_regressors)

    bread = np.kron(np.eye(n_assets), bread_one)
    covariance = bread @ meat @ bread.T
    return (covariance + covariance.T) / 2.0


def linear_restriction_wald(covariance: np.ndarray, parameters: np.ndarray, positions: np.ndarray):
    selected_parameters = parameters[positions]
    selected_covariance = covariance[np.ix_(positions, positions)]
    selected_covariance = (selected_covariance + selected_covariance.T) / 2.0
    try:
        statistic = float(selected_parameters.T @ np.linalg.solve(selected_covariance, selected_parameters))
    except np.linalg.LinAlgError as exc:
        raise ValueError("HAC Wald restriction covariance matrix is singular.") from exc
    df = len(positions)
    p_value = float(stats.chi2.sf(statistic, df))
    return statistic, p_value, df


# ---------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------
def run_model():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "results" / "ff3_plus_rmw_cma"
    output_dir.mkdir(parents=True, exist_ok=True)

    dates, factor_frame, raw_returns, portfolios, diagnostics = load_nested_data(project_root)

    factor_names = ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
    factor_values = factor_frame[factor_names].to_numpy(dtype=float)
    rf = factor_frame["RF"].to_numpy(dtype=float)

    # The estimation sample contains raw portfolio returns. Subtract FF3 RF once.
    excess_returns = raw_returns.to_numpy(dtype=float) - rf[:, None]

    design_frame = sm.add_constant(factor_frame[factor_names].astype(float), has_constant="add")
    design = design_frame.to_numpy(dtype=float)

    regression_rows = []
    residuals_by_asset = []
    fitted_by_asset = []
    parameter_vectors = []
    alpha_values = []

    for asset_index, portfolio in enumerate(portfolios):
        y = excess_returns[:, asset_index]
        ols = sm.OLS(y, design_frame, missing="raise").fit()
        hac6 = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_PRIMARY_LAGS, use_correction=True, use_t=True)
        hac12 = ols.get_robustcov_results(cov_type="HAC", maxlags=HAC_ROBUSTNESS_LAGS, use_correction=True, use_t=True)

        params = np.asarray(ols.params, dtype=float)
        se6 = np.asarray(hac6.bse, dtype=float)
        t6 = np.asarray(hac6.tvalues, dtype=float)
        p6 = np.asarray(hac6.pvalues, dtype=float)
        se12 = np.asarray(hac12.bse, dtype=float)
        t12 = np.asarray(hac12.tvalues, dtype=float)
        p12 = np.asarray(hac12.pvalues, dtype=float)

        row = {
            "portfolio": portfolio,
            "alpha_pct_month": params[0],
            "alpha_pct_year_simple": params[0] * 12.0,
            "alpha_hac6_se": se6[0],
            "alpha_hac6_t": t6[0],
            "alpha_hac6_p": p6[0],
            "alpha_hac12_se": se12[0],
            "alpha_hac12_t": t12[0],
            "alpha_hac12_p": p12[0],
            "r_squared": float(ols.rsquared),
            "adjusted_r_squared": float(ols.rsquared_adj),
            "observations": int(ols.nobs),
        }
        for j, factor in enumerate(factor_names, start=1):
            row[f"beta_{factor}"] = params[j]
            row[f"beta_{factor}_hac6_se"] = se6[j]
            row[f"beta_{factor}_hac6_t"] = t6[j]
            row[f"beta_{factor}_hac6_p"] = p6[j]
            row[f"beta_{factor}_hac12_se"] = se12[j]
            row[f"beta_{factor}_hac12_t"] = t12[j]
            row[f"beta_{factor}_hac12_p"] = p12[j]

        regression_rows.append(row)
        parameter_vectors.append(params)
        alpha_values.append(float(params[0]))
        residuals_by_asset.append(np.asarray(ols.resid, dtype=float))
        fitted_by_asset.append(np.asarray(ols.fittedvalues, dtype=float))

    regression_table = pd.DataFrame(regression_rows)
    residual_matrix = np.column_stack(residuals_by_asset)
    fitted_matrix = np.column_stack(fitted_by_asset)
    parameter_matrix = np.vstack(parameter_vectors)
    alpha_vector = np.asarray(alpha_values, dtype=float)
    stacked_parameters = parameter_matrix.reshape(-1)

    grs, grs_p, grs_num_df, grs_den_df = calculate_grs(alpha_vector, residual_matrix, factor_values)

    hac6_cov = system_hac_covariance(design, residual_matrix, HAC_PRIMARY_LAGS)
    hac12_cov = system_hac_covariance(design, residual_matrix, HAC_ROBUSTNESS_LAGS)

    # Intercept positions in the asset-by-asset stacked coefficient vector.
    alpha_positions = np.arange(len(portfolios)) * design.shape[1]
    hac6_alpha_wald, hac6_alpha_p, hac6_alpha_df = linear_restriction_wald(
        hac6_cov, stacked_parameters, alpha_positions
    )
    hac12_alpha_wald, hac12_alpha_p, hac12_alpha_df = linear_restriction_wald(
        hac12_cov, stacked_parameters, alpha_positions
    )

    # Added-factor restrictions: 25 RMW loadings, 25 CMA loadings, or all 50 jointly.
    k_reg = design.shape[1]
    rmw_positions = np.arange(len(portfolios)) * k_reg + 4
    cma_positions = np.arange(len(portfolios)) * k_reg + 5
    added_positions = np.concatenate([rmw_positions, cma_positions])

    added_wald, added_p, added_df = linear_restriction_wald(
        hac6_cov, stacked_parameters, added_positions
    )
    rmw_wald, rmw_p, rmw_df = linear_restriction_wald(
        hac6_cov, stacked_parameters, rmw_positions
    )
    cma_wald, cma_p, cma_df = linear_restriction_wald(
        hac6_cov, stacked_parameters, cma_positions
    )

    mean_abs_alpha = float(np.mean(np.abs(alpha_vector)))
    rmse_alpha = float(np.sqrt(np.mean(alpha_vector ** 2)))
    max_abs_alpha = float(np.max(np.abs(alpha_vector)))

    model_summary = pd.DataFrame([{
        "model": "FF3+RMW+CMA",
        "sample_start": dates.min().strftime("%Y-%m"),
        "sample_end": dates.max().strftime("%Y-%m"),
        "observations": len(dates),
        "test_assets": len(portfolios),
        "number_of_factors": len(factor_names),
        "mean_r_squared": float(regression_table["r_squared"].mean()),
        "median_r_squared": float(regression_table["r_squared"].median()),
        "mean_abs_alpha_pct_month": mean_abs_alpha,
        "mean_abs_alpha_pct_year_simple": mean_abs_alpha * 12.0,
        "rmse_alpha_pct_month": rmse_alpha,
        "max_abs_alpha_pct_month": max_abs_alpha,
        "grs_f_statistic": grs,
        "grs_p_value": grs_p,
        "grs_numerator_df": grs_num_df,
        "grs_denominator_df": grs_den_df,
        "hac6_joint_alpha_wald_chi2": hac6_alpha_wald,
        "hac6_joint_alpha_wald_p": hac6_alpha_p,
        "hac12_joint_alpha_wald_chi2": hac12_alpha_wald,
        "hac12_joint_alpha_wald_p": hac12_alpha_p,
        "hac6_added_rmw_cma_wald_chi2": added_wald,
        "hac6_added_rmw_cma_wald_p": added_p,
        "hac6_added_rmw_wald_chi2": rmw_wald,
        "hac6_added_rmw_wald_p": rmw_p,
        "hac6_added_cma_wald_chi2": cma_wald,
        "hac6_added_cma_wald_p": cma_p,
        "risk_free_source": "Official FF3 RF",
        "original_three_factor_source": "Official FF3 sample",
        "added_factor_source": "Official FF5 sample RMW and CMA",
    }])

    grs_table = pd.DataFrame([{
        "model": "FF3+RMW+CMA",
        "grs_f_statistic": grs,
        "p_value": grs_p,
        "numerator_df": grs_num_df,
        "denominator_df": grs_den_df,
        "test_assets": len(portfolios),
        "factors": len(factor_names),
        "observations": len(dates),
        "null_hypothesis": "All 25 portfolio alphas are jointly zero",
        "method": "Classical GRS; not HAC-adjusted",
    }])

    alpha_test_table = pd.DataFrame([
        {"test": "All 25 alphas = 0", "hac_lags": 6, "wald_chi2": hac6_alpha_wald, "df": hac6_alpha_df, "p_value": hac6_alpha_p},
        {"test": "All 25 alphas = 0", "hac_lags": 12, "wald_chi2": hac12_alpha_wald, "df": hac12_alpha_df, "p_value": hac12_alpha_p},
    ])

    added_factor_test_table = pd.DataFrame([
        {"test": "All 25 RMW and 25 CMA loadings = 0", "hac_lags": 6, "wald_chi2": added_wald, "df": added_df, "p_value": added_p},
        {"test": "All 25 RMW loadings = 0", "hac_lags": 6, "wald_chi2": rmw_wald, "df": rmw_df, "p_value": rmw_p},
        {"test": "All 25 CMA loadings = 0", "hac_lags": 6, "wald_chi2": cma_wald, "df": cma_df, "p_value": cma_p},
    ])

    residual_table = pd.DataFrame(residual_matrix, columns=portfolios)
    residual_table.insert(0, "Date", dates.dt.strftime("%Y-%m-%d"))
    fitted_table = pd.DataFrame(fitted_matrix, columns=portfolios)
    fitted_table.insert(0, "Date", dates.dt.strftime("%Y-%m-%d"))

    # Save results.
    regression_table.to_csv(output_dir / "portfolio_regressions.csv", index=False, float_format="%.10f")
    model_summary.to_csv(output_dir / "model_summary.csv", index=False, float_format="%.12g")
    grs_table.to_csv(output_dir / "grs_test.csv", index=False, float_format="%.12g")
    alpha_test_table.to_csv(output_dir / "hac_joint_alpha_test.csv", index=False, float_format="%.12g")
    added_factor_test_table.to_csv(output_dir / "added_factor_joint_tests.csv", index=False, float_format="%.12g")
    residual_table.to_csv(output_dir / "regression_residuals.csv", index=False, float_format="%.10f")
    fitted_table.to_csv(output_dir / "fitted_excess_returns.csv", index=False, float_format="%.10f")

    report = [
        "NESTED FF3 + RMW + CMA MODEL",
        "=" * 68,
        f"Sample: {diagnostics['first_month']} to {diagnostics['last_month']}",
        f"Observations: {diagnostics['observations']}",
        f"Test assets: {diagnostics['portfolios']}",
        "Factors: official FF3 Mkt-RF, SMB, HML + official FF5 RMW, CMA",
        "Risk-free rate: official FF3 RF",
        "Estimator: 25 separate OLS time-series regressions",
        "Individual inference: Newey-West HAC(6) and HAC(12)",
        "Joint alpha tests: stacked cross-equation HAC Wald",
        "Additional-factor tests: stacked cross-equation HAC Wald on 50 added loadings",
        "GRS: classical finite-sample F test; not HAC-adjusted",
        "Units: percentage points per month",
        "",
        f"Mean R-squared: {model_summary.loc[0, 'mean_r_squared']:.8f}",
        f"Mean absolute alpha: {mean_abs_alpha:.8f}% per month",
        f"RMSE alpha: {rmse_alpha:.8f}% per month",
        f"Maximum absolute alpha: {max_abs_alpha:.8f}% per month",
        f"GRS F({grs_num_df}, {grs_den_df}): {grs:.8f}; p-value={grs_p:.8g}",
        f"HAC(6) joint alpha Wald chi-square: {hac6_alpha_wald:.8f}; p-value={hac6_alpha_p:.8g}",
        f"HAC(12) joint alpha Wald chi-square: {hac12_alpha_wald:.8f}; p-value={hac12_alpha_p:.8g}",
        "",
        "INCREMENTAL FACTOR TESTS (HAC 6)",
        f"Joint 50 restrictions (all RMW and CMA loadings zero): chi-square({added_df})={added_wald:.8f}; p={added_p:.8g}",
        f"RMW 25 restrictions: chi-square({rmw_df})={rmw_wald:.8f}; p={rmw_p:.8g}",
        f"CMA 25 restrictions: chi-square({cma_df})={cma_wald:.8f}; p={cma_p:.8g}",
        "",
        "DATA CONSTRUCTION",
        "The 25 raw portfolio return series are taken from the FF3 estimation sample.",
        "Excess returns are calculated as raw portfolio return minus FF3 RF.",
        "Mkt-RF, SMB and HML are taken from the official FF3 sample.",
        "RMW and CMA are taken from the official FF5 sample.",
        f"Maximum absolute difference between the two raw portfolio panels: {diagnostics['max_abs_difference_raw_portfolio_returns']:.12g}",
        f"Maximum absolute difference between FF3 and FF5 RF: {diagnostics['max_abs_difference_RF_FF3_vs_FF5']:.12g}",
        "This is a nested diagnostic extension, not the official Fama-French five-factor model.",
    ]
    (output_dir / "model_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nNESTED FF3 + RMW + CMA ESTIMATION COMPLETE")
    print(f"Sample: {diagnostics['first_month']} to {diagnostics['last_month']} | {len(dates)} months | {len(portfolios)} portfolios")
    print(model_summary.to_string(index=False))
    print("\nAdded-factor joint tests:")
    print(added_factor_test_table.to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    run_model()
