from __future__ import annotations

"""Standalone ECO1400 Fama-French Five-Factor Model.

Estimates 25 portfolio time-series regressions using:
- OLS
- Newey-West HAC(6)
- Newey-West HAC(12)
- Classical GRS joint-alpha test
- Cross-equation HAC joint-alpha Wald test

Sample: July 1963 to June 2026.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


# ============================================================
# RESEARCH DESIGN
# ============================================================

START_DATE = "1963-07-01"
END_DATE = "2026-06-01"

EXPECTED_OBSERVATIONS = 756
EXPECTED_PORTFOLIOS = 25

HAC_PRIMARY_LAGS = 6
HAC_ROBUSTNESS_LAGS = 12

SAMPLE_DIRECTORY = "ff3_vs_ff5_estimation_sample"


# ============================================================
# 1. CLASSICAL GRS TEST
# ============================================================

def calculate_grs(
    alphas: np.ndarray,
    residuals: np.ndarray,
    factors: np.ndarray,
) -> tuple[float, float, int, int]:

    observations, n_assets = residuals.shape
    n_factors = factors.shape[1]

    denominator_df = observations - n_assets - n_factors

    if denominator_df <= 0:
        raise ValueError("Insufficient observations for GRS test.")

    residual_covariance = (
        residuals.T @ residuals
    ) / observations

    demeaned_factors = factors - factors.mean(axis=0)

    factor_covariance = (
        demeaned_factors.T @ demeaned_factors
    ) / observations

    mean_factors = factors.mean(axis=0)

    try:
        alpha_term = float(
            alphas.T @ np.linalg.solve(
                residual_covariance,
                alphas
            )
        )

        factor_term = float(
            mean_factors.T @ np.linalg.solve(
                factor_covariance,
                mean_factors
            )
        )

    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "GRS covariance matrix is singular."
        ) from exc

    grs_statistic = (
        (denominator_df / n_assets)
        * alpha_term
        / (1.0 + factor_term)
    )

    grs_p_value = float(
        stats.f.sf(
            grs_statistic,
            n_assets,
            denominator_df
        )
    )

    return (
        float(grs_statistic),
        grs_p_value,
        n_assets,
        denominator_df
    )


# ============================================================
# 2. SYSTEM-WIDE HAC JOINT ALPHA WALD TEST
# ============================================================

def calculate_system_hac_alpha_wald(
    design: np.ndarray,
    residuals: np.ndarray,
    alphas: np.ndarray,
    max_lags: int,
) -> tuple[float, float, int]:

    observations, n_assets = residuals.shape
    n_regressors = design.shape[1]

    try:
        bread_one = np.linalg.inv(
            design.T @ design
        )

    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Factor design matrix is singular."
        ) from exc

    score_blocks = [
        design * residuals[:, [asset_index]]
        for asset_index in range(n_assets)
    ]

    scores = np.concatenate(
        score_blocks,
        axis=1
    )

    meat = scores.T @ scores

    used_lags = min(
        max_lags,
        observations - 1
    )

    for lag in range(1, used_lags + 1):

        bartlett_weight = (
            1.0 - lag / (max_lags + 1.0)
        )

        lagged_cross_product = (
            scores[lag:].T @ scores[:-lag]
        )

        meat += bartlett_weight * (
            lagged_cross_product
            + lagged_cross_product.T
        )

    if observations > n_regressors:
        meat *= (
            observations
            / (observations - n_regressors)
        )

    bread = np.kron(
        np.eye(n_assets),
        bread_one
    )

    parameter_covariance = (
        bread @ meat @ bread.T
    )

    alpha_positions = (
        np.arange(n_assets) * n_regressors
    )

    alpha_covariance = parameter_covariance[
        np.ix_(
            alpha_positions,
            alpha_positions
        )
    ]

    alpha_covariance = (
        alpha_covariance
        + alpha_covariance.T
    ) / 2.0

    try:
        wald_statistic = float(
            alphas.T @ np.linalg.solve(
                alpha_covariance,
                alphas
            )
        )

    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "HAC joint-alpha covariance matrix is singular."
        ) from exc

    degrees_of_freedom = n_assets

    p_value = float(
        stats.chi2.sf(
            wald_statistic,
            degrees_of_freedom
        )
    )

    return (
        wald_statistic,
        p_value,
        degrees_of_freedom
    )


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_and_validate_sample(
    project_root: Path,
    sample_filename: str,
    factor_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:

    input_path = (
        project_root
        / "datasets"
        / SAMPLE_DIRECTORY
        / sample_filename
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Estimation sample not found: {input_path}\n"
            "Run the estimation-sample preparation script first."
        )

    data = pd.read_csv(input_path)

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    if "Date" not in data.columns:
        raise ValueError(
            "The estimation sample must contain a Date column."
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    if data["Date"].isna().any():
        raise ValueError(
            "Unparseable dates found in estimation sample."
        )

    data = (
        data.sort_values("Date")
        .reset_index(drop=True)
    )

    if data["Date"].duplicated().any():
        raise ValueError(
            "Duplicate monthly dates found."
        )

    expected_dates = pd.date_range(
        START_DATE,
        END_DATE,
        freq="MS"
    )

    actual_dates = pd.DatetimeIndex(
        data["Date"]
    )

    if (
        len(data) != EXPECTED_OBSERVATIONS
        or not actual_dates.equals(expected_dates)
    ):

        first_date = (
            data["Date"].min().strftime("%Y-%m")
            if len(data) else "none"
        )

        last_date = (
            data["Date"].max().strftime("%Y-%m")
            if len(data) else "none"
        )

        raise ValueError(
            f"Expected {EXPECTED_OBSERVATIONS} consecutive "
            f"monthly observations from {START_DATE[:7]} "
            f"through {END_DATE[:7]}.\n"
            f"Found {len(data)} observations from "
            f"{first_date} through {last_date}."
        )

    required_columns = (
        ["RF"] + factor_columns
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing factor columns: {missing_columns}"
        )

    excluded_columns = {
        "Date",
        "RF",
        *factor_columns
    }

    portfolio_columns = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

    if len(portfolio_columns) != EXPECTED_PORTFOLIOS:
        raise ValueError(
            f"Expected 25 portfolios but found "
            f"{len(portfolio_columns)}."
        )

    numeric_columns = (
        required_columns + portfolio_columns
    )

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    numeric_values = data[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Missing, non-numeric or infinite values "
            "found in estimation sample."
        )

    return data, portfolio_columns


# ============================================================
# 4. RUN FF5 MODEL
# ============================================================

def run_model():

    project_root = Path(__file__).resolve().parents[2]

    output_directory = (
        project_root
        / "results"
        / "ff5"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    factor_columns = [
        "Mkt_RF",
        "SMB",
        "HML",
        "RMW",
        "CMA"
    ]

    factor_labels = [
        "Mkt_RF",
        "SMB",
        "HML",
        "RMW",
        "CMA"
    ]

    # --------------------------------------------------------
    # Load estimation sample
    # --------------------------------------------------------

    data, portfolio_columns = load_and_validate_sample(
        project_root,
        "ff5_estimation_sample.csv",
        factor_columns
    )

    print("\nFF5 DATA VALIDATION SUCCESSFUL")

    print(
        f"Sample: {data['Date'].min():%Y-%m} "
        f"to {data['Date'].max():%Y-%m}"
    )

    print(
        f"Observations: {len(data)}"
    )

    print(
        f"Portfolios: {len(portfolio_columns)}"
    )

    # --------------------------------------------------------
    # Construct excess portfolio returns
    # --------------------------------------------------------

    # The estimation sample contains raw portfolio returns.
    # Subtract RF exactly once.

    risk_free = data[
        "RF"
    ].to_numpy(dtype=float)

    portfolio_returns = data[
        portfolio_columns
    ].to_numpy(dtype=float)

    excess_returns = (
        portfolio_returns
        - risk_free[:, None]
    )

    # --------------------------------------------------------
    # Factor matrix and regression design
    # --------------------------------------------------------

    factors = data[
        factor_columns
    ].to_numpy(dtype=float)

    design_frame = sm.add_constant(
        data[factor_columns].astype(float),
        has_constant="add"
    )

    design = design_frame.to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Store regression outputs
    # --------------------------------------------------------

    regression_rows = []

    residual_columns = []
    fitted_columns = []
    alpha_values = []

    # --------------------------------------------------------
    # Estimate 25 individual regressions
    # --------------------------------------------------------

    for asset_index, portfolio_name in enumerate(
        portfolio_columns
    ):

        print(
            f"Estimating FF5: {portfolio_name}"
        )

        dependent_variable = excess_returns[
            :,
            asset_index
        ]

        ols_result = sm.OLS(
            dependent_variable,
            design_frame,
            missing="raise"
        ).fit()

        # HAC with six monthly lags
        hac6_result = (
            ols_result.get_robustcov_results(
                cov_type="HAC",
                maxlags=HAC_PRIMARY_LAGS,
                use_correction=True,
                use_t=True
            )
        )

        # HAC with twelve monthly lags
        hac12_result = (
            ols_result.get_robustcov_results(
                cov_type="HAC",
                maxlags=HAC_ROBUSTNESS_LAGS,
                use_correction=True,
                use_t=True
            )
        )

        coefficients = np.asarray(
            ols_result.params,
            dtype=float
        )

        hac6_se = np.asarray(
            hac6_result.bse,
            dtype=float
        )

        hac6_t = np.asarray(
            hac6_result.tvalues,
            dtype=float
        )

        hac6_p = np.asarray(
            hac6_result.pvalues,
            dtype=float
        )

        hac12_se = np.asarray(
            hac12_result.bse,
            dtype=float
        )

        hac12_t = np.asarray(
            hac12_result.tvalues,
            dtype=float
        )

        hac12_p = np.asarray(
            hac12_result.pvalues,
            dtype=float
        )

        # ----------------------------------------------------
        # Portfolio regression results
        # ----------------------------------------------------

        row = {
            "portfolio": portfolio_name,

            "alpha_pct_month": coefficients[0],

            "alpha_pct_year_simple": (
                coefficients[0] * 12.0
            ),

            "alpha_hac6_se": hac6_se[0],
            "alpha_hac6_t": hac6_t[0],
            "alpha_hac6_p": hac6_p[0],

            "alpha_hac12_se": hac12_se[0],
            "alpha_hac12_t": hac12_t[0],
            "alpha_hac12_p": hac12_p[0],

            "r_squared": float(
                ols_result.rsquared
            ),

            "adjusted_r_squared": float(
                ols_result.rsquared_adj
            ),

            "observations": int(
                ols_result.nobs
            )
        }

        # Factor loadings and inference
        for factor_index, factor_label in enumerate(
            factor_labels,
            start=1
        ):

            row[
                f"beta_{factor_label}"
            ] = coefficients[factor_index]

            row[
                f"beta_{factor_label}_hac6_se"
            ] = hac6_se[factor_index]

            row[
                f"beta_{factor_label}_hac6_t"
            ] = hac6_t[factor_index]

            row[
                f"beta_{factor_label}_hac6_p"
            ] = hac6_p[factor_index]

            row[
                f"beta_{factor_label}_hac12_se"
            ] = hac12_se[factor_index]

            row[
                f"beta_{factor_label}_hac12_t"
            ] = hac12_t[factor_index]

            row[
                f"beta_{factor_label}_hac12_p"
            ] = hac12_p[factor_index]

        regression_rows.append(row)

        alpha_values.append(
            float(coefficients[0])
        )

        residual_columns.append(
            np.asarray(
                ols_result.resid,
                dtype=float
            )
        )

        fitted_columns.append(
            np.asarray(
                ols_result.fittedvalues,
                dtype=float
            )
        )

    # --------------------------------------------------------
    # Combine regression results
    # --------------------------------------------------------

    regression_table = pd.DataFrame(
        regression_rows
    )

    residual_matrix = np.column_stack(
        residual_columns
    )

    fitted_matrix = np.column_stack(
        fitted_columns
    )

    alpha_vector = np.asarray(
        alpha_values,
        dtype=float
    )

    # --------------------------------------------------------
    # GRS joint pricing test
    # --------------------------------------------------------

    grs_statistic, grs_p_value, grs_num_df, grs_den_df = (
        calculate_grs(
            alpha_vector,
            residual_matrix,
            factors
        )
    )

    # --------------------------------------------------------
    # HAC joint alpha Wald tests
    # --------------------------------------------------------

    hac6_wald, hac6_wald_p, hac6_df = (
        calculate_system_hac_alpha_wald(
            design,
            residual_matrix,
            alpha_vector,
            HAC_PRIMARY_LAGS
        )
    )

    hac12_wald, hac12_wald_p, hac12_df = (
        calculate_system_hac_alpha_wald(
            design,
            residual_matrix,
            alpha_vector,
            HAC_ROBUSTNESS_LAGS
        )
    )

    # --------------------------------------------------------
    # Pricing-error statistics
    # --------------------------------------------------------

    mean_absolute_alpha = float(
        np.mean(np.abs(alpha_vector))
    )

    rmse_alpha = float(
        np.sqrt(
            np.mean(alpha_vector ** 2)
        )
    )

    maximum_absolute_alpha = float(
        np.max(np.abs(alpha_vector))
    )

    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    model_summary = pd.DataFrame([{

        "model": "FF5",

        "sample_start": (
            data["Date"].min().strftime("%Y-%m")
        ),

        "sample_end": (
            data["Date"].max().strftime("%Y-%m")
        ),

        "observations": len(data),

        "test_assets": len(portfolio_columns),

        "number_of_factors": len(factor_columns),

        "primary_hac_lags": HAC_PRIMARY_LAGS,

        "robustness_hac_lags": HAC_ROBUSTNESS_LAGS,

        "mean_r_squared": float(
            regression_table["r_squared"].mean()
        ),

        "median_r_squared": float(
            regression_table["r_squared"].median()
        ),

        "mean_abs_alpha_pct_month": (
            mean_absolute_alpha
        ),

        "mean_abs_alpha_pct_year_simple": (
            mean_absolute_alpha * 12.0
        ),

        "rmse_alpha_pct_month": rmse_alpha,

        "max_abs_alpha_pct_month": (
            maximum_absolute_alpha
        ),

        "grs_f_statistic": grs_statistic,

        "grs_p_value": grs_p_value,

        "grs_numerator_df": grs_num_df,

        "grs_denominator_df": grs_den_df,

        "hac6_joint_alpha_wald_chi2": (
            hac6_wald
        ),

        "hac6_joint_alpha_wald_p": (
            hac6_wald_p
        ),

        "hac12_joint_alpha_wald_chi2": (
            hac12_wald
        ),

        "hac12_joint_alpha_wald_p": (
            hac12_wald_p
        )

    }])

    # --------------------------------------------------------
    # GRS results
    # --------------------------------------------------------

    grs_table = pd.DataFrame([{

        "model": "FF5",

        "grs_f_statistic": grs_statistic,

        "p_value": grs_p_value,

        "numerator_df": grs_num_df,

        "denominator_df": grs_den_df,

        "test_assets": len(portfolio_columns),

        "factors": len(factor_columns),

        "observations": len(data),

        "null_hypothesis": (
            "All 25 portfolio alphas are jointly zero"
        ),

        "covariance_method": (
            "Classical GRS with T-denominator "
            "residual and factor covariance estimates; "
            "not HAC-adjusted"
        )

    }])

    # --------------------------------------------------------
    # HAC joint alpha results
    # --------------------------------------------------------

    hac_test_table = pd.DataFrame([

        {
            "model": "FF5",

            "hac_lags": HAC_PRIMARY_LAGS,

            "wald_chi2": hac6_wald,

            "degrees_freedom": hac6_df,

            "p_value": hac6_wald_p,

            "null_hypothesis": (
                "All 25 portfolio alphas are jointly zero"
            ),

            "covariance_method": (
                "Stacked cross-equation Newey-West HAC"
            )
        },

        {
            "model": "FF5",

            "hac_lags": HAC_ROBUSTNESS_LAGS,

            "wald_chi2": hac12_wald,

            "degrees_freedom": hac12_df,

            "p_value": hac12_wald_p,

            "null_hypothesis": (
                "All 25 portfolio alphas are jointly zero"
            ),

            "covariance_method": (
                "Stacked cross-equation Newey-West HAC"
            )
        }

    ])

    # --------------------------------------------------------
    # Residuals and fitted returns
    # --------------------------------------------------------

    date_strings = (
        data["Date"].dt.strftime("%Y-%m-%d")
    )

    residual_table = pd.DataFrame(
        residual_matrix,
        columns=portfolio_columns
    )

    residual_table.insert(
        0,
        "Date",
        date_strings
    )

    fitted_table = pd.DataFrame(
        fitted_matrix,
        columns=portfolio_columns
    )

    fitted_table.insert(
        0,
        "Date",
        date_strings
    )

    # --------------------------------------------------------
    # Export all results
    # --------------------------------------------------------

    regression_table.to_csv(
        output_directory / "portfolio_regressions.csv",
        index=False,
        float_format="%.10f"
    )

    model_summary.to_csv(
        output_directory / "model_summary.csv",
        index=False,
        float_format="%.10f"
    )

    grs_table.to_csv(
        output_directory / "grs_test.csv",
        index=False,
        float_format="%.10f"
    )

    hac_test_table.to_csv(
        output_directory / "hac_joint_alpha_test.csv",
        index=False,
        float_format="%.10f"
    )

    residual_table.to_csv(
        output_directory / "regression_residuals.csv",
        index=False,
        float_format="%.10f"
    )

    fitted_table.to_csv(
        output_directory / "fitted_excess_returns.csv",
        index=False,
        float_format="%.10f"
    )

    # --------------------------------------------------------
    # Human-readable report
    # --------------------------------------------------------

    report_lines = [

        "FAMA-FRENCH FIVE-FACTOR MODEL",
        "=" * 64,

        f"Sample: {data['Date'].min():%Y-%m} "
        f"to {data['Date'].max():%Y-%m}",

        f"Monthly observations: {len(data)}",

        f"Test assets: {len(portfolio_columns)}",

        "Estimator: OLS time-series regressions",

        "Dependent variable: portfolio return minus RF",

        "Units: percentage points per month",

        f"Factors: {', '.join(factor_labels)}",

        "Primary inference: Newey-West HAC(6)",

        "Robustness inference: Newey-West HAC(12)",

        "GRS: classical finite-sample F test",

        "Additional test: stacked cross-equation HAC Wald",

        "",

        f"Mean R-squared: "
        f"{model_summary.loc[0, 'mean_r_squared']:.6f}",

        f"Median R-squared: "
        f"{model_summary.loc[0, 'median_r_squared']:.6f}",

        f"Mean absolute alpha: "
        f"{mean_absolute_alpha:.6f}% per month",

        f"RMSE alpha: "
        f"{rmse_alpha:.6f}% per month",

        f"Maximum absolute alpha: "
        f"{maximum_absolute_alpha:.6f}% per month",

        f"GRS F({grs_num_df}, {grs_den_df}): "
        f"{grs_statistic:.6f}",

        f"GRS p-value: {grs_p_value:.8g}",

        f"HAC(6) joint alpha Wald: "
        f"{hac6_wald:.6f}",

        f"HAC(6) p-value: {hac6_wald_p:.8g}",

        f"HAC(12) joint alpha Wald: "
        f"{hac12_wald:.6f}",

        f"HAC(12) p-value: {hac12_wald_p:.8g}",

        "",

        "Interpretation: assess pricing performance "
        "using alphas and joint tests, not R-squared alone.",

        "Official FF5 factor series retained."

    ]

    (
        output_directory / "model_report.txt"
    ).write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nFF5 ESTIMATION COMPLETE")

    print(
        model_summary.to_string(index=False)
    )

    print(
        f"\nResults saved to: {output_directory}"
    )

    print(
        "\nPortfolio excess returns calculated "
        "by subtracting RF exactly once."
    )


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    run_model()