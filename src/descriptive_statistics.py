"""Descriptive summaries and confidence intervals for the UCI heart data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t

from src.data_loading import load_heart_disease_data


TARGET_COLUMN = "num"
SOURCE_COLUMN = "source"
DISEASE_COLUMN = "disease_status"
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CONFIDENCE_LEVEL = 0.95


@dataclass
class DescriptiveStatistics:
    """Tables and plain-language interpretations of descriptive results."""

    numerical_summary: pd.DataFrame
    prevalence_summary: pd.DataFrame
    interpretations: list[str]


def _mean_confidence_interval(
    values: pd.Series, confidence_level: float
) -> tuple[float, float]:
    """Calculate a two-sided Student t confidence interval for a sample mean."""
    observed = values.dropna().astype(float)
    if len(observed) < 2:
        return float("nan"), float("nan")

    standard_error = observed.std(ddof=1) / np.sqrt(len(observed))
    margin = t.ppf((1 + confidence_level) / 2, df=len(observed) - 1) * standard_error
    return float(observed.mean() - margin), float(observed.mean() + margin)


def _summarize_numerical_features(
    data: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    """Summarize continuous features overall and by binary disease group."""
    groups = [
        ("Overall", data),
        ("No disease", data.loc[data[DISEASE_COLUMN] == 0]),
        ("Heart disease", data.loc[data[DISEASE_COLUMN] == 1]),
    ]
    rows = []
    for group_name, group_data in groups:
        for feature in NUMERICAL_FEATURES:
            values = group_data[feature].dropna().astype(float)
            ci_low, ci_high = _mean_confidence_interval(values, confidence_level)
            rows.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "n": len(values),
                    "mean": values.mean(),
                    "mean_ci_lower": ci_low,
                    "mean_ci_upper": ci_high,
                    "median": values.median(),
                    "variance": values.var(ddof=1),
                    "standard_deviation": values.std(ddof=1),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                }
            )
    return pd.DataFrame(rows)


def _summarize_prevalence(
    data: pd.DataFrame, confidence_level: float
) -> pd.DataFrame:
    """Estimate disease prevalence overall and within each source."""
    rows = []
    groups = [("Overall", data)]
    groups.extend(
        (str(source), source_data)
        for source, source_data in data.groupby(SOURCE_COLUMN, sort=True)
    )

    for group_name, group_data in groups:
        outcomes = group_data[DISEASE_COLUMN].dropna().astype(int)
        total = len(outcomes)
        cases = int(outcomes.sum())
        interval = binomtest(cases, total).proportion_ci(
            confidence_level=confidence_level,
            method="wilson",
        )
        rows.append(
            {
                "group": group_name,
                "n": total,
                "disease_cases": cases,
                "prevalence": cases / total,
                "ci_lower": interval.low,
                "ci_upper": interval.high,
                "confidence_level": confidence_level,
            }
        )
    return pd.DataFrame(rows)


def _create_interpretations(
    numerical_summary: pd.DataFrame,
    prevalence_summary: pd.DataFrame,
    confidence_level: float,
    data: pd.DataFrame,
) -> list[str]:
    """Explain estimates and their uncertainty without causal interpretation."""
    percent = confidence_level * 100
    overall_prevalence = prevalence_summary.loc[
        prevalence_summary["group"] == "Overall"
    ].iloc[0]
    interpretations = [
        (
            f"Observed disease prevalence was {overall_prevalence['prevalence']:.1%} "
            f"({int(overall_prevalence['disease_cases'])} of "
            f"{int(overall_prevalence['n'])} records); its {percent:.0f}% Wilson "
            f"confidence interval is "
            f"{overall_prevalence['ci_lower']:.1%} to "
            f"{overall_prevalence['ci_upper']:.1%}."
        ),
        (
            "The prevalence interval quantifies sampling uncertainty under a "
            "binomial model; it does not account for selection or differences "
            "among the contributing hospital samples."
        ),
        (
            "Source-specific observed prevalence ranges from "
            f"{prevalence_summary.loc[prevalence_summary['group'] != 'Overall', 'prevalence'].min():.1%} "
            "to "
            f"{prevalence_summary.loc[prevalence_summary['group'] != 'Overall', 'prevalence'].max():.1%}. "
            "The pooled estimate reflects the mix of these source datasets and "
            "should not be treated as population prevalence."
        ),
        (
            "Numerical-feature intervals are Student t confidence intervals "
            "for group means, using non-missing observations for each feature. "
            "They describe uncertainty in means, not the spread of individual "
            "measurements."
        ),
    ]

    zero_coded_measurements = ["trestbps", "chol", "thalach"]
    zero_counts = data[zero_coded_measurements].eq(0).sum()
    zero_counts = zero_counts[zero_counts > 0]
    if not zero_counts.empty:
        counts = ", ".join(
            f"{feature}: {int(count)}" for feature, count in zero_counts.items()
        )
        interpretations.append(
            "Recorded zero values were retained because the supplied files use "
            f"'?' to mark missingness ({counts}). Some may be invalid or "
            "source-specific missing-value codes, so verify their meaning before "
            "interpreting those feature summaries."
        )

    overall_rows = numerical_summary.loc[numerical_summary["group"] == "Overall"]
    for _, row in overall_rows.iterrows():
        if pd.notna(row["mean_ci_lower"]) and pd.notna(row["mean_ci_upper"]):
            interpretations.append(
                f"{row['feature']}: mean {row['mean']:.2f} "
                f"(n={int(row['n'])}; {percent:.0f}% t interval "
                f"{row['mean_ci_lower']:.2f} to {row['mean_ci_upper']:.2f})."
            )
    return interpretations


def calculate_descriptive_statistics(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> DescriptiveStatistics:
    """Calculate summaries and confidence intervals, and save result tables.

    Numerical features are summarized overall and separately by disease
    status. Prevalence is estimated overall and by dataset source. Missing
    predictor observations are omitted feature-by-feature; records without an
    observed target are excluded from prevalence and group summaries.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")
    if data is None:
        data = load_heart_disease_data()

    required_columns = NUMERICAL_FEATURES + [TARGET_COLUMN, SOURCE_COLUMN]
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    analysis_data = data.dropna(subset=[TARGET_COLUMN, SOURCE_COLUMN]).copy()
    if analysis_data.empty:
        raise ValueError("No rows with observed targets and sources are available.")
    analysis_data[DISEASE_COLUMN] = analysis_data[TARGET_COLUMN].gt(0).astype(int)

    numerical_summary = _summarize_numerical_features(
        analysis_data, confidence_level
    )
    prevalence_summary = _summarize_prevalence(
        analysis_data, confidence_level
    )
    interpretations = _create_interpretations(
        numerical_summary, prevalence_summary, confidence_level, analysis_data
    )

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        result_directory = project_root / "results" / "statistical_results"
    else:
        result_directory = Path(output_dir)
    result_directory.mkdir(parents=True, exist_ok=True)
    numerical_summary.to_csv(
        result_directory / "descriptive_numerical_summary.csv", index=False
    )
    prevalence_summary.to_csv(
        result_directory / "disease_prevalence_summary.csv", index=False
    )
    (result_directory / "descriptive_interpretation.txt").write_text(
        "\n".join(f"- {text}" for text in interpretations) + "\n",
        encoding="utf-8",
    )

    print("Numerical feature summaries (means with confidence intervals):")
    print(
        numerical_summary[
            [
                "group",
                "feature",
                "n",
                "mean",
                "mean_ci_lower",
                "mean_ci_upper",
                "median",
                "variance",
                "standard_deviation",
                "q1",
                "q3",
                "minimum",
                "maximum",
            ]
        ].round(3).to_string(index=False)
    )
    print("\nDisease prevalence and Wilson confidence intervals:")
    print(prevalence_summary.round(4).to_string(index=False))
    print("\nStatistical interpretation:")
    for interpretation in interpretations:
        print(f"- {interpretation}")
    print(f"\nResult tables saved to: {result_directory}")

    return DescriptiveStatistics(
        numerical_summary=numerical_summary,
        prevalence_summary=prevalence_summary,
        interpretations=interpretations,
    )


if __name__ == "__main__":
    calculate_descriptive_statistics()
