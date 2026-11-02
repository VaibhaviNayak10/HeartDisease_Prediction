"""Investigate zero and missing values in selected original UCI features."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.data_loading import load_heart_disease_data


FEATURES_TO_ANALYZE = ["trestbps", "chol"]
SOURCE_COLUMN = "source"
TARGET_COLUMN = "num"
RECORD_COLUMNS = ["source", "age", "sex", "trestbps", "chol", "num"]
TARGET_CATEGORIES = [0, 1, 2, 3, 4]
SUMMARY_FILENAME = "zero_value_summary.csv"


@dataclass
class ZeroValueAnalysis:
    """Tables produced by the zero-value investigation."""

    summary: pd.DataFrame
    target_distribution: pd.DataFrame
    affected_records: dict[str, pd.DataFrame]


def _validate_input(data: pd.DataFrame) -> None:
    """Raise a useful error when expected source or feature columns are absent."""
    required_columns = set(RECORD_COLUMNS + FEATURES_TO_ANALYZE)
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")


def _build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Count zeros, non-zeros, and missing values overall and by source.

    Zero percentage uses all records in the group as its denominator, including
    rows where the feature is missing. Missing values are reported separately.
    """
    groups = [("Overall", data)]
    groups.extend(
        (str(source), source_data)
        for source, source_data in data.groupby(SOURCE_COLUMN, dropna=False, sort=True)
    )

    rows = []
    for feature in FEATURES_TO_ANALYZE:
        for source_name, group_data in groups:
            values = group_data[feature]
            total_records = len(group_data)
            zero_count = int(values.eq(0).sum())
            missing_count = int(values.isna().sum())
            nonzero_count = int((values.notna() & values.ne(0)).sum())
            zero_percentage = (
                float(np.divide(zero_count, total_records) * 100)
                if total_records
                else 0.0
            )
            rows.append(
                {
                    "Feature": feature,
                    "Source": source_name,
                    "Total Records": total_records,
                    "Zero Count": zero_count,
                    "Zero Percentage": zero_percentage,
                    "Non-zero Count": nonzero_count,
                    "Missing Count": missing_count,
                }
            )
    return pd.DataFrame(rows)


def _build_target_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Count zero-valued records in original target classes, overall and by source."""
    groups = [("Overall", data)]
    groups.extend(
        (str(source), source_data)
        for source, source_data in data.groupby(SOURCE_COLUMN, dropna=False, sort=True)
    )

    rows = []
    for feature in FEATURES_TO_ANALYZE:
        for source_name, group_data in groups:
            group_zeros = group_data.loc[group_data[feature].eq(0)]
            target_counts = group_zeros[TARGET_COLUMN].value_counts(dropna=False)
            row = {
                "Feature": feature,
                "Source": source_name,
                "Zero Records": len(group_zeros),
            }
            for category in TARGET_CATEGORIES:
                row[f"num_{category}"] = int(target_counts.get(category, 0))
            row["Missing num"] = int(group_zeros[TARGET_COLUMN].isna().sum())
            rows.append(row)

    return pd.DataFrame(rows)


def _get_affected_records(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return original rows with zero in each feature, without modifying input."""
    return {
        feature: data.loc[
            data[feature].eq(0), RECORD_COLUMNS
        ].copy()
        for feature in FEATURES_TO_ANALYZE
    }


def _print_interpretation(summary: pd.DataFrame) -> None:
    """Print a concise evidence-based interpretation of the observed counts."""
    overall = summary.loc[summary["Source"] == "Overall"].set_index("Feature")
    print("\nInterpretation:")
    for feature in FEATURES_TO_ANALYZE:
        stats = overall.loc[feature]
        print(
            f"- {feature}: {int(stats['Zero Count'])} zero value(s) "
            f"({stats['Zero Percentage']:.2f}% of "
            f"{int(stats['Total Records'])} records), "
            f"{int(stats['Non-zero Count'])} non-zero value(s), and "
            f"{int(stats['Missing Count'])} missing value(s)."
        )
    print(
        "- These counts describe the supplied records only. A zero is reported "
        "separately from a missing value; this analysis does not determine "
        "whether any zero is an error or a missing-value code. No values or "
        "records have been changed or removed."
    )


def analyze_zero_values(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> ZeroValueAnalysis:
    """Analyze original zero values and save an overall/source summary CSV."""
    if data is None:
        data = load_heart_disease_data()
    _validate_input(data)

    summary = _build_summary(data)
    target_distribution = _build_target_distribution(data)
    affected_records = _get_affected_records(data)

    print("Zero-value summary (percentage denominator: all records per group):")
    print(summary.round({"Zero Percentage": 2}).to_string(index=False))

    print("\nZero-value records by source and original num category:")
    print(target_distribution.to_string(index=False))

    for feature, records in affected_records.items():
        print(f"\nRecords with {feature} == 0:")
        if records.empty:
            print("No affected records.")
        else:
            print(records.to_string(index=False))

    _print_interpretation(summary)

    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        result_directory = project_root / "results" / "statistical_results"
    else:
        result_directory = Path(output_dir)
    result_directory.mkdir(parents=True, exist_ok=True)
    output_path = result_directory / SUMMARY_FILENAME
    summary.to_csv(output_path, index=False)
    print(f"\nSummary CSV saved to: {output_path}")

    return ZeroValueAnalysis(
        summary=summary,
        target_distribution=target_distribution,
        affected_records=affected_records,
    )


if __name__ == "__main__":
    analyze_zero_values()
