"""Visualizations of the original UCI Heart Disease data."""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

from src.data_loading import load_heart_disease_data


NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ZERO_AS_MISSING_FEATURES = ["trestbps", "chol"]
TARGET_COLUMN = "num"
SOURCE_COLUMN = "source"
CONFIDENCE_LEVEL = 0.95
DEFAULT_PLOT_DIRECTORY = Path(__file__).resolve().parents[1] / "results" / "plots"


def plot_numerical_histograms(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    bins: int = 20,
) -> Path:
    """Plot original-data histograms for the five numerical features.

    Zero values in ``trestbps`` and ``chol`` are excluded from their
    histograms by treating them as missing. No imputation, scaling, or
    modification of the supplied DataFrame is performed.
    """
    if data is None:
        data = load_heart_disease_data()

    missing_columns = sorted(set(NUMERICAL_FEATURES).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing numerical columns: {missing_columns}")
    if bins < 1:
        raise ValueError("bins must be a positive integer.")

    histogram_data = data[NUMERICAL_FEATURES].copy()
    for feature in NUMERICAL_FEATURES:
        try:
            histogram_data[feature] = pd.to_numeric(
                histogram_data[feature], errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Feature '{feature}' contains values that cannot be plotted numerically."
            ) from error

    for feature in ZERO_AS_MISSING_FEATURES:
        histogram_data[feature] = histogram_data[feature].mask(
            histogram_data[feature].eq(0)
        )

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, feature in zip(axes.flat, NUMERICAL_FEATURES):
        values = histogram_data[feature].dropna()
        axis.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
        axis.set_title(f"{feature} distribution")
        axis.set_xlabel(feature)
        axis.set_ylabel("Number of records")
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.98,
            0.95,
            f"n = {len(values)}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )

    axes.flat[-1].set_visible(False)
    figure.suptitle("Distributions of Numerical Features", fontsize=15)
    figure.tight_layout()

    plot_directory = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIRECTORY
    plot_directory.mkdir(parents=True, exist_ok=True)
    output_path = plot_directory / "numerical_feature_histograms.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Numerical feature histograms saved to: {output_path}")
    print("Records included in each histogram after missing-value exclusions:")
    print(histogram_data.notna().sum().to_string())
    return output_path


def plot_numerical_boxplots_by_disease(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Compare original numerical-feature distributions by binary disease status.

    The boxplots show the median, interquartile range, and 1.5-IQR whiskers.
    Zero values in ``trestbps`` and ``chol`` are treated as missing for this
    visualization only; other values and the original DataFrame are unchanged.
    """
    if data is None:
        data = load_heart_disease_data()

    required_columns = NUMERICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    plot_data = data[required_columns].copy()
    for feature in NUMERICAL_FEATURES:
        try:
            plot_data[feature] = pd.to_numeric(plot_data[feature], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Feature '{feature}' contains values that cannot be plotted numerically."
            ) from error
    try:
        plot_data[TARGET_COLUMN] = pd.to_numeric(
            plot_data[TARGET_COLUMN], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains values that cannot be interpreted numerically."
        ) from error

    plot_data = plot_data.dropna(subset=[TARGET_COLUMN])
    plot_data["disease_status"] = plot_data[TARGET_COLUMN].gt(0).map(
        {False: "No disease", True: "Heart disease"}
    )
    for feature in ZERO_AS_MISSING_FEATURES:
        plot_data[feature] = plot_data[feature].mask(plot_data[feature].eq(0))

    group_names = ["No disease", "Heart disease"]
    if not plot_data["disease_status"].isin(group_names).any():
        raise ValueError("No records with an observed disease target are available.")

    figure, axes = plt.subplots(1, len(NUMERICAL_FEATURES), figsize=(17, 5))
    group_colors = ["#72B7B2", "#E45756"]
    for axis, feature in zip(axes, NUMERICAL_FEATURES):
        grouped_values = [
            plot_data.loc[
                plot_data["disease_status"] == group_name, feature
            ].dropna()
            for group_name in group_names
        ]
        if not any(len(values) for values in grouped_values):
            raise ValueError(
                f"No usable observations for feature '{feature}' after missing-value exclusions."
            )

        boxplot = axis.boxplot(
            grouped_values,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box, color in zip(boxplot["boxes"], group_colors):
            box.set_facecolor(color)
            box.set_alpha(0.8)

        axis.set_title(feature)
        axis.set_ylabel("Value")
        axis.set_xticks([1, 2], group_names)
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.98,
            0.95,
            f"n = {len(grouped_values[0])}, {len(grouped_values[1])}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    figure.suptitle("Numerical Features by Heart Disease Status", fontsize=15)
    figure.tight_layout()

    plot_directory = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIRECTORY
    plot_directory.mkdir(parents=True, exist_ok=True)
    output_path = plot_directory / "numerical_features_by_disease_boxplots.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Disease-group boxplots saved to: {output_path}")
    print("Non-missing observations per feature (No disease, Heart disease):")
    for feature in NUMERICAL_FEATURES:
        counts = [
            int(
                plot_data.loc[
                    plot_data["disease_status"] == group_name, feature
                ].notna().sum()
            )
            for group_name in group_names
        ]
        print(f"  {feature}: {counts[0]}, {counts[1]}")
    return output_path


def plot_pearson_correlation_heatmap(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Plot pairwise Pearson correlations for the five original numerical features.

    Correlations are calculated pairwise from available observations. Zeros in
    ``trestbps`` and ``chol`` are treated as missing for this plot only.
    """
    if data is None:
        data = load_heart_disease_data()

    missing_columns = sorted(set(NUMERICAL_FEATURES).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing numerical columns: {missing_columns}")

    correlation_data = data[NUMERICAL_FEATURES].copy()
    for feature in NUMERICAL_FEATURES:
        try:
            correlation_data[feature] = pd.to_numeric(
                correlation_data[feature], errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Feature '{feature}' contains values that cannot be analyzed numerically."
            ) from error

    for feature in ZERO_AS_MISSING_FEATURES:
        correlation_data[feature] = correlation_data[feature].mask(
            correlation_data[feature].eq(0)
        )

    correlation_matrix = correlation_data.corr(method="pearson", min_periods=2)
    pairwise_counts = correlation_data.notna().astype(int).T.dot(
        correlation_data.notna().astype(int)
    )
    if correlation_matrix.isna().all().all():
        raise ValueError(
            "Pearson correlations could not be calculated; check for insufficient "
            "observations or constant numerical features."
        )

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(
        correlation_matrix.to_numpy(),
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )
    axis.set_xticks(range(len(NUMERICAL_FEATURES)), NUMERICAL_FEATURES)
    axis.set_yticks(range(len(NUMERICAL_FEATURES)), NUMERICAL_FEATURES)
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_title("Pearson Correlation of Numerical Features")

    for row in range(len(NUMERICAL_FEATURES)):
        for column in range(len(NUMERICAL_FEATURES)):
            value = correlation_matrix.iloc[row, column]
            annotation = f"{value:.2f}" if pd.notna(value) else "N/A"
            axis.text(
                column,
                row,
                annotation,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation coefficient")
    figure.tight_layout()

    plot_directory = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIRECTORY
    plot_directory.mkdir(parents=True, exist_ok=True)
    output_path = plot_directory / "pearson_correlation_heatmap.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Pearson correlation heatmap saved to: {output_path}")
    print("Pairwise observation counts after missing-value exclusions:")
    print(pairwise_counts.to_string())
    return output_path


def plot_hospital_disease_prevalence(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Path:
    """Plot disease prevalence by source with two-sided Wilson confidence intervals.

    Disease is defined from the original target as ``num > 0``. Confidence
    intervals describe binomial sampling uncertainty within each source group.
    """
    if data is None:
        data = load_heart_disease_data()

    required_columns = [SOURCE_COLUMN, TARGET_COLUMN]
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")

    prevalence_data = data[required_columns].copy()
    try:
        prevalence_data[TARGET_COLUMN] = pd.to_numeric(
            prevalence_data[TARGET_COLUMN], errors="raise"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains values that cannot be interpreted numerically."
        ) from error

    prevalence_data = prevalence_data.dropna(subset=required_columns)
    if prevalence_data.empty:
        raise ValueError("No records with both source and target are available.")

    rows = []
    for source, source_data in prevalence_data.groupby(SOURCE_COLUMN, sort=True):
        outcomes = source_data[TARGET_COLUMN].gt(0).astype(int)
        total = len(outcomes)
        cases = int(outcomes.sum())
        interval = binomtest(cases, total).proportion_ci(
            confidence_level=confidence_level,
            method="wilson",
        )
        rows.append(
            {
                SOURCE_COLUMN: source,
                "records": total,
                "disease_cases": cases,
                "prevalence": cases / total,
                "ci_lower": interval.low,
                "ci_upper": interval.high,
            }
        )
    prevalence_summary = pd.DataFrame(rows)

    figure, axis = plt.subplots(figsize=(9, 6))
    positions = range(len(prevalence_summary))
    lower_errors = prevalence_summary["prevalence"] - prevalence_summary["ci_lower"]
    upper_errors = prevalence_summary["ci_upper"] - prevalence_summary["prevalence"]
    axis.errorbar(
        positions,
        prevalence_summary["prevalence"],
        yerr=[lower_errors, upper_errors],
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=6,
        markersize=7,
        linewidth=1.5,
    )
    axis.set_xticks(
        list(positions),
        [
            f"{source}\n(n={count})"
            for source, count in zip(
                prevalence_summary[SOURCE_COLUMN],
                prevalence_summary["records"],
            )
        ],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Observed disease prevalence")
    axis.set_title(
        f"Hospital-wise Disease Prevalence "
        f"({confidence_level:.0%} Wilson confidence intervals)"
    )
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    plot_directory = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIRECTORY
    plot_directory.mkdir(parents=True, exist_ok=True)
    output_path = plot_directory / "hospital_disease_prevalence.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Hospital-wise disease prevalence plot saved to: {output_path}")
    print(
        f"Prevalence and {confidence_level:.0%} Wilson confidence intervals "
        "(disease defined as num > 0):"
    )
    print(
        prevalence_summary.assign(
            prevalence_percent=prevalence_summary["prevalence"] * 100,
            ci_lower_percent=prevalence_summary["ci_lower"] * 100,
            ci_upper_percent=prevalence_summary["ci_upper"] * 100,
        )[
            [
                SOURCE_COLUMN,
                "records",
                "disease_cases",
                "prevalence_percent",
                "ci_lower_percent",
                "ci_upper_percent",
            ]
        ].round(2).to_string(index=False)
    )
    return output_path


def plot_binary_target_distribution(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Plot counts for no disease (num == 0) and disease (num > 0)."""
    if data is None:
        data = load_heart_disease_data()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Input data is missing target column '{TARGET_COLUMN}'.")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise").dropna()
    if target.empty:
        raise ValueError("No records with an observed target are available.")

    binary_target = target.gt(0).map(
        {False: "No heart disease", True: "Heart disease"}
    )
    categories = ["No heart disease", "Heart disease"]
    counts = binary_target.value_counts().reindex(categories, fill_value=0)

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        categories,
        counts.values,
        color=["#72B7B2", "#E45756"],
        edgecolor="white",
    )
    axis.set_title("Binary Heart Disease Target Distribution")
    axis.set_ylabel("Number of records")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.bar_label(bars, labels=[f"{int(count)}" for count in counts.values], padding=3)
    axis.set_ylim(0, max(counts.max() * 1.12, 1))
    figure.tight_layout()

    plot_directory = Path(output_dir) if output_dir is not None else DEFAULT_PLOT_DIRECTORY
    plot_directory.mkdir(parents=True, exist_ok=True)
    output_path = plot_directory / "binary_target_distribution.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Binary target distribution plot saved to: {output_path}")
    print("Target counts (missing num values excluded):")
    print(counts.to_string())
    return output_path


if __name__ == "__main__":
    plot_numerical_histograms()
    plot_numerical_boxplots_by_disease()
    plot_pearson_correlation_heatmap()
    plot_hospital_disease_prevalence()
    plot_binary_target_distribution()
