"""Interpret fitted logistic-regression coefficients and project findings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PreprocessedData,
)
from src.hypothesis_testing import AUCComparison
from src.model_evaluation import ModelEvaluation
from src.model_training import TrainedModelArtifact


CONFIDENCE_LEVEL = 0.95
DEFAULT_RESULT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "statistical_results"
)


@dataclass
class StatisticalInterpretation:
    """Coefficient estimates, odds ratios, confidence intervals, and narrative."""

    coefficient_summary: pd.DataFrame
    interpretations: list[str]


def _wald_coefficient_summary(
    model: TrainedModelArtifact,
    data: PreprocessedData,
    confidence_level: float,
) -> pd.DataFrame:
    """Compute approximate penalized-Wald intervals from training information."""
    estimator = model.estimator
    if not isinstance(estimator, LogisticRegression):
        raise TypeError("Odds-ratio interpretation requires a Logistic Regression model.")
    if not data.X_train.index.equals(data.y_train.index):
        raise ValueError("X_train and y_train must have aligned indices.")
    if tuple(map(str, data.X_train.columns)) != model.feature_names:
        raise ValueError("Training feature columns do not match the Logistic Regression model.")
    if estimator.coef_.shape != (1, data.X_train.shape[1]):
        raise ValueError("Expected a fitted binary Logistic Regression coefficient vector.")
    if not np.array_equal(np.asarray(estimator.classes_), np.array([0, 1])):
        raise ValueError("The Logistic Regression model must use classes [0, 1].")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")

    X = data.X_train.to_numpy(dtype=float)
    y = data.y_train.to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training target must contain both binary classes.")

    fitted_probabilities = estimator.predict_proba(data.X_train)[:, 1]
    design_matrix = np.column_stack((np.ones(len(X)), X))
    weights = fitted_probabilities * (1 - fitted_probabilities)
    information = design_matrix.T @ (design_matrix * weights[:, np.newaxis])

    # scikit-learn's L2 penalty is inverse to C and does not penalize the intercept.
    regularization = np.zeros(information.shape[0])
    regularization[1:] = 1 / estimator.C
    penalized_information = information + np.diag(regularization)
    covariance = np.linalg.pinv(penalized_information, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance)[1:], 0))
    coefficients = estimator.coef_[0]
    if not np.isfinite(standard_errors).all() or (standard_errors <= 0).any():
        raise ValueError(
            "Could not estimate finite positive coefficient standard errors "
            "from the fitted model's information matrix."
        )

    z_critical = norm.ppf((1 + confidence_level) / 2)
    ci_lower = coefficients - z_critical * standard_errors
    ci_upper = coefficients + z_critical * standard_errors
    odds_ratios = np.exp(coefficients)
    odds_ratio_lower = np.exp(ci_lower)
    odds_ratio_upper = np.exp(ci_upper)

    rows = []
    for feature_name, coefficient, standard_error, low, high, odds_ratio, or_low, or_high in zip(
        model.feature_names,
        coefficients,
        standard_errors,
        ci_lower,
        ci_upper,
        odds_ratios,
        odds_ratio_lower,
        odds_ratio_upper,
    ):
        if feature_name.startswith("numeric__"):
            raw_feature = feature_name.removeprefix("numeric__")
            category_value = None
            is_numeric = raw_feature in NUMERICAL_FEATURES
        else:
            category_feature = next(
                (
                    feature
                    for feature in CATEGORICAL_FEATURES
                    if feature_name.startswith(f"{feature}_")
                ),
                None,
            )
            if category_feature is None:
                raise ValueError(
                    f"Cannot map encoded feature name '{feature_name}' to a source feature."
                )
            raw_feature = category_feature
            category_value = feature_name[len(category_feature) + 1 :]
            is_numeric = False
        if is_numeric:
            contrast = "per 1 standard-deviation increase"
            note = (
                "Numerical predictors were standardized; the odds ratio is for "
                "a one-standard-deviation increase, conditional on other encoded predictors."
            )
        else:
            contrast = f"indicator for category {category_value}"
            note = (
                "Categorical variables use full one-hot encoding without a dropped "
                "reference level; interpret this coefficient as a regularized "
                "encoded-coordinate contrast, not a direct category-versus-reference effect."
            )
        rows.append(
            {
                "encoded_feature": feature_name,
                "feature": raw_feature,
                "contrast": contrast,
                "coefficient": coefficient,
                "standard_error": standard_error,
                "coefficient_ci_lower": low,
                "coefficient_ci_upper": high,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_lower": or_low,
                "odds_ratio_ci_upper": or_high,
                "ci_excludes_or_1": bool(or_low > 1 or or_high < 1),
                "interpretation_note": note,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "coefficient", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


def _build_interpretations(
    coefficient_summary: pd.DataFrame,
    confidence_level: float,
    evaluation: Optional[ModelEvaluation],
    auc_comparison: Optional[AUCComparison],
) -> list[str]:
    """Write cautious interpretations of model associations and test findings."""
    interpretations = [
        (
            "The Logistic Regression coefficients describe conditional associations "
            "with the model's estimated odds of num > 0, holding the other encoded "
            "predictors fixed. These are associations, not causal or clinical effects."
        ),
        (
            f"Reported odds-ratio intervals are approximate {confidence_level:.0%} "
            "penalized-Wald intervals based on the training-set observed information "
            "with the model's L2 penalty included. They are conditional on this "
            "fitted model and do not account for model selection, data-source "
            "heterogeneity, or uncertainty from preprocessing."
        ),
        (
            "Numerical features were standardized, so their odds ratios correspond "
            "to a one-standard-deviation increase rather than a one-unit increase "
            "in the original measurement."
        ),
        (
            "Categorical features were fully one-hot encoded without a dropped "
            "reference category. Their individual coefficients are regularized "
            "encoded-coordinate effects and should not be read as simple "
            "category-versus-reference comparisons."
        ),
    ]

    interval_associations = coefficient_summary.loc[
        coefficient_summary["ci_excludes_or_1"]
    ]
    if interval_associations.empty:
        interpretations.append(
            f"None of the approximate {confidence_level:.0%} odds-ratio intervals "
            "exclude 1. This is descriptive coefficient uncertainty, not a "
            "multiplicity-adjusted hypothesis test."
        )
    else:
        interpretations.append(
            f"{len(interval_associations)} encoded coefficient interval(s) exclude "
            "an odds ratio of 1. This is an exploratory model-based observation, "
            "not a multiplicity-adjusted test or evidence of causation."
        )
        for _, row in interval_associations.iterrows():
            interpretations.append(
                f"{row['encoded_feature']}: odds ratio "
                f"{row['odds_ratio']:.3f} ({confidence_level:.0%} approximate "
                f"interval {row['odds_ratio_ci_lower']:.3f} to "
                f"{row['odds_ratio_ci_upper']:.3f}). "
                f"{row['interpretation_note']}"
            )

    if not coefficient_summary.empty:
        strongest = coefficient_summary.iloc[0]
        direction = (
            "higher"
            if strongest["coefficient"] > 0
            else "lower"
        )
        interpretations.append(
            f"The largest absolute fitted coefficient is "
            f"'{strongest['encoded_feature']}' "
            f"(odds ratio {strongest['odds_ratio']:.3f}; "
            f"{confidence_level:.0%} interval "
            f"{strongest['odds_ratio_ci_lower']:.3f} to "
            f"{strongest['odds_ratio_ci_upper']:.3f}), corresponding to "
            f"{direction} model-estimated odds for its specified encoded contrast. "
            "Coefficient magnitude alone does not establish practical or clinical importance."
        )

    if evaluation is not None:
        required_models = {"Logistic Regression", "Random Forest"}
        missing_models = sorted(required_models.difference(evaluation.metrics.index))
        if missing_models:
            raise ValueError(f"Evaluation summary is missing models: {missing_models}")
        logistic_auc = float(evaluation.metrics.loc["Logistic Regression", "roc_auc"])
        forest_auc = float(evaluation.metrics.loc["Random Forest", "roc_auc"])
        interpretations.append(
            f"On the shared held-out test set, Logistic Regression ROC-AUC was "
            f"{logistic_auc:.4f} and Random Forest ROC-AUC was {forest_auc:.4f}."
        )

    if auc_comparison is not None:
        interpretations.append(
            f"The paired ROC-AUC comparison estimated Logistic Regression minus "
            f"Random Forest AUC as {auc_comparison.auc_difference:.4f} "
            f"(95% bootstrap CI {auc_comparison.confidence_interval[0]:.4f} to "
            f"{auc_comparison.confidence_interval[1]:.4f}; "
            f"p={auc_comparison.p_value:.4f}). Decision: "
            f"{auc_comparison.decision}. Failure to reject the null, if applicable, "
            "does not prove equal model performance."
        )
    return interpretations


def interpret_logistic_regression(
    model: TrainedModelArtifact,
    data: PreprocessedData,
    evaluation: Optional[ModelEvaluation] = None,
    auc_comparison: Optional[AUCComparison] = None,
    result_dir: Optional[Union[str, Path]] = None,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> StatisticalInterpretation:
    """Calculate Logistic Regression odds ratios and summarize project findings."""
    coefficient_summary = _wald_coefficient_summary(
        model=model,
        data=data,
        confidence_level=confidence_level,
    )
    interpretations = _build_interpretations(
        coefficient_summary,
        confidence_level,
        evaluation,
        auc_comparison,
    )

    result_directory = (
        Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIRECTORY
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    coefficient_path = result_directory / "logistic_regression_odds_ratios.csv"
    interpretation_path = result_directory / "statistical_interpretation.txt"
    coefficient_summary.to_csv(coefficient_path, index=False)
    interpretation_path.write_text(
        "\n".join(f"- {text}" for text in interpretations) + "\n",
        encoding="utf-8",
    )

    print("Logistic Regression coefficients, odds ratios, and confidence intervals:")
    print(
        coefficient_summary[
            [
                "encoded_feature",
                "contrast",
                "coefficient",
                "odds_ratio",
                "odds_ratio_ci_lower",
                "odds_ratio_ci_upper",
                "ci_excludes_or_1",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nStatistical interpretation:")
    for interpretation in interpretations:
        print(f"- {interpretation}")
    print(f"\nOdds-ratio table saved to: {coefficient_path}")
    print(f"Interpretation saved to: {interpretation_path}")

    return StatisticalInterpretation(
        coefficient_summary=coefficient_summary,
        interpretations=interpretations,
    )


if __name__ == "__main__":
    from src.data_loading import load_heart_disease_data
    from src.hypothesis_testing import test_model_auc_difference
    from src.model_evaluation import evaluate_models
    from src.model_training import train_models
    from src.data_preprocessing import preprocess_heart_disease_data

    raw_data = load_heart_disease_data()
    preprocessed_data = preprocess_heart_disease_data(raw_data)
    trained_models = train_models(preprocessed_data)
    evaluation_results = evaluate_models(preprocessed_data, trained_models)
    auc_results = test_model_auc_difference(preprocessed_data, evaluation_results)
    interpret_logistic_regression(
        model=trained_models["Logistic Regression"],
        data=preprocessed_data,
        evaluation=evaluation_results,
        auc_comparison=auc_results,
    )
