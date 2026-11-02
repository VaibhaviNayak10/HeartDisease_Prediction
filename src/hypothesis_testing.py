"""Paired bootstrap comparison of the classifiers' ROC-AUC values."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import PreprocessedData, preprocess_heart_disease_data
from src.model_evaluation import ModelEvaluation, evaluate_models
from src.model_training import train_models


ALPHA = 0.05
CONFIDENCE_LEVEL = 0.95
N_BOOTSTRAP = 10_000
RANDOM_STATE = 42
RESULT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "statistical_results"
)
DIFFERENCE_DEFINITION = "Logistic Regression AUC - Random Forest AUC"


@dataclass
class AUCComparison:
    """Paired ROC-AUC estimates, bootstrap interval, and test conclusion."""

    logistic_regression_auc: float
    random_forest_auc: float
    auc_difference: float
    confidence_interval: tuple[float, float]
    p_value: float
    alpha: float
    decision: str
    interpretation: str
    bootstrap_differences: np.ndarray


def _validate_comparison_inputs(
    targets: Union[pd.Series, np.ndarray],
    probabilities: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check that paired probabilities align with a binary test target."""
    required_models = {"Logistic Regression", "Random Forest"}
    missing_models = sorted(required_models.difference(probabilities))
    if missing_models:
        raise ValueError(f"Missing probability outputs for models: {missing_models}")

    y_true = np.asarray(targets)
    logistic_scores = np.asarray(probabilities["Logistic Regression"], dtype=float)
    forest_scores = np.asarray(probabilities["Random Forest"], dtype=float)
    if y_true.ndim != 1:
        raise ValueError("targets must be one-dimensional.")
    if logistic_scores.ndim != 1 or forest_scores.ndim != 1:
        raise ValueError("Each model's positive-class probabilities must be one-dimensional.")
    if not (len(y_true) == len(logistic_scores) == len(forest_scores)):
        raise ValueError(
            "Targets and both model probability arrays must have equal lengths "
            "and refer to the same test observations."
        )
    if len(y_true) == 0 or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("AUC comparison requires observations from both target classes.")
    if not np.isfinite(logistic_scores).all() or not np.isfinite(forest_scores).all():
        raise ValueError("Model probabilities must all be finite.")
    if (
        (logistic_scores < 0).any()
        or (logistic_scores > 1).any()
        or (forest_scores < 0).any()
        or (forest_scores > 1).any()
    ):
        raise ValueError("Model probabilities must lie between 0 and 1.")
    return y_true.astype(int), logistic_scores, forest_scores


def _paired_stratified_bootstrap_differences(
    targets: np.ndarray,
    logistic_scores: np.ndarray,
    forest_scores: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> np.ndarray:
    """Resample the same records within each class for both models."""
    rng = np.random.default_rng(random_state)
    class_indices = {
        label: np.flatnonzero(targets == label) for label in (0, 1)
    }
    bootstrap_differences = np.empty(n_bootstrap, dtype=float)

    for replicate in range(n_bootstrap):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices.values()
            ]
        )
        sampled_targets = targets[sampled_indices]
        logistic_auc = roc_auc_score(
            sampled_targets, logistic_scores[sampled_indices]
        )
        forest_auc = roc_auc_score(
            sampled_targets, forest_scores[sampled_indices]
        )
        bootstrap_differences[replicate] = logistic_auc - forest_auc
    return bootstrap_differences


def compare_model_auc(
    targets: Union[pd.Series, np.ndarray],
    probabilities: dict[str, np.ndarray],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
    alpha: float = ALPHA,
    random_state: int = RANDOM_STATE,
    output_dir: Optional[Union[str, Path]] = None,
) -> AUCComparison:
    """Test equality of paired model AUCs using a stratified paired bootstrap.

    The reported difference is Logistic Regression AUC minus Random Forest
    AUC. The confidence interval is percentile-based. For the two-sided
    bootstrap p-value, bootstrap deviations are centered at the observed
    difference to approximate the null distribution at a zero difference.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be greater than 0 and less than 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be greater than 0 and less than 1.")

    y_true, logistic_scores, forest_scores = _validate_comparison_inputs(
        targets, probabilities
    )
    logistic_auc = float(roc_auc_score(y_true, logistic_scores))
    forest_auc = float(roc_auc_score(y_true, forest_scores))
    observed_difference = logistic_auc - forest_auc

    bootstrap_differences = _paired_stratified_bootstrap_differences(
        y_true,
        logistic_scores,
        forest_scores,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    tail_probability = (1 - confidence_level) / 2
    ci_low, ci_high = np.quantile(
        bootstrap_differences, [tail_probability, 1 - tail_probability]
    )

    null_centered_errors = bootstrap_differences - observed_difference
    extreme_count = np.count_nonzero(
        np.abs(null_centered_errors) >= abs(observed_difference)
    )
    p_value = float((extreme_count + 1) / (n_bootstrap + 1))
    decision = (
        "Reject H0"
        if p_value < alpha
        else "Fail to reject H0"
    )
    if decision == "Reject H0":
        interpretation = (
            f"At alpha={alpha:.3f}, the paired bootstrap provides evidence "
            "that the models' ROC-AUC values differ."
        )
    else:
        interpretation = (
            f"At alpha={alpha:.3f}, the paired bootstrap does not provide "
            "sufficient evidence that the models' ROC-AUC values differ. "
            "This is not proof that the models have equal AUC."
        )

    result = AUCComparison(
        logistic_regression_auc=logistic_auc,
        random_forest_auc=forest_auc,
        auc_difference=float(observed_difference),
        confidence_interval=(float(ci_low), float(ci_high)),
        p_value=p_value,
        alpha=alpha,
        decision=decision,
        interpretation=interpretation,
        bootstrap_differences=bootstrap_differences,
    )

    result_directory = Path(output_dir) if output_dir is not None else RESULT_DIRECTORY
    result_directory.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "comparison": DIFFERENCE_DEFINITION,
                "logistic_regression_auc": logistic_auc,
                "random_forest_auc": forest_auc,
                "auc_difference": observed_difference,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "confidence_level": confidence_level,
                "p_value": p_value,
                "alpha": alpha,
                "decision": decision,
                "n_test_records": len(y_true),
                "n_bootstrap": n_bootstrap,
                "random_state": random_state,
                "method": "Paired stratified percentile bootstrap; null-centered two-sided test",
            }
        ]
    )
    summary.to_csv(result_directory / "paired_auc_bootstrap_test.csv", index=False)
    (result_directory / "paired_auc_bootstrap_interpretation.txt").write_text(
        "\n".join(
            [
                "H0: Logistic Regression AUC = Random Forest AUC",
                "H1: Logistic Regression AUC != Random Forest AUC",
                f"Difference definition: {DIFFERENCE_DEFINITION}",
                f"Logistic Regression AUC: {logistic_auc:.6f}",
                f"Random Forest AUC: {forest_auc:.6f}",
                f"AUC difference: {observed_difference:.6f}",
                (
                    f"{confidence_level:.0%} bootstrap confidence interval: "
                    f"[{ci_low:.6f}, {ci_high:.6f}]"
                ),
                f"Two-sided p-value: {p_value:.6f}",
                f"Alpha: {alpha:.3f}",
                f"Decision: {decision}",
                f"Interpretation: {interpretation}",
                (
                    "Method: paired stratified bootstrap, resampling identical "
                    "test observations for both models within target class; "
                    "percentile interval and null-centered two-sided p-value."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Paired bootstrap comparison of test ROC-AUC:")
    print(f"  H0: Logistic Regression AUC = Random Forest AUC")
    print(f"  H1: Logistic Regression AUC != Random Forest AUC")
    print(f"  Logistic Regression AUC: {logistic_auc:.4f}")
    print(f"  Random Forest AUC:       {forest_auc:.4f}")
    print(f"  Difference (Logistic - Random Forest): {observed_difference:.4f}")
    print(
        f"  {confidence_level:.0%} bootstrap CI for difference: "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )
    print(f"  Two-sided p-value: {p_value:.4f}")
    print(f"  Decision at alpha={alpha:.3f}: {decision}")
    print(f"  Interpretation: {interpretation}")
    print(f"  Results saved to: {result_directory}")
    return result


def test_model_auc_difference(
    data: PreprocessedData,
    evaluation: ModelEvaluation,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> AUCComparison:
    """Run the paired test using outputs from the shared model evaluation."""
    if len(data.y_test) != len(evaluation.probabilities["Logistic Regression"]):
        raise ValueError("Evaluation probabilities do not align with y_test.")
    if not data.y_test.index.equals(data.X_test.index):
        raise ValueError("y_test must align with the held-out feature rows.")
    return compare_model_auc(
        targets=data.y_test,
        probabilities=evaluation.probabilities,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )


if __name__ == "__main__":
    from src.data_loading import load_heart_disease_data

    preprocessed_data = preprocess_heart_disease_data(load_heart_disease_data())
    trained_models = train_models(preprocessed_data)
    evaluation_results = evaluate_models(preprocessed_data, trained_models)
    test_model_auc_difference(preprocessed_data, evaluation_results)
