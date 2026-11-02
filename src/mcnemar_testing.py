"""Paired McNemar test of classification errors on the shared test set."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.data_preprocessing import PreprocessedData, preprocess_heart_disease_data
from src.model_evaluation import ModelEvaluation, evaluate_models
from src.model_training import TrainedModelArtifact, train_models


ALPHA = 0.05
EXACT_DISCORDANT_THRESHOLD = 25
DEFAULT_RESULT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "statistical_results"
)


@dataclass
class McNemarResult:
    """Paired classification-error contingency table and McNemar test result."""

    contingency_table: pd.DataFrame
    discordant_pairs: int
    test_statistic: float
    p_value: float
    alpha: float
    method: str
    decision: str
    interpretation: str


def _validate_predictions(
    y_test: Union[pd.Series, np.ndarray],
    y_pred_lr: Union[pd.Series, np.ndarray],
    y_pred_rf: Union[pd.Series, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate labels and ensure predictions refer to the same test records."""
    inputs = (y_test, y_pred_lr, y_pred_rf)
    if all(isinstance(value, pd.Series) for value in inputs):
        indices = [value.index for value in inputs]
        if not indices[0].equals(indices[1]) or not indices[0].equals(indices[2]):
            raise ValueError(
                "y_test, y_pred_lr, and y_pred_rf must have identical indices "
                "and ordering."
            )

    arrays = tuple(np.asarray(value) for value in inputs)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("y_test and both prediction arrays must be one-dimensional.")
    if not arrays[0].size or not (
        len(arrays[0]) == len(arrays[1]) == len(arrays[2])
    ):
        raise ValueError(
            "y_test and both prediction arrays must have the same non-zero length."
        )
    if not np.isin(arrays[0], [0, 1]).all():
        raise ValueError("y_test must contain only binary labels 0 and 1.")
    if not np.isin(arrays[1], [0, 1]).all():
        raise ValueError("Logistic Regression predictions must contain only 0 and 1.")
    if not np.isin(arrays[2], [0, 1]).all():
        raise ValueError("Random Forest predictions must contain only 0 and 1.")
    return tuple(array.astype(int) for array in arrays)


def run_mcnemar_test(
    y_test: Union[pd.Series, np.ndarray],
    y_pred_lr: Union[pd.Series, np.ndarray],
    y_pred_rf: Union[pd.Series, np.ndarray],
    alpha: float = ALPHA,
    result_dir: Optional[Union[str, Path]] = None,
) -> McNemarResult:
    """Compare model error probabilities using paired correctness outcomes.

    Table rows indicate Logistic Regression correctness and columns indicate
    Random Forest correctness. The exact binomial version is used when there
    are at most 25 discordant pairs; otherwise statsmodels' continuity-corrected
    asymptotic chi-square version is used.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be greater than 0 and less than 1.")
    y_true, logistic_predictions, forest_predictions = _validate_predictions(
        y_test, y_pred_lr, y_pred_rf
    )

    logistic_correct = logistic_predictions == y_true
    forest_correct = forest_predictions == y_true
    table_values = np.array(
        [
            [
                np.count_nonzero(logistic_correct & forest_correct),
                np.count_nonzero(logistic_correct & ~forest_correct),
            ],
            [
                np.count_nonzero(~logistic_correct & forest_correct),
                np.count_nonzero(~logistic_correct & ~forest_correct),
            ],
        ],
        dtype=int,
    )
    discordant_pairs = int(table_values[0, 1] + table_values[1, 0])
    exact = discordant_pairs <= EXACT_DISCORDANT_THRESHOLD
    test_result = mcnemar(
        table_values,
        exact=exact,
        correction=not exact,
    )
    method = (
        "Exact binomial McNemar test"
        if exact
        else "Continuity-corrected asymptotic McNemar chi-square test"
    )

    p_value = float(test_result.pvalue)
    decision = "Reject H0" if p_value < alpha else "Fail to reject H0"
    if decision == "Reject H0":
        interpretation = (
            f"At alpha={alpha:.3f}, the paired test provides evidence that "
            "Logistic Regression and Random Forest have different classification "
            "error probabilities on this held-out test sample."
        )
    else:
        interpretation = (
            f"At alpha={alpha:.3f}, the paired test does not provide sufficient "
            "evidence that the models have different classification error "
            "probabilities on this held-out test sample. This is not proof that "
            "their error probabilities are equal."
        )

    contingency_table = pd.DataFrame(
        table_values,
        index=["Logistic Regression correct", "Logistic Regression incorrect"],
        columns=["Random Forest correct", "Random Forest incorrect"],
    )
    result = McNemarResult(
        contingency_table=contingency_table,
        discordant_pairs=discordant_pairs,
        test_statistic=float(test_result.statistic),
        p_value=p_value,
        alpha=alpha,
        method=method,
        decision=decision,
        interpretation=interpretation,
    )

    output_directory = (
        Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIRECTORY
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    contingency_table.to_csv(output_directory / "mcnemar_contingency_table.csv")
    summary = pd.DataFrame(
        [
            {
                "both_correct": int(table_values[0, 0]),
                "logistic_correct_forest_incorrect": int(table_values[0, 1]),
                "logistic_incorrect_forest_correct": int(table_values[1, 0]),
                "both_incorrect": int(table_values[1, 1]),
                "discordant_pairs": discordant_pairs,
                "test_statistic": result.test_statistic,
                "p_value": p_value,
                "alpha": alpha,
                "method": method,
                "decision": decision,
                "test_records": len(y_true),
            }
        ]
    )
    summary.to_csv(output_directory / "mcnemar_test_results.csv", index=False)
    (output_directory / "mcnemar_test_interpretation.txt").write_text(
        "\n".join(
            [
                "H0: Both models have the same probability of classification error.",
                "H1: Both models have different probabilities of classification error.",
                "Contingency table rows: Logistic Regression correctness.",
                "Contingency table columns: Random Forest correctness.",
                f"Discordant pairs: {discordant_pairs}",
                f"Test statistic: {result.test_statistic:.6f}",
                f"P-value: {p_value:.6f}",
                f"Alpha: {alpha:.3f}",
                f"Method: {method}",
                f"Decision: {decision}",
                f"Interpretation: {interpretation}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Paired classification correctness contingency table:")
    print(contingency_table.to_string())
    print(f"\nDiscordant pairs: {discordant_pairs}")
    print(f"Test statistic: {result.test_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance level: alpha={alpha:.3f}")
    print(f"Test used: {method}")
    print(f"Decision: {decision}")
    print(f"Interpretation: {interpretation}")
    print(f"Results saved to: {output_directory}")
    return result


def test_evaluated_models(
    data: PreprocessedData,
    evaluation: ModelEvaluation,
    alpha: float = ALPHA,
    result_dir: Optional[Union[str, Path]] = None,
) -> McNemarResult:
    """Apply McNemar's test to predictions returned by existing evaluation."""
    required_models = {"Logistic Regression", "Random Forest"}
    missing_models = sorted(required_models.difference(evaluation.predictions))
    if missing_models:
        raise ValueError(f"Evaluation is missing predictions for: {missing_models}")
    if not data.y_test.index.equals(data.X_test.index):
        raise ValueError("y_test must align with the held-out feature rows.")
    return run_mcnemar_test(
        y_test=data.y_test,
        y_pred_lr=evaluation.predictions["Logistic Regression"],
        y_pred_rf=evaluation.predictions["Random Forest"],
        alpha=alpha,
        result_dir=result_dir,
    )


if __name__ == "__main__":
    from src.data_loading import load_heart_disease_data

    preprocessed_data = preprocess_heart_disease_data(load_heart_disease_data())
    saved_models = train_models(preprocessed_data)
    test_evaluation = evaluate_models(preprocessed_data, saved_models)
    test_evaluated_models(preprocessed_data, test_evaluation)
