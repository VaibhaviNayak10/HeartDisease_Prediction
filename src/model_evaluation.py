"""Evaluate the trained classifiers on their shared held-out test data."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.data_preprocessing import PreprocessedData, preprocess_heart_disease_data
from src.model_training import TrainedModelArtifact, train_models


DEFAULT_RESULT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "results" / "statistical_results"
)
DEFAULT_PLOT_DIRECTORY = Path(__file__).resolve().parents[1] / "results" / "plots"
MODEL_NAMES = ("Logistic Regression", "Random Forest")


@dataclass
class ModelEvaluation:
    """Test metrics and outputs for each evaluated classifier."""

    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    confusion_matrices: dict[str, np.ndarray]


def _validate_test_data(data: PreprocessedData) -> None:
    """Ensure the held-out predictors and target can support all requested metrics."""
    if data.X_test.empty or data.y_test.empty:
        raise ValueError("Test data is empty.")
    if len(data.X_test) != len(data.y_test):
        raise ValueError("X_test and y_test must contain the same number of rows.")
    if not data.X_test.index.equals(data.y_test.index):
        raise ValueError("X_test and y_test indices must be aligned.")
    if data.X_test.isna().any().any():
        raise ValueError("X_test still contains missing values.")
    if set(data.y_test.unique()) != {0, 1}:
        raise ValueError(
            "ROC-AUC evaluation requires both binary target classes in y_test."
        )


def _positive_class_probability(artifact: TrainedModelArtifact, X_test: pd.DataFrame) -> np.ndarray:
    """Return probabilities associated specifically with target class 1."""
    if tuple(map(str, X_test.columns)) != artifact.feature_names:
        raise ValueError(
            f"Test feature columns do not match the saved {artifact.name} model."
        )
    classes = np.asarray(artifact.estimator.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError(
            f"{artifact.name} does not expose exactly one probability column "
            "for positive class 1."
        )
    predict_proba = getattr(artifact.estimator, "predict_proba", None)
    if predict_proba is None:
        raise TypeError(f"{artifact.name} does not provide class probabilities.")
    probabilities = predict_proba(X_test)
    return probabilities[:, int(positive_positions[0])]


def _plot_confusion_matrices(
    matrices: dict[str, np.ndarray], output_path: Path
) -> None:
    """Save side-by-side confusion matrices with fixed negative/positive order."""
    figure, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(11, 4.5))
    for axis, name in zip(axes, MODEL_NAMES):
        matrix = matrices[name]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks([0, 1], ["No disease (0)", "Disease (1)"], rotation=15)
        axis.set_yticks([0, 1], ["No disease (0)", "Disease (1)"])
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Confusion Matrices on the Shared Test Set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_roc_curves(
    targets: pd.Series,
    probabilities: dict[str, np.ndarray],
    auc_values: dict[str, float],
    output_path: Path,
) -> None:
    """Save ROC curves for all models on the same held-out observations."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for name in MODEL_NAMES:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            targets, probabilities[name], pos_label=1
        )
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{name} (AUC = {auc_values[name]:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC Curves on the Shared Test Set")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def evaluate_models(
    data: PreprocessedData,
    models: dict[str, TrainedModelArtifact],
    result_dir: Optional[Union[str, Path]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
) -> ModelEvaluation:
    """Calculate standard classification metrics and save tables and plots.

    Both models are evaluated on the exact same preprocessed ``X_test`` rows.
    Test targets are used only for evaluation, never for model fitting.
    """
    _validate_test_data(data)
    missing_models = sorted(set(MODEL_NAMES).difference(models))
    if missing_models:
        raise ValueError(f"Missing trained models for evaluation: {missing_models}")

    truth = data.y_test.to_numpy()
    metric_rows = []
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    matrices: dict[str, np.ndarray] = {}
    auc_values: dict[str, float] = {}

    for name in MODEL_NAMES:
        artifact = models[name]
        if not isinstance(artifact, TrainedModelArtifact):
            raise TypeError(f"Model '{name}' is not a TrainedModelArtifact.")
        if artifact.name != name:
            raise ValueError(
                f"Model dictionary key '{name}' does not match artifact "
                f"name '{artifact.name}'."
            )

        model_prediction = np.asarray(artifact.estimator.predict(data.X_test))
        positive_probability = _positive_class_probability(artifact, data.X_test)
        if len(model_prediction) != len(truth) or len(positive_probability) != len(truth):
            raise ValueError(f"{name} returned predictions with an invalid length.")
        if not np.isfinite(positive_probability).all():
            raise ValueError(f"{name} returned non-finite class probabilities.")

        model_matrix = confusion_matrix(truth, model_prediction, labels=[0, 1])
        model_auc = float(auc(*roc_curve(truth, positive_probability, pos_label=1)[:2]))
        predictions[name] = model_prediction
        probabilities[name] = positive_probability
        matrices[name] = model_matrix
        auc_values[name] = model_auc
        metric_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(truth, model_prediction),
                "precision": precision_score(
                    truth, model_prediction, pos_label=1, zero_division=0
                ),
                "recall": recall_score(
                    truth, model_prediction, pos_label=1, zero_division=0
                ),
                "f1_score": f1_score(
                    truth, model_prediction, pos_label=1, zero_division=0
                ),
                "roc_auc": model_auc,
                "test_records": len(truth),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("model")
    output_directory = Path(result_dir) if result_dir is not None else DEFAULT_RESULT_DIRECTORY
    plot_directory = Path(plot_dir) if plot_dir is not None else DEFAULT_PLOT_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_directory.mkdir(parents=True, exist_ok=True)

    metrics_path = output_directory / "model_evaluation_metrics.csv"
    metrics.to_csv(metrics_path)
    for name, matrix in matrices.items():
        safe_name = name.lower().replace(" ", "_")
        pd.DataFrame(
            matrix,
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(output_directory / f"{safe_name}_confusion_matrix.csv")

    confusion_plot_path = plot_directory / "model_confusion_matrices.png"
    roc_plot_path = plot_directory / "model_roc_curves.png"
    _plot_confusion_matrices(matrices, confusion_plot_path)
    _plot_roc_curves(data.y_test, probabilities, auc_values, roc_plot_path)

    print("Held-out test metrics:")
    print(metrics.round(4).to_string())
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrices saved to: {confusion_plot_path}")
    print(f"ROC curves saved to: {roc_plot_path}")

    return ModelEvaluation(
        metrics=metrics,
        predictions=predictions,
        probabilities=probabilities,
        confusion_matrices=matrices,
    )


if __name__ == "__main__":
    preprocessed_data = preprocess_heart_disease_data()
    trained_models = train_models(preprocessed_data)
    evaluate_models(preprocessed_data, trained_models)
