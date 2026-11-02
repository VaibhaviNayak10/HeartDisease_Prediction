"""Train and persist heart disease classifiers and their fitted preprocessing."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import (
    RANDOM_STATE as PREPROCESSING_RANDOM_STATE,
    HospitalPreprocessor,
    PreprocessedData,
    preprocess_heart_disease_data,
)


DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parents[1] / "saved_models"
ARTIFACT_FORMAT_VERSION = 1
MODEL_CONFIGURATIONS = {
    "Logistic Regression": {
        "filename": "logistic_regression.joblib",
        "estimator_type": LogisticRegression,
    },
    "Random Forest": {
        "filename": "random_forest.joblib",
        "estimator_type": RandomForestClassifier,
    },
}


@dataclass
class TrainedModelArtifact:
    """A fitted classifier bundled with the matching preprocessing transformer."""

    name: str
    estimator: BaseEstimator
    preprocessor: HospitalPreprocessor
    feature_names: tuple[str, ...]
    training_data_signature: str
    random_state: int

    def predict(self, X: pd.DataFrame, source: pd.Series) -> np.ndarray:
        """Predict labels from raw feature values and separately supplied sources."""
        transformed = self.preprocessor.transform(X, source)
        return self.estimator.predict(transformed)

    def predict_proba(self, X: pd.DataFrame, source: pd.Series) -> np.ndarray:
        """Predict class probabilities from raw feature values and sources."""
        transformed = self.preprocessor.transform(X, source)
        predict_proba = getattr(self.estimator, "predict_proba", None)
        if predict_proba is None:
            raise TypeError(f"{self.name} does not provide class probabilities.")
        return predict_proba(transformed)


def _training_data_signature(data: PreprocessedData) -> str:
    """Fingerprint transformed training predictors and labels for cache checks."""
    digest = sha256()
    digest.update(pd.util.hash_pandas_object(data.X_train, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(data.y_train, index=True).values.tobytes())
    digest.update("\0".join(map(str, data.X_train.columns)).encode("utf-8"))
    return digest.hexdigest()


def _validate_artifact(
    artifact: object,
    name: str,
    expected_estimator_type: type[BaseEstimator],
    feature_names: tuple[str, ...],
    training_signature: str,
    path: Path,
) -> TrainedModelArtifact:
    """Ensure a cached model matches the requested model and training data."""
    if not isinstance(artifact, dict) or artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Saved artifact at '{path}' is not a recognized model artifact. "
            "Use retrain=True to replace it."
        )
    required_keys = {
        "name",
        "estimator",
        "preprocessor",
        "feature_names",
        "training_data_signature",
        "random_state",
    }
    if not required_keys.issubset(artifact):
        raise ValueError(
            f"Saved artifact at '{path}' is incomplete. Use retrain=True to replace it."
        )
    if not isinstance(artifact["estimator"], expected_estimator_type):
        raise ValueError(
            f"Saved artifact at '{path}' does not contain {name}. "
            "Use retrain=True to replace it."
        )
    if artifact["name"] != name:
        raise ValueError(
            f"Saved artifact at '{path}' has an inconsistent model name. "
            "Use retrain=True to replace it."
        )
    if not isinstance(artifact["preprocessor"], HospitalPreprocessor):
        raise ValueError(
            f"Saved artifact at '{path}' does not contain the expected fitted "
            "preprocessor. Use retrain=True to replace it."
        )
    if tuple(artifact["feature_names"]) != feature_names:
        raise ValueError(
            f"Saved {name} has different feature columns than the current data. "
            "Use retrain=True to replace it."
        )
    if artifact["training_data_signature"] != training_signature:
        raise ValueError(
            f"Saved {name} was trained on different training data. "
            "Use retrain=True to train on the current split."
        )
    if (
        not hasattr(artifact["estimator"], "classes_")
        or getattr(artifact["estimator"], "n_features_in_", None)
        != len(feature_names)
    ):
        raise ValueError(
            f"Saved {name} estimator is not fitted for the expected features. "
            "Use retrain=True to replace it."
        )
    return TrainedModelArtifact(
        name=artifact["name"],
        estimator=artifact["estimator"],
        preprocessor=artifact["preprocessor"],
        feature_names=tuple(artifact["feature_names"]),
        training_data_signature=artifact["training_data_signature"],
        random_state=artifact["random_state"],
    )


def _create_estimator(name: str, random_state: int) -> BaseEstimator:
    """Create one classifier with explicit reproducible settings."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model name: {name}")


def train_models(
    data: PreprocessedData,
    model_dir: Optional[Union[str, Path]] = None,
    random_state: int = PREPROCESSING_RANDOM_STATE,
    retrain: bool = False,
) -> dict[str, TrainedModelArtifact]:
    """Train or load both classifiers using one shared train/test split.

    Only ``X_train`` and ``y_train`` are used for fitting. A valid matching
    saved artifact is reused unless ``retrain`` is true. Each saved artifact
    includes the fitted preprocessor so new raw records can be transformed
    consistently by calling its ``predict`` or ``predict_proba`` method.
    """
    if data.X_train.empty or data.y_train.empty:
        raise ValueError("Training data is empty.")
    if len(data.X_train) != len(data.y_train):
        raise ValueError("X_train and y_train must contain the same number of rows.")
    if not data.X_train.index.equals(data.y_train.index):
        raise ValueError("X_train and y_train indices must be aligned.")
    if data.X_train.isna().any().any():
        raise ValueError("X_train still contains missing values.")
    if data.y_train.nunique() != 2:
        raise ValueError("Training target must contain both binary classes.")

    feature_names = tuple(map(str, data.X_train.columns))
    training_signature = _training_data_signature(data)
    output_directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIRECTORY
    output_directory.mkdir(parents=True, exist_ok=True)

    trained_models: dict[str, TrainedModelArtifact] = {}
    for name, configuration in MODEL_CONFIGURATIONS.items():
        artifact_path = output_directory / configuration["filename"]
        estimator_type = configuration["estimator_type"]

        if artifact_path.is_file() and not retrain:
            artifact = joblib.load(artifact_path)
            validated_artifact = _validate_artifact(
                artifact=artifact,
                name=name,
                expected_estimator_type=estimator_type,
                feature_names=feature_names,
                training_signature=training_signature,
                path=artifact_path,
            )
            trained_models[name] = validated_artifact
            print(f"Reused saved {name} model: {artifact_path}")
            continue

        estimator = _create_estimator(name, random_state)
        estimator.fit(data.X_train, data.y_train)
        artifact = TrainedModelArtifact(
            name=name,
            estimator=estimator,
            preprocessor=data.preprocessor,
            feature_names=feature_names,
            training_data_signature=training_signature,
            random_state=random_state,
        )
        joblib.dump(
            {
                "format_version": ARTIFACT_FORMAT_VERSION,
                "name": artifact.name,
                "estimator": artifact.estimator,
                "preprocessor": artifact.preprocessor,
                "feature_names": artifact.feature_names,
                "training_data_signature": artifact.training_data_signature,
                "random_state": artifact.random_state,
            },
            artifact_path,
        )
        trained_models[name] = artifact
        print(f"Trained and saved {name} model: {artifact_path}")

    print(
        f"\nModels were fitted using {len(data.X_train)} training records "
        f"and {len(feature_names)} encoded features."
    )
    print(
        f"The separate test split ({len(data.X_test)} records) was not used "
        "during model fitting."
    )
    return trained_models


if __name__ == "__main__":
    preprocessed_data = preprocess_heart_disease_data()
    train_models(preprocessed_data)
