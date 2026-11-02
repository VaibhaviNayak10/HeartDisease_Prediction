"""Prepare leakage-safe training and testing data for model development."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loading import load_heart_disease_data


TARGET_COLUMN = "num"
SOURCE_COLUMN = "source"
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
ZERO_AS_MISSING_FEATURES = ["trestbps", "chol"]
HIGH_MISSINGNESS_THRESHOLD = 0.30
TEST_SIZE = 0.20
RANDOM_STATE = 42


@dataclass
class PreprocessedData:
    """Train/test data and fitted preprocessing information."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    source_train: pd.Series
    source_test: pd.Series
    preprocessor: "HospitalPreprocessor"


@dataclass
class HospitalPreprocessor:
    """Fitted hospital-aware imputation and feature transformation."""

    hospital_statistics: dict[str, dict[str, float]]
    global_statistics: dict[str, float]
    fallback_features: dict[str, list[str]]
    scaler: StandardScaler
    encoder: OneHotEncoder
    feature_names: np.ndarray

    def _impute(self, X: pd.DataFrame, source: pd.Series) -> pd.DataFrame:
        """Impute rows from learned hospital statistics, without fitting."""
        feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        missing_columns = sorted(set(feature_columns).difference(X.columns))
        if missing_columns:
            raise ValueError(f"Input data is missing predictor columns: {missing_columns}")
        if len(X) != len(source):
            raise ValueError("X and source must contain the same number of rows.")
        if source.isna().any():
            raise ValueError("Source values must be present to perform hospital-specific imputation.")

        imputed = _replace_invalid_zeros(X[feature_columns])
        feature_positions = {
            feature: imputed.columns.get_loc(feature) for feature in feature_columns
        }
        for position, hospital in enumerate(source.tolist()):
            statistics = self.hospital_statistics.get(
                hospital, self.global_statistics
            )
            for feature in feature_columns:
                feature_position = feature_positions[feature]
                if pd.isna(imputed.iat[position, feature_position]):
                    imputed.iat[position, feature_position] = statistics[feature]
        return imputed

    def transform(self, X: pd.DataFrame, source: pd.Series) -> pd.DataFrame:
        """Impute and encode new raw records using training-fitted parameters."""
        imputed = self._impute(X, source)
        numeric_values = self.scaler.transform(imputed[NUMERICAL_FEATURES])
        categorical_values = self.encoder.transform(imputed[CATEGORICAL_FEATURES])
        transformed = np.column_stack((numeric_values, categorical_values))
        return pd.DataFrame(
            transformed, index=X.index, columns=self.feature_names
        )


def _report_missingness(data: pd.DataFrame) -> None:
    """Print overall missingness and source-specific rates for high-missingness features."""
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_counts = data[feature_columns].isna().sum()
    missing_rates = data[feature_columns].isna().mean()
    report = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": (missing_rates * 100).round(2),
        }
    ).sort_values("missing_percent", ascending=False)

    print("Missing predictor values (overall):")
    print(report.to_string())

    high_missing_features = missing_rates[
        missing_rates > HIGH_MISSINGNESS_THRESHOLD
    ].index.tolist()
    if high_missing_features:
        print(
            "\nFeatures above "
            f"{HIGH_MISSINGNESS_THRESHOLD:.0%} missingness, by source (%):"
        )
        source_rates = (
            data.groupby(SOURCE_COLUMN, observed=True)[high_missing_features]
            .agg(lambda values: values.isna().mean() * 100)
            .round(2)
        )
        print(source_rates.to_string())
    else:
        print(
            "\nNo predictors exceed "
            f"{HIGH_MISSINGNESS_THRESHOLD:.0%} missingness."
        )


def _replace_invalid_zeros(X: pd.DataFrame) -> pd.DataFrame:
    """Treat zero in the specified measured features as missing on a copy."""
    converted = X.copy()
    for feature in ZERO_AS_MISSING_FEATURES:
        if feature in converted.columns:
            converted[feature] = converted[feature].mask(converted[feature].eq(0))
    return converted


def _report_zero_conversion(data: pd.DataFrame) -> pd.DataFrame:
    """Report zero counts by source before conversion and missing counts after."""
    before_conversion = pd.DataFrame(
        {
            feature: data[feature].eq(0).groupby(data[SOURCE_COLUMN]).sum()
            for feature in ZERO_AS_MISSING_FEATURES
        }
    ).fillna(0).astype(int)
    print("Zero values before conversion, by source:")
    print(before_conversion.to_string())

    converted_features = _replace_invalid_zeros(data[ZERO_AS_MISSING_FEATURES])
    missing_after_conversion = pd.DataFrame(
        {
            "zero_converted": data[ZERO_AS_MISSING_FEATURES].eq(0).sum(),
            "missing_after_conversion": converted_features.isna().sum(),
        }
    )
    print("\nZero values converted to missing values:")
    print(missing_after_conversion["zero_converted"].to_string())
    print("\nMissing values after zero conversion, by source:")
    print(
        converted_features.assign(**{SOURCE_COLUMN: data[SOURCE_COLUMN]})
        .groupby(SOURCE_COLUMN, observed=True)[ZERO_AS_MISSING_FEATURES]
        .agg(lambda values: values.isna().sum())
        .to_string()
    )
    print("\nMissing values after zero conversion, overall:")
    print(missing_after_conversion["missing_after_conversion"].to_string())
    return converted_features


def _calculate_imputation_statistics(
    X_train: pd.DataFrame, source_train: pd.Series
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, list[str]]]:
    """Calculate hospital and global fallback statistics from training rows."""
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    global_statistics: dict[str, float] = {}
    for feature in feature_columns:
        observed = X_train[feature].dropna()
        if observed.empty:
            raise ValueError(
                f"Cannot impute feature '{feature}': it has no observed values "
                "in the training data."
            )
        if feature in NUMERICAL_FEATURES:
            global_statistics[feature] = float(observed.median())
        else:
            global_statistics[feature] = float(observed.mode().iloc[0])

    hospital_statistics: dict[str, dict[str, float]] = {}
    fallback_features: dict[str, list[str]] = {}
    for hospital in sorted(source_train.unique()):
        hospital_rows = X_train.loc[source_train == hospital]
        hospital_statistics[hospital] = {}
        fallback_features[hospital] = []
        for feature in feature_columns:
            observed = hospital_rows[feature].dropna()
            if observed.empty:
                value = global_statistics[feature]
                fallback_features[hospital].append(feature)
            elif feature in NUMERICAL_FEATURES:
                value = float(observed.median())
            else:
                value = float(observed.mode().iloc[0])
            hospital_statistics[hospital][feature] = value

    return hospital_statistics, global_statistics, fallback_features


def _report_imputation_statistics(
    hospital_statistics: dict[str, dict[str, float]],
    fallback_features: dict[str, list[str]],
) -> None:
    """Print the learned statistic for each feature and hospital."""
    print("\nTraining-derived imputation values by hospital:")
    for hospital, statistics in hospital_statistics.items():
        print(f"\n{hospital}:")
        for feature, value in statistics.items():
            fallback_marker = " (global fallback)" if feature in fallback_features[hospital] else ""
            print(f"  {feature}: {value:g}{fallback_marker}")

    print("\nFeatures requiring global fallback:")
    any_fallback = False
    for hospital, features in fallback_features.items():
        if features:
            any_fallback = True
            print(f"  {hospital}: {', '.join(features)}")
    if not any_fallback:
        print("  None")


def _print_distributions(
    name: str, source: pd.Series, target: pd.Series
) -> None:
    print(f"\n{name} source distribution:")
    print(source.value_counts().sort_index().to_string())
    print(f"{name} target distribution:")
    print(target.value_counts().sort_index().to_string())


def preprocess_heart_disease_data(
    data: Optional[pd.DataFrame] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessedData:
    """Create a stratified split and fit preprocessing on training data only.

    The binary target is 0 for ``num == 0`` and 1 for ``num > 0``. The source
    label is retained alongside each split but is not included in predictors.
    """
    if data is None:
        data = load_heart_disease_data()

    required_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET_COLUMN, SOURCE_COLUMN]
    )
    missing_columns = sorted(set(required_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    usable_data = data.dropna(subset=[TARGET_COLUMN]).copy()
    if usable_data.empty:
        raise ValueError("No rows with an observed target are available.")

    converted_features = _report_zero_conversion(usable_data)
    usable_data.loc[:, ZERO_AS_MISSING_FEATURES] = converted_features
    _report_missingness(usable_data)

    y = usable_data[TARGET_COLUMN].gt(0).astype("int64").rename("target")
    X = usable_data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    source = usable_data[SOURCE_COLUMN].copy().rename(SOURCE_COLUMN)

    X_train_raw, X_test_raw, y_train, y_test, source_train, source_test = (
        train_test_split(
            X,
            y,
            source,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    (
        hospital_statistics,
        global_statistics,
        fallback_features,
    ) = _calculate_imputation_statistics(X_train_raw, source_train)
    _report_imputation_statistics(hospital_statistics, fallback_features)

    preprocessor = HospitalPreprocessor(
        hospital_statistics=hospital_statistics,
        global_statistics=global_statistics,
        fallback_features=fallback_features,
        scaler=StandardScaler(),
        encoder=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        feature_names=np.array([], dtype=object),
    )
    X_train_imputed = preprocessor._impute(X_train_raw, source_train)
    X_test_imputed = preprocessor._impute(X_test_raw, source_test)

    X_train_numeric = preprocessor.scaler.fit_transform(
        X_train_imputed[NUMERICAL_FEATURES]
    )
    X_test_numeric = preprocessor.scaler.transform(
        X_test_imputed[NUMERICAL_FEATURES]
    )
    X_train_categorical = preprocessor.encoder.fit_transform(
        X_train_imputed[CATEGORICAL_FEATURES]
    )
    X_test_categorical = preprocessor.encoder.transform(
        X_test_imputed[CATEGORICAL_FEATURES]
    )

    preprocessor.feature_names = np.concatenate(
        (
            np.asarray([f"numeric__{feature}" for feature in NUMERICAL_FEATURES]),
            preprocessor.encoder.get_feature_names_out(CATEGORICAL_FEATURES),
        )
    )
    X_train = pd.DataFrame(
        np.column_stack((X_train_numeric, X_train_categorical)),
        index=X_train_raw.index,
        columns=preprocessor.feature_names,
    )
    X_test = pd.DataFrame(
        np.column_stack((X_test_numeric, X_test_categorical)),
        index=X_test_raw.index,
        columns=preprocessor.feature_names,
    )

    post_imputation_missing = pd.DataFrame(
        {
            "training_missing": X_train_imputed.isna().sum(),
            "testing_missing": X_test_imputed.isna().sum(),
        }
    )
    print("\nMissing predictor values remaining after imputation, by feature:")
    print(post_imputation_missing.to_string())
    print(
        "Total missing values after imputation: "
        f"training={int(post_imputation_missing['training_missing'].sum())}, "
        f"testing={int(post_imputation_missing['testing_missing'].sum())}"
    )
    print(f"\nTraining predictors shape: {X_train.shape}")
    print(f"Testing predictors shape: {X_test.shape}")
    _print_distributions("Training", source_train, y_train)
    _print_distributions("Testing", source_test, y_test)

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        source_train=source_train,
        source_test=source_test,
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    preprocess_heart_disease_data()
    
