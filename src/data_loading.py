"""Load and combine the four processed UCI Heart Disease datasets."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


DATASET_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def load_heart_disease_data(
    dataset_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load all UCI source files and return one combined DataFrame.

    By default, the ``Dataset`` directory is resolved relative to the project
    root, so the function does not depend on the process's current directory.
    The original ``num`` target is retained; binary conversion belongs in the
    preprocessing module.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_directory = Path(dataset_dir) if dataset_dir is not None else project_root / "Dataset"

    source_frames = []
    for source, filename in DATASET_FILES.items():
        file_path = data_directory / filename
        if not file_path.is_file():
            raise FileNotFoundError(
                "Required UCI dataset file was not found: "
                f"{file_path}. Place all four processed files in the Dataset directory."
            )

        source_frame = pd.read_csv(
            file_path,
            header=None,
            names=COLUMNS,
            na_values="?",
            skipinitialspace=True,
        )
        source_frame["source"] = source
        source_frames.append(source_frame)

        print(f"\n{source} dataset: {source_frame.shape[0]} records, "
              f"{source_frame.shape[1] - 1} clinical columns")
        print("Original num class distribution:")
        print(source_frame["num"].value_counts(dropna=False).sort_index())

    combined_data = pd.concat(source_frames, ignore_index=True)

    print(f"\nCombined dataset: {combined_data.shape[0]} records, "
          f"{combined_data.shape[1]} columns including source")
    print("Combined original num class distribution:")
    print(combined_data["num"].value_counts(dropna=False).sort_index())

    return combined_data


if __name__ == "__main__":
    load_heart_disease_data()
