
import pandas as pd
import os


def load_dataset(file_path):
    """Load a CSV dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def validate_dataset(df):
    """Validate the dataset structure and quality."""

    required_columns = ["subject", "body", "queue", "language"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    results = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "languages": df["language"].value_counts().to_dict(),
        "categories": df["queue"].value_counts().to_dict()
    }

    return results


def inspect_data(df):
    """Display basic dataset information."""

    print("Dataset Preview:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe(include="all"))
