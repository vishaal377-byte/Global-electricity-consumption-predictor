"""Train a Random Forest regressor for primary energy consumption.

Loads an Excel or CSV database, uses an 80:20 random split, encodes
categorical inputs, reports regression metrics, and saves the fitted pipeline.

Example from the project root::

    .venv/bin/python .venv/ML/RandomForest/train_energy_random_forest.py
    .venv/bin/python .venv/ML/RandomForest/train_energy_random_forest.py --input other.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "ISDBcleaned_DB.xlsx").is_file()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_INPUT = PROJECT_ROOT / "ISDBcleaned_DB.xlsx"
DEFAULT_TARGET = "primary_energy_consumption"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "artifacts" / "random_forest_energy_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input .xlsx, .xls, or .csv file")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Numeric target column to predict")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT, help="Path for the saved model")
    return parser.parse_args()


def read_database(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be a .csv, .xlsx, or .xls file.")


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Database file not found: {args.input}")

    data = read_database(args.input)
    if args.target not in data.columns:
        raise ValueError(
            f"Target column {args.target!r} was not found. Available columns: "
            + ", ".join(map(str, data.columns))
        )

    y = pd.to_numeric(data[args.target], errors="coerce")
    valid = y.notna()
    dropped = int((~valid).sum())
    X = data.loc[valid].drop(columns=[args.target]).copy()
    y = y.loc[valid].astype(float)
    if len(X) < 2:
        raise ValueError("At least two rows with numeric target values are required.")
    if X.shape[1] == 0:
        raise ValueError("At least one feature column besides the target is required.")

    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessing = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Input: {args.input}")
    print(f"Target: {args.target}")
    train_percent = 100 * len(X_train) / len(X)
    test_percent = 100 * len(X_test) / len(X)
    print(f"Split: random (train={len(X_train)} / {train_percent:.1f}%, test={len(X_test)} / {test_percent:.1f}%)")
    if dropped:
        print(f"Dropped {dropped} row(s) with missing/non-numeric target.")
    print(f"Features: {len(X.columns)} ({len(numeric_columns)} numeric, {len(categorical_columns)} categorical)")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.4f}")
    print(f"R²:   {r2_score(y_test, predictions):.4f}")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "target": args.target,
        "feature_columns": X.columns.tolist(),
        "training_rows": len(X_train),
        "test_size": args.test_size,
    }, args.model_output)
    print(f"Saved model: {args.model_output}")


if __name__ == "__main__":
    main()
