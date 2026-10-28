"""Train and evaluate a LightGBM model for primary energy consumption.

The model learns the change from the previous year's consumption. It also
reports a carry-forward baseline using that previous year's value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "ISDBcleaned_DB.xlsx").is_file()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_INPUT = PROJECT_ROOT / "ISDBcleaned_DB.xlsx"
DEFAULT_TARGET = "primary_energy_consumption"
DEFAULT_MODEL_OUTPUT = PROJECT_ROOT / "artifacts" / "energy_consumption_model.joblib"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Excel workbook to load")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to predict")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of years reserved for testing")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
        help=f"Where to save the fitted model and preprocessing (default: {DEFAULT_MODEL_OUTPUT})",
    )
    return parser.parse_args()

def make_lag_features(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Add exact previous-calendar-year target values for each country."""
    result = data.copy()
    lag_columns = [f"{target}_lag1", f"{target}_lag2"]
    if "year" not in result.columns:
        for column in lag_columns:
            result[column] = np.nan
    else:
        lag_data = pd.DataFrame(
            {
                "country": result["country"],
                "year": pd.to_numeric(result["year"], errors="coerce"),
                "target": result[target],
            },
            index=result.index,
        ).sort_values(["country", "year"], kind="mergesort")
        grouped = lag_data.groupby("country", sort=False)
        for lag, column in enumerate(lag_columns, start=1):
            prior_target = grouped["target"].shift(lag)
            prior_year = grouped["year"].shift(lag)
            result[column] = prior_target.where(
                lag_data["year"].sub(prior_year).eq(lag)
            )
    result["delta_last_year"] = result[lag_columns[0]] - result[lag_columns[1]]
    return result

def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Excel file not found: {args.input}")

    data = pd.read_excel(args.input)
    if args.target not in data.columns:
        raise ValueError(
            f"Target column {args.target!r} was not found in {args.input.name}. "
            f"Available columns: {', '.join(map(str, data.columns))}"
        )
    if "country" not in data.columns:
        raise ValueError("Required categorical feature 'country' was not found.")

    data[args.target] = pd.to_numeric(data[args.target], errors="coerce")
    data = make_lag_features(data, args.target)
    rows_before = len(data)
    data = data.dropna(subset=[args.target]).copy()
    if len(data) < 2:
        raise ValueError("At least two rows with numeric target values are required.")
    if len(data) != rows_before:
        print(f"Dropped {rows_before - len(data)} row(s) with missing/non-numeric target.")

    X = data.drop(columns=[args.target]).copy()
    y = data[args.target].astype(float)
    lag1 = f"{args.target}_lag1"
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    if "year" in data.columns and data["year"].nunique() >= 2:
        known_year = data["year"].notna()
        years = sorted(data.loc[known_year, "year"].unique().tolist())
        test_year_count = min(
            max(1, int(np.ceil(len(years) * args.test_size))), len(years) - 1
        )
        test_years = years[-test_year_count:]
        train_years = years[:-test_year_count]
        train_mask = data["year"].isin(train_years)
        test_mask = data["year"].isin(test_years)
        X_train, X_test = X.loc[train_mask].copy(), X.loc[test_mask].copy()
        y_train, y_test = y.loc[train_mask], y.loc[test_mask]
        split_description = "chronological (latest years held out)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )
        test_years = []
        split_description = "random"

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    if categorical_columns:
        train_encoded = encoder.fit_transform(X_train[categorical_columns])
        test_encoded = encoder.transform(X_test[categorical_columns])
        for index, column in enumerate(categorical_columns):
            X_train[column] = train_encoded[:, index]
            X_test[column] = test_encoded[:, index]

    residual_train_mask = X_train[lag1].notna()
    if not residual_train_mask.any():
        raise ValueError("No training rows have a usable previous-year target value.")

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=25,
    )
    model.fit(
        X_train.loc[residual_train_mask],
        y_train.loc[residual_train_mask] - X_train.loc[residual_train_mask, lag1],
        categorical_feature=categorical_columns,
    )

    fallback_model = lgb.LGBMRegressor(
        objective="regression", random_state=42, n_jobs=-1
    )
    fallback_model.fit(X_train, y_train, categorical_feature=categorical_columns)

    predictions = fallback_model.predict(X_test)
    lag_available = X_test[lag1].notna()
    if lag_available.any():
        predictions[lag_available.to_numpy()] = (
            X_test.loc[lag_available, lag1].to_numpy()
            + model.predict(X_test.loc[lag_available])
        )

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"Input: {args.input}")
    print(f"Target: {args.target}")
    print(f"Split: {split_description} (train={len(X_train)}, test={len(X_test)})")
    if test_years:
        print(f"Test years: {test_years}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    if lag_available.any():
        actual = y_test.loc[lag_available]
        carry_forward = X_test.loc[lag_available, lag1]
        model_on_lag_rows = pd.Series(predictions, index=y_test.index).loc[lag_available]
        print(f"Carry-forward baseline coverage: {int(lag_available.sum())}/{len(X_test)} test rows")
        print(f"Carry-forward baseline MAE: {mean_absolute_error(actual, carry_forward):.4f}")
        print(f"LightGBM MAE on same rows: {mean_absolute_error(actual, model_on_lag_rows):.4f}")

    deployment_features = X.copy()
    deployment_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    if categorical_columns:
        encoded = deployment_encoder.fit_transform(deployment_features[categorical_columns])
        for index, column in enumerate(categorical_columns):
            deployment_features[column] = encoded[:, index]

    deployment_lag_available = deployment_features[lag1].notna()
    deployment_model = lgb.LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=25,
    )
    deployment_model.fit(
        deployment_features.loc[deployment_lag_available],
        y.loc[deployment_lag_available]
        - deployment_features.loc[deployment_lag_available, lag1],
        categorical_feature=categorical_columns,
    )
    deployment_fallback_model = lgb.LGBMRegressor(
        objective="regression", random_state=42, n_jobs=-1
    )
    deployment_fallback_model.fit(
        deployment_features, y, categorical_feature=categorical_columns
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": deployment_model,
            "fallback_model": deployment_fallback_model,
            "encoder": deployment_encoder if categorical_columns else None,
            "categorical_columns": categorical_columns,
            "feature_columns": X.columns.tolist(),
            "target": args.target,
            "training_rows": len(deployment_features),
            "prediction_mode": "delta_from_previous_year",
        },
        args.model_output,
    )
    print(f"Saved model trained on {len(deployment_features)} row(s): {args.model_output}")

if __name__ == "__main__":
    main()

# temporary input probe
