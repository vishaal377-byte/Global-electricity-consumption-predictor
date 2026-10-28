"""Predict primary energy consumption for every row in an Excel workbook.

Run from the project root after training MLcreation.py::

    .venv/bin/python .venv/ML/batch_predict.py --input new_data.xlsx

The output is written beside the input as new_data_predictions.xlsx unless
--output is provided. The input workbook is never modified. Provide the lag
columns directly or include country-year rows with observed target values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "ISDBcleaned_DB.xlsx").is_file()),
    SCRIPT_PATH.parents[3],
)
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "energy_consumption_model.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input .xlsx workbook")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved model bundle")
    parser.add_argument("--output", type=Path, help="Output .xlsx path")
    return parser.parse_args()


def add_lag_features(data: pd.DataFrame, target: str, required: list[str]) -> pd.DataFrame:
    result = data.copy()
    lag_columns = [f"{target}_lag1", f"{target}_lag2"]
    missing_lags = [column for column in lag_columns if column in required and column not in result]
    if missing_lags:
        needed = {"country", "year", target}
        if not needed.issubset(result.columns):
            raise ValueError(
                f"Input is missing lag feature(s): {', '.join(missing_lags)}. Add those columns, "
                f"or include country, year, and observed {target} values for prior years."
            )
        lag_data = pd.DataFrame({
            "country": result["country"],
            "year": pd.to_numeric(result["year"], errors="coerce"),
            "target": pd.to_numeric(result[target], errors="coerce"),
        }, index=result.index).sort_values(["country", "year"], kind="mergesort")
        grouped = lag_data.groupby("country", sort=False)
        for lag, column in enumerate(lag_columns, start=1):
            if column not in missing_lags:
                continue
            prior_target = grouped["target"].shift(lag)
            prior_year = grouped["year"].shift(lag)
            result[column] = prior_target.where(lag_data["year"].sub(prior_year).eq(lag))

    delta_column = "delta_last_year"
    if delta_column in required and delta_column not in result:
        if not set(lag_columns).issubset(result.columns):
            raise ValueError(f"Provide {delta_column} or both lag columns: {', '.join(lag_columns)}.")
        result[delta_column] = (
            pd.to_numeric(result[lag_columns[0]], errors="coerce")
            - pd.to_numeric(result[lag_columns[1]], errors="coerce")
        )
    return result


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input workbook not found: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Saved model not found: {args.model}. Train MLcreation.py first.")

    bundle = joblib.load(args.model)
    data = add_lag_features(pd.read_excel(args.input), bundle["target"], bundle["feature_columns"])
    feature_columns = bundle["feature_columns"]
    categorical_columns = bundle["categorical_columns"]
    missing = [column for column in feature_columns if column not in data.columns]
    if missing:
        raise ValueError("Input workbook is missing required model feature(s): " + ", ".join(missing))

    features = data[feature_columns].copy()
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    for column in numeric_columns:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Feature {column!r} has {int(invalid.sum())} non-numeric value(s).")
        features[column] = converted

    if categorical_columns:
        encoded = bundle["encoder"].transform(features[categorical_columns])
        for index, column in enumerate(categorical_columns):
            features[column] = encoded[:, index]

    predictions = bundle["fallback_model"].predict(features)
    lag1 = f"{bundle['target']}_lag1"
    has_lag = features[lag1].notna()
    if has_lag.any():
        lgb_residual = bundle["model"].predict(features.loc[has_lag])
        if "secondary_model" in bundle:
            rf_residual = bundle["secondary_model"].predict(features.loc[has_lag])
            weights = bundle.get("ensemble_weights", {"lightgbm": 0.75, "random_forest": 0.25})
            residual = weights["lightgbm"] * lgb_residual + weights["random_forest"] * rf_residual
        else:
            # Continue to support model bundles created before ensemble blending.
            residual = lgb_residual
        residual = bundle.get("residual_scale", 1.0) * residual
        predictions[has_lag.to_numpy()] = features.loc[has_lag, lag1].to_numpy() + residual

    result = data.copy()
    prediction_column = f"predicted_{bundle['target']}"
    result[prediction_column] = predictions
    output = args.output or args.input.with_name(f"{args.input.stem}_predictions.xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output, index=False)
    print(f"Predicted {len(result)} row(s).")
    print(f"Prediction column: {prediction_column}")
    print(f"Saved results: {output}")


if __name__ == "__main__":
    main()
