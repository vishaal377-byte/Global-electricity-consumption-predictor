"""FastAPI service for the LightGBM energy consumption model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import json
import re
import sqlite3
import tempfile
from io import BytesIO
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "energy_consumption_model.joblib"
DATA_PATH = ROOT / "ISDBcleaned_DB.xlsx"
FRONTEND_PATH = ROOT / ".venv" / "ML" / "Frontend" / "Webapp"

app = FastAPI(title="Energy Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    iso3: str
    country: str
    year: int = Field(ge=1900, le=2200)
    renewables_share_elec: float = Field(ge=0)
    greenhouse_gas_emissions: float = Field(ge=0)
    electricity_access_pct: float = Field(ge=0, le=100)
    gdp_per_capita: float = Field(ge=0)
    urban_pop_growth: float
    previous_year_consumption: float = Field(gt=0)
    two_years_ago_consumption: float | None = Field(default=None, gt=0)


def load_assets() -> tuple[dict[str, Any], pd.DataFrame]:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}. Run MLcreation.py first.")
    if not DATA_PATH.exists():
        raise RuntimeError(f"Training data not found: {DATA_PATH}")
    return joblib.load(MODEL_PATH), pd.read_excel(DATA_PATH)


bundle, training_data = load_assets()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "LightGBM"}


@app.get("/api/options")
def options() -> dict[str, Any]:
    countries = (
        training_data[["iso3", "country"]]
        .dropna()
        .drop_duplicates()
        .sort_values("country")
        .to_dict(orient="records")
    )
    numeric = training_data.select_dtypes(include="number")
    return {
        "countries": countries,
        "year_min": int(training_data["year"].min()),
        "year_max": int(training_data["year"].max()) + 10,
        "data_year": int(training_data["year"].max()),
        "ranges": {
            column: {"min": float(numeric[column].min()), "max": float(numeric[column].max())}
            for column in numeric.columns
            if column != "year"
        },
        "training_rows": bundle["training_rows"],
    }


@app.get("/api/country/{iso3}")
def country_defaults(iso3: str) -> dict[str, Any]:
    rows = training_data.loc[training_data["iso3"].str.upper() == iso3.upper()].sort_values("year")
    if rows.empty:
        raise HTTPException(status_code=404, detail="Country not found in the training data.")
    latest = rows.iloc[-1]
    previous = rows.loc[rows["year"] == latest["year"] - 1]
    record = latest.to_dict()
    record["previous_consumption"] = (
        float(previous.iloc[-1]["primary_energy_consumption"]) if not previous.empty else None
    )
    return {key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value) for key, value in record.items()}


@app.post("/api/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    data = payload.model_dump()
    data["primary_energy_consumption_lag1"] = data.pop("previous_year_consumption")
    data["primary_energy_consumption_lag2"] = data.pop("two_years_ago_consumption")
    prediction, has_history = predict_features(data)
    lag1 = data["primary_energy_consumption_lag1"]
    if not pd.notna(prediction):
        raise HTTPException(status_code=422, detail="The model could not produce a prediction for these inputs.")
    return {
        "prediction": prediction,
        "country": payload.country,
        "year": payload.year,
        "unit": "TWh",
        "method": "LightGBM year-over-year forecast" if has_history else "LightGBM estimate (limited history)",
        "previous_year_consumption": lag1,
        "change_percent": (prediction - lag1) / lag1 * 100,
    }


def predict_features(data: dict[str, Any]) -> tuple[float, bool]:
    raw_lag1 = data.get("primary_energy_consumption_lag1")
    lag1 = float(raw_lag1) if raw_lag1 is not None and pd.notna(raw_lag1) else None
    lag2 = data.get("primary_energy_consumption_lag2")
    has_history = lag1 is not None and lag2 is not None and pd.notna(lag2)
    data["delta_last_year"] = lag1 - float(lag2) if has_history else float("nan")
    row = pd.DataFrame([{column: data.get(column) for column in bundle["feature_columns"]}])
    categorical = bundle["categorical_columns"]
    encoder = bundle.get("encoder")
    if categorical and encoder is not None:
        row[categorical] = encoder.transform(row[categorical])
    model = bundle["model"] if has_history else bundle["fallback_model"]
    raw = float(model.predict(row)[0])
    return (lag1 + raw if has_history else raw), bool(has_history)


CANONICAL_ALIASES = {
    "iso3": ["iso3", "iso_code", "country_code", "code"],
    "country": ["country", "nation", "country_name"],
    "year": ["year", "forecast_year", "time"],
    "renewables_share_elec": ["renewables_share_elec", "renewables_share", "renewable_share", "renewables_percent"],
    "greenhouse_gas_emissions": ["greenhouse_gas_emissions", "ghg_emissions", "emissions"],
    "electricity_access_pct": ["electricity_access_pct", "electricity_access", "access_pct"],
    "gdp_per_capita": ["gdp_per_capita", "gdp_pc", "income_per_capita"],
    "urban_pop_growth": ["urban_pop_growth", "urban_population_growth", "urban_growth"],
    "primary_energy_consumption_lag1": ["primary_energy_consumption_lag1", "previous_year_consumption", "lag1", "energy_last_year"],
    "primary_energy_consumption_lag2": ["primary_energy_consumption_lag2", "two_years_ago_consumption", "lag2", "energy_two_years_ago"],
    "primary_energy_consumption": ["primary_energy_consumption", "actual_consumption", "energy_consumption", "target"],
}
REQUIRED_UPLOAD_FIELDS = [
    "country", "year", "renewables_share_elec", "greenhouse_gas_emissions",
    "electricity_access_pct", "gdp_per_capita", "urban_pop_growth",
]


def read_uploaded_dataset(content: bytes, filename: str, table: str | None = None) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(BytesIO(content))
        if suffix == ".xlsx":
            return pd.read_excel(BytesIO(content))
        if suffix in (".db", ".sqlite", ".sqlite3"):
            with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
                temp.write(content)
                temp.flush()
                connection = sqlite3.connect(f"file:{temp.name}?mode=ro", uri=True)
                try:
                    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                    if not tables:
                        raise HTTPException(status_code=400, detail="The SQLite database has no tables.")
                    selected = table or tables[0]
                    if selected not in tables:
                        raise HTTPException(status_code=400, detail="Select a valid table from the uploaded database.")
                    escaped = selected.replace('"', '""')
                    return pd.read_sql_query(f'SELECT * FROM "{escaped}"', connection)
                finally:
                    connection.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded data: {exc}") from exc
    raise HTTPException(status_code=415, detail="Upload a CSV, Excel, or SQLite database file.")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def guess_mapping(columns: list[str]) -> dict[str, str]:
    by_normalized = {normalize_name(column): column for column in columns}
    result = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if normalize_name(alias) in by_normalized:
                result[canonical] = by_normalized[normalize_name(alias)]
                break
    return result


def sqlite_tables(content: bytes, filename: str) -> list[str]:
    if Path(filename).suffix.lower() not in (".db", ".sqlite", ".sqlite3"):
        return []
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as temp:
        temp.write(content)
        temp.flush()
        connection = sqlite3.connect(f"file:{temp.name}?mode=ro", uri=True)
        try:
            return [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        finally:
            connection.close()


@app.post("/api/upload/inspect")
async def inspect_upload(file: UploadFile = File(...), table: str | None = Form(default=None)) -> dict[str, Any]:
    content = await file.read()
    tables = sqlite_tables(content, file.filename or "")
    data = read_uploaded_dataset(content, file.filename or "", table)
    columns = [str(column) for column in data.columns]
    return {
        "filename": file.filename,
        "tables": tables,
        "columns": columns,
        "mapping": guess_mapping(columns),
        "missing_fields": [field for field in REQUIRED_UPLOAD_FIELDS if field not in guess_mapping(columns)],
        "rows": len(data),
        "preview": json.loads(data.head(5).to_json(orient="records")),
    }


@app.post("/api/upload/predict")
async def predict_upload(
    file: UploadFile = File(...),
    table: str | None = Form(default=None),
    mapping_json: str = Form(default="{}"),
) -> dict[str, Any]:
    content = await file.read()
    data_frame = read_uploaded_dataset(content, file.filename or "", table)
    try:
        mapping = {**guess_mapping([str(c) for c in data_frame.columns]), **json.loads(mapping_json)}
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Column mapping must be valid JSON.") from exc
    missing = [field for field in REQUIRED_UPLOAD_FIELDS if not mapping.get(field) or mapping[field] not in data_frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Map these required fields to your columns: {', '.join(missing)}")
    if len(data_frame) > 5000:
        raise HTTPException(status_code=413, detail="This preview supports up to 5,000 rows per upload.")

    results = []
    actual_column = mapping.get("primary_energy_consumption")
    history: dict[tuple[str, int], float] = {}
    if actual_column and actual_column in data_frame.columns:
        for _, historic_row in data_frame.iterrows():
            historic_year = pd.to_numeric(pd.Series([historic_row[mapping["year"]]]), errors="coerce").iloc[0]
            historic_value = pd.to_numeric(pd.Series([historic_row[actual_column]]), errors="coerce").iloc[0]
            historic_country = historic_row[mapping["country"]]
            if pd.notna(historic_year) and pd.notna(historic_value) and pd.notna(historic_country):
                history[(str(historic_country), int(historic_year))] = float(historic_value)
    for index, source_row in data_frame.iterrows():
        row_data: dict[str, Any] = {}
        row_data["iso3"] = source_row[mapping["iso3"]] if mapping.get("iso3") in data_frame.columns else "UNK"
        for canonical in REQUIRED_UPLOAD_FIELDS:
            row_data[canonical] = source_row[mapping[canonical]]
        if pd.isna(row_data["country"]):
            raise HTTPException(status_code=422, detail=f"Row {index + 1} has no country value.")
        optional_lag1 = mapping.get("primary_energy_consumption_lag1")
        optional_lag2 = mapping.get("primary_energy_consumption_lag2")
        row_data["primary_energy_consumption_lag1"] = source_row[optional_lag1] if optional_lag1 and optional_lag1 in data_frame.columns else None
        row_data["primary_energy_consumption_lag2"] = source_row[optional_lag2] if optional_lag2 and optional_lag2 in data_frame.columns else None
        for numeric in REQUIRED_UPLOAD_FIELDS:
            if numeric == "country":
                continue
            row_data[numeric] = pd.to_numeric(pd.Series([row_data[numeric]]), errors="coerce").iloc[0]
        if any(pd.isna(row_data[field]) for field in REQUIRED_UPLOAD_FIELDS if field != "country"):
            raise HTTPException(status_code=422, detail=f"Row {index + 1} has missing or non-numeric required values.")
        if pd.isna(row_data["primary_energy_consumption_lag1"]):
            row_data["primary_energy_consumption_lag1"] = history.get((str(row_data["country"]), int(row_data["year"]) - 1))
        if pd.isna(row_data["primary_energy_consumption_lag2"]):
            row_data["primary_energy_consumption_lag2"] = history.get((str(row_data["country"]), int(row_data["year"]) - 2))
        for lag_field in ("primary_energy_consumption_lag1", "primary_energy_consumption_lag2"):
            if row_data[lag_field] is not None:
                row_data[lag_field] = pd.to_numeric(pd.Series([row_data[lag_field]]), errors="coerce").iloc[0]
                if pd.isna(row_data[lag_field]) or row_data[lag_field] <= 0:
                    row_data[lag_field] = None
        prediction, has_history = predict_features(row_data)
        results.append({
            "row": int(index) if isinstance(index, (int,)) else len(results) + 1,
            "country": str(row_data["country"]),
            "year": int(row_data["year"]),
            "prediction": prediction,
            "actual": (
                float(pd.to_numeric(pd.Series([source_row[actual_column]]), errors="coerce").iloc[0])
                if actual_column and pd.notna(source_row[actual_column])
                and pd.notna(pd.to_numeric(pd.Series([source_row[actual_column]]), errors="coerce").iloc[0])
                else None
            ),
            "previous": float(row_data["primary_energy_consumption_lag1"]) if row_data["primary_energy_consumption_lag1"] is not None else None,
            "method": "Year-over-year" if has_history else "Limited history",
        })
    return {"filename": file.filename, "table": table, "rows": results, "count": len(results)}


app.mount("/", StaticFiles(directory=FRONTEND_PATH, html=True), name="frontend")
