# Global Electricity Consumption Predictor

An interactive web app for exploring country-level energy data and forecasting primary energy consumption. The project combines a FastAPI backend, a LightGBM forecasting model, and a browser-based dashboard.

## Features

- Select a country and review its latest energy profile.
- Adjust renewable energy share, emissions, electricity access, GDP per capita, and other assumptions.
- Generate a consumption forecast in TWh.
- Upload CSV, Excel, or SQLite data to inspect and predict from a custom dataset.
- Compare additional regression models and training scripts in the project files.

## Requirements

- Python 3.10 or newer
- Internet access for the interactive country map and globe textures

Install the Python dependencies from the project root:

```bash
python3 -m pip install -r requirements.txt
```

## Run the web app

From the project root, start the API and static frontend together:

```bash
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000>. Keep the terminal running while using the app; press **Ctrl+C** to stop it.

The API loads the saved model at `artifacts/energy_consumption_model.joblib` and training data at `ISDBcleaned_DB.xlsx`. Both are included in this repository.

## Retrain the LightGBM model

To regenerate the app's model artifact from the included workbook:

```bash
python3 MLcreation.py
```

The script writes the trained model to `artifacts/energy_consumption_model.joblib` by default. You can override the input workbook or output path with `--input` and `--model-output`.

## API routes

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Check API status |
| `GET /api/options` | Load country choices and model ranges |
| `GET /api/country/{iso3}` | Load a country's recent data |
| `POST /api/predict` | Generate a forecast |
| `POST /api/upload/inspect` | Inspect an uploaded dataset |
| `POST /api/upload/predict` | Generate predictions from an uploaded dataset |

## Data and large files

The repository includes the Excel workbooks used by the project and the LightGBM model used by the web app. Some additional Random Forest model artifacts exceed GitHub's 100 MB regular-file limit and are stored with Git LFS. Install Git LFS before cloning if you need those files:

```bash
git lfs install
git clone https://github.com/vishaal377-byte/Global-electricity-consumption-predictor.git
```

The web app accepts CSV, `.xlsx`, and SQLite (`.db`, `.sqlite`, `.sqlite3`) uploads. Uploads are limited to 5,000 rows and are processed for prediction without being saved by the API.
