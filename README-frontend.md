# TerraCast energy forecast UI

This adds a browser interface for the saved LightGBM energy model.

## Run it

From the project root, install the API requirements into the Python environment that has the model dependencies, then start the server:

```bash
python -m pip install -r requirements.txt
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000>. The frontend files are saved in `.venv/ML/Frontend/Webapp` and the API serves them from the same process. The service loads `artifacts/energy_consumption_model.joblib` and `ISDBcleaned_DB.xlsx` on startup. If the model artifact needs to be regenerated, run `python MLcreation.py` first.

The country selector fills the form from each country's most recent workbook row. Users can adjust the assumptions and recent consumption values before requesting a forecast. The **Your data** tab accepts CSV, Excel (`.xlsx`), and SQLite (`.db`, `.sqlite`, `.sqlite3`) files, supports selecting a SQLite table and mapping dataset columns, and returns downloadable predictions without saving the uploaded data. Uploads are limited to 5,000 rows. Upload model input units: percentages, emissions in MtCO₂e, GDP per capita in USD, and consumption in TWh. If a dataset contains country, year, and historical consumption, prior-year values are derived automatically. Two years of history enable the year-over-year LightGBM path; otherwise the saved fallback LightGBM model is used.
