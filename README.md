# Rossmann Store Sales and Customers Forecasting

## Project Overview

An interactive Streamlit dashboard for forecasting daily Sales and Customers for Rossmann stores using two trained LSTM deep learning models.

The application supports store selection, future-date CSV upload, recursive forecasting, visualizations, and forecast CSV downloads.

## Application Features

- Store-wise Sales forecasting using an LSTM model.
- Store-wise Customers forecasting using a separate LSTM model.
- Recursive multi-day forecasting.
- Input CSV validation.
- Daily Sales and Customers forecast tables.
- Sales prediction interval visualization.
- Sales and Customers trend charts.
- Combined forecast CSV download.

## Project Files

- `app.py` — Streamlit application.
- `rossmann_lstm_model.keras` — trained Sales LSTM model.
- `rossmann_customers_lstm_model.keras` — trained Customers LSTM model.
- `forecast_preprocessing_bundle.joblib` — Sales preprocessing and historical data bundle, including deployment configuration.
- `rossmann_customers_scaler.joblib` — Customers preprocessing scaler.
- `sample_future_dates.csv` — example future-date input.
- `requirements.txt` — Python dependencies.

## Run Locally

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

## Input CSV Format

Required column:

- `Date`

Optional columns:

- `IsHoliday`
- `IsWeekend`
- `IsPromo`
- `Open`

Dates must be continuous daily dates and must occur after the selected store's last known historical date.

Binary input columns, when provided, must contain only 0 or 1.

## Forecasting Methodology

The application uses two separate univariate LSTM models:

1. Sales LSTM for daily Sales prediction.
2. Customers LSTM for daily Customers prediction.

The models use historical target values and recursive forecasting.

Calendar columns are validated but are not currently used as direct model input features.

## Important Limitations

- Forecasts are model-based estimates and may differ from actual store performance.
- Negative predictions are clipped to zero.
- Closed stores are assigned zero predicted Sales and Customers when `Open = 0`.
- Prediction intervals are empirical estimates and are not guaranteed 95% confidence intervals.
- Forecast quality may vary across stores and forecast horizons.

## Technology Stack

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Project Status

The Sales and Customers LSTM models have been trained and integrated into the Streamlit dashboard. Public deployment is a separate project step.
