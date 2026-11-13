
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Rossmann Store Sales Forecasting")

st.write(
    "A daily Sales forecasting dashboard powered by a trained LSTM model."
)

st.divider()


# ============================================================
# LOAD TRAINED MODEL AND PREPROCESSING OBJECTS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "rossmann_lstm_model.keras"

CUSTOMER_MODEL_PATH = (
    APP_DIR / "rossmann_customers_lstm_model.keras"
)

BUNDLE_PATH = APP_DIR / "forecast_preprocessing_bundle.joblib"


@st.cache_resource
def load_assets():

    sales_model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    customer_model = tf.keras.models.load_model(
        CUSTOMER_MODEL_PATH,
        compile=False
    )

    bundle = joblib.load(BUNDLE_PATH)

    return sales_model, customer_model, bundle


try:

    model, customer_model, bundle = load_assets()

except Exception as error:

    st.error(
        f"Unable to load forecasting assets: {error}"
    )

    st.stop()


scalers_by_status = bundle["scalers_by_status"]

forecast_store_status = bundle["forecast_store_status"]

historical_sales_by_store = (
    bundle["historical_sales_by_store"]
)

historical_end_dates = (
    bundle["historical_end_dates"]
)

lookback_window = int(
    bundle["lookback_window"]
)

store_error_margin_lookup = (
    bundle.get("store_error_margin_lookup", {})
)

global_error_margin = float(
    bundle.get("global_error_margin", 0.0)
)

customer_scaler = joblib.load(
    APP_DIR / bundle["customer_scaler_file"]
)

customer_lookback_window = int(
    bundle["customer_lookback_window"]
)

historical_customers_by_store = (
    bundle["customer_historical_customers_by_store"]
)

customer_historical_end_dates = (
    bundle["customer_historical_end_dates"]
)

customer_prediction_available = bool(
    bundle.get("customer_prediction_available", False)
)

available_stores = sorted(
    int(store_id)
    for store_id in bundle["forecast_store_ids"]
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Forecast Configuration")

selected_store = st.sidebar.selectbox(
    "Select Store ID",
    available_stores
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Future Dates CSV",
    type=["csv"]
)

st.sidebar.caption(
    "Required: Date | Optional: IsHoliday, IsWeekend, IsPromo, Open"
)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(input_df):

    input_df = input_df.copy()

    input_df.columns = (
        input_df.columns.str.strip()
    )

    required_columns = ["Date"]

    missing = [
        col for col in required_columns
        if col not in input_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    input_df["Date"] = pd.to_datetime(
        input_df["Date"],
        errors="coerce"
    )

    if input_df["Date"].isna().any():
        raise ValueError(
            "Invalid or missing Date values found."
        )

    if input_df["Date"].duplicated().any():
        raise ValueError(
            "Duplicate dates are not allowed."
        )

    input_df = input_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    if input_df.empty:
        raise ValueError(
            "The uploaded CSV is empty."
        )

    # Validate optional binary columns.

    binary_columns = [
        "IsHoliday",
        "IsWeekend",
        "IsPromo",
        "Open"
    ]

    for col in binary_columns:

        if col in input_df.columns:

            input_df[col] = pd.to_numeric(
                input_df[col],
                errors="coerce"
            )

            if input_df[col].isna().any():
                raise ValueError(
                    f"{col} contains missing or invalid values."
                )

            if not input_df[col].isin([0, 1]).all():
                raise ValueError(
                    f"{col} must contain only 0 or 1."
                )

        else:

            input_df[col] = np.nan


    # Require continuous daily dates because this model
    # performs one recursive step per calendar day.

    expected_dates = pd.date_range(
        start=input_df["Date"].min(),
        end=input_df["Date"].max(),
        freq="D"
    )

    if not input_df["Date"].reset_index(drop=True).equals(
        pd.Series(expected_dates)
    ):
        raise ValueError(
            "Dates must be continuous daily dates without gaps."
        )


    # Forecast dates must start after known history.

    last_history_date = pd.Timestamp(
        historical_end_dates[int(selected_store)]
    )

    if input_df["Date"].min() <= last_history_date:
        raise ValueError(
            "Forecast dates must be later than the last "
            "known historical date for this store."
        )


    return input_df


# ============================================================
# RECURSIVE SALES FORECAST FUNCTION
# ============================================================

def generate_sales_forecast(store_id, input_df):

    store_id = int(store_id)

    if store_id not in historical_sales_by_store:
        raise ValueError(
            "Historical Sales data is unavailable for this store."
        )

    status = forecast_store_status[store_id]

    scaler = scalers_by_status[status]

    historical_values = np.asarray(
        historical_sales_by_store[store_id],
        dtype=np.float64
    )

    if status == "Stationary":

        transformed_history = historical_values[
            -lookback_window:
        ]

    elif status == "Non-Stationary":

        transformed_history = np.diff(
            historical_values
        )[-lookback_window:]

    else:

        raise ValueError(
            "Unknown stationarity status."
        )


    history = scaler.transform(
        transformed_history.reshape(-1, 1)
    ).ravel().astype(np.float32)


    if len(history) != lookback_window:
        raise ValueError(
            "Historical lookback length is incorrect."
        )


    previous_sales = float(
        historical_values[-1]
    )

    error_margin = float(
        store_error_margin_lookup.get(
            store_id,
            global_error_margin
        )
    )

    records = []


    for day_number, row in enumerate(
        input_df.itertuples(index=False),
        start=1
    ):

        model_input = history.reshape(
            1,
            lookback_window,
            1
        ).astype(np.float32)


        scaled_prediction = float(
            model.predict(
                model_input,
                verbose=0
            ).reshape(-1)[0]
        )


        transformed_prediction = float(
            scaler.inverse_transform(
                np.array(
                    [[scaled_prediction]],
                    dtype=np.float64
                )
            )[0, 0]
        )


        if status == "Stationary":

            predicted_sales = transformed_prediction

        else:

            predicted_sales = (
                previous_sales + transformed_prediction
            )


        predicted_sales = max(
            0.0,
            predicted_sales
        )


        open_value = getattr(
            row,
            "Open",
            np.nan
        )

        if pd.notna(open_value) and int(open_value) == 0:
            predicted_sales = 0.0


        lower_bound = max(
            0.0,
            predicted_sales - error_margin
        )

        upper_bound = (
            predicted_sales + error_margin
        )


        if pd.notna(open_value) and int(open_value) == 0:

            lower_bound = 0.0
            upper_bound = 0.0


        # Update recursive input history.

        if status == "Stationary":

            next_transformed_value = predicted_sales

        else:

            next_transformed_value = (
                predicted_sales - previous_sales
            )


        next_scaled_value = float(
            scaler.transform(
                np.array(
                    [[next_transformed_value]],
                    dtype=np.float64
                )
            )[0, 0]
        )


        history = np.concatenate([

            history[1:],

            np.array(
                [next_scaled_value],
                dtype=np.float32
            )

        ])


        records.append({

            "Store": store_id,

            "Date": pd.Timestamp(row.Date),

            "Forecast_Day": day_number,

            "Predicted_Sales": float(predicted_sales),

            "Lower_Bound": float(lower_bound),

            "Upper_Bound": float(upper_bound),

            "Open": open_value,

            "Customers_Prediction": np.nan,

            "Customer_Prediction_Status":
                "Not available - no trained customer model"

        })


        previous_sales = predicted_sales


    return pd.DataFrame(records)




# ============================================================
# RECURSIVE CUSTOMERS FORECAST FUNCTION
# ============================================================

def generate_customers_forecast(store_id, input_df):

    store_id = int(store_id)

    history_mapping = historical_customers_by_store

    if store_id in history_mapping:
        historical_values = history_mapping[store_id]

    elif str(store_id) in history_mapping:
        historical_values = history_mapping[str(store_id)]

    else:
        raise ValueError(
            "Historical Customers data is unavailable for this store."
        )

    history = np.asarray(
        historical_values,
        dtype=np.float32
    )

    if len(history) < customer_lookback_window:
        raise ValueError(
            "Insufficient historical Customers observations."
        )

    history = history[-customer_lookback_window:]

    records = []

    for day_number, row in enumerate(
        input_df.itertuples(index=False),
        start=1
    ):

        model_input = customer_scaler.transform(
            history.reshape(-1, 1)
        ).reshape(
            1,
            customer_lookback_window,
            1
        ).astype(np.float32)

        scaled_prediction = float(
            customer_model(
                tf.convert_to_tensor(
                    model_input,
                    dtype=tf.float32
                ),
                training=False
            ).numpy().reshape(-1)[0]
        )

        predicted_customers = float(
            customer_scaler.inverse_transform(
                np.array(
                    [[scaled_prediction]],
                    dtype=np.float32
                )
            )[0, 0]
        )

        predicted_customers = max(
            0.0,
            predicted_customers
        )

        open_value = getattr(
            row,
            "Open",
            np.nan
        )

        if (
            pd.notna(open_value)
            and int(open_value) == 0
        ):
            predicted_customers = 0.0

        predicted_customers = int(
            np.rint(predicted_customers)
        )

        records.append({

            "Store": store_id,

            "Date": pd.Timestamp(row.Date),

            "Predicted_Customers": predicted_customers

        })

        # Recursive update.
        history = np.append(
            history[1:],
            np.float32(predicted_customers)
        )

    return pd.DataFrame(records)


# ============================================================
# DASHBOARD WORKFLOW
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV file containing future dates to generate a forecast."
    )

    st.subheader("Sample CSV format")

    sample_df = pd.DataFrame({

        "Date": pd.date_range(
            start=pd.Timestamp(
                historical_end_dates[selected_store]
            ) + pd.Timedelta(days=1),
            periods=7,
            freq="D"
        ),

        "IsHoliday": [0, 0, 0, 0, 0, 0, 0],

        "IsWeekend": [0, 0, 0, 0, 0, 1, 1],

        "IsPromo": [0, 1, 0, 0, 1, 0, 0],

        "Open": [1, 1, 1, 1, 1, 0, 0]

    })

    st.dataframe(
        sample_df,
        use_container_width=True
    )


else:

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        st.subheader("Uploaded CSV Preview")

        st.dataframe(
            uploaded_df.head(10),
            use_container_width=True
        )

        validated_df = validate_input(
            uploaded_df
        )

        st.success(
            f"Input validation successful: {len(validated_df)} forecast dates."
        )

        st.info(
            "Sales and Customers are forecast by separate historical "
            "univariate LSTM models. Holiday, weekend and promotion "
            "columns are validated but are not currently model input features."
        )


        if st.button(
            "Generate Sales Forecast",
            type="primary"
        ):

            with st.spinner(
                "Generating recursive Sales predictions..."
            ):

                sales_result_df = generate_sales_forecast(
                    selected_store,
                    validated_df
                )

                customers_result_df = generate_customers_forecast(
                    selected_store,
                    validated_df
                )

                result_df = sales_result_df.merge(
                    customers_result_df[
                        [
                            "Store",
                            "Date",
                            "Predicted_Customers"
                        ]
                    ],
                    on=["Store", "Date"],
                    how="inner",
                    validate="one_to_one"
                )

                if len(result_df) != len(validated_df):
                    raise ValueError(
                        "Sales and Customers forecast rows do not match."
                    )


            st.success(
                "Sales and Customers forecasts generated successfully."
            )


            # ------------------------------------------------
            # KPI METRICS
            # ------------------------------------------------

            total_sales = float(
                result_df["Predicted_Sales"].sum()
            )

            average_sales = float(
                result_df["Predicted_Sales"].mean()
            )

            total_customers = int(
                result_df["Predicted_Customers"].sum()
            )

            average_customers = float(
                result_df["Predicted_Customers"].mean()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Store ID",
                selected_store
            )

            col2.metric(
                "Forecast Days",
                len(result_df)
            )

            col3.metric(
                "Total Predicted Sales",
                f"{total_sales:,.2f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Average Daily Sales",
                f"{average_sales:,.2f}"
            )

            col5.metric(
                "Total Predicted Customers",
                f"{total_customers:,}"
            )

            col6.metric(
                "Average Daily Customers",
                f"{average_customers:,.0f}"
            )


            # ------------------------------------------------
            # FORECAST TABLE
            # ------------------------------------------------

            st.subheader("Daily Sales Forecast")

            st.dataframe(
                result_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # SALES GRAPH
            # ------------------------------------------------

            st.subheader("Predicted Sales Trend")

            chart_df = result_df.set_index(
                "Date"
            )[[
                "Predicted_Sales",
                "Lower_Bound",
                "Upper_Bound"
            ]]

            st.line_chart(
                chart_df,
                use_container_width=True
            )

            st.subheader("Predicted Customers Trend")

            customer_chart_df = result_df.set_index(
                "Date"
            )[[
                "Predicted_Customers"
            ]]

            st.line_chart(
                customer_chart_df,
                use_container_width=True
            )


            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv_data = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="Download Forecast CSV",

                data=csv_data,

                file_name=(
                    f"rossmann_store_{selected_store}_sales_customers_forecast.csv"
                ),

                mime="text/csv"

            )


    except Exception as error:

        st.error(
            f"Unable to generate forecast: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Project 6 | Rossmann Store Sales Forecasting | "
    "LSTM Deep Learning Dashboard"
)

st.caption(
    "Forecast intervals are empirical estimates and are not guaranteed "
    "95% confidence intervals."
)
