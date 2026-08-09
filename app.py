import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Fever Prediction in Children",
    page_icon="🌡️",
    layout="wide"
)


# ============================================================
# Title & Introduction
# ============================================================

st.title("📈 Fever Prediction in Children")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the current day**
to predict whether a fever may occur in the coming day.

**Input Options:**  
Manual entry: edit temperatures directly in the table below.

**Input Criteria:**  
1. The interval between any two consecutive temperature measurements must not exceed **8 hours**.
2. The interval between the first and last temperature measurements must be at least **19 hours**.
3. Body temperature must be between **35°C and 43°C**.

**Disclaimer:**  
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional medical judgment.
Clinical decisions should always be made by qualified healthcare professionals based on comprehensive clinical evaluation.
""")


# ============================================================
# Manual Data Entry
# ============================================================

st.subheader("Manual Data Entry (editable table)")

day1_times = [f"{h:02d}:00" for h in range(8, 24)]
day2_times = [f"{h:02d}:00" for h in range(0, 8)]

all_times = (
    [("Day1", t) for t in day1_times]
    + [("Day2", t) for t in day2_times]
)

manual_df = pd.DataFrame(
    all_times,
    columns=["Day", "Time"]
)

# Explicitly specify float dtype for Temperature
# to avoid st.data_editor type inference problems.
manual_df["Temperature"] = pd.Series(
    [np.nan] * len(manual_df),
    dtype="float64"
)


# ============================================================
# Data Editor
# ============================================================

try:

    edited_df = st.data_editor(
        manual_df,
        column_config={
            "Day": st.column_config.TextColumn(
                "Day",
                disabled=True
            ),
            "Time": st.column_config.TextColumn(
                "Time",
                disabled=True
            ),
            "Temperature": st.column_config.NumberColumn(
                "Temperature (°C)",
                min_value=35.0,
                max_value=43.0,
                step=0.1,
                format="%.1f"
            )
        },
        disabled=["Day", "Time"],
        hide_index=True,
        num_rows="fixed",
        use_container_width=True
    )

except Exception as e:

    st.error(
        "Unable to display the temperature input table."
    )

    st.exception(e)

    st.stop()


# ============================================================
# Convert Temperature to Numeric
# ============================================================

edited_df["Temperature"] = pd.to_numeric(
    edited_df["Temperature"],
    errors="coerce"
)

edited_df = edited_df.dropna(
    subset=["Temperature"]
).copy()


# ============================================================
# Create DateTime
# ============================================================

df = pd.DataFrame()

if not edited_df.empty:

    df = edited_df.copy()

    # Use today's date as the reference date
    today = datetime.today().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    def create_datetime(row):

        if row["Day"] == "Day1":
            base_date = today - timedelta(days=1)
        else:
            base_date = today

        hour = int(row["Time"][:2])
        minute = int(row["Time"][3:])

        return base_date + timedelta(
            hours=hour,
            minutes=minute
        )

    df["DateTime"] = df.apply(
        create_datetime,
        axis=1
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)


# ============================================================
# Proceed if Data Exists
# ============================================================

if not df.empty:

    # ========================================================
    # Define 24-hour observation window
    # 08:00 previous day → 08:00 current day
    # ========================================================

    last_date = df["DateTime"].dt.date.max()

    end_time = (
        datetime.combine(
            last_date,
            datetime.min.time()
        )
        + timedelta(hours=8)
    )

    start_time = end_time - timedelta(hours=24)

    df_24h = df[
        (df["DateTime"] >= start_time)
        & (df["DateTime"] <= end_time)
    ].copy()

    df_24h = df_24h.sort_values(
        "DateTime"
    ).reset_index(drop=True)


    # ========================================================
    # Check if data are available
    # ========================================================

    if df_24h.empty:

        st.warning(
            "No data available in the last 24 hours "
            "(08:00 → 08:00)."
        )

        st.stop()


    # ========================================================
    # Check Number of Measurements
    # ========================================================

    if len(df_24h) < 2:

        st.error(
            "At least two temperature measurements are required."
        )

        st.stop()


    # ========================================================
    # Check Temperature Range
    # ========================================================

    invalid_temperature = (
        (df_24h["Temperature"] < 35)
        | (df_24h["Temperature"] > 43)
    )

    if invalid_temperature.any():

        invalid_values = df_24h.loc[
            invalid_temperature,
            "Temperature"
        ].tolist()

        st.error(
            "Invalid temperature value(s): "
            f"{invalid_values}. "
            "Temperature must be between 35°C and 43°C."
        )

        st.stop()


    # ========================================================
    # Calculate Intervals Between Consecutive Measurements
    # ========================================================

    time_diff_hours = (
        df_24h["DateTime"]
        .diff()
        .dt.total_seconds()
        / 3600
    )

    # First row has no previous measurement
    consecutive_intervals = time_diff_hours.iloc[1:]

    max_gap_hours = consecutive_intervals.max()


    # ========================================================
    # Calculate Total Observation Duration
    # ========================================================

    first_datetime = df_24h["DateTime"].min()
    last_datetime = df_24h["DateTime"].max()

    total_duration_hours = (
        last_datetime - first_datetime
    ).total_seconds() / 3600


    # ========================================================
    # Check 8-hour Maximum Gap
    # ========================================================

    if max_gap_hours > 8:

        st.error(
            f"Invalid input: The maximum interval between "
            f"consecutive temperature measurements is "
            f"{max_gap_hours:.1f} hours. "
            f"The interval must not exceed 8 hours."
        )

        st.stop()


    # ========================================================
    # Check 19-hour Minimum Duration
    # ========================================================

    if total_duration_hours < 19:

        st.error(
            f"Insufficient observation period: The interval "
            f"between the first and last temperature "
            f"measurements is {total_duration_hours:.1f} hours. "
            f"It must be at least 19 hours."
        )

        st.stop()


    # ========================================================
    # Input Criteria Passed
    # ========================================================

    st.success(
        f"✓ Input criteria satisfied — "
        f"maximum interval: {max_gap_hours:.1f} h; "
        f"observation duration: {total_duration_hours:.1f} h."
    )


    # ========================================================
    # Calculate Hours
    # ========================================================

    df_24h["Hours"] = (
        df_24h["DateTime"]
        - df_24h["DateTime"].min()
    ).dt.total_seconds() / 3600


    # ========================================================
    # Feature Extraction
    # ========================================================

    max_bt = df_24h["Temperature"].max()

    min_bt = df_24h["Temperature"].min()

    mean_bt = df_24h["Temperature"].mean()

    std_bt = df_24h["Temperature"].std()


    # --------------------------------------------------------
    # Temperature Slope
    # --------------------------------------------------------

    X = df_24h["Hours"].values.reshape(-1, 1)

    y = df_24h["Temperature"].values

    slope = LinearRegression().fit(
        X,
        y
    ).coef_[0]


    # --------------------------------------------------------
    # Maximum Temperature in Last 8 Hours
    # --------------------------------------------------------

    last_time = df_24h["Hours"].max()

    last_8h = df_24h[
        df_24h["Hours"] >= last_time - 8
    ]

    max_last8 = last_8h["Temperature"].max()


    # --------------------------------------------------------
    # Temperature Difference
    # --------------------------------------------------------

    range_bt = max_bt - min_bt


    # IMPORTANT:
    # This definition was confirmed by the user.
    #
    # T_Diff_MaxLast8H_Max =
    # T_MaxLast8H - T_Max
    # --------------------------------------------------------

    diff_last8_allmax = max_last8 - max_bt


    # ========================================================
    # Feature Vector
    # ========================================================

    features = [
        max_bt,
        min_bt,
        mean_bt,
        std_bt,
        slope,
        range_bt,
        max_last8,
        diff_last8_allmax
    ]


    # ========================================================
    # Model Prediction
    # ========================================================

    st.subheader("🤖 Prediction Result")

    try:

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        scaler = joblib.load(
            "scaler.pkl"
        )


        # ----------------------------------------------------
        # Load SVM model
        # ----------------------------------------------------

        svm_model = joblib.load(
            "svm_model.pkl"
        )


        # ----------------------------------------------------
        # Convert features to numpy array
        # ----------------------------------------------------

        features_array = np.array(
            features,
            dtype=float
        ).reshape(1, -1)


        # ----------------------------------------------------
        # Scale features
        # ----------------------------------------------------

        features_scaled = scaler.transform(
            features_array
        )


        # ----------------------------------------------------
        # SVM Probability
        # ----------------------------------------------------

        if not hasattr(
            svm_model,
            "predict_proba"
        ):

            st.error(
                "The loaded SVM model does not support "
                "probability prediction."
            )

            st.stop()


        pred_prob = svm_model.predict_proba(
            features_scaled
        )[0][1]


        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        threshold = 0.5


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if pred_prob >= threshold:

            st.success(
                f"Prediction: Fever expected in the coming day "
                f"(Probability = {pred_prob:.3f} ≥ {threshold})"
            )

        else:

            st.info(
                f"Prediction: No fever expected in the coming day "
                f"(Probability = {pred_prob:.3f} < {threshold})"
            )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        st.metric(
            label="Predicted Probability of Fever",
            value=f"{pred_prob:.3f}"
        )


    except FileNotFoundError as e:

        st.error(
            "Model or scaler file not found. "
            "Please make sure both "
            "`scaler.pkl` and `svm_model.pkl` "
            "are included in the Streamlit deployment."
        )

        st.exception(e)

    except Exception as e:

        st.error(
            "An error occurred during model prediction."
        )

        st.exception(e)


    # ========================================================
    # Temperature Trend
    # ========================================================

    st.subheader(
        "📉 Temperature Trend (Last 24h)"
    )

    from matplotlib.dates import (
        HourLocator,
        DateFormatter
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    # --------------------------------------------------------
    # Temperature Curve
    # --------------------------------------------------------

    ax.plot(
        df_24h["DateTime"],
        df_24h["Temperature"],
        marker="o",
        label="Temperature"
    )


    # --------------------------------------------------------
    # Fever Threshold
    # --------------------------------------------------------

    ax.axhline(
        y=38,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label="Fever Threshold (38°C)"
    )


    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    ax.xaxis.set_major_locator(
        HourLocator(interval=1)
    )

    ax.xaxis.set_major_formatter(
        DateFormatter("%H:%M")
    )


    # --------------------------------------------------------
    # Y-axis
    # --------------------------------------------------------

    ax.set_ylim(
        35,
        43
    )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Temperature (°C)"
    )


    # --------------------------------------------------------
    # Grid & Legend
    # --------------------------------------------------------

    ax.grid(
        True
    )

    ax.legend()


    # --------------------------------------------------------
    # Rotate X-axis labels
    # --------------------------------------------------------

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()


    # --------------------------------------------------------
    # Display Figure
    # --------------------------------------------------------

    st.pyplot(
        fig
    )
