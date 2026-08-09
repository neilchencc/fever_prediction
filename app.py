import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib


# ---------------------------------------------------
# Title & Introduction
# ---------------------------------------------------
st.title("📈 Fever Prediction in Children")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the current day**
to predict whether a fever may occur in the coming day.

**Input Options:**  
Manual entry: edit temperatures directly in the table below.

**Input Criteria:**  
(1) The interval between any two consecutive temperature measurements must not exceed 8 hours.  
(2) The interval between the first and last temperature measurements must be at least 19 hours.  
(3) Body temperature must be between 35°C and 43°C.

**Disclaimer:**  
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional medical judgment.
Clinical decisions should always be made by qualified healthcare professionals based on comprehensive clinical evaluation.
""")


# ---------------------------------------------------
# Manual Data Entry
# ---------------------------------------------------
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

manual_df["Temperature"] = np.nan

edited_df = st.data_editor(
    manual_df,
    num_rows="dynamic",
    use_container_width=True
)


# ---------------------------------------------------
# Convert Temperature to Numeric
# ---------------------------------------------------
edited_df["Temperature"] = pd.to_numeric(
    edited_df["Temperature"],
    errors="coerce"
)

edited_df = edited_df.dropna(
    subset=["Temperature"]
).copy()


# ---------------------------------------------------
# Create DateTime
# ---------------------------------------------------
df = pd.DataFrame()

if not edited_df.empty:

    df = edited_df.copy()

    today = datetime.today().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    df["DateTime"] = df.apply(
        lambda row: (
            today - timedelta(days=1)
            if row["Day"] == "Day1"
            else today
        )
        + timedelta(
            hours=int(row["Time"][:2]),
            minutes=int(row["Time"][3:])
        ),
        axis=1
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)


# ---------------------------------------------------
# Proceed if Data Exists
# ---------------------------------------------------
if not df.empty:

    # ---------------------------------------------------
    # Define the 24-hour observation window
    # 08:00 previous day → 08:00 current day
    # ---------------------------------------------------

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

    df_24h = df_24h.reset_index(drop=True)


    # ---------------------------------------------------
    # Check whether data exist
    # ---------------------------------------------------
    if df_24h.empty:

        st.warning(
            "No data available in the last 24 hours "
            "(08:00 → 08:00)."
        )

    else:

        # ---------------------------------------------------
        # Check Temperature Range
        # ---------------------------------------------------

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


        # ---------------------------------------------------
        # Check 8-hour Maximum Gap
        # ---------------------------------------------------

        time_diff_hours = (
            df_24h["DateTime"]
            .diff()
            .dt.total_seconds()
            / 3600
        )

        # The first observation has no preceding observation
        max_gap_hours = time_diff_hours.iloc[1:].max()

        # If only one measurement is entered
        if len(df_24h) < 2:

            st.error(
                "At least two temperature measurements "
                "are required."
            )

            st.stop()


        # ---------------------------------------------------
        # Check 19-hour Minimum Duration
        # ---------------------------------------------------

        total_duration_hours = (
            df_24h["DateTime"].max()
            - df_24h["DateTime"].min()
        ).total_seconds() / 3600


        # ---------------------------------------------------
        # Apply Inclusion Criteria
        # ---------------------------------------------------

        if max_gap_hours > 8:

            st.error(
                f"Invalid input: The maximum interval between "
                f"consecutive temperature measurements is "
                f"{max_gap_hours:.1f} hours. "
                f"The interval must not exceed 8 hours."
            )

            st.stop()


        if total_duration_hours < 19:

            st.error(
                f"Insufficient observation period: The interval "
                f"between the first and last temperature "
                f"measurements is {total_duration_hours:.1f} hours. "
                f"It must be at least 19 hours."
            )

            st.stop()


        # ---------------------------------------------------
        # Display Data Quality Information
        # ---------------------------------------------------

        st.success(
            f"Input criteria satisfied: "
            f"maximum measurement interval = "
            f"{max_gap_hours:.1f} h; "
            f"observation duration = "
            f"{total_duration_hours:.1f} h."
        )


        # ---------------------------------------------------
        # Calculate Hours
        # ---------------------------------------------------

        df_24h["Hours"] = (
            df_24h["DateTime"]
            - df_24h["DateTime"].min()
        ).dt.total_seconds() / 3600


        # ===================================================
        # Features
        # ===================================================

        max_bt = df_24h["Temperature"].max()

        min_bt = df_24h["Temperature"].min()

        mean_bt = df_24h["Temperature"].mean()

        std_bt = df_24h["Temperature"].std()


        # ---------------------------------------------------
        # Temperature Slope
        # ---------------------------------------------------

        X = df_24h["Hours"].values.reshape(-1, 1)

        y = df_24h["Temperature"].values

        slope = LinearRegression().fit(
            X,
            y
        ).coef_[0]


        # ---------------------------------------------------
        # Last 8-hour Maximum Temperature
        # ---------------------------------------------------

        last_time = df_24h["Hours"].max()

        last_8h = df_24h[
            df_24h["Hours"] >= last_time - 8
        ]

        max_last8 = last_8h["Temperature"].max()


        # ---------------------------------------------------
        # Temperature Range
        # ---------------------------------------------------

        range_bt = max_bt - min_bt


        # ---------------------------------------------------
        # Difference Between Last 8h Maximum and Overall Maximum
        #
        # Confirmed by user:
        # T_Diff_MaxLast8H_Max = T_MaxLast8H - T_Max
        # ---------------------------------------------------

        diff_last8_allmax = max_last8 - max_bt


        # ---------------------------------------------------
        # Feature Vector
        # ---------------------------------------------------

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


        # ===================================================
        # Model Prediction
        # ===================================================

        try:

            scaler = joblib.load(
                "scaler.pkl"
            )

            svm_model = joblib.load(
                "svm_model.pkl"
            )


            # ---------------------------------------------------
            # Scale Features
            # ---------------------------------------------------

            features_scaled = scaler.transform(
                np.array(features).reshape(1, -1)
            )


            # ---------------------------------------------------
            # Prediction Result
            # ---------------------------------------------------

            st.subheader("🤖 Prediction Result")


            # ---------------------------------------------------
            # Obtain Probability
            # ---------------------------------------------------

            if hasattr(
                svm_model,
                "predict_proba"
            ):

                pred_prob = svm_model.predict_proba(
                    features_scaled
                )[0][1]

            else:

                st.error(
                    "The loaded SVM model does not support "
                    "probability prediction (predict_proba)."
                )

                st.stop()


            # ---------------------------------------------------
            # Classification Threshold
            # ---------------------------------------------------

            threshold = 0.5


            # ---------------------------------------------------
            # Display Prediction
            # ---------------------------------------------------

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


            # ---------------------------------------------------
            # Display Probability
            # ---------------------------------------------------

            st.metric(
                label="Predicted Probability of Fever",
                value=f"{pred_prob:.3f}"
            )


        except FileNotFoundError as e:

            st.error(
                f"Model or scaler file not found: {e}"
            )

        except Exception as e:

            st.error(
                f"Error loading scaler or model: {e}"
            )


        # ===================================================
        # Temperature Trend
        # ===================================================

        st.subheader(
            "📉 Temperature Trend (Last 24h)"
        )

        from matplotlib.dates import (
            HourLocator,
            DateFormatter
        )

        fig, ax = plt.subplots()


        # ---------------------------------------------------
        # Temperature Curve
        # ---------------------------------------------------

        ax.plot(
            df_24h["DateTime"],
            df_24h["Temperature"],
            marker="o",
            label="Temperature"
        )


        # ---------------------------------------------------
        # Fever Threshold
        # ---------------------------------------------------

        ax.axhline(
            y=38,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label="Fever Threshold (38°C)"
        )


        # ---------------------------------------------------
        # X-axis: Hourly Tick
        # ---------------------------------------------------

        ax.xaxis.set_major_locator(
            HourLocator(interval=1)
        )

        ax.xaxis.set_major_formatter(
            DateFormatter("%H:%M")
        )


        # ---------------------------------------------------
        # Axis Settings
        # ---------------------------------------------------

        ax.set_ylim(
            35,
            43
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Temperature (°C)"
        )

        ax.grid(
            True
        )

        ax.legend()


        plt.xticks(
            rotation=45,
            ha="left"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )
