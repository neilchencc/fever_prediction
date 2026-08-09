import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
import joblib


# ============================================================
# Title & Introduction
# ============================================================

st.title("📈 Fever Prediction in Children")

st.markdown("""
**App Description:**

This app uses historical body temperature records from **08:00 of the previous day
to 08:00 of the last day** to predict whether a fever may occur in the coming day.

**Input Method:**

Enter the child's temperature measurements manually in the table below.

**Note:**
1. Please enter temperature measurements between 08:00 of the previous day and 08:00 of today.
2. The interval between any two consecutive temperature measurements should not exceed 8 hours.
3. The interval between the first and last temperature measurements should be at least 19 hours.

**Disclaimer:**
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional
medical judgment. Clinical decisions should always be made by qualified healthcare
professionals based on comprehensive clinical evaluation.
""")


# ============================================================
# Manual Entry
# ============================================================

st.subheader("🌡️ Manual Data Entry")

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
    width="stretch"
)

edited_df = edited_df.dropna(subset=["Temperature"]).copy()


# ============================================================
# Create DateTime
# ============================================================

df = pd.DataFrame()

if not edited_df.empty:

    today = datetime.today().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    def create_datetime(row):
        base_date = (
            today - timedelta(days=1)
            if row["Day"] == "Day1"
            else today
        )

        hour = int(row["Time"][:2])
        minute = int(row["Time"][3:])

        return base_date + timedelta(
            hours=hour,
            minutes=minute
        )

    df = edited_df.copy()

    df["DateTime"] = df.apply(
        create_datetime,
        axis=1
    )

    df["Temperature"] = pd.to_numeric(
        df["Temperature"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Temperature"]
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)


# ============================================================
# Proceed if Data Exists
# ============================================================

if df.empty:

    st.info(
        "⬆️ Please enter temperature measurements above "
        "to begin analysis."
    )

else:

    # --------------------------------------------------------
    # Last 24 hours: 08:00 → 08:00
    # --------------------------------------------------------

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


    if df_24h.empty:

        st.warning(
            "No data available in the last 24 hours "
            "(08:00 → 08:00)."
        )

    else:

        # ====================================================
        # Check Measurement Interval
        # ====================================================

        if len(df_24h) >= 2:

            intervals = (
                df_24h["DateTime"]
                .diff()
                .dt.total_seconds()
                / 3600
            )

            max_interval = intervals.iloc[1:].max()

            if max_interval > 8:
                st.warning(
                    f"⚠️ The longest interval between "
                    f"temperature measurements is "
                    f"{max_interval:.1f} hours, which exceeds "
                    f"the recommended 8-hour interval."
                )


        # ====================================================
        # Check Observation Duration
        # ====================================================

        observation_hours = (
            (
                df_24h["DateTime"].max()
                - df_24h["DateTime"].min()
            ).total_seconds()
            / 3600
        )

        if observation_hours < 19:

            st.warning(
                f"⚠️ The observation period is only "
                f"{observation_hours:.1f} hours. "
                f"At least 19 hours of observation is recommended."
            )


        # ====================================================
        # Features
        # ====================================================

        df_24h["Hours"] = (
            (
                df_24h["DateTime"]
                - df_24h["DateTime"].min()
            )
            .dt.total_seconds()
            / 3600
        )

        max_bt = df_24h["Temperature"].max()
        min_bt = df_24h["Temperature"].min()
        mean_bt = df_24h["Temperature"].mean()
        std_bt = df_24h["Temperature"].std()

        # Avoid NaN when there is only one measurement
        if pd.isna(std_bt):
            std_bt = 0.0

        X = df_24h["Hours"].values.reshape(-1, 1)
        y = df_24h["Temperature"].values

        if len(df_24h) >= 2:

            slope = (
                LinearRegression()
                .fit(X, y)
                .coef_[0]
            )

        else:

            slope = 0.0


        last_time = df_24h["Hours"].max()

        last_8h = df_24h[
            df_24h["Hours"] >= last_time - 8
        ]

        max_last8 = last_8h["Temperature"].max()

        range_bt = max_bt - min_bt

        diff_last8_allmax = (
            max_last8 - max_bt
        )


        # ----------------------------------------------------
        # Feature names
        # ----------------------------------------------------

        feature_names = [
            "max_bt",
            "min_bt",
            "mean_bt",
            "std_bt",
            "slope",
            "range_bt",
            "max_last8",
            "diff_last8_allmax"
        ]

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

        features_df = pd.DataFrame(
            [features],
            columns=feature_names
        )


        # ====================================================
        # Model Prediction
        # ====================================================

        st.subheader("🤖 Prediction Result")

        try:

            scaler = joblib.load(
                "scaler.pkl"
            )

            svm_model = joblib.load(
                "svm_model.pkl"
            )


            # ------------------------------------------------
            # Keep feature names consistent with the scaler
            # ------------------------------------------------

            if hasattr(
                scaler,
                "feature_names_in_"
            ):

                scaler_features = list(
                    scaler.feature_names_in_
                )

                features_for_scaler = features_df[
                    scaler_features
                ]

            else:

                features_for_scaler = features_df


            features_scaled = scaler.transform(
                features_for_scaler
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            if hasattr(
                svm_model,
                "predict_proba"
            ):

                pred_prob = (
                    svm_model
                    .predict_proba(features_scaled)[0][1]
                )

                threshold = 0.5

                if pred_prob >= threshold:

                    st.success(
                        f"Prediction: Fever expected in "
                        f"the coming day "
                        f"(Probability = {pred_prob:.3f}, "
                        f"threshold = {threshold})"
                    )

                else:

                    st.info(
                        f"Prediction: No fever expected "
                        f"in the coming day "
                        f"(Probability = {pred_prob:.3f}, "
                        f"threshold = {threshold})"
                    )


            else:

                # SVC without probability=True
                decision_score = (
                    svm_model
                    .decision_function(
                        features_scaled
                    )[0]
                )

                # For SVC decision_function,
                # 0 is the usual classification boundary.
                threshold = 0.0

                if decision_score >= threshold:

                    st.success(
                        f"Prediction: Fever expected in "
                        f"the coming day "
                        f"(Decision score = "
                        f"{decision_score:.3f})"
                    )

                else:

                    st.info(
                        f"Prediction: No fever expected "
                        f"in the coming day "
                        f"(Decision score = "
                        f"{decision_score:.3f})"
                    )


        except FileNotFoundError as e:

            st.error(
                f"Missing model file: {e.filename}"
            )

        except Exception as e:

            st.error(
                f"Error loading scaler or model: {e}"
            )


        # ====================================================
        # Data Preview
        # ====================================================

        st.subheader(
            "🧾 Data Preview (Last 24h)"
        )

        df_preview = df_24h.copy()

        df_preview["Date"] = (
            df_preview["DateTime"]
            .dt.strftime("%Y-%m-%d")
        )

        df_preview["Time"] = (
            df_preview["DateTime"]
            .dt.strftime("%H:%M")
        )

        df_preview["Temperature"] = (
            df_preview["Temperature"]
            .map(lambda x: f"{x:.1f}")
        )


        def highlight_temp(val):

            try:

                if (
                    float(val) < 35
                    or float(val) > 43
                ):

                    return (
                        "color: red; "
                        "font-weight: bold"
                    )

            except Exception:

                pass

            return ""


        st.dataframe(
            df_preview[
                [
                    "Date",
                    "Time",
                    "Temperature"
                ]
            ].style.map(
                highlight_temp
            ),
            width="stretch"
        )


        # ====================================================
        # Temperature Trend
        # ====================================================

        st.subheader(
            "📉 Temperature Trend (Last 24h)"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            df_24h["DateTime"],
            df_24h["Temperature"],
            marker="o",
            linewidth=2,
            label="Temperature"
        )

        ax.axhline(
            y=38,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label="Fever Threshold (38°C)"
        )

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
            True,
            alpha=0.3
        )

        ax.legend()

        # Prevent Matplotlib from generating
        # tens of thousands of ticks.
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=8)
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)
```

