import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib


# ============================================================
# Title & Introduction
# ============================================================

st.title("📈 Fever Prediction in Children")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day
to 08:00 of the current day** to predict whether a fever may occur in the coming day.

**Input Method:**  
Enter body temperatures directly in the table below, then press
**"Start Prediction"** to perform the prediction.

**Note:**  
(1) The interval between any two consecutive temperature measurements
should not exceed 8 hours.

(2) The interval between the first and last temperature measurements
should be at least 19 hours.

**Disclaimer:**  
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional
medical judgment.

Clinical decisions should always be made by qualified healthcare professionals
based on comprehensive clinical evaluation.
""")


# ============================================================
# Manual Temperature Entry
# ============================================================

st.subheader("🌡️ Manual Temperature Entry")

day1_times = [
    f"{h:02d}:00"
    for h in range(8, 24)
]

day2_times = [
    f"{h:02d}:00"
    for h in range(0, 8)
]

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
    num_rows="fixed",
    use_container_width=True,
    hide_index=True,
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
            min_value=30.0,
            max_value=45.0,
            step=0.1,
            format="%.1f"
        )
    }
)


# ============================================================
# Start Prediction Button
# ============================================================

st.markdown("---")

start_prediction = st.button(
    "🔮 Start Prediction",
    type="primary",
    use_container_width=True
)


# ============================================================
# Prediction
# ============================================================

if start_prediction:

    # --------------------------------------------------------
    # Remove rows without temperature
    # --------------------------------------------------------

    edited_df = edited_df.dropna(
        subset=["Temperature"]
    ).copy()


    # --------------------------------------------------------
    # Check whether temperature data exists
    # --------------------------------------------------------

    if edited_df.empty:

        st.warning(
            "⚠️ Please enter body temperature data "
            "before starting the prediction."
        )

    else:

        # ----------------------------------------------------
        # Make sure temperature is numeric
        # ----------------------------------------------------

        edited_df["Temperature"] = pd.to_numeric(
            edited_df["Temperature"],
            errors="coerce"
        )

        edited_df = edited_df.dropna(
            subset=["Temperature"]
        ).copy()


        # ====================================================
        # Define Day1 / Day2
        # ====================================================

        today = datetime.today().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        day1_date = today - timedelta(days=1)
        day2_date = today


        # ====================================================
        # Create DateTime
        # ====================================================

        def create_datetime(row):

            time_value = datetime.strptime(
                row["Time"],
                "%H:%M"
            ).time()

            if row["Day"] == "Day1":
                date_value = day1_date
            else:
                date_value = day2_date

            return datetime.combine(
                date_value.date(),
                time_value
            )


        edited_df["DateTime"] = edited_df.apply(
            create_datetime,
            axis=1
        )


        edited_df = edited_df.sort_values(
            "DateTime"
        ).reset_index(drop=True)


        # ====================================================
        # Fixed 24-hour Observation Period
        #
        # Day1 08:00 → Day2 08:00
        # ====================================================

        start_time = today - timedelta(hours=16)

        end_time = today + timedelta(hours=8)


        # ====================================================
        # Select observation period
        # ====================================================

        df_24h = edited_df[
            (edited_df["DateTime"] >= start_time)
            &
            (edited_df["DateTime"] <= end_time)
        ].copy()

        df_24h = df_24h.sort_values(
            "DateTime"
        ).reset_index(drop=True)


        # ====================================================
        # Check data
        # ====================================================

        if df_24h.empty:

            st.warning(
                "⚠️ No temperature data is available "
                "in the observation period "
                "(Day1 08:00 → Day2 08:00)."
            )

        elif len(df_24h) < 2:

            st.warning(
                "⚠️ At least two temperature measurements "
                "are required for prediction."
            )

        else:

            # =================================================
            # Data Quality Check
            # =================================================

            time_diffs = (
                df_24h["DateTime"]
                .diff()
                .dropna()
                .dt.total_seconds()
                / 3600
            )

            max_interval = time_diffs.max()


            if max_interval > 8:

                st.warning(
                    f"⚠️ The maximum interval between consecutive "
                    f"temperature measurements is "
                    f"{max_interval:.1f} hours. "
                    f"It should not exceed 8 hours."
                )


            # -------------------------------------------------
            # First-to-last measurement interval
            # -------------------------------------------------

            total_duration = (
                df_24h["DateTime"].max()
                - df_24h["DateTime"].min()
            ).total_seconds() / 3600


            if total_duration < 19:

                st.warning(
                    f"⚠️ The interval between the first and last "
                    f"temperature measurements is only "
                    f"{total_duration:.1f} hours. "
                    f"It should be at least 19 hours."
                )


            # =================================================
            # Calculate Hours
            # =================================================

            df_24h["Hours"] = (
                df_24h["DateTime"]
                - df_24h["DateTime"].min()
            ).dt.total_seconds() / 3600


            # =================================================
            # Features
            # =================================================

            max_bt = df_24h["Temperature"].max()

            min_bt = df_24h["Temperature"].min()

            mean_bt = df_24h["Temperature"].mean()

            std_bt = df_24h["Temperature"].std()

            if pd.isna(std_bt):
                std_bt = 0.0


            # =================================================
            # Temperature Slope
            # =================================================

            X = df_24h[
                "Hours"
            ].values.reshape(-1, 1)

            y = df_24h[
                "Temperature"
            ].values


            linear_model = LinearRegression()

            linear_model.fit(
                X,
                y
            )

            slope = linear_model.coef_[0]


            # =================================================
            # Last 8 Hours
            #
            # Fixed definition:
            # Day2 00:00 → Day2 08:00
            #
            # Actual measurement points:
            # 00:00, 01:00, ..., 07:00
            # =================================================

            last_8h_start = today

            last_8h_end = today + timedelta(hours=8)


            last_8h = df_24h[
                (df_24h["DateTime"] >= last_8h_start)
                &
                (df_24h["DateTime"] < last_8h_end)
            ].copy()


            # =================================================
            # Check Last 8-hour Data
            # =================================================

            if last_8h.empty:

                st.error(
                    "❌ No temperature measurements were entered "
                    "during Day2 00:00–07:00. "
                    "The Last 8-hour features cannot be calculated."
                )

                max_last8 = np.nan

            else:

                max_last8 = last_8h[
                    "Temperature"
                ].max()


            # =================================================
            # Additional Features
            # =================================================

            range_bt = (
                max_bt
                - min_bt
            )


            if pd.isna(max_last8):

                diff_last8_allmax = np.nan

            else:

                diff_last8_allmax = (
                    max_last8
                    - max_bt
                )


            # =================================================
            # Final 8 Features
            # =================================================

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


            # =================================================
            # Check Features
            # =================================================

            if any(
                pd.isna(x)
                for x in features
            ):

                st.error(
                    "❌ Some required features could not be calculated. "
                    "Please make sure that temperature measurements "
                    "are available during Day2 00:00–07:00."
                )

            else:

                # =================================================
                # SVM Prediction
                # =================================================

                try:

                    # ------------------------------------------------
                    # Load scaler
                    # ------------------------------------------------

                    scaler = joblib.load(
                        "scaler.pkl"
                    )


                    # ------------------------------------------------
                    # Load SVM model
                    # ------------------------------------------------

                    svm_model = joblib.load(
                        "svm_model.pkl"
                    )


                    # ------------------------------------------------
                    # Convert features
                    # ------------------------------------------------

                    features_array = np.array(
                        features,
                        dtype=float
                    ).reshape(1, -1)


                    # ------------------------------------------------
                    # Standardize features
                    # ------------------------------------------------

                    features_scaled = scaler.transform(
                        features_array
                    )


                    # ------------------------------------------------
                    # SVM prediction
                    # ------------------------------------------------

                    prediction = svm_model.predict(
                        features_scaled
                    )[0]


                    # ------------------------------------------------
                    # Fever probability
                    # ------------------------------------------------

                    pred_prob = svm_model.predict_proba(
                        features_scaled
                    )[0][1]


                    # =================================================
                    # Prediction Result
                    # =================================================

                    st.subheader(
                        "🤖 Prediction Result"
                    )


                    if prediction == 1:

                        result_text = (
                            "🌡️ Prediction: Fever expected "
                            "in the coming day"
                        )

                        result_color = "#b91c1c"

                    else:

                        result_text = (
                            "✅ Prediction: No fever expected "
                            "in the coming day"
                        )

                        result_color = "#15803d"


                    # -------------------------------------------------
                    # Large Prediction + Probability
                    # -------------------------------------------------

                    st.markdown(
                        f"""
                        <div style="
                            font-size: 28px;
                            font-weight: 700;
                            color: {result_color};
                            margin-top: 10px;
                            margin-bottom: 18px;
                            line-height: 1.5;
                        ">
                            {result_text}
                            <span style="
                                margin-left: 25px;
                                font-size: 26px;
                                font-weight: 700;
                            ">
                                Probability: {pred_prob:.1%}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                   
                # =================================================
                # Error Handling
                # =================================================

                except FileNotFoundError as e:

                    st.error(
                        f"❌ Missing model file: {e.filename}"
                    )


                except Exception as e:

                    st.error(
                        f"❌ Error loading scaler or SVM model: {e}"
                    )


                # =================================================
                # Temperature Trend
                # =================================================

                st.subheader(
                    "📉 Temperature Trend"
                )


                fig, ax = plt.subplots(
                    figsize=(14, 5)
                )


                # -------------------------------------------------
                # Temperature line
                # -------------------------------------------------

                ax.plot(
                    df_24h["DateTime"],
                    df_24h["Temperature"],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label="Temperature"
                )


                # -------------------------------------------------
                # Fever threshold
                # -------------------------------------------------

                ax.axhline(
                    y=38,
                    color="darkred",
                    linestyle="--",
                    linewidth=2,
                    label="Fever Threshold (38°C)"
                )


                # =================================================
                # X-axis
                #
                # Day1 08:00
                # Day1 09:00
                # ...
                # Day1 23:00
                # Day2 00:00
                # ...
                # Day2 07:00
                # Day2 08:00
                # =================================================

                hourly_times = pd.date_range(
                    start=start_time,
                    end=end_time,
                    freq="1h"
                )


                hourly_labels = []


                for t in hourly_times:

                    if t.date() == day1_date.date():

                        day_label = "Day1"

                    else:

                        day_label = "Day2"


                    hourly_labels.append(
                        f"{day_label} {t.strftime('%H:%M')}"
                    )


                ax.set_xticks(
                    hourly_times
                )


                ax.set_xticklabels(
                    hourly_labels,
                    rotation=45,
                    ha="right"
                )


                # -------------------------------------------------
                # Y-axis
                # -------------------------------------------------

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


                # -------------------------------------------------
                # Grid
                # -------------------------------------------------

                ax.grid(
                    True,
                    which="major",
                    axis="both",
                    alpha=0.3
                )


                # -------------------------------------------------
                # Legend
                # -------------------------------------------------

                ax.legend()


                plt.tight_layout()


                st.pyplot(fig)


# ============================================================
# Before Prediction
# ============================================================

else:

    st.info(
        "⬆️ Please enter the child's temperature data "
        "and press **Start Prediction** to begin the analysis."
    )
