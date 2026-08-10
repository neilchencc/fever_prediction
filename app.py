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
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the last day**
to predict whether a fever may occur in the coming day.

**Input Method:**  
Enter body temperatures directly in the table below, then press
**"Start Prediction"** to perform the prediction.

**Note:**  
(1) The measurement interval between any two consecutive temperature records
should not exceed 8 hours.  
(2) The interval between the first and last temperature measurements
should be at least 19 hours.

**Disclaimer:**  
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional medical judgment.
Clinical decisions should always be made by qualified healthcare professionals based on comprehensive clinical evaluation.
""")


# ---------------------------------------------------
# Manual Data Entry
# ---------------------------------------------------

st.subheader("🌡️ Manual Temperature Entry")

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
    use_container_width=True,
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
    },
    hide_index=True
)


# ---------------------------------------------------
# Start Prediction Button
# ---------------------------------------------------

st.markdown("---")

start_prediction = st.button(
    "🔮 Start Prediction",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if start_prediction:

    # Remove rows without temperature
    edited_df = edited_df.dropna(
        subset=["Temperature"]
    ).copy()

    # Check whether any temperature was entered
    if edited_df.empty:

        st.warning(
            "⚠️ Please enter at least one body temperature "
            "before starting the prediction."
        )

    else:

        # ---------------------------------------------------
        # Create DateTime
        # ---------------------------------------------------

        today = datetime.today().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0
        )

        edited_df["DateTime"] = edited_df.apply(
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

        edited_df = edited_df.sort_values(
            "DateTime"
        ).reset_index(drop=True)


        # ---------------------------------------------------
        # Check measurement interval
        # ---------------------------------------------------

        if len(edited_df) >= 2:

            time_diffs = (
                edited_df["DateTime"]
                .diff()
                .dropna()
                .dt.total_seconds()
                / 3600
            )

            max_interval = time_diffs.max()

            if max_interval > 8:

                st.warning(
                    f"⚠️ The interval between consecutive "
                    f"temperature measurements is {max_interval:.1f} hours. "
                    f"It should not exceed 8 hours."
                )


        # ---------------------------------------------------
        # Check total observation period
        # ---------------------------------------------------

        if len(edited_df) >= 2:

            total_duration = (
                edited_df["DateTime"].max()
                - edited_df["DateTime"].min()
            ).total_seconds() / 3600

            if total_duration < 19:

                st.warning(
                    f"⚠️ The interval between the first and last "
                    f"temperature measurements is only "
                    f"{total_duration:.1f} hours. "
                    f"It should be at least 19 hours."
                )


        # ---------------------------------------------------
        # Select Last 24 Hours
        # ---------------------------------------------------

        last_date = edited_df["DateTime"].dt.date.max()

        end_time = (
            datetime.combine(
                last_date,
                datetime.min.time()
            )
            + timedelta(hours=8)
        )

        start_time = end_time - timedelta(hours=24)

        df_24h = edited_df[
            (edited_df["DateTime"] >= start_time)
            & (edited_df["DateTime"] <= end_time)
        ].copy().reset_index(drop=True)


        # ---------------------------------------------------
        # Check Data
        # ---------------------------------------------------

        if df_24h.empty:

            st.warning(
                "⚠️ No data available in the last 24 hours "
                "(08:00 → 08:00)."
            )

        elif len(df_24h) < 2:

            st.warning(
                "⚠️ At least two temperature measurements "
                "are required for prediction."
            )

        else:

            # ---------------------------------------------------
            # Calculate Hours
            # ---------------------------------------------------

            df_24h["Hours"] = (
                df_24h["DateTime"]
                - df_24h["DateTime"].min()
            ).dt.total_seconds() / 3600


            # ---------------------------------------------------
            # Features
            # ---------------------------------------------------

            max_bt = df_24h["Temperature"].max()

            min_bt = df_24h["Temperature"].min()

            mean_bt = df_24h["Temperature"].mean()

            std_bt = df_24h["Temperature"].std()

            # If only two measurements exist, std is still valid.
            # If necessary, replace NaN with 0.
            if pd.isna(std_bt):
                std_bt = 0.0


            # ---------------------------------------------------
            # Temperature Slope
            # ---------------------------------------------------

            X = df_24h["Hours"].values.reshape(-1, 1)

            y = df_24h["Temperature"].values

            linear_model = LinearRegression()

            linear_model.fit(X, y)

            slope = linear_model.coef_[0]


            # ---------------------------------------------------
            # Last 8 Hours
            # ---------------------------------------------------

            last_time = df_24h["Hours"].max()

            last_8h = df_24h[
                df_24h["Hours"] >= last_time - 8
            ]

            max_last8 = last_8h["Temperature"].max()


            # ---------------------------------------------------
            # Additional Features
            # ---------------------------------------------------


            range_bt = max_bt - min_bt

            diff_last8_allmax = (
                max_last8 - max_bt
            )


            # ---------------------------------------------------
            # Final 8 Features
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


            # ---------------------------------------------------
            # Display Features
            # ---------------------------------------------------

            st.subheader("📊 Calculated Features")

            feature_df = pd.DataFrame({
                "Feature": [
                    "Maximum temperature",
                    "Minimum temperature",
                    "Mean temperature",
                    "Temperature SD",
                    "Temperature slope",
                    "Temperature range",
                    "Maximum temperature in last 8h",
                    "Last 8h max - overall max"
                ],
                "Value": features
            })

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------------------
            # SVM Model Prediction
            # ---------------------------------------------------

            try:

                scaler = joblib.load(
                    "scaler.pkl"
                )

                svm_model = joblib.load(
                    "svm_model.pkl"
                )


                # Scale features
                features_array = np.array(
                    features
                ).reshape(1, -1)

                features_scaled = scaler.transform(
                    features_array
                )


                # ---------------------------------------------------
                # Prediction Result
                # ---------------------------------------------------

                st.subheader("🤖 Prediction Result")


                # Your SVM uses probability=True
                pred_prob = svm_model.predict_proba(
                    features_scaled
                )[0][1]

                # Binary prediction
                prediction = svm_model.predict(
                    features_scaled
                )[0]


                # ---------------------------------------------------
                # Display Prediction
                # ---------------------------------------------------

                if prediction == 1:

                    st.success(
                        f"🌡️ Prediction: Fever expected in the coming day\n\n"
                        f"Probability of fever: "
                        f"**{pred_prob:.1%}**"
                    )

                else:

                    st.info(
                        f"✅ Prediction: No fever expected in the coming day\n\n"
                        f"Probability of fever: "
                        f"**{pred_prob:.1%}**"
                    )


                # ---------------------------------------------------
                # Probability Bar
                # ---------------------------------------------------

                st.progress(
                    float(pred_prob)
                )

                st.caption(
                    f"Fever probability: {pred_prob:.3f}"
                )


            except FileNotFoundError as e:

                st.error(
                    f"❌ Missing model file: {e.filename}"
                )

            except Exception as e:

                st.error(
                    f"❌ Error loading scaler or SVM model: {e}"
                )


            # ---------------------------------------------------
            # Data Preview
            # ---------------------------------------------------

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


            # ---------------------------------------------------
            # Highlight Abnormal Temperature
            # ---------------------------------------------------

            def highlight_temp(val):

                try:

                    if float(val) < 35 or float(val) > 43:

                        return (
                            "color: red; "
                            "font-weight: bold"
                        )

                except Exception:

                    pass

                return ""


            st.dataframe(
                df_preview[
                    ["Date", "Time", "Temperature"]
                ].style.map(highlight_temp),
                use_container_width=True
            )


            # ---------------------------------------------------
            # Temperature Trend
            # ---------------------------------------------------

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

            plt.xticks(
                rotation=45,
                ha="left"
            )

            plt.tight_layout()

            st.pyplot(fig)
