import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib
from dateutil import parser

# ---------------------------------------------------
# Title & Introduction
# ---------------------------------------------------
st.title("📈 Fever Prediction in Children")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the last day** 
to predict whether a fever may occur in the coming days.

**Input Options:**  
Manual entry: edit temperatures directly in the table below.

**Note:**  
(1) The interval between any two consecutive temperature measurements did not exceed 8 hours.  
(2) The interval between the first and last temperature measurements was at least 19 hours. 

**Disclaimer:**  
The prediction results provided by this app are for research and informational purposes only.
They should not be considered as medical advice, diagnosis, or a substitute for professional medical judgment. 
Clinical decisions should always be made by qualified healthcare professionals based on comprehensive clinical evaluation.
""")

# ----------------------
# Manual Entry Only
# ----------------------
st.subheader("Manual Data Entry (editable table)")

day1_times = [f"{h:02d}:00" for h in range(8,24)]
day2_times = [f"{h:02d}:00" for h in range(0,8)]
all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]

manual_df = pd.DataFrame(all_times, columns=["Day", "Time"])
manual_df["Temperature"] = np.nan

edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)
edited_df = edited_df.dropna(subset=["Temperature"])

df = pd.DataFrame()

if not edited_df.empty:
    df = edited_df.copy()
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    df["DateTime"] = df.apply(lambda row: (
        today - timedelta(days=1) if row["Day"]=="Day1" else today
    ) + timedelta(hours=int(row["Time"][:2]), minutes=int(row["Time"][3:])), axis=1)
    df = df.sort_values("DateTime").reset_index(drop=True)

# ----------------------
# Proceed if Data Exists
# ----------------------
if not df.empty:
    last_date = df["DateTime"].dt.date.max()
    end_time = datetime.combine(last_date, datetime.min.time()) + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)
    df_24h = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)].copy().reset_index(drop=True)

    if df_24h.empty:
        st.warning("No data available in the last 24 hours (08:00 → 08:00).")
    else:
        df_24h["Hours"] = (df_24h["DateTime"] - df_24h["DateTime"].min()).dt.total_seconds()/3600

        # ---------------------- Features ----------------------
        max_bt = df_24h["Temperature"].max()
        min_bt = df_24h["Temperature"].min()
        mean_bt = df_24h["Temperature"].mean()
        std_bt = df_24h["Temperature"].std()

        X = df_24h["Hours"].values.reshape(-1,1)
        y = df_24h["Temperature"].values
        slope = LinearRegression().fit(X, y).coef_[0]

        last_time = df_24h["Hours"].max()
        last_8h = df_24h[df_24h["Hours"] >= last_time-8]
        max_last8 = last_8h["Temperature"].max()
        range_bt = max_bt - min_bt
        diff_last8_allmax = max_last8 - max_bt

        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # ---------------------- Model Prediction ----------------------
        try:
            scaler = joblib.load("scaler.pkl")
            svm_model = joblib.load("svm_model.pkl")
            features_scaled = scaler.transform(np.array(features).reshape(1,-1))

            st.subheader("🤖 Prediction Result")
            if hasattr(svm_model, "predict_proba"):
                pred_prob = svm_model.predict_proba(features_scaled)[0][1]
            else:
                pred_prob = svm_model.decision_function(features_scaled)[0]

            threshold = 0.5
            if pred_prob >= threshold:
                st.success(f"Prediction: Fever expected in the coming day (Score/Probability={pred_prob:.3f} ≥ {threshold})")
            else:
                st.info(f"Prediction: No fever expected in the coming day (Score/Probability={pred_prob:.3f} < {threshold})")

        except Exception as e:
            st.error(f"Error loading scaler or model: {e}")

        # ---------------------- Temperature Trend ----------------------
        st.subheader("📉 Temperature Trend (Last 24h)")

        from matplotlib.dates import HourLocator, DateFormatter
        fig, ax = plt.subplots()

        ax.plot(df_24h["DateTime"], df_24h["Temperature"], marker='o', label="Temperature")
        ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")

        # X 軸：每小時一個刻度
        ax.xaxis.set_major_locator(HourLocator(interval=1))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))

        ax.set_ylim(35, 43)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True)
        ax.legend()

        plt.xticks(rotation=45, ha='left')
        st.pyplot(fig)

#else:
#    st.info("⬆️ Please fill in temperatures manually to begin analysis.")


