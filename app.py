import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib  # for loading pkl models

# ---------------------------------------------------
# 📘 Streamlit Title & App Introduction
# ---------------------------------------------------
st.title("📈 Body Temperature Analysis Tool (CSV Upload / Manual Entry + Normalization + Prediction)")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 to 08:00 the following day** 
to predict whether a fever may occur in the coming days.

**Input Options:**  
1. Upload a CSV file with three columns: `Date`, `Time`, `Temperature`  
   - `Date`: in `YYYYMMDD` format (e.g., 20251111)  
   - `Time`: in `HHMM` format (e.g., 0830 for 08:30)  
   - `Temperature`: in Celsius (e.g., 36.5)  
2. Or enter the data manually using the form below:  
   - `Date`: Day 1 or Day 2  
   - `Time`: 08:00, 09:00, …, 23:00  
   - `Temperature`: numeric input in Celsius
""")

# ---------------------------------------------------
# Input choice
# ---------------------------------------------------
input_method = st.radio("Select input method:", ["Upload CSV file", "Manual Entry"])

# ---------------------------------------------------
# Initialize empty dataframe
# ---------------------------------------------------
df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# ---------------------------------------------------
# CSV Upload
# ---------------------------------------------------
if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Please upload a CSV file with three columns: Date, Time, Temperature", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        expected_cols = ["Date", "Time", "Temperature"]
        df.columns = expected_cols[:len(df.columns)]
        df.columns = [c.strip() for c in df.columns]
        # Convert to datetime
        df["DateTime"] = df.apply(
            lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
            axis=1
        )
        df = df.sort_values("DateTime").reset_index(drop=True)

# ---------------------------------------------------
# Manual Entry
# ---------------------------------------------------
elif input_method == "Manual Entry":
    st.subheader("Manual Data Entry")
    manual_data = []
    for i in range(1, 3):  # Day 1 and Day 2
        st.markdown(f"**{i}. Day {i} entries**")
        n_rows = st.number_input(f"How many entries for Day {i}?", min_value=1, max_value=24, value=1, key=f"rows_{i}")
        for j in range(int(n_rows)):
            col1, col2, col3 = st.columns(3)
            with col1:
                date_label = f"Day {i} (logical date)"
                date_val = st.text_input(f"{date_label} entry {j+1}", value=f"Day {i}", key=f"day_{i}_{j}")
            with col2:
                time_val = st.selectbox(f"Time for Day {i} entry {j+1}", 
                                        [f"{h:02d}:00" for h in range(8,24)], key=f"time_{i}_{j}")
            with col3:
                temp_val = st.number_input(f"Temperature (°C) entry {j+1}", min_value=30.0, max_value=43.0, value=36.5, step=0.1, key=f"temp_{i}_{j}")
            manual_data.append({"Date": f"Day{i}", "Time": time_val.replace(":", ""), "Temperature": temp_val})
    if manual_data:
        df = pd.DataFrame(manual_data)
        # Map Day1/Day2 to logical datetime
        base_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        df["DateTime"] = df.apply(lambda row: base_date + timedelta(days=int(row["Date"][-1])-1, 
                                                                hours=int(row["Time"][:2]), 
                                                                minutes=int(row["Time"][2:])), axis=1)
        df = df.sort_values("DateTime").reset_index(drop=True)

# ---------------------------------------------------
# If dataframe is not empty, proceed
# ---------------------------------------------------
if not df.empty:

    # ------------------------------
    # 🤖 Prediction (before Data Preview)
    # ------------------------------
    st.subheader("🤖 Prediction Result")
    try:
        # Feature engineering
        t0 = df["DateTime"].min()
        df["Hours"] = (df["DateTime"] - t0).dt.total_seconds() / 3600

        max_bt = df["Temperature"].max()
        min_bt = df["Temperature"].min()
        mean_bt = df["Temperature"].mean()
        std_bt = df["Temperature"].std()

        X = df["Hours"].values.reshape(-1, 1)
        y = df["Temperature"].values
        model_lr = LinearRegression().fit(X, y)
        slope = model_lr.coef_[0]

        last_time = df["Hours"].max()
        last_8h = df[df["Hours"] >= last_time - 8]
        max_last8 = last_8h["Temperature"].max()

        range_bt = max_bt - min_bt
        diff_last8_allmax = max_last8 - max_bt

        # Build features
        features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]

        # Load models
        scaler = joblib.load("scaler.pkl")
        svm_model = joblib.load("svm_model.pkl")

        features_array = np.array(features).reshape(1,-1)
        features_scaled = scaler.transform(features_array)

        if hasattr(svm_model, "predict_proba"):
            pred_prob = svm_model.predict_proba(features_scaled)[0][1]
        else:
            pred_prob = svm_model.decision_function(features_scaled)[0]

        threshold = 0.5
        if pred_prob >= threshold:
            st.success(f"Prediction: Fever likely (Score/Probability={pred_prob:.3f} ≥ {threshold})")
        else:
            st.info(f"Prediction: No fever expected (Score/Probability={pred_prob:.3f} < {threshold})")

    except FileNotFoundError as e:
        st.error(f"Missing required model file: {e.filename}")
    except Exception as e:
        st.error(f"Error during model loading or prediction: {e}")

    # ------------------------------
    # 🧾 Data Preview
    # ------------------------------
    st.write("### 🧾 Data Preview:")
    st.dataframe(df)

    # ------------------------------
    # Statistical Summary & Trend
    # ------------------------------
    unique_dates = df["Date"].unique()
    start_time = df["DateTime"].min()
    end_time = df["DateTime"].max()
    st.write(f"### ⏱ Analysis Range: {start_time} – {end_time}")

    # Statistical summary
    max_bt = df["Temperature"].max()
    min_bt = df["Temperature"].min()
    mean_bt = df["Temperature"].mean()
    std_bt = df["Temperature"].std()
    X = df["Hours"].values.reshape(-1,1)
    y = df["Temperature"].values
    model_lr = LinearRegression().fit(X,y)
    slope = model_lr.coef_[0]
    last_time = df["Hours"].max()
    last_8h = df[df["Hours"] >= last_time - 8]
    max_last8 = last_8h["Temperature"].max()
    range_bt = max_bt - min_bt
    diff_last8_allmax = max_last8 - max_bt

    feature_names = [
        "Maximum (max)", "Minimum (min)", "Average (mean)", "Standard Deviation (std)",
        "Slope", "Max - Min", "Max of Last 8 Hours", "Last 8h Max - Overall Max"
    ]
    features_values = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]
    result_table = pd.DataFrame({"Feature": feature_names, "Value":[f"{v:.4f}" for v in features_values]})
    st.subheader("📊 Statistical Summary")
    st.table(result_table)

    # Temperature trend plot
    st.subheader("📉 Temperature Trend")
    fig, ax = plt.subplots()
    ax.plot(df["DateTime"], df["Temperature"], marker='o', label="Temperature")
    ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")
    ax.set_ylim(35, 43)
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

else:
    st.info("⬆️ Please upload a CSV file or enter data manually to begin analysis.")






