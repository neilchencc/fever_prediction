import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from dateutil import parser

# -----------------------
# Title & Introduction
# -----------------------
st.title("📈 Body Temperature Analysis Tool (Last 24h Prediction + Feedback & Retraining)")

st.markdown("""
**App Description:**  
This app uses historical body temperature records from **08:00 of the previous day to 08:00 of the last day** 
to predict whether a fever may occur in the coming days.  

You can provide feedback on the prediction and retrain the model with new data.
""")

# -----------------------
# Helper Function for Parsing
# -----------------------
def parse_datetime(date_str, time_str):
    time_str = str(time_str).strip()
    if time_str in ["", "nan", "NaN"]:
        time_str = "00:00"
    elif time_str.isdigit():
        time_str = time_str.zfill(4)
        time_str = time_str[:2] + ":" + time_str[2:]

    try:
        dt_date = parser.parse(str(date_str), dayfirst=False, fuzzy=True)
        dt_time = parser.parse(time_str, fuzzy=True).time()
    except Exception as e:
        raise ValueError(f"Unrecognized date/time: {date_str} {time_str}")
    return datetime.combine(dt_date.date(), dt_time)

# -----------------------
# Input method
# -----------------------
input_method = st.radio("Select input method:", ["Upload CSV file", "Manual Entry"])
df = pd.DataFrame(columns=["Date", "Time", "Temperature"])

# CSV upload
if input_method == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload CSV with columns: Date, Time, Temperature", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = ["Date", "Time", "Temperature"][:len(df.columns)]
        df.columns = [c.strip() for c in df.columns]
        try:
            df["Date"] = df["Date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df["Time"] = df["Time"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df["DateTime"] = df.apply(lambda row: parse_datetime(row["Date"], row["Time"]), axis=1)
            df = df.sort_values("DateTime").reset_index(drop=True)
        except Exception as e:
            st.error(f"Date/Time parsing error: {e}")
            df = pd.DataFrame(columns=["Date", "Time", "Temperature", "DateTime"])

# Manual entry
elif input_method == "Manual Entry":
    day1_times = [f"{h:02d}:00" for h in range(8,24)]
    day2_times = [f"{h:02d}:00" for h in range(0,8)]
    all_times = [("Day1", t) for t in day1_times] + [("Day2", t) for t in day2_times]
    manual_df = pd.DataFrame(all_times, columns=["Day", "Time"])
    manual_df["Temperature"] = np.nan
    edited_df = st.data_editor(manual_df, num_rows="dynamic", use_container_width=True)
    edited_df = edited_df.dropna(subset=["Temperature"])
    if not edited_df.empty:
        df = edited_df.copy()
        today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        df["DateTime"] = df.apply(lambda row: (
            today - timedelta(days=1) if row["Day"]=="Day1" else today
        ) + timedelta(hours=int(row["Time"][:2]), minutes=int(row["Time"][3:])), axis=1)
        df = df.sort_values("DateTime").reset_index(drop=True)

# -----------------------
# Process last 24h
# -----------------------
if not df.empty:
    last_date = df["DateTime"].dt.date.max()
    end_time = datetime.combine(last_date, datetime.min.time()) + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)
    df_24h = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)].copy().reset_index(drop=True)
    
    if df_24h.empty:
        st.warning("No data in last 24 hours (08:00 → 08:00).")
    else:
        df_24h["Hours"] = (df_24h["DateTime"] - df_24h["DateTime"].min()).dt.total_seconds()/3600
        # Features
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

        # -----------------------
        # Prediction
        # -----------------------
        try:
            scaler = joblib.load("scaler.pkl")
            svm_model = joblib.load("svm_model.pkl")
            features_scaled = scaler.transform(np.array(features).reshape(1,-1))
            if hasattr(svm_model, "predict_proba"):
                pred_prob = svm_model.predict_proba(features_scaled)[0][1]
            else:
                pred_prob = svm_model.decision_function(features_scaled)[0]

            threshold = 0.5
            if pred_prob >= threshold:
                st.success(f"Prediction: Fever likely (Score={pred_prob:.3f})")
                prediction_label = 1
            else:
                st.info(f"Prediction: No fever expected (Score={pred_prob:.3f})")
                prediction_label = 0

        except FileNotFoundError:
            st.warning("Model or scaler not found. You can train a new model below.")
            features_scaled = None
            prediction_label = None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            features_scaled = None
            prediction_label = None

        # -----------------------
        # Data Preview
        # -----------------------
        st.subheader("🧾 Data Preview (Last 24h)")
        df_preview = df_24h.copy()
        df_preview["Date"] = df_preview["DateTime"].dt.strftime("%Y-%m-%d")
        df_preview["Time"] = df_preview["DateTime"].dt.strftime("%H:%M")
        st.dataframe(df_preview[["Date", "Time", "Temperature"]])

        # -----------------------
        # Temperature Trend
        # -----------------------
        st.subheader("📉 Temperature Trend (Last 24h)")
        fig, ax = plt.subplots()
        ax.plot(df_24h["DateTime"], df_24h["Temperature"], marker='o', label="Temperature")
        ax.axhline(y=38, color='darkred', linestyle='--', linewidth=2, label="Fever Threshold (38°C)")
        ax.set_ylim(35, 43)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45, ha='left')
        st.pyplot(fig)

        # -----------------------
        # Feedback & Retraining
        # -----------------------
        st.subheader("📝 Feedback")
        feedback = st.radio("Was the prediction correct?", ["Yes", "No"])
        correct_label = st.selectbox("Correct label (0=No fever, 1=Fever)", options=[0,1], index=prediction_label if prediction_label is not None else 0)

        if st.button("Submit feedback and retrain model"):
            # Load previous training data if exists
            try:
                training_df = pd.read_csv("training_data.csv")
            except:
                training_df = pd.DataFrame(columns=[*['max_bt','min_bt','mean_bt','std_bt','slope','range_bt','max_last8','diff_last8_allmax'], 'label'])
            
            # Add new data using pd.concat
            new_row_df = pd.DataFrame([dict(zip(['max_bt','min_bt','mean_bt','std_bt','slope','range_bt','max_last8','diff_last8_allmax'], features), **{'label': correct_label})])
            training_df = pd.concat([training_df, new_row_df], ignore_index=True)
            training_df.to_csv("training_data.csv", index=False)

            # Retrain model only if there are at least 2 classes
            X_train = training_df.iloc[:,:-1].values
            y_train = training_df['label'].values

            if len(np.unique(y_train)) < 2:
                st.warning("Not enough class diversity to retrain model. Need both 0 and 1 labels.")
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                svm_model = SVC(probability=True)
                svm_model.fit(X_train_scaled, y_train)
                joblib.dump(scaler, "scaler.pkl")
                joblib.dump(svm_model, "svm_model.pkl")
                st.success("Model retrained and saved successfully! ✅")









