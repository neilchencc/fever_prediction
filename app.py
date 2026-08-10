import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import joblib


# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("📈 Fever Prediction in Children")

st.markdown("""
This app uses temperature records from **08:00 of the previous day
to 08:00 of the last day** to predict whether fever may occur in the coming day.

**Input:** Enter temperatures manually in the table below.

**Disclaimer:**  
The prediction results are for research and informational purposes only.
They should not be considered medical advice or a substitute for professional
medical judgment.
""")


# ---------------------------------------------------
# Manual Entry
# ---------------------------------------------------
st.subheader("Manual Data Entry")

times = (
    [f"{h:02d}:00" for h in range(8, 24)] +
    [f"{h:02d}:00" for h in range(0, 8)]
)

days = (
    ["Day1"] * 16 +
    ["Day2"] * 8
)

manual_df = pd.DataFrame({
    "Day": days,
    "Time": times,
    "Temperature": np.nan
})

edited_df = st.data_editor(
    manual_df,
    use_container_width=True
)

edited_df = edited_df.dropna(
    subset=["Temperature"]
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if len(edited_df) > 0:

    df = edited_df.copy()

    today = datetime.today().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    def make_datetime(row):
        if row["Day"] == "Day1":
            date = today - timedelta(days=1)
        else:
            date = today

        hour = int(row["Time"][:2])
        minute = int(row["Time"][3:])

        return date + timedelta(
            hours=hour,
            minutes=minute
        )

    df["DateTime"] = df.apply(
        make_datetime,
        axis=1
    )

    df["Temperature"] = pd.to_numeric(
        df["Temperature"]
    )

    df = df.sort_values(
        "DateTime"
    ).reset_index(drop=True)


    # ---------------------------------------------------
    # Last 24 Hours
    # ---------------------------------------------------
    end_time = today + timedelta(hours=8)
    start_time = end_time - timedelta(hours=24)

    df = df[
        (df["DateTime"] >= start_time) &
        (df["DateTime"] <= end_time)
    ].copy()


    if len(df) < 2:

        st.warning(
            "Please enter at least 2 temperature measurements."
        )

    else:

        # ---------------------------------------------------
        # Features
        # ---------------------------------------------------
        df["Hours"] = (
            df["DateTime"] - df["DateTime"].min()
        ).dt.total_seconds() / 3600

        max_bt = df["Temperature"].max()
        min_bt = df["Temperature"].min()
        mean_bt = df["Temperature"].mean()
        std_bt = df["Temperature"].std()

        X = df["Hours"].values.reshape(-1, 1)
        y = df["Temperature"].values

        slope = LinearRegression().fit(
            X, y
        ).coef_[0]

        last_time = df["Hours"].max()

        last_8h = df[
            df["Hours"] >= last_time - 8
        ]

        max_last8 = last_8h["Temperature"].max()

        range_bt = max_bt - min_bt

        diff_last8_allmax = (
            max_last8 - max_bt
        )

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
        # Model
        # ---------------------------------------------------
        try:

            scaler = joblib.load("scaler.pkl")
            model = joblib.load("svm_model.pkl")

            features_scaled = scaler.transform(
                np.array(features).reshape(1, -1)
            )

            if hasattr(model, "predict_proba"):

                probability = model.predict_proba(
                    features_scaled
                )[0][1]

            else:

                probability = model.decision_function(
                    features_scaled
                )[0]

            st.subheader("🤖 Prediction Result")

            if probability >= 0.5:

                st.success(
                    f"Fever expected in the coming day "
                    f"(Score = {probability:.3f})"
                )

            else:

                st.info(
                    f"No fever expected in the coming day "
                    f"(Score = {probability:.3f})"
                )


        except Exception as e:

            st.error(
                f"Model error: {e}"
            )


        # ---------------------------------------------------
        # Data
        # ---------------------------------------------------
        st.subheader("🧾 Temperature Data")

        show_df = df.copy()

        show_df["Date"] = (
            show_df["DateTime"]
            .dt.strftime("%Y-%m-%d")
        )

        show_df["Time"] = (
            show_df["DateTime"]
            .dt.strftime("%H:%M")
        )

        show_df["Temperature"] = (
            show_df["Temperature"]
            .round(1)
        )

        st.dataframe(
            show_df[
                ["Date", "Time", "Temperature"]
            ],
            use_container_width=True
        )

else:

    st.info(
        "Please enter temperature values above."
    )
