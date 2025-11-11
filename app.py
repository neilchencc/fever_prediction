import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import joblib  # 載入 pkl 模型

# ---------------------------------------------------
# 📘 Streamlit 標題
# ---------------------------------------------------
st.title("📈 體溫紀錄分析工具（CSV 上傳 + 標準化 + 預測）")

# ---------------------------------------------------
# 📂 上傳 CSV 檔案
# ---------------------------------------------------
uploaded_file = st.file_uploader("請上傳包含 Date, Time, BT 欄位的 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    # 讀取並清理欄位名稱
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]

    # 建立 DateTime 欄位
    df["DateTime"] = df.apply(
        lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
        axis=1
    )
    df = df.sort_values("DateTime").reset_index(drop=True)

    st.write("### 🧾 原始資料預覽：")
    st.dataframe(df)

    # ---------------------------------------------------
    # 🧮 資料檢查與時間範圍設定
    # ---------------------------------------------------
    unique_dates = sorted(df["Date"].unique())
    if len(unique_dates) < 2:
        st.error("⚠️ 資料不足，請至少包含兩個不同日期。")
    else:
        second_last_date = unique_dates[-2]
        last_date = unique_dates[-1]

        start_time = datetime.strptime(str(second_last_date) + "0800", "%Y%m%d%H%M")
        end_time = datetime.strptime(str(last_date) + "2359", "%Y%m%d%H%M")

        df_range = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)]

        if df_range.empty:
            st.warning("⚠️ 此時間區間內沒有資料。")
        else:
            st.write(f"### ⏱ 分析範圍：{start_time} ～ {end_time}")
            st.dataframe(df_range)

            # ---------------------------------------------------
            # 🧩 特徵工程
            # ---------------------------------------------------
            t0 = df_range["DateTime"].min()
            df_range["Hours"] = (df_range["DateTime"] - t0).dt.total_seconds() / 3600

            max_bt = df_range["BT"].max()
            min_bt = df_range["BT"].min()
            mean_bt = df_range["BT"].mean()
            std_bt = df_range["BT"].std()

            X = df_range["Hours"].values.reshape(-1, 1)
            y = df_range["BT"].values
            model_lr = LinearRegression().fit(X, y)
            slope = model_lr.coef_[0]

            last_time = df_range["Hours"].max()
            last_8h = df_range[df_range["Hours"] >= last_time - 8]
            max_last8 = last_8h["BT"].max()

            range_bt = max_bt - min_bt
            diff_last8_allmax = max_last8 - max_bt

            # 建立特徵列表
            features = [max_bt, min_bt, mean_bt, std_bt, slope, range_bt, max_last8, diff_last8_allmax]
            feature_names = [
                "最大值 (max)", "最小值 (min)", "平均值 (mean)", "標準差 (std)",
                "斜率 (slope)", "max - min", "最後8小時的 max", "最後8小時 max - 全部 max"
            ]

            result_table = pd.DataFrame({
                "指標": feature_names,
                "數值": [f"{v:.4f}" for v in features]
            })
            st.subheader("📊 統計結果")
            st.table(result_table)

            # ---------------------------------------------------
            # 🤖 模型預測
            # ---------------------------------------------------
            st.subheader("🤖 預測結果")

            try:
                # 載入 scaler 與 SVM 模型
                scaler = joblib.load("scaler.pkl")
                svm_model = joblib.load("svm_model.pkl")

                # 標準化輸入特徵
                features_array = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features_array)

                # 模型預測
                if hasattr(svm_model, "predict_proba"):
                    pred_prob = svm_model.predict_proba(features_scaled)[0][1]
                else:
                    pred_prob = svm_model.decision_function(features_scaled)[0]

                threshold = 0.5
                if pred_prob >= threshold:
                    st.success(f"預測結果：未來可能會發燒 (分數/機率={pred_prob:.3f} ≥ {threshold})")
                else:
                    st.info(f"預測結果：未來不會發燒 (分數/機率={pred_prob:.3f} < {threshold})")

            except FileNotFoundError as e:
                st.error(f"找不到必要的模型檔案：{e.filename}")
            except Exception as e:
                st.error(f"載入或預測時發生錯誤：{e}")

            # ---------------------------------------------------
            # 📉 體溫變化圖
            # ---------------------------------------------------
            st.subheader("📉 體溫變化圖")
            st.line_chart(df_range.set_index("DateTime")["BT"])

else:
    st.info("⬆️ 請上傳一個 CSV 檔以開始分析。")


