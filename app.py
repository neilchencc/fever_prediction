import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

st.title("📈 體溫紀錄分析工具（CSV 上傳）")

uploaded_file = st.file_uploader("請上傳包含 Date, Time, BT 欄位的 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 清理欄位名稱
    df.columns = [c.strip() for c in df.columns]

    # 建立 datetime 欄
    df["DateTime"] = df.apply(
        lambda row: datetime.strptime(str(int(row["Date"])) + f"{int(row['Time']):04d}", "%Y%m%d%H%M"),
        axis=1
    )
    df = df.sort_values("DateTime").reset_index(drop=True)

    st.write("### 🧾 原始資料預覽：")
    st.dataframe(df)

    # 找出倒數第二個與最後一個日期
    unique_dates = sorted(df["Date"].unique())
    if len(unique_dates) < 2:
        st.error("⚠️ 資料不足，請至少包含兩個不同日期。")
    else:
        second_last_date = unique_dates[-2]
        last_date = unique_dates[-1]

        # 分析區間：倒數第二日 08:00 至最後一日結束
        start_time = datetime.strptime(str(second_last_date) + "0800", "%Y%m%d%H%M")
        end_time = datetime.strptime(str(last_date) + "2359", "%Y%m%d%H%M")

        df_range = df[(df["DateTime"] >= start_time) & (df["DateTime"] <= end_time)]

        if df_range.empty:
            st.warning("⚠️ 此時間區間內沒有資料。")
        else:
            st.write(f"### ⏱ 分析範圍：{start_time} ～ {end_time}")
            st.dataframe(df_range)

            # 時間轉為相對小時
            t0 = df_range["DateTime"].min()
            df_range["Hours"] = (df_range["DateTime"] - t0).dt.total_seconds() / 3600

            # 基本統計
            max_bt = df_range["BT"].max()
            min_bt = df_range["BT"].min()
            mean_bt = df_range["BT"].mean()
            std_bt = df_range["BT"].std()

            # 線性回歸
            X = df_range["Hours"].values.reshape(-1, 1)
            y = df_range["BT"].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]

            # 最後 8 小時
            last_time = df_range["Hours"].max()
            last_8h = df_range[df_range["Hours"] >= last_time - 8]
            max_last8 = last_8h["BT"].max()

            # 額外計算
            range_bt = max_bt - min_bt
            diff_last8_allmax = max_last8 - max_bt

            # 顯示結果
            st.subheader("📊 統計結果")
            result_table = pd.DataFrame({
                "指標": [
                    "最大值 (max)",
                    "最小值 (min)",
                    "平均值 (mean)",
                    "標準差 (std)",
                    "線性回歸斜率 (slope, °C/hour)",
                    "max - min",
                    "最後8小時的 max",
                    "最後8小時的 max - 全部的 max"
                ],
                "數值": [
                    f"{max_bt:.2f}",
                    f"{min_bt:.2f}",
                    f"{mean_bt:.2f}",
                    f"{std_bt:.2f}",
                    f"{slope:.4f}",
                    f"{range_bt:.2f}",
                    f"{max_last8:.2f}",
                    f"{diff_last8_allmax:.2f}"
                ]
            })
            st.table(result_table)

            # 繪圖
            st.subheader("📉 體溫變化圖")
            st.line_chart(df_range.set_index("DateTime")["BT"])

else:
    st.info("⬆️ 請上傳一個 CSV 檔以開始分析。")
