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


