# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor (XGBoost)")
st.write("Enter a few details and get a predicted house price.")

# Load model + columns
model = joblib.load("models/xgb_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

# --- User inputs ---
sqft_living = st.number_input("Sqft Living", min_value=200, max_value=10000, value=1800, step=50)
bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
grade = st.number_input("Grade (1–13)", min_value=1, max_value=13, value=7, step=1)
lat = st.number_input("Latitude", value=47.55, format="%.4f")
long = st.number_input("Longitude", value=-122.25, format="%.4f")

zipcode = st.text_input("Zipcode (e.g., 98178)", value="98178")

# Build 1-row input with ALL feature columns (default 0)
X_input = pd.DataFrame([{c: 0 for c in feature_cols}])


for col, val in {
    "sqft_living": sqft_living,
    "bathrooms": bathrooms,
    "bedrooms": bedrooms,
    "grade": grade,
    "lat": lat,
    "long": long,
}.items():
    if col in X_input.columns:
        X_input.at[0, col] = val


zip_col = f"zipcode_{zipcode}"
if zip_col in X_input.columns:
    X_input.at[0, zip_col] = 1
elif "zipcode" in X_input.columns:
    try:
        X_input.at[0, "zipcode"] = int(zipcode)
    except:
        pass

st.divider()

if st.button("Predict Price"):
    # Predict log(price)
    pred_log = model.predict(X_input)[0]
    # Convert back to raw price
    pred_price = np.expm1(pred_log)

    st.success(f"✅ Predicted Price: ${pred_price:,.0f}")
    st.caption("Note: This is a model estimate based on the dataset features.")
