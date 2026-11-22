import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "storage_ann.keras"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

st.set_page_config(page_title="Post-Harvest Storage Loss Predictor", page_icon="🌾", layout="centered")
st.title("🌾 Post-Harvest Crop Storage Loss Predictor")
st.write("Estimate crop weight loss from storage conditions using a trained ANN regression model.")
st.warning("The bundled dataset is synthetic demonstration data. Predictions are educational demonstrations, not validated agricultural advice.")

if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    st.error("Model files not found. Run notebooks/model_training.ipynb through the save-model section first.")
    st.stop()

@st.cache_resource
def load_assets():
    return tf.keras.models.load_model(MODEL_PATH), joblib.load(PREPROCESSOR_PATH)

model, preprocessor = load_assets()

with st.form("prediction_form"):
    crop = st.selectbox("Crop type", ["Wheat", "Rice", "Maize", "Pulses", "Groundnuts", "Soybean"])
    duration = st.number_input("Storage duration (days)", min_value=1, max_value=1000, value=45)
    temp = st.number_input("Average temperature (°C)", min_value=-5.0, max_value=60.0, value=27.0)
    rh = st.number_input("Average relative humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
    moisture = st.number_input("Initial moisture content (%)", min_value=0.0, max_value=50.0, value=12.5)
    method = st.selectbox("Storage method", ["Hermetic bag", "Ordinary bag", "Warehouse", "Metal silo"])
    pest = st.selectbox("Pest level", ["None", "Low", "Medium", "High"])
    ventilation = st.selectbox("Ventilation", ["Good", "Moderate", "Poor"])
    initial_weight = st.number_input("Initial net crop weight (kg)", min_value=0.1, value=100.0)
    submitted = st.form_submit_button("Predict storage loss")

if submitted:
    case = pd.DataFrame([{
        "storage_duration_days": duration,
        "average_temperature_c": temp,
        "average_relative_humidity_pct": rh,
        "initial_moisture_content_pct": moisture,
        "crop_type": crop,
        "storage_method": method,
        "pest_level": pest,
        "ventilation": ventilation,
    }])
    processed = preprocessor.transform(case).astype("float32")
    loss = float(np.clip(model.predict(processed, verbose=0)[0, 0], 0, 100))
    remaining = initial_weight * (1 - loss / 100)
    c1, c2 = st.columns(2)
    c1.metric("Estimated weight loss", f"{loss:.2f}%")
    c2.metric("Estimated remaining weight", f"{remaining:.2f} kg")
    st.caption("Estimate only. Reliability depends on the quality, size and representativeness of the real training dataset.")
