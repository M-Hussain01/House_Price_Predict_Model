# app.py
import streamlit as st
import pandas as pd
import numpy as np
from src.ingestion import load_data
from src.validation import validate_data
from src.transformation import transform_data
from src.model import train_model
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="🏠 California House Predictor",
    layout="wide",
    page_icon="🏡"
)

st.markdown("""
<div style='background: linear-gradient(90deg, #4CAF50, #81C784); padding: 20px; border-radius: 15px;'>
    <h1 style='color:white; text-align:center;'>🏠 California Housing Price Predictor</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("")

@st.cache_data(show_spinner=True)
def prepare_model():
    df = load_data()
    df = validate_data(df)
    X_scaled, y, scaler = transform_data(df)
    model = train_model(X_scaled, y)
    r2 = r2_score(y, model.predict(X_scaled))
    return df, scaler, model, r2

df, scaler, model, r2 = prepare_model()

st.subheader("🛠️ Enter Features for Prediction")

def user_input_features():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        MedInc = st.slider("Median Income (MedInc)",
                           float(df['MedInc'].min()), float(df['MedInc'].max()), float(df['MedInc'].mean()), step=0.01)
        HouseAge = st.slider("House Age (HouseAge)",
                             float(df['HouseAge'].min()), float(df['HouseAge'].max()), float(df['HouseAge'].mean()), step=0.01)
    with col2:
        AveRooms = st.slider("Average Rooms (AveRooms)",
                             float(df['AveRooms'].min()), float(df['AveRooms'].max()), float(df['AveRooms'].mean()), step=0.01)
        AveBedrms = st.slider("Average Bedrooms (AveBedrms)",
                              float(df['AveBedrms'].min()), float(df['AveBedrms'].max()), float(df['AveBedrms'].mean()), step=0.01)
    with col3:
        Population = st.slider("Population",
                               float(df['Population'].min()), float(df['Population'].max()), float(df['Population'].mean()), step=1.0)
        AveOccup = st.slider("Average Occupancy (AveOccup)",
                             float(df['AveOccup'].min()), float(df['AveOccup'].max()), float(df['AveOccup'].mean()), step=0.01)
    with col4:
        Latitude = st.slider("Latitude",
                             float(df['Latitude'].min()), float(df['Latitude'].max()), float(df['Latitude'].mean()), step=0.001)
        Longitude = st.slider("Longitude",
                              float(df['Longitude'].min()), float(df['Longitude'].max()), float(df['Longitude'].mean()), step=0.001)
    
    features = pd.DataFrame({
        "MedInc": [MedInc],
        "HouseAge": [HouseAge],
        "AveRooms": [AveRooms],
        "AveBedrms": [AveBedrms],
        "Population": [Population],
        "AveOccup": [AveOccup],
        "Latitude": [Latitude],
        "Longitude": [Longitude]
    })
    return features

input_df = user_input_features()

st.markdown("---")

if st.button("💡 Predict House Value"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    st.subheader("💰 Prediction Result")
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #FFEB3B, #FFF176); padding: 25px; border-radius: 15px; text-align:center;'>
        <h2 style='color:#4CAF50;'>Predicted Median House Value: <strong>${prediction[0]:.2f}k</strong></h2>
        <p style='color:#000;'>Model Accuracy (R² Score): <strong>{r2:.2f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 Show Entered Features"):
        st.dataframe(input_df)

