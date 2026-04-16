import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Delhi Power AI", layout="wide")

# 1. CSV INTEGRATION (Yahan data load hota hai)
try:
    df = pd.read_csv('delhi_electricity_data.csv')
    X = df[['temperature']]
    y = df['demand']
    
    # 2. AI MODEL TRAINING
    model = LinearRegression()
    model.fit(X, y)
    data_loaded = True
except Exception as e:
    st.error(f"CSV File Error: Make sure 'delhi_electricity_data.csv' is in the same folder. Error: {e}")
    data_loaded = False

# 3. UI LAYOUT
st.title("⚡ Delhi Electricity Demand Predictor")

if data_loaded:
    tab1, tab2 = st.tabs(["Manual Prediction", "Live API Feed"])

    with tab1:
        st.subheader("Predict Demand based on Temp")
        temp_val = st.slider("Select Temperature (°C)", 20, 50, 35)
        
        prediction = model.predict([[temp_val]])[0]
        
        c1, c2 = st.columns(2)
        c1.metric("Predicted Peak Load", f"{round(prediction, 2)} MW")
        
        # Graph Integration
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', alpha=0.3, label='Past Data')
        ax.plot(X, model.predict(X), color='red', label='AI Trend')
        ax.scatter([temp_val], [prediction], color='green', s=100, label='Current Point')
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Demand (MW)")
        ax.legend()
        c2.pyplot(fig)

    with tab2:
        st.subheader("Real-Time Data Integration")
        api_key = st.text_input("Enter OpenWeather API Key", type="password")
        
        if st.button("Fetch & Predict"):
            if api_key:
                try:
                    url = f"http://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={api_key}&units=metric"
                    res = requests.get(url).json()
                    live_t = res['main']['temp']
                    live_p = model.predict([[live_t]])[0]
                    
                    st.success(f"Current Delhi Temp: {live_t}°C")
                    st.metric("Live Predicted Demand", f"{round(live_p, 2)} MW")
                except:
                    st.error("API Key is invalid or not yet active.")
            else:
                st.warning("Please enter your API Key.")

else:
    st.info("Check your VS Code folder for the CSV file.")