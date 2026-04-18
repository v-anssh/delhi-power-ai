import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="SmartLoad AI", layout="wide")

# 1. CSV INTEGRATION
try:
    df = pd.read_csv('delhi_electricity_data.csv')
    X = df[['temperature']]
    y = df['demand']
    
    # 2. AI MODEL TRAINING
    model = LinearRegression()
    model.fit(X, y)
    data_loaded = True
except Exception as e:
    st.error(f"CSV File Error: {e}")
    data_loaded = False

# 3. SIDEBAR SETTINGS (Advanced Features)
st.sidebar.title("⚙️ Control Room")
time_slot = st.sidebar.selectbox("Select Time of Day", ["Normal", "Peak (12 PM - 4 PM)", "Off-Peak (2 AM - 6 AM)"])
heatwave_trigger = st.sidebar.button("🔥 Simulate Extreme Heatwave")

# 4. UI LAYOUT
st.title("⚡ SmartLoad AI: Delhi Power Predictor")

if data_loaded:
    tab1, tab2 = st.tabs(["Manual Prediction", "Live API Feed"])

    with tab1:
        st.subheader("Predict Demand based on Temp")
        
        # Logic for Slider or Heatwave
        default_temp = 35
        if heatwave_trigger:
            temp_val = 49.5
            st.sidebar.warning("Heatwave Mode Active!")
        else:
            temp_val = st.slider("Select Temperature (°C)", 20, 50, default_temp)
        
        # Base Prediction
        base_prediction = model.predict([[temp_val]])[0]
        
        # Apply Time Multipliers
        multiplier = 1.0
        if "Peak" in time_slot:
            multiplier = 1.15  # 15% extra
        elif "Off-Peak" in time_slot:
            multiplier = 0.80  # 20% less
            
        final_demand = base_prediction * multiplier
        if heatwave_trigger:
            final_demand = final_demand * 1.10 # Extra stress for heatwave
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Peak Load", f"{round(final_demand, 2)} MW")
            st.info(f"Condition: {time_slot} settings applied.")
            
            # --- AI Recommendations Section ---
            st.markdown("---")
            st.subheader("💡 Smart Grid Advice")
            if final_demand > 8000:
                st.error("🚨 CRITICAL: High Overload Risk! Recommendation: Start rotational load shedding.")
            elif final_demand > 6500:
                st.warning("⚠️ ALERT: High Demand. Recommendation: Activate backup gas turbines.")
            else:
                st.success("✅ STABLE: Grid is operating within safety limits.")

        with col2:
            fig, ax = plt.subplots()
            ax.scatter(X, y, color='blue', alpha=0.3, label='Past Data')
            ax.plot(X, model.predict(X), color='red', label='AI Trend')
            ax.scatter([temp_val], [final_demand], color='green', s=150, label='Current Point')
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Demand (MW)")
            ax.legend()
            st.pyplot(fig)

    with tab2:
        st.subheader("Real-Time Data Integration")
        api_key = st.text_input("Enter OpenWeather API Key", type="password")
        
        if st.button("Fetch & Predict"):
            if api_key:
                try:
                    url = f"http://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={api_key}&units=metric"
                    res = requests.get(url).json()
                    live_t = res['main']['temp']
                    live_p = model.predict([[live_t]])[0] * multiplier
                    
                    st.success(f"Current Delhi Temp: {live_t}°C")
                    st.metric("Live Predicted Demand", f"{round(live_p, 2)} MW")
                except:
                    st.error("API Key is invalid or not yet active.")
            else:
                st.warning("Please enter your API Key.")
else:
    st.info("Check your folder for the CSV file.")
