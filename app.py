import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import requests
import sqlite3
from datetime import datetime
import time

# --- 1. DB & MODEL SETUP ---
def init_db():
    conn = sqlite3.connect('smartgrid_pulse.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pulse_logs 
                 (timestamp TEXT, temp REAL, load REAL, status TEXT)''')
    conn.commit()
    conn.close()

def log_pulse(temp, load, status):
    conn = sqlite3.connect('smartgrid_pulse.db')
    c = conn.cursor()
    now = datetime.now().strftime("%H:%M:%S")
    c.execute("INSERT INTO pulse_logs VALUES (?, ?, ?, ?)", (now, temp, load, status))
    conn.commit()
    conn.close()

@st.cache_resource
def train_pro_model():
    df = pd.read_csv('delhi_electricity_data.csv')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['temperature']].values, df['demand'])
    return model, df

model, historical_df = train_pro_model()
init_db()

# --- 2. UI CONFIG ---
st.set_page_config(page_title="SmartLoad AI: Living Grid", layout="wide")

if 'live_data' not in st.session_state:
    st.session_state.live_data = pd.DataFrame(columns=['Time', 'Load'])

st.sidebar.title("🎮 Control Room")
app_mode = st.sidebar.radio("Navigation", ["Live Pulse Dashboard", "Manual Stress-Test Hub", "Audit Logs"])
api_key = st.sidebar.text_input("OpenWeather API Key", type="password")

# --- 3. THE MAGIC FIX: @st.fragment ---
# Ye decorator ensures ki sirf ye function refresh ho, pura page nahi!
@st.fragment(run_every=5) # Har 5 second mein apne aap chalega bina flicker ke
def live_heartbeat_ui():
    if api_key:
        try:
            # Fetch & Predict
            url = f"http://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={api_key}&units=metric"
            res = requests.get(url).json()
            t = res['main']['temp']
            p = model.predict([[t]])[0]
            s = "CRITICAL" if p > 7800 else "STABLE"
            
            # Log to DB
            log_pulse(t, p, s)
            
            # Update Session State
            new_entry = pd.DataFrame({'Time': [datetime.now().strftime("%H:%M:%S")], 'Load': [p]})
            st.session_state.live_data = pd.concat([st.session_state.live_data, new_entry]).tail(20)

            # UI Update inside fragment
            c1, c2, c3 = st.columns(3)
            c1.metric("Live Delhi Temp", f"{t}°C")
            c2.metric("AI Prediction", f"{round(p, 2)} MW")
            c3.metric("Grid Status", s)

            st.markdown("#### Real-time Load Stream")
            st.line_chart(st.session_state.live_data.set_index('Time'))
            
            st.caption(f"Last Auto-Sync: {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            st.error(f"Syncing... (Waiting for API response or Key)")
    else:
        st.warning("Enter API Key in sidebar to start the Heartbeat.")

# --- 4. APP NAVIGATION ---
if app_mode == "Live Pulse Dashboard":
    st.title("📡 Live Grid Pulse")
    live_heartbeat_ui() # Call the fragment

elif app_mode == "Manual Stress-Test Hub":
    st.title("🔥 Heatwave Simulation")
    sim_t = st.slider("Select Simulation Temperature (°C)", 20.0, 50.0, 35.0)
    if st.button("Run AI Stress Test"):
        pred = model.predict([[sim_t]])[0]
        st.metric("Simulated Peak Load", f"{round(pred, 2)} MW")
        # Regression Plot
        fig, ax = plt.subplots(facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.scatter(historical_df['temperature'], historical_df['demand'], alpha=0.1, color='cyan')
        ax.scatter([sim_t], [pred], color='red', s=200)
        st.pyplot(fig)

elif app_mode == "Audit Logs":
    st.title("📋 Audit Logs")
    conn = sqlite3.connect('smartgrid_pulse.db')
    db_df = pd.read_sql_query("SELECT * FROM pulse_logs ORDER BY timestamp DESC", conn)
    conn.close()
    st.dataframe(db_df, use_container_width=True)
