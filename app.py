from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import requests
import sqlite3
from datetime import datetime
import time

# --- 1. DATABASE PERSISTENCE LAYER ---
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

# --- 2. AI INFERENCE ENGINE ---
@st.cache_resource
def train_pro_model():
    try:
        df = pd.read_csv('delhi_electricity_data.csv')
        # Random Forest for high-accuracy non-linear prediction
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df[['temperature']].values, df['demand'])
        return model, df, True
    except FileNotFoundError:
        return None, None, False

# Initialize
model, historical_df, data_ready = train_pro_model()
init_db()

# --- 3. UI CONFIGURATION ---
st.set_page_config(page_title="SmartLoad AI: Urban Pulse", layout="wide")

# Persistent buffer for the "Sliding Window" ECG effect
if 'pulse_buffer' not in st.session_state:
    st.session_state.pulse_buffer = pd.DataFrame({
        'Time': [datetime.now().strftime("%H:%M:%S")] * 40, 
        'Load': [0.0] * 40
    })

# --- HARDCODED CONFIGURATION ---
MY_API_KEY = "66ac3134640c7b2f57d1dd5add7cee13" 

st.sidebar.title("🩺 Grid Vitality Hub")
app_mode = st.sidebar.radio("Navigation", ["Live Urban Pulse", "Strategic Simulation", "Audit Ledger"])
st.sidebar.markdown("---")
st.sidebar.success("📡 Satellite Sync: ACTIVE")
st.sidebar.info("Model: RF Ensemble (100 Trees)")

# --- 4. DYNAMIC PULSE FRAGMENT (The ECG Effect) ---
@st.fragment(run_every=3) # Faster refresh for sharp visualization
def live_pulse_monitor():
    if not data_ready:
        st.error("Dataset 'delhi_electricity_data.csv' missing!")
        return

    try:
        # API Telemetry
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={MY_API_KEY}&units=metric"
        res = requests.get(url).json()
        temp = res['main']['temp']
        
        # AI Forecast
        prediction = model.predict([[temp]])[0]
        status = "CRITICAL" if prediction > 7800 else "STABLE"
        
        # Log to Database
        log_pulse(temp, prediction, status)
        
        # Update Sliding Window (Shift left, append right)
        new_row = pd.DataFrame({'Time': [datetime.now().strftime("%H:%M:%S")], 'Load': [prediction]})
        updated_buffer = pd.concat([st.session_state.pulse_buffer, new_row], ignore_index=True)
        st.session_state.pulse_buffer = updated_buffer.iloc[1:] # Maintain 40-point window

        # Metrics HUD
        c1, c2, c3 = st.columns(3)
        c1.metric("Satellite Temp", f"{temp}°C")
        
        # Calculate Delta
        prev_load = st.session_state.pulse_buffer['Load'].iloc[-2]
        delta = prediction - prev_load if prev_load > 0 else 0
        c2.metric("Grid Load Forecast", f"{round(prediction, 2)} MW", delta=f"{round(delta, 2)} MW")
        
        status_color = "red" if status == "CRITICAL" else "#00FF85"
        c3.markdown(f"**Grid Health:** <span style='color:{status_color}; font-size:24px;'>{status}</span>", unsafe_allow_html=True)

        # PLOTLY ECG VISUALIZATION
        fig = go.Figure()
        
        # Neon Green Sharp Pulse Line
        fig.add_trace(go.Scatter(
            x=st.session_state.pulse_buffer['Time'], 
            y=st.session_state.pulse_buffer['Load'],
            mode='lines',
            line=dict(color='#00FF85', width=3),
            fill=None,
            hoverinfo='text',
            text=[f"Load: {round(l, 2)} MW" for l in st.session_state.pulse_buffer['Load']]
        ))

        # Styling the "Hospital Monitor" Grid
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,1)', # Solid black background
            plot_bgcolor='rgba(0,0,0,1)',
            xaxis=dict(showgrid=True, gridcolor='#222', zeroline=False, color='#666', title="Time Pulse"),
            yaxis=dict(showgrid=True, gridcolor='#222', zeroline=False, color='#666', title="Load (MW)", range=[0, 10000]),
            margin=dict(l=0, r=0, t=20, b=0),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.caption(f"STREAMING TELEMETRY | WINDOW: 40 TICKS | SYNC: {datetime.now().strftime('%H:%M:%S')}")

    except Exception:
        st.warning("Connecting to Satellite Feed... Ensure API Key is valid.")

# --- 5. APP MODES ---
if app_mode == "Live Urban Pulse":
    st.title("🩺 Urban Grid Heartbeat Monitor")
    live_pulse_monitor()

elif app_mode == "Strategic Simulation":
    st.title("🔥 Heatwave Stress Simulation")
    sim_t = st.slider("Target Temperature (°C)", 20.0, 50.0, 38.0)
    
    if st.button("Simulate Grid Impact"):
        pred = model.predict([[sim_t]])[0]
        st.metric("Forecasted Peak Demand", f"{round(pred, 2)} MW")
        
        # Static Regression View
        fig, ax = plt.subplots(facecolor='#0a0a0a')
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        ax.scatter(historical_df['temperature'], historical_df['demand'], alpha=0.1, color='cyan')
        ax.scatter([sim_t], [pred], color='#00FF85', s=200, label="Simulation Point")
        ax.set_xlabel("Temp", color='white')
        ax.set_ylabel("Demand", color='white')
        st.pyplot(fig)

elif app_mode == "Audit Ledger":
    st.title("📋 Grid Market Transaction Ledger")
    conn = sqlite3.connect('smartgrid_pulse.db')
    db_df = pd.read_sql_query("SELECT * FROM pulse_logs ORDER BY timestamp DESC", conn)
    conn.close()
    st.dataframe(db_df, use_container_width=True)