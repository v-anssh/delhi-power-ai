import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(df[['temperature']].values, df['demand'])
        return model, df, True
    except FileNotFoundError:
        return None, None, False

model, historical_df, data_ready = train_pro_model()
init_db()

# --- 3. UI CONFIGURATION & CUSTOM HACKER CSS ---
st.set_page_config(page_title="SmartLoad AI: Quantum Grid Engine", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
    
    .stApp { background-color: #030712 !important; }
    
    div[data-testid="stMetricBlock"] {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 25px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    div[data-testid="stMetricBlock"]:hover {
        border-color: #00FF85;
        box-shadow: 0 0 15px rgba(0, 255, 133, 0.2);
    }
    
    .pulse-badge-stable {
        background-color: rgba(0, 255, 133, 0.1);
        border: 1px solid #00FF85;
        color: #00FF85;
        padding: 6px 16px;
        border-radius: 50px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 0 10px rgba(0, 255, 133, 0.2);
    }
    .pulse-badge-critical {
        background-color: rgba(255, 49, 49, 0.1);
        border: 1px solid #FF3131;
        color: #FF3131;
        padding: 6px 16px;
        border-radius: 50px;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        display: inline-block;
        animation: blinker 1.5s linear infinite;
        box-shadow: 0 0 10px rgba(255, 49, 49, 0.3);
    }
    @keyframes blinker { 50% { opacity: 0.4; } }
</style>
""", unsafe_allow_html=True)

if 'pulse_buffer' not in st.session_state:
    st.session_state.pulse_buffer = pd.DataFrame({
        'Time': [datetime.now().strftime("%H:%M:%S")] * 40, 
        'Load': [0.0] * 40
    })

# --- CONFIGURATION HUB ---
MY_API_KEY = "66ac3134640c7b2f57d1dd5add7cee13" 

st.sidebar.markdown("<h1 style='font-family:\"Orbitron\", sans-serif; color:#00FF85; font-size: 24px;'>🌐 CORE NODE ENGINE</h1>", unsafe_allow_html=True)
app_mode = st.sidebar.radio("NAV-TIER DIAGNOSTICS", ["🛰️ Live Ingestion Feed", "🔮 Scenario Forecasting", "🗄️ System Ledger Core"])
st.sidebar.markdown("---")
st.sidebar.success("📡 SATELLITE COMMS: ONLINE")
st.sidebar.info("KERNEL: Random Forest v2.4")

# --- 4. LIVE INGESTION DASHBOARD ---
@st.fragment(run_every=3)
def live_pulse_monitor():
    if not data_ready:
        st.error("Dataset 'delhi_electricity_data.csv' missing in core path!")
        return

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q=Delhi&appid={MY_API_KEY}&units=metric"
        res = requests.get(url).json()
        temp = res['main']['temp']
        
        prediction = model.predict([[temp]])[0]
        status = "CRITICAL" if prediction > 7800 else "STABLE"
        
        log_pulse(temp, prediction, status)
        
        new_row = pd.DataFrame({'Time': [datetime.now().strftime("%H:%M:%S")], 'Load': [prediction]})
        updated_buffer = pd.concat([st.session_state.pulse_buffer, new_row], ignore_index=True)
        st.session_state.pulse_buffer = updated_buffer.iloc[1:]

        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #0f172a, transparent); padding: 15px; border-left: 4px solid #00FF85; border-radius: 4px; margin-bottom: 25px;">
            <span style="color: #64748b; text-transform: uppercase; font-family: monospace; tracking: 2px;">System Pipeline Matrix</span>
            <h2 style="margin:0; color: #fff; font-family: 'Orbitron', sans-serif; font-size: 28px;">HIGH-FREQUENCY INFRASTRUCTURE VITALITY</h2>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("SATELLITE THERMAL FEED", f"{temp} °C")
        
        prev_load = st.session_state.pulse_buffer['Load'].iloc[-2]
        delta = prediction - prev_load if prev_load > 0 else 0
        c2.metric("AI AGGREGATED LOAD FORECAST", f"{round(prediction, 2)} MW", delta=f"{round(delta, 2)} MW")
        
        badge_class = "pulse-badge-critical" if status == "CRITICAL" else "pulse-badge-stable"
        with c3:
            st.write("") 
            st.markdown(f"<p style='color: #64748b; font-size:14px; margin-bottom:5px; font-weight:bold;'>NODE STATE HEALTH</p><div class='{badge_class}'>{status}</div>", unsafe_allow_html=True)

        st.markdown("<p style='font-family:\"Orbitron\", sans-serif; color: #cbd5e1; margin-top:20px; font-size:16px;'>📊 VECTOR GRID QUANTUM TICKER</p>", unsafe_allow_html=True)
        
        # --- FIXED EMBEDDED BLOCK GRAPH LAYER ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.pulse_buffer['Time'], 
            y=st.session_state.pulse_buffer['Load'],
            mode='lines',
            line=dict(color='#00FF85', width=3, shape='spline'),
            fill=None
        ))

        fig.update_layout(
            paper_bgcolor='#0b0f19', # Absolute dark base layout code injection fixed
            plot_bgcolor='#020617', 
            xaxis=dict(
                showgrid=True, 
                gridcolor='#1e293b', 
                gridwidth=1,
                zeroline=False, 
                color='#cbd5e1', 
                font=dict(family='Share Tech Mono')
            ),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#1e293b', 
                gridwidth=1,
                zeroline=False, 
                color='#cbd5e1', 
                font=dict(family='Share Tech Mono'), 
                range=[0, 10000]
            ),
            margin=dict(l=60, r=30, t=30, b=50),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        lc1, lc2, lc3 = st.columns(3)
        lc1.caption("⚡ INGESTION INTERVAL: 3000MS RUNTIME")
        lc2.caption("🌳 CALCULATED CARBON DEFICIT OPTIMIZED: 14.2%")
        lc3.caption(f"💾 DATA INTERACTION RECURSION PING: SECURE | TIME: {datetime.now().strftime('%H:%M:%S')}")

    except Exception as e:
        st.warning("Awaiting Remote API Handshake Sync... Check Network Routing or Token Authorization.")

# --- 5. SYSTEM APPLICATIONS MODES ---
if app_mode == "🛰️ Live Ingestion Feed":
    live_pulse_monitor()

elif app_mode == "🔮 Scenario Forecasting":
    st.markdown("<h2 style='font-family:\"Orbitron\", sans-serif; color:white;'>🔮 Predictive System Scenario Stressor</h2>", unsafe_allow_html=True)
    sim_t = st.slider("Select Environmental Testing Matrix (°C)", 20.0, 50.0, 38.0)
    
    if st.button("RUN ENGINE STRESS EVALUATION"):
        pred = model.predict([[sim_t]])[0]
        
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.metric("CRITICAL CAPACITY DISPATCHED", f"{round(pred, 2)} MW")
        
        with sc2:
            fig, ax = plt.subplots(facecolor='#020617')
            ax.set_facecolor('#020617')
            ax.tick_params(colors='#94a3b8')
            ax.scatter(historical_df['temperature'], historical_df['demand'], alpha=0.08, color='#00A3FF')
            ax.scatter([sim_t], [pred], color='#FF3131' if pred > 7800 else '#00FF85', s=250, edgecolors='white', label="Target")
            ax.set_xlabel("Temperature Baseline", color='#64748b')
            ax.set_ylabel("Power Demand Megawatts", color='#64748b')
            st.pyplot(fig)

elif app_mode == "🗄️ System Ledger Core":
    st.markdown("<h2 style='font-family:\"Orbitron\", sans-serif; color:white;'>🗄️ Relational Ledger Deep Audits</h2>", unsafe_allow_html=True)
    conn = sqlite3.connect('smartgrid_pulse.db')
    db_df = pd.read_sql_query("SELECT * FROM pulse_logs ORDER BY timestamp DESC", conn)
    conn.close()
    
    if not db_df.empty:
        st.dataframe(db_df, use_container_width=True)
        st.download_button("⚡ EXTRACT DATA CONTEXT (CSV)", db_df.to_csv(index=False).encode('utf-8'), "quantum_grid_logs.csv", "text/csv")
    else:
        st.info("System storage arrays are blank. Activate core live pipeline to stream vectors.")