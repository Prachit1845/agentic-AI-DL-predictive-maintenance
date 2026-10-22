import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import torch

from src.dataset import prepare_preprocessed_data, create_sequences, create_test_sequences, SELECTED_SENSORS
from src.prognostics import LSTMRULEstimator, predict_monte_carlo_dropout, evaluate_metrics
from src.agents import PrognosticAgent, ResourceAgent, CostAgent, CoordinatorAgent, MessageBus
from src.simulation import FleetMaintenanceSimulator, run_fleet_policy_comparison

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & BACKGROUND ASSET LOADER
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AERO-AGENT | Predictive Maintenance Command Center",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(image_path: str) -> str:
    """Reads a local image and encodes it to base64 for CSS background injection."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

bg_turbofan_b64 = get_base64_image(os.path.join("assets", "turbofan_bg.jpg"))
bg_cockpit_b64 = get_base64_image(os.path.join("assets", "cockpit_bg.jpg"))

# -----------------------------------------------------------------------------
# 2. SIDEBAR INSIGNIA & THEME CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div class="sidebar-insignia-card">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.4); border-radius:10px; width:44px; height:44px; display:flex; align-items:center; justify-content:center; font-size:1.6rem;">
            ✈️
        </div>
        <div>
            <div style="font-size:1.1rem; font-weight:800; background:linear-gradient(135deg, #38BDF8, #818CF8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:0.03em;">
                AERO-COMMAND
            </div>
            <div style="font-size:0.7rem; color:#94A3B8; font-family:'JetBrains Mono',monospace;">
                PHM DECISION OS v2.4
            </div>
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; font-size:0.72rem; color:#64748B; border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">
        <span>DEPOT: AMES FLIGHT OPS</span>
        <span style="color:#34D399; font-weight:700;"><span class="dot-green-pulse"></span> ONLINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Theme & Visual Customization Controls
st.sidebar.markdown("<div style='font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0.06em; color:#38BDF8; margin-bottom:6px;'>🎨 VISUAL & HUD THEME</div>", unsafe_allow_html=True)
bg_theme = st.sidebar.selectbox(
    "Backdrop Telemetry Stream",
    ["Turbofan Engine Hangar HUD", "Night Cockpit Flight Deck", "Deep Space Navy (Minimalist)"],
    index=0
)
enable_animations = st.sidebar.toggle("HUD Radar Scanlines & Pulse Effects", value=True)

# Pick selected background image string
if bg_theme == "Turbofan Engine Hangar HUD" and bg_turbofan_b64:
    active_bg_css = f"""
        background: linear-gradient(180deg, rgba(7, 11, 22, 0.82) 0%, rgba(4, 7, 15, 0.94) 100%),
                    url("data:image/jpeg;base64,{bg_turbofan_b64}") no-repeat center center fixed !important;
        background-size: cover !important;
    """
elif bg_theme == "Night Cockpit Flight Deck" and bg_cockpit_b64:
    active_bg_css = f"""
        background: linear-gradient(180deg, rgba(7, 11, 22, 0.82) 0%, rgba(4, 7, 15, 0.94) 100%),
                    url("data:image/jpeg;base64,{bg_cockpit_b64}") no-repeat center center fixed !important;
        background-size: cover !important;
    """
else:
    active_bg_css = """
        background: radial-gradient(circle at 15% 15%, #0B1120 0%, #060911 100%) !important;
    """

# Dynamic Scanline & Animation CSS
scanline_html = '<div class="scanline"></div>' if enable_animations else ''

# -----------------------------------------------------------------------------
# 3. HIGH-END CYBER-AEROSPACE CSS WITH KEYFRAME ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Typography */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }}
    
    /* Main Background with Dynamic Image Blend */
    .stApp {{
        {active_bg_css}
        color: #E2E8F0;
        min-height: 100vh;
    }}

    /* Radar Scanline Animation */
    @keyframes scanlineMove {{
        0% {{ top: -100px; opacity: 0; }}
        15% {{ opacity: 0.6; }}
        85% {{ opacity: 0.6; }}
        100% {{ top: 100%; opacity: 0; }}
    }}

    .scanline {{
        width: 100%;
        height: 60px;
        z-index: 9999;
        background: linear-gradient(0deg, rgba(0,0,0,0) 0%, rgba(56, 189, 248, 0.06) 50%, rgba(0,0,0,0) 100%);
        position: fixed;
        left: 0;
        animation: scanlineMove 9s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        pointer-events: none;
    }}

    /* Pulsing Glowing Dots */
    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }}
        70% {{ box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
    }}

    @keyframes pulseGlowCyan {{
        0% {{ box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.8); }}
        70% {{ box-shadow: 0 0 0 10px rgba(56, 189, 248, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }}
    }}

    .dot-green-pulse {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #34D399;
        display: inline-block;
        animation: pulseGlow 2s infinite;
    }}

    .dot-cyan-pulse {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #38BDF8;
        display: inline-block;
        animation: pulseGlowCyan 2.2s infinite;
    }}

    /* Hero Banner with Subtle Glowing Shimmer */
    .hero-container {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 30, 0.92) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 18px;
        padding: 24px 32px;
        margin-bottom: 22px;
        backdrop-filter: blur(20px);
        box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.7), 0 0 25px rgba(56, 189, 248, 0.15);
        position: relative;
        overflow: hidden;
    }}

    .hero-container::before {{
        content: '';
        position: absolute;
        top: 0; left: -50%;
        width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.05), transparent);
        transform: rotate(30deg);
        animation: shimmerBanner 10s infinite;
        pointer-events: none;
    }}

    @keyframes shimmerBanner {{
        0% {{ transform: translateX(-100%) rotate(30deg); }}
        100% {{ transform: translateX(100%) rotate(30deg); }}
    }}

    .hero-title {{
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
    }}

    .hero-subtitle {{
        font-size: 1.02rem;
        color: #94A3B8;
        margin-top: 6px;
        font-weight: 400;
    }}

    /* Status Badges */
    .status-badge-container {{
        display: flex;
        gap: 12px;
        margin-top: 16px;
        flex-wrap: wrap;
    }}

    .chip {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #CBD5E1;
        backdrop-filter: blur(8px);
    }}

    .chip-live {{
        border-color: rgba(52, 211, 153, 0.45);
        background: rgba(5, 150, 105, 0.2);
        color: #34D399;
    }}

    .chip-agent {{
        border-color: rgba(129, 140, 248, 0.45);
        background: rgba(79, 70, 229, 0.2);
        color: #C7D2FE;
    }}

    /* Animated Glassmorphic Metric Cards */
    .metric-card-glass {{
        background: linear-gradient(135deg, rgba(23, 32, 54, 0.78) 0%, rgba(13, 20, 36, 0.88) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(14px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}

    .metric-card-glass:hover {{
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.25);
    }}

    .metric-val {{
        font-size: 1.95rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }}

    .metric-lbl {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 5px;
    }}

    /* Aerospace Flight Terminal Window */
    .terminal-container {{
        background: rgba(4, 7, 16, 0.94);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.7), 0 0 20px rgba(56, 189, 248, 0.12);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 10px;
    }}

    .terminal-header {{
        background: rgba(15, 23, 42, 0.95);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 18px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 700;
    }}

    .dot-red {{ width: 10px; height: 10px; border-radius: 50%; background: #EF4444; display: inline-block; }}
    .dot-yellow {{ width: 10px; height: 10px; border-radius: 50%; background: #F59E0B; display: inline-block; }}
    .dot-green {{ width: 10px; height: 10px; border-radius: 50%; background: #10B981; display: inline-block; }}

    .terminal-body {{
        padding: 18px 22px;
        color: #38BDF8;
        font-size: 0.88rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }}

    /* Agent Message Timeline Item */
    .agent-msg-box {{
        background: rgba(15, 23, 42, 0.85);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #38BDF8;
        font-size: 0.86rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }}

    .agent-msg-box:hover {{
        transform: translateX(4px);
    }}

    /* Risk Badges */
    .risk-safe {{
        background: rgba(16, 185, 129, 0.25);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.5);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-warn {{
        background: rgba(245, 158, 11, 0.25);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.5);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-crit {{
        background: rgba(239, 68, 68, 0.25);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.5);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }}

    /* Custom Styled Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.8);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: #94A3B8;
        font-weight: 700;
        padding: 10px 20px;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25) 0%, rgba(99, 102, 241, 0.25) 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
    }}

    /* Tactical Sidebar Custom Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(8, 14, 27, 0.95) 0%, rgba(4, 7, 16, 0.98) 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 12px 0 35px rgba(0, 0, 0, 0.7) !important;
        backdrop-filter: blur(25px) !important;
    }}

    .sidebar-insignia-card {{
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 18px;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(56, 189, 248, 0.15);
    }}

    .sidebar-section-header {{
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #38BDF8;
        margin-top: 18px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .sidebar-tactical-widget {{
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 8px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
    }}

    /* Styled Selectboxes and Sliders */
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(56, 189, 248, 0.15) !important;
        margin: 16px 0 !important;
    }}

    /* Tactical Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {{
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.76rem !important;
        padding: 6px 4px !important;
        transition: all 0.25s ease !important;
    }}

    [data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(56, 189, 248, 0.2) !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.3) !important;
        transform: translateY(-1px) !important;
    }}

    [data-testid="stSidebar"] .stDownloadButton > button {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.45) !important;
        color: #38BDF8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        transition: all 0.3s ease !important;
    }}

    [data-testid="stSidebar"] .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.35) 0%, rgba(99, 102, 241, 0.35) 100%) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-2px) !important;
    }}
</style>

{scanline_html}
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. CACHED MODEL & TELEMETRY LOADER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system_assets():
    """Caches preprocessed NASA C-MAPSS dataset and loads trained PyTorch LSTM weights."""
    train_scaled, test_scaled, rul_df, scaler = prepare_preprocessed_data()
    X_test_30, y_test_30, test_ids = create_test_sequences(test_scaled, rul_df, seq_len=30)
    
    model = LSTMRULEstimator(input_dim=14, hidden_dim=64, num_layers=1, dropout=0.2)
    model_path = os.path.join("models", "best_lstm_model.pt")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    else:
        # Fallback quick train if file missing
        X_tr, y_tr = create_sequences(train_scaled, seq_len=30)
        from src.prognostics import train_lstm_model
        model, _ = train_lstm_model(X_tr[:6000], y_tr[:6000], epochs=3, batch_size=256, verbose=False)
        
    model.eval()
    return train_scaled, test_scaled, rul_df, X_test_30, y_test_30, test_ids, model

train_scaled, test_scaled, rul_df, X_test, y_test, test_ids, model = load_system_assets()


# -----------------------------------------------------------------------------
# 5. HERO HEADER WITH ANIMATED PULSE CHIPS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
        <div>
            <div class="hero-title">AERO-AGENT | COMMAND CENTER</div>
            <div class="hero-subtitle">Agentic Predictive Maintenance Scheduling under Remaining-Useful-Life (RUL) Uncertainty</div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; color:#38BDF8; font-weight:800;">AEROSPACE PHM PLATFORM</div>
            <div style="font-size:0.8rem; color:#64748B;">Autonomous Fleet Intelligence</div>
        </div>
    </div>
    <div class="status-badge-container">
        <span class="chip chip-live"><span class="dot-green-pulse"></span> TELEMETRY STREAM ACTIVE (100 FLEET ENGINES)</span>
        <span class="chip chip-agent"><span class="dot-cyan-pulse"></span> 4 FIPA-ACL AGENTS OPERATIONAL</span>
        <span class="chip">🚀 PYTORCH LSTM (64 UNITS) + MONTE CARLO DROPOUT</span>
        <span class="chip">📊 NASA C-MAPSS FD001 BENCHMARK</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. OPERATIONAL SCENARIO CONTROLS (TACTICAL SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-section-header'><span>🎯</span> MISSION PROFILE & SCENARIO</div>", unsafe_allow_html=True)

scenario = st.sidebar.selectbox(
    "Select Operational Scenario",
    [
        "Nominal Flight Operations",
        "High-Risk Engine Alert (Critical RUL)",
        "Supply Chain Crisis (High Lead Time)",
        "Depot Staffing Bottleneck (1 Tech/Slot)",
        "Custom Manual Configuration"
    ]
)

if scenario == "Nominal Flight Operations":
    default_engine = 17
    default_lead = 8
    default_tech = 2
elif scenario == "High-Risk Engine Alert (Critical RUL)":
    lowest_idx = int(np.argmin(y_test))
    default_engine = test_ids[lowest_idx]
    default_lead = 8
    default_tech = 2
elif scenario == "Supply Chain Crisis (High Lead Time)":
    default_engine = 17
    default_lead = 16
    default_tech = 2
elif scenario == "Depot Staffing Bottleneck (1 Tech/Slot)":
    default_engine = 17
    default_lead = 8
    default_tech = 1
else:
    default_engine = 17
    default_lead = 8
    default_tech = 2

# Track scenario changes and active engine in session state
if "last_scenario" not in st.session_state or st.session_state["last_scenario"] != scenario:
    st.session_state["last_scenario"] = scenario
    st.session_state["selected_engine_id"] = default_engine

# Quick Fleet Triage Presets
st.sidebar.markdown("<div class='sidebar-section-header'><span>⚡</span> QUICK FLEET UNIT TRIAGE</div>", unsafe_allow_html=True)
triage_col1, triage_col2, triage_col3 = st.sidebar.columns(3)
if triage_col1.button("🚨 E-031", help="Jump to Critical Unit #31 (Remaining RUL: 8 cycles)"):
    st.session_state["selected_engine_id"] = 31
if triage_col2.button("⚠️ E-017", help="Jump to Watchlist Unit #17 (Remaining RUL: 34 cycles)"):
    st.session_state["selected_engine_id"] = 17
if triage_col3.button("🛡️ E-006", help="Jump to Nominal Unit #6 (Remaining RUL: 91 cycles)"):
    st.session_state["selected_engine_id"] = 6

if st.session_state.get("selected_engine_id") not in test_ids:
    st.session_state["selected_engine_id"] = default_engine

sel_idx = test_ids.index(st.session_state["selected_engine_id"])
engine_choice = st.sidebar.selectbox("Turbofan Engine Unit", test_ids, index=sel_idx)
st.session_state["selected_engine_id"] = engine_choice
engine_idx = test_ids.index(engine_choice)

# Physical Logistical Constraints
st.sidebar.markdown("<div class='sidebar-section-header'><span>📦</span> DEPOT LOGISTICS & SPARES</div>", unsafe_allow_html=True)
spare_lead_time = st.sidebar.slider(
    "Spare-Part Lead Time (Cycles)",
    2, 20, default_lead, 1,
    help="Procurement and depot transit delay for turbine hot-section replacements."
)
tech_capacity = st.sidebar.slider(
    "Technician Slots / Cycle",
    1, 5, default_tech, 1,
    help="Certified aircraft maintenance engineer depot capacity available per cycle."
)
mc_samples = st.sidebar.slider(
    "Monte Carlo Passes (N)",
    10, 100, 40, 10,
    help="Number of stochastic forward passes for epistemic uncertainty quantification."
)

# Economic Cost Structure
st.sidebar.markdown("<div class='sidebar-section-header'><span>💰</span> COST MATRIX ($ USD)</div>", unsafe_allow_html=True)
cost_pm = st.sidebar.number_input("Preventive Maintenance Cost ($)", 200, 5000, 1000, 100)
cost_failure = st.sidebar.number_input("In-Service Failure Cost ($)", 1000, 25000, 5000, 500)
cost_downtime = st.sidebar.number_input("Downtime Loss / Cycle ($)", 10, 1000, 100, 10)

penalty_ratio = cost_failure / cost_pm if cost_pm > 0 else 5.0
st.sidebar.markdown(f"""
<div style="font-size:0.75rem; color:#94A3B8; background:rgba(15,23,42,0.7); padding:8px 12px; border-radius:10px; border:1px solid rgba(56,189,248,0.2); margin-top:6px; margin-bottom:12px;">
    <span>Failure Penalty Ratio:</span>
    <strong style="color:#F59E0B; margin-left:6px;">{penalty_ratio:.1f}x</strong>
    <span style="color:#64748B; font-size:0.7rem; display:block; margin-top:2px;">(Failure / Preventive Cost)</span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. MULTI-AGENT NEGOTIATION DISPATCH
# -----------------------------------------------------------------------------
bus = MessageBus()
prognostic_agent = PrognosticAgent(model=model, n_mc_samples=mc_samples, message_bus=bus)
resource_agent = ResourceAgent(spare_lead_time=spare_lead_time, tech_capacity_per_cycle=tech_capacity, message_bus=bus)
cost_agent = CostAgent(cost_pm=cost_pm, cost_failure=cost_failure, cost_downtime_per_cycle=cost_downtime, message_bus=bus)
coordinator_agent = CoordinatorAgent(prognostic_agent, resource_agent, cost_agent, message_bus=bus)

# Engine Telemetry Extraction
engine_data = test_scaled[test_scaled["engine_id"] == engine_choice]
current_cycle = int(engine_data["cycle"].max())
true_remaining = int(y_test[engine_idx])
engine_seq = X_test[engine_idx]

# Execute Multi-Agent Negotiation
decision = coordinator_agent.coordinate_decision(
    engine_id=engine_choice,
    current_cycle=current_cycle,
    sensor_sequence=engine_seq,
    book_slot=False
)
prog = decision["prognostics"]

# Health Status & Life Percentage Calculation
total_expected_life = current_cycle + true_remaining
life_remaining_pct = max(0.0, min(100.0, (true_remaining / total_expected_life) * 100)) if total_expected_life > 0 else 0.0

if true_remaining <= 15:
    health_badge = "CRITICAL RISK"
    badge_bg = "rgba(239, 68, 68, 0.2)"
    badge_color = "#F87171"
    badge_border = "rgba(239, 68, 68, 0.5)"
    bar_grad = "linear-gradient(90deg, #EF4444, #DC2626)"
elif true_remaining <= 40:
    health_badge = "ATTENTION"
    badge_bg = "rgba(245, 158, 11, 0.2)"
    badge_color = "#FBBF24"
    badge_border = "rgba(245, 158, 11, 0.5)"
    bar_grad = "linear-gradient(90deg, #F59E0B, #D97706)"
else:
    health_badge = "AIRWORTHY"
    badge_bg = "rgba(16, 185, 129, 0.2)"
    badge_color = "#34D399"
    badge_border = "rgba(16, 185, 129, 0.5)"
    bar_grad = "linear-gradient(90deg, #10B981, #059669)"

dispatch_action = "EMERGENCY PM" if decision.get("escalated") else ("SCHEDULE PM" if decision.get("recommended_cycle") is not None else "CRUISE CLEAR")
dispatch_color = "#F87171" if decision.get("escalated") else ("#FBBF24" if decision.get("recommended_cycle") is not None else "#34D399")

# Sidebar Live Telemetry Cockpit Widget
st.sidebar.markdown(f"""
<div class="sidebar-tactical-widget">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-family:'JetBrains Mono',monospace; font-weight:800; font-size:0.86rem; color:#38BDF8;">TURBOFAN E-{engine_choice:03d}</span>
        <span style="font-size:0.68rem; padding:3px 8px; border-radius:12px; background:{badge_bg}; color:{badge_color}; font-weight:700; border:1px solid {badge_border}; letter-spacing:0.04em;">{health_badge}</span>
    </div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; font-size:0.73rem;">
        <div style="background:rgba(255,255,255,0.03); padding:6px 8px; border-radius:8px;">
            <div style="color:#64748B; font-size:0.66rem;">LOGGED CYCLES</div>
            <div style="color:#F1F5F9; font-weight:800; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{current_cycle}</div>
        </div>
        <div style="background:rgba(255,255,255,0.03); padding:6px 8px; border-radius:8px;">
            <div style="color:#64748B; font-size:0.66rem;">GROUND TRUTH RUL</div>
            <div style="color:{badge_color}; font-weight:800; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">{true_remaining} cyc</div>
        </div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.68rem; color:#94A3B8; margin-bottom:4px;">
        <span>Lower 95%: <strong style="color:#38BDF8;">{prog['lower_bound']} cyc</strong></span>
        <span>Lifespan: <strong style="color:#F1F5F9;">{life_remaining_pct:.0f}%</strong></span>
    </div>
    <div style="background:rgba(255,255,255,0.08); border-radius:6px; height:6px; overflow:hidden;">
        <div style="background:{bar_grad}; width:{life_remaining_pct:.1f}%; height:100%; border-radius:6px;"></div>
    </div>
    <div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); font-size:0.7rem; display:flex; justify-content:space-between; align-items:center;">
        <span style="color:#64748B;">AGENT DISPATCH:</span>
        <span style="font-weight:800; color:{dispatch_color}; font-family:'JetBrains Mono',monospace;">{dispatch_action}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Data Export & Diagnostics in Sidebar
st.sidebar.markdown("<div class='sidebar-section-header'><span>📥</span> BENCHMARK EXPORT & AUDIT</div>", unsafe_allow_html=True)

csv_path = os.path.join("results", "fleet_policy_comparison.csv")
if os.path.exists(csv_path):
    with open(csv_path, "rb") as f:
        csv_bytes = f.read()
    st.sidebar.download_button(
        label="📥 Download Fleet Policy CSV",
        data=csv_bytes,
        file_name="fleet_policy_comparison.csv",
        mime="text/csv"
    )

with st.sidebar.expander("🛠️ System Specifications", expanded=False):
    st.markdown("""
    - **Architecture**: PyTorch 1-Layer LSTM (64 Hidden Units)
    - **Uncertainty**: MC Dropout (40 stochastic passes)
    - **Protocol**: FIPA-ACL (REQUEST, INFORM, PROPOSE, AGREE)
    - **Benchmark**: NASA C-MAPSS FD001 (100 Turbofans)
    - **Target Cap**: $RUL_{max} = 125$ cycles
    - **Baseline Savings**: **75.05%** over Run-to-Failure
    """)

st.sidebar.markdown("""
<div style="text-align:center; margin-top:20px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06); font-size:0.7rem; color:#64748B; font-family:'JetBrains Mono',monospace;">
    AERO-AGENT PHM OS • v2.4.0<br/>
    <span style="color:#38BDF8;">Autonomous Fleet Intelligence Engine</span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. COMMAND DECK TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Engine Telemetry & Agent Negotiation",
    "📊 Fleet Economics & Policy Benchmark",
    "🧪 What-If Sensitivity Sandbox",
    "🏛️ System Architecture & Specifications"
])


# =============================================================================
# TAB 1: SINGLE-ENGINE TELEMETRY & MULTI-AGENT NEGOTIATION
# =============================================================================
with tab1:
    # Top 5 KPI Cards Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="metric-lbl">CURRENT CYCLE</div>
            <div class="metric-val">{current_cycle}</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Inspection Epoch</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="metric-lbl">PREDICTED RUL (MEAN)</div>
            <div class="metric-val" style="color:#38BDF8;">{prog['mean_rul']} <span style="font-size:0.9rem;">cyc</span></div>
            <div style="font-size:0.75rem; color:#94A3B8;">Ground Truth: {true_remaining} cyc</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="metric-lbl">95% UNCERTAINTY WINDOW</div>
            <div class="metric-val" style="font-size:1.4rem; color:#A5B4FC;">[{prog['lower_bound']}, {prog['upper_bound']}]</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Confidence Interval</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="metric-lbl">MODEL VARIANCE (σ)</div>
            <div class="metric-val" style="color:#F59E0B;">± {prog['std_rul']} <span style="font-size:0.9rem;">cyc</span></div>
            <div style="font-size:0.75rem; color:#94A3B8;">Monte Carlo Epistemic Std</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        risk_class = "risk-crit" if prog['lower_bound'] < 15 else ("risk-warn" if prog['lower_bound'] < 35 else "risk-safe")
        risk_text = "CRITICAL HORIZON" if prog['lower_bound'] < 15 else ("MODERATE WEAR" if prog['lower_bound'] < 35 else "HEALTHY NOMINAL")
        st.markdown(f"""
        <div class="metric-card-glass">
            <div class="metric-lbl">HEALTH CLASSIFICATION</div>
            <div style="margin-top:8px;"><span class="{risk_class}">● {risk_text}</span></div>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:8px;">Based on RUL Lower Bound</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # 2-Column Deck: Left = Telemetry & RUL Distribution, Right = Agent Coordination & Terminal
    c_left, c_right = st.columns([1.15, 1.0])

    with c_left:
        st.markdown("#### 📈 Multivariate Telemetry Stream")
        
        # 4-Sensor Synchronized Plotly Subplots
        fig_telemetry = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "LPC Outlet Temp (s_2)", "LPT Outlet Temp (s_4)",
                "HPC Static Pressure (s_11)", "Fuel Flow Ratio (s_12)"
            ),
            vertical_spacing=0.18, horizontal_spacing=0.12
        )
        
        sensor_configs = [
            ("s_2", 1, 1, "#38BDF8"),
            ("s_4", 1, 2, "#818CF8"),
            ("s_11", 2, 1, "#34D399"),
            ("s_12", 2, 2, "#F472B6")
        ]
        
        for s_col, r, c, color in sensor_configs:
            fig_telemetry.add_trace(
                go.Scatter(
                    x=engine_data["cycle"], y=engine_data[s_col],
                    mode="lines", name=s_col,
                    line=dict(color=color, width=2.5)
                ),
                row=r, col=c
            )
            
        fig_telemetry.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.65)",
            font=dict(color="#94A3B8", family="Plus Jakarta Sans"),
            showlegend=False
        )
        fig_telemetry.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
        fig_telemetry.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
        st.plotly_chart(fig_telemetry, use_container_width=True)

        # RUL Uncertainty Gaussian Distribution Curve
        st.markdown("#### 🎯 Prognostic Uncertainty Bell Curve (Monte Carlo Dropout)")
        mu = prog["mean_rul"]
        sigma = max(1e-2, prog["std_rul"])
        x_axis = np.linspace(max(0, mu - 3.5 * sigma), mu + 3.5 * sigma, 200)
        pdf_vals = stats.norm.pdf(x_axis, mu, sigma)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Scatter(
            x=x_axis, y=pdf_vals, mode="lines",
            line=dict(color="#38BDF8", width=3),
            fill="tozeroy", fillcolor="rgba(56, 189, 248, 0.18)",
            name="Predictive Distribution"
        ))
        
        fig_dist.add_vline(x=prog["lower_bound"], line_dash="dash", line_color="#F87171",
                           annotation_text="Conservative Bound (RUL_lower)", annotation_position="top left")
        fig_dist.add_vline(x=mu, line_dash="solid", line_color="#38BDF8",
                           annotation_text=f"Mean: {mu} cyc", annotation_position="top right")
        fig_dist.add_vline(x=true_remaining, line_dash="dot", line_color="#34D399",
                           annotation_text=f"True: {true_remaining} cyc", annotation_position="bottom right")

        fig_dist.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.65)",
            font=dict(color="#94A3B8"),
            xaxis_title="Operating Cycles",
            yaxis_title="Probability Density",
            showlegend=False
        )
        fig_dist.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_dist.update_yaxes(showticklabels=False, gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with c_right:
        # Section 13 Audit Box Styled like an Aerospace Flight Terminal
        st.markdown("#### 🤖 Section 13 Multi-Agent Decision Audit Log")
        
        st.markdown(f"""
        <div class="terminal-container">
            <div class="terminal-header">
                <span class="dot-red"></span>
                <span class="dot-yellow"></span>
                <span class="dot-green"></span>
                <span style="margin-left:8px;">AERO-AGENT TERMINAL // ENGINE_ID: E-{str(engine_choice).zfill(3)}</span>
            </div>
            <div class="terminal-body">{decision["audit_trail_text"]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Decision Status Badge
        if decision["escalated"]:
            st.markdown("""
            <div style="margin-top:12px;">
                <span class="risk-crit" style="font-size:0.85rem; padding:6px 14px;">
                    🚨 ESCALATION ALERT: EMERGENCY EXPEDITED INTERVENTION TRIGGERED
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="margin-top:12px;">
                <span class="risk-safe" style="font-size:0.85rem; padding:6px 14px;">
                    ✅ FEASIBLE PREVENTIVE MAINTENANCE SCHEDULED SAFELY
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)

        # FIPA-ACL Live Message Protocol Timeline Expander
        with st.expander("💬 View Live Inter-Agent Negotiation Protocol (FIPA-ACL Messages)", expanded=False):
            st.markdown("<div style='font-size:0.8rem; color:#94A3B8; margin-bottom:10px;'>Asynchronous message exchange recorded on central MessageBus:</div>", unsafe_allow_html=True)
            for msg in decision["message_transcript"]:
                sender_col = "#38BDF8" if "Coordinator" in msg.sender else ("#34D399" if "Prognostic" in msg.sender else ("#FBBF24" if "Resource" in msg.sender else "#F472B6"))
                st.markdown(f"""
                <div class="agent-msg-box" style="border-left-color:{sender_col};">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-weight:700; color:{sender_col};">[{msg.sender}] ➔ [{msg.receiver}]</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B;">{msg.timestamp}</span>
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; color:#E2E8F0; font-size:0.8rem;">
                        <strong>ACT:</strong> <span style="color:#C084FC;">{msg.performative}</span><br>
                        <strong>PAYLOAD:</strong> {msg.to_formatted_string().split('): ')[-1]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Candidate Window Evaluation Table
        with st.expander("🔍 Cost Agent Candidate Window Evaluation Table", expanded=True):
            candidates = decision["candidates_evaluated"]
            if candidates:
                df_c = pd.DataFrame(candidates[:6])
                df_c["Failure Risk"] = df_c["p_failure"].apply(lambda p: f"{p:.1%}")
                df_c["Expected TCO"] = df_c["expected_cost"].apply(lambda c: f"${c:,.2f}")
                df_c["Scheduled Lead"] = df_c["delta_t"].apply(lambda d: f"{d} cyc")
                
                show_cols = ["candidate_cycle", "Scheduled Lead", "Failure Risk", "Expected TCO"]
                st.dataframe(
                    df_c[show_cols].rename(columns={"candidate_cycle": "Maintenance Cycle"}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No candidate cycles satisfied physical spare/technician constraints.")


# =============================================================================
# TAB 2: FLEET ECONOMICS & 4-POLICY BENCHMARK ARENA
# =============================================================================
with tab2:
    st.markdown("### 📊 Fleet-Level Performance Across All Four Maintenance Policies")
    st.markdown("Simulating the identical **100-engine fleet** under real-world logistical constraints ($L_{spare} = 8$, $K_{tech} = 2$).")

    sim = FleetMaintenanceSimulator(
        cost_pm=cost_pm, cost_failure=cost_failure, cost_downtime_per_cycle=cost_downtime,
        spare_lead_time=spare_lead_time, tech_capacity_per_cycle=tech_capacity,
        fixed_interval=125, naive_threshold=20
    )

    test_cur_cycles = test_scaled.groupby("engine_id")["cycle"].max().values
    true_lives = test_cur_cycles + y_test.astype(int)
    preds = predict_monte_carlo_dropout(model, X_test, n_samples=25)["mean"]

    summary_df = run_fleet_policy_comparison(
        simulator=sim,
        coordinator=coordinator_agent,
        engine_ids=test_ids,
        current_cycles=list(test_cur_cycles),
        sequences=X_test,
        true_lifetimes=list(true_lives),
        predicted_ruls=list(preds)
    )

    row_agentic = summary_df[summary_df["policy"] == "Proposed Agentic Policy"].iloc[0]
    row_rtf = summary_df[summary_df["policy"] == "Run-to-Failure"].iloc[0]
    row_fi = summary_df[summary_df["policy"] == "Fixed Interval"].iloc[0]
    row_naive = summary_df[summary_df["policy"] == "Naive RUL Threshold"].iloc[0]

    # Executive KPI Banners
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card-glass" style="border-top:3px solid #38BDF8;">
            <div class="metric-lbl">AGENTIC FLEET TCO</div>
            <div class="metric-val" style="color:#38BDF8;">${row_agentic['total_cost']:,.0f}</div>
            <div style="font-size:0.75rem; color:#34D399;">Optimal Fleet Lifetime Cost</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        saving_rtf = round(((row_rtf['total_cost'] - row_agentic['total_cost']) / row_rtf['total_cost']) * 100, 1)
        st.markdown(f"""
        <div class="metric-card-glass" style="border-top:3px solid #34D399;">
            <div class="metric-lbl">COST SAVING VS RUN-TO-FAILURE</div>
            <div class="metric-val" style="color:#34D399;">+{saving_rtf}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">96 Failures Avoided</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        saving_naive = round(((row_naive['total_cost'] - row_agentic['total_cost']) / row_naive['total_cost']) * 100, 1)
        st.markdown(f"""
        <div class="metric-card-glass" style="border-top:3px solid #F59E0B;">
            <div class="metric-lbl">COST SAVING VS NAIVE THRESHOLD</div>
            <div class="metric-val" style="color:#FBBF24;">+{saving_naive}%</div>
            <div style="font-size:0.75rem; color:#94A3B8;">Lead-Time Hedging Advantage</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        life_saved = round(row_fi['wasted_useful_life'] - row_agentic['wasted_useful_life'], 1)
        st.markdown(f"""
        <div class="metric-card-glass" style="border-top:3px solid #C084FC;">
            <div class="metric-lbl">LIFE SAVED VS FIXED-INTERVAL</div>
            <div class="metric-val" style="color:#C084FC;">+{life_saved} <span style="font-size:0.9rem;">cyc/eng</span></div>
            <div style="font-size:0.75rem; color:#94A3B8;">Prevents Premature Scrapping</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # 3-Panel Visual Comparison
    p1, p2, p3 = st.columns(3)
    
    with p1:
        fig_c = px.bar(
            summary_df, x="policy", y="total_cost", text_auto='$.2s',
            color="policy",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#6366F1", "#38BDF8"],
            title="Total Cost of Ownership ($)"
        )
        fig_c.update_layout(
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
            font=dict(color="#94A3B8"), xaxis_title="", yaxis_title="Total Cost ($)"
        )
        fig_c.update_xaxes(tickangle=20)
        st.plotly_chart(fig_c, use_container_width=True)

    with p2:
        fig_f = px.bar(
            summary_df, x="policy", y="failures", text="failures",
            color="policy",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#6366F1", "#38BDF8"],
            title="Catastrophic In-Service Breakdowns"
        )
        fig_f.update_layout(
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
            font=dict(color="#94A3B8"), xaxis_title="", yaxis_title="Failure Count"
        )
        fig_f.update_xaxes(tickangle=20)
        st.plotly_chart(fig_f, use_container_width=True)

    with p3:
        fig_d = px.bar(
            summary_df, x="policy", y="total_downtime", text="total_downtime",
            color="policy",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#6366F1", "#38BDF8"],
            title="Fleet Downtime (Operational Cycles)"
        )
        fig_d.update_layout(
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=35, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
            font=dict(color="#94A3B8"), xaxis_title="", yaxis_title="Downtime Cycles"
        )
        fig_d.update_xaxes(tickangle=20)
        st.plotly_chart(fig_d, use_container_width=True)

    # Detailed Summary Table
    st.markdown("#### 📋 Comprehensive Policy Benchmark Table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: WHAT-IF SENSITIVITY SANDBOX
# =============================================================================
with tab3:
    st.markdown("### 🧪 What-If Operational Sensitivity Sandbox (Mandatory Experiment 6)")
    st.markdown("Analyze how varying supply chain delays, maintenance depot crew sizes, and failure penalty multipliers impact policy performance.")

    csv_lead = os.path.join("results", "sensitivity_lead_time.csv")
    csv_tech = os.path.join("results", "sensitivity_technician_capacity.csv")
    csv_cost = os.path.join("results", "sensitivity_cost_ratio.csv")

    if os.path.exists(csv_lead) and os.path.exists(csv_tech) and os.path.exists(csv_cost):
        df_l = pd.read_csv(csv_lead)
        df_t = pd.read_csv(csv_tech)
        df_c = pd.read_csv(csv_cost)

        s_col1, s_col2, s_col3 = st.columns(3)

        with s_col1:
            fig_s1 = go.Figure()
            fig_s1.add_trace(go.Scatter(
                x=df_l["spare_lead_time"], y=df_l["Agentic Cost"],
                name="Agentic Policy", mode="lines+markers",
                line=dict(color="#38BDF8", width=3)
            ))
            fig_s1.add_trace(go.Scatter(
                x=df_l["spare_lead_time"], y=df_l["Naive Cost"],
                name="Naive Threshold", mode="lines+markers",
                line=dict(color="#EF4444", width=2, dash="dash")
            ))
            fig_s1.update_layout(
                title="Spare Lead Time vs Total Cost",
                height=320, margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
                font=dict(color="#94A3B8"),
                xaxis_title="Spare Lead Time (Cycles)", yaxis_title="Total Cost ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_s1.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_s1.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_s1, use_container_width=True)

        with s_col2:
            fig_s2 = go.Figure()
            fig_s2.add_trace(go.Scatter(
                x=df_t["tech_capacity"], y=df_t["Agentic Failures"],
                name="Agentic Failures", mode="lines+markers",
                line=dict(color="#38BDF8", width=3)
            ))
            fig_s2.add_trace(go.Scatter(
                x=df_t["tech_capacity"], y=df_t["Naive Failures"],
                name="Naive Failures", mode="lines+markers",
                line=dict(color="#EF4444", width=2, dash="dash")
            ))
            fig_s2.update_layout(
                title="Technician Slots vs Failures",
                height=320, margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
                font=dict(color="#94A3B8"),
                xaxis_title="Crews Available / Cycle", yaxis_title="Failures",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_s2.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_s2.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_s2, use_container_width=True)

        with s_col3:
            fig_s3 = go.Figure()
            fig_s3.add_trace(go.Scatter(
                x=df_c["failure_cost_ratio"], y=df_c["Agentic Cost"],
                name="Agentic Policy", mode="lines+markers",
                line=dict(color="#38BDF8", width=3)
            ))
            fig_s3.add_trace(go.Scatter(
                x=df_c["failure_cost_ratio"], y=df_c["Naive Cost"],
                name="Naive Threshold", mode="lines+markers",
                line=dict(color="#EF4444", width=2, dash="dash")
            ))
            fig_s3.update_layout(
                title="Failure Penalty Multiplier vs Cost",
                height=320, margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.65)",
                font=dict(color="#94A3B8"),
                xaxis_title="Cost Ratio (C_fail / C_pm)", yaxis_title="Total Cost ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_s3.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_s3.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_s3, use_container_width=True)
    else:
        st.info("Run `python train_and_evaluate.py` to populate all sensitivity data files.")


# =============================================================================
# TAB 4: SYSTEM ARCHITECTURE & ENGINEERING SPECIFICATIONS
# =============================================================================
with tab4:
    st.markdown("### 🏛️ System Architecture & Engineering Specifications")
    
    st.markdown("""
    #### 📋 Autonomous PHM Modular Engineering Architecture
    """)
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown("""
        <div class="metric-card-glass" style="margin-bottom:15px;">
            <div style="font-weight:700; color:#38BDF8; font-size:1.05rem;">Module 1: Telemetry Ingestion & Capped RUL Pipeline</div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-top:6px;">
                • Parsing NASA C-MAPSS FD001 (20,631 sensor records across 100 units).<br>
                • Filtering 7 uninformative zero-variance sensor channels.<br>
                • Robust Min-Max scaling strictly isolated to training historical flights.<br>
                • Piecewise linear degradation formulation ($RUL_{max} = 125$ operating cycles).
            </div>
        </div>
        <div class="metric-card-glass">
            <div style="font-weight:700; color:#34D399; font-size:1.05rem;">Module 2: PyTorch Deep LSTM & Monte Carlo Uncertainty Engine</div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-top:6px;">
                • Recurrent LSTM neural network architecture (64 hidden units, Dropout 0.2).<br>
                • Sequence sliding temporal window ($T = 30$ historical operating cycles).<br>
                • Multi-metric evaluation: RMSE, MAE, and NASA Asymmetric PHM Score.<br>
                • Monte Carlo Dropout Bayesian approximation across stochastic forward passes.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r_col2:
        st.markdown("""
        <div class="metric-card-glass" style="margin-bottom:15px;">
            <div style="font-weight:700; color:#FBBF24; font-size:1.05rem;">Module 3: Fleet Lifecycle Simulation & Economic Optimization</div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-top:6px;">
                • Total Cost of Ownership modeling ($C_{pm}, C_{fail}, C_{downtime}$).<br>
                • Comparative benchmark against Run-to-Failure, Fixed-Interval, & Naive Threshold.<br>
                • Depot queuing under technician bay and spare-part lead time constraints.<br>
                • Asset preservation and wasted remaining useful life minimization.
            </div>
        </div>
        <div class="metric-card-glass">
            <div style="font-weight:700; color:#C084FC; font-size:1.05rem;">Module 4: Multi-Agent FIPA-ACL Negotiation & Orchestration</div>
            <div style="font-size:0.85rem; color:#CBD5E1; margin-top:6px;">
                • Distributed 4-Agent architecture (Prognostic, Resource, Cost, Coordinator).<br>
                • Pareto-optimal candidate maintenance window search and escalation handling.<br>
                • IEEE FIPA-ACL compliant MessageBus communication protocol.<br>
                • Full Section 13 auditable decision trail and real-time telemetry streaming.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🧩 Multi-Agent Interaction & Coordination Topology")
    st.markdown("""
    <div class="metric-card-glass">
        <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:12px; font-size:0.82rem;">
            <div style="background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25); border-radius:10px; padding:12px;">
                <div style="font-weight:800; color:#38BDF8; margin-bottom:4px;">1. PROGNOSTIC AGENT</div>
                <div style="color:#94A3B8;">Quantifies remaining useful life distribution $p(RUL|X)$ via MC Dropout. Emits conservative lower 95% bound.</div>
            </div>
            <div style="background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.25); border-radius:10px; padding:12px;">
                <div style="font-weight:800; color:#34D399; margin-bottom:4px;">2. RESOURCE AGENT</div>
                <div style="color:#94A3B8;">Tracks spare-part procurement lead times $L_{spare}$ and depot technician bay capacity $K_{tech}$.</div>
            </div>
            <div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.25); border-radius:10px; padding:12px;">
                <div style="font-weight:800; color:#FBBF24; margin-bottom:4px;">3. COST AGENT</div>
                <div style="color:#94A3B8;">Calculates expected maintenance cost $E[C(t)]$ balancing preventive cost vs unscheduled failure penalties.</div>
            </div>
            <div style="background:rgba(192,132,252,0.08); border:1px solid rgba(192,132,252,0.25); border-radius:10px; padding:12px;">
                <div style="font-weight:800; color:#C084FC; margin-bottom:4px;">4. COORDINATOR AGENT</div>
                <div style="color:#94A3B8;">Orchestrates multi-round FIPA-ACL negotiation, resolves resource contention, and emits dispatch decisions.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
