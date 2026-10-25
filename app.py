import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="OR Optimizer | Hospital Operations",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      [data-testid="stSidebar"] { background: #101d35; }
      [data-testid="stSidebar"] * { color: #f3f6fc !important; }
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg, #102344 0%, #176b87 100%);
        color: white; margin-bottom: 1.1rem;
      }
      .hero h1 { margin: 0; font-size: 2rem; color: white; }
      .hero p { margin: .45rem 0 0 0; color: #dceaf5; }
      .metric-card {
        background: white; padding: 1rem 1.1rem;
        border: 1px solid #e4eaf2; border-radius: 14px;
        box-shadow: 0 2px 10px rgba(16,35,68,.04);
      }
      .metric-label { color: #64748b; font-size: .85rem; }
      .metric-value { color: #102344; font-size: 1.65rem; font-weight: 750; }
      .metric-note { color: #64748b; font-size: .78rem; }
      .section-title { color: #102344; font-size: 1.15rem; font-weight: 700; }
      .small-note { color: #64748b; font-size: .86rem; }
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
      .stButton button, .stDownloadButton button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
SPECIALTIES = ["General", "Orthopedic", "Cardiac"]
ROOMS = {
    "OR-1 · General": "General",
    "OR-2 · Orthopedic": "Orthopedic",
    "OR-3 · Cardiac": "Cardiac",
}


def make_surgeries(n: int, seed: int) -> pd.DataFrame:
    """Create reproducible synthetic surgery requests."""
    rng = np.random.default_rng(seed)
    specialties = rng.choice(SPECIALTIES, size=n, p=[0.45, 0.35, 0.20])
    urgency = rng.choice([1, 2, 3], size=n, p=[0.35, 0.40, 0.25])
    estimated = rng.integers(45, 241, size=n)
    actual = np.maximum(20, (estimated * rng.normal(1.0, 0.12, size=n)).astype(int))
    arrival = rng.integers(0, 181, size=n)
    recovery = rng.integers(30, 121, size=n)
    return pd.DataFrame({
        "Surgery ID": [f"S{i:03d}" for i in range(1, n + 1)],
        "Specialty": specialties,
        "Urgency": urgency,
        "Arrival (min)": arrival,
        "Estimated duration (min)": estimated,
        "Actual duration (min)": actual,
        "Recovery (min)": recovery,
    }).sort_values(["Arrival (min)", "Surgery ID"]).reset_index(drop=True)


def schedule_surgeries(data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Baseline-only demonstration scheduler with specialty-compatible rooms."""
    work = data.copy()
    if strategy == "Urgency-First":
        work = work.sort_values(
            ["Urgency", "Arrival (min)"], ascending=[False, True]
        )
    elif strategy == "Shortest-Estimated-Time":
        work = work.sort_values(
            ["Estimated duration (min)", "Arrival (min)"], ascending=[True, True]
        )
    else:  # FCFS
        work = work.sort_values(["Arrival (min)", "Surgery ID"])

    room_free = {room: 0 for room in ROOMS}
    output = []
