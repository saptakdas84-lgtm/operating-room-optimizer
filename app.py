import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="OR Scheduling Optimizer", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(120,120,120,.08); padding: 14px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🏥 Operating-Room Scheduling Optimizer")
st.caption("Interactive simulation dashboard · Synthetic data · Research prototype")

with st.sidebar:
    st.header("Simulation settings")
    n_surgeries = st.slider("Number of surgeries", 8, 40, 18)
    n_rooms = st.slider("Operating rooms", 1, 6, 3)
    workday = st.slider("Workday length (minutes)", 240, 720, 480, step=30)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)
    st.markdown("---")
    st.info("This dashboard uses synthetic data and simple scheduling heuristics. It is not a clinical decision-support tool.")

@st.cache_data
def make_surgeries(n, seed):
    rng = np.random.default_rng(seed)
    specialties = ["General", "Orthopedic", "Cardiac"]
    rows = []
    for i in range(n):
        duration = int(rng.integers(45, 181))
        urgency = int(rng.integers(1, 6))
        rows.append({
            "Surgery": f"S{i+1:02d}",
            "Specialty": str(rng.choice(specialties)),
            "Duration (min)": duration,
            "Urgency (1–5)": urgency,
            "Arrival (min)": int(rng.integers(0, 121))
        })
    return pd.DataFrame(rows)

def schedule_surgeries(df, n_rooms, method):
    data = df.copy()
    if method == "FCFS":
        data = data.sort_values(["Arrival (min)", "Surgery"])
    elif method == "Urgency-First":
        data = data.sort_values(["Urgency (1–5)", "Arrival (min)"], ascending=[False, True])
    elif method == "Shortest-Estimated-Time":
        data = data.sort_values(["Duration (min)", "Arrival (min)"])
    room_end = [0] * n_rooms
    room_rows = []
    for _, row in data.iterrows():
        # Choose the room that becomes available earliest.
        room = int(np.argmin(room_end))
        start = max(int(row["Arrival (min)"]), room_end[room])
        end = start + int(row["Duration (min)"])
        room_end[room] = end
        room_rows.append({
            **row.to_dict(),
            "Room": f"OR {room+1}",
            "Start (min)": start,
            "End (min)": end,
            "Waiting (min)": start - int(row["Arrival (min)"]),
            "Overtime (min)": max(0, end - workday)
        })
    return pd.DataFrame(room_rows)

surgeries = make_surgeries(n_surgeries, int(seed))
methods = ["FCFS", "Urgency-First", "Shortest-Estimated-Time"]
schedules = {m: schedule_surgeries(surgeries, n_rooms, m) for m in methods}

def metrics(sched, method):
    return {
        "Algorithm": method,
        "Average waiting (min)": round(sched["Waiting (min)"].mean(), 1),
        "Total waiting (min)": int(sched["Waiting (min)"].sum()),
        "Scheduled within workday (%)": round((sched["End (min)"] <= workday).mean() * 100, 1),
        "Total overtime (min)": int(sched["Overtime (min)"].sum()),
        "Makespan (min)": int(sched["End (min)"].max())
    }

summary = pd.DataFrame([metrics(schedules[m], m) for m in methods])
best = summary.loc[summary["Average waiting (min)"].idxmin()]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗓️ Schedule timeline", "⚖️ Compare algorithms", "📋 Surgery data"])

with tab1:
    st.subheader("At-a-glance results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Surgeries", n_surgeries)
    c2.metric("Operating rooms", n_rooms)
    c3.metric("Lowest average wait", f"{best['Average waiting (min)']} min", str(best["Algorithm"]))
    c4.metric("Best within-workday rate", f"{summary['Scheduled within workday (%)'].max()}%")
