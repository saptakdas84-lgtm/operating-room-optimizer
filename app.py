import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from stable_baselines3 import DQN, PPO

st.set_page_config(page_title="Operating Room Optimizer", page_icon="🏥", layout="wide")
st.title("🏥 Operating-Room Scheduling Optimizer")
st.caption("Synthetic surgery scheduling simulation with baseline, DQN, and PPO options")

with st.sidebar:
    st.header("Simulation Settings")
    number_of_surgeries = st.slider("Number of surgeries", 10, 50, 18)
    number_of_rooms = st.slider("Number of operating rooms", 1, 6, 3)
    workday_minutes = st.slider("Workday length (minutes)", 240, 720, 480, step=30)
    seed = st.number_input("Random seed", min_value=0, value=42)
    strategy = st.selectbox(
        "Scheduling strategy",
        ["FCFS", "Urgency-First", "Shortest-Duration", "DQN", "PPO"]
    )
    st.caption("DQN/PPO require compatible model ZIP files and the matching training environment.")

@st.cache_data
def generate_surgeries(n, seed):
    rng = np.random.default_rng(int(seed))
    return pd.DataFrame({
        "Surgery ID": [f"S{i:03d}" for i in range(1, n + 1)],
        "Urgency": rng.integers(1, 4, n),
        "Arrival (min)": rng.integers(0, 180, n),
        "Duration (min)": rng.integers(45, 241, n),
        "Recovery (min)": rng.integers(30, 121, n),
        "Specialty": rng.choice(["General", "Orthopedic", "Cardiac"], n),
    }).sort_values(["Arrival (min)", "Surgery ID"]).reset_index(drop=True)


def baseline_order(data, method):
    if method == "Urgency-First":
        return data.sort_values(["Urgency", "Arrival (min)"],
                                ascending=[False, True]).reset_index(drop=True)
    if method == "Shortest-Duration":
        return data.sort_values(["Duration (min)", "Arrival (min)"]).reset_index(drop=True)
    return data.sort_values(["Arrival (min)", "Surgery ID"]).reset_index(drop=True)


def schedule_baseline(data, method, n_rooms, workday):
    ordered = baseline_order(data, method)
    room_free = [0] * n_rooms
    output = []
    for _, s in ordered.iterrows():
        room_index = int(np.argmin(room_free))
        start = max(int(s["Arrival (min)"]), room_free[room_index])
        finish = start + int(s["Duration (min)"])
        room_free[room_index] = finish
        output.append({
            **s.to_dict(),
            "Operating Room": f"OR {room_index + 1}",
            "Start (min)": start,
            "Finish (min)": finish,
            "Waiting (min)": start - int(s["Arrival (min)"]),
            "Overtime (min)": max(0, finish - workday),
            "Scheduling Method": method,
        })
    return pd.DataFrame(output).sort_values(
        ["Operating Room", "Start (min)"]
    ).reset_index(drop=True)


def load_rl_model(method):
    model_path = "dqn_model.zip" if method == "DQN" else "ppo_model.zip"
    if not os.path.exists(model_path):
        return None, f"{model_path} was not found in the app's GitHub repository."
    try:
        model = DQN.load(model_path) if method == "DQN" else PPO.load(model_path)
        return model, None
    except Exception as exc:
        return None, f"Could not load {model_path}: {exc}"


def schedule_rl(data, method, n_rooms, workday):
    """
    IMPORTANT: Generic action adapter for a discrete-action model.
    This assumes the model takes a 1-D observation of:
    [urgency, arrival, duration, recovery, room_free_times...]
    and predicts an action in [0, n_rooms-1] indicating the room.
    It only works if training used this exact observation/action meaning.
    """
    model, error = load_rl_model(method)
    if error:
        raise ValueError(error)

    room_free = [0] * n_rooms
    ordered = data.sort_values(["Arrival (min)", "Surgery ID"]).reset_index(drop=True)
    output = []
