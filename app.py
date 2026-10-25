
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Operating Room Optimizer",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Operating-Room Scheduling Optimizer")
st.caption("Reinforcement Learning Project | DQN and PPO integration planned")

st.sidebar.header("Simulation Settings")

number_of_surgeries = st.sidebar.slider(
    "Number of surgeries", 10, 50, 18
)

strategy = st.sidebar.selectbox(
    "Scheduling strategy",
    ["FCFS", "Urgency-First"]
)

seed = st.sidebar.number_input(
    "Random seed", min_value=0, value=42
)

if st.sidebar.button("Generate Schedule"):
    st.session_state["run_simulation"] = True

if st.sidebar.button("Reset"):
    st.session_state["run_simulation"] = False

if st.session_state.get("run_simulation", False):
    rng = np.random.default_rng(int(seed))

    data = pd.DataFrame({
        "Surgery ID": [
            f"S{i:03d}" for i in range(1, number_of_surgeries + 1)
        ],
        "Urgency": rng.integers(1, 4, number_of_surgeries),
        "Arrival (min)": rng.integers(0, 180, number_of_surgeries),
        "Duration (min)": rng.integers(45, 241, number_of_surgeries),
        "Recovery (min)": rng.integers(30, 121, number_of_surgeries),
        "Specialty": rng.choice(
            ["General", "Orthopedic", "Cardiac"],
            number_of_surgeries
        )
    })

    data = data.sort_values("Arrival (min)").reset_index(drop=True)

    if strategy == "Urgency-First":
        data = data.sort_values(
            ["Urgency", "Arrival (min)"],
            ascending=[False, True]
        ).reset_index(drop=True)

    rooms = {
        "General": 0,
        "Orthopedic": 0,
        "Cardiac": 0
    }

    schedule = []

    for _, surgery in data.iterrows():
        specialty = surgery["Specialty"]
        start = max(
            rooms[specialty],
            int(surgery["Arrival (min)"])
        )
        finish = start + int(surgery["Duration (min)"])
        rooms[specialty] = finish

        schedule.append({
            **surgery.to_dict(),
            "Operating Room": specialty,
            "Start (min)": start,
            "Finish (min)": finish,
            "Waiting (min)": start - int(surgery["Arrival (min)"]),
            "Overtime (min)": max(0, finish - 480)
        })

    results = pd.DataFrame(schedule)

    average_wait = results["Waiting (min)"].mean()
    overtime = results.groupby("Operating Room")["Finish (min)"].max()
    total_overtime = (overtime - 480).clip(lower=0).sum()
    utilization = (
        results["Duration (min)"].sum() / (3 * 480) * 100
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Surgeries", len(results))
    c2.metric("Average waiting", f"{average_wait:.1f} min")
    c3.metric("Room utilization", f"{utilization:.1f}%")
    c4.metric("Total room overtime", f"{total_overtime:.0f} min")

    st.subheader("Surgery Schedule")
    st.dataframe(results, use_container_width=True)

    st.subheader("Surgery Waiting Times")
    fig = px.bar(
        results,
        x="Surgery ID",
        y="Waiting (min)",
        color="Urgency",
        title="Waiting time for each surgery"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Operating-Room Timeline")
    fig2 = px.timeline(
        results,
        x_start="Start (min)",
        x_end="Finish (min)",
        y="Operating Room",
        color="Surgery ID",
        hover_data=["Urgency", "Waiting (min)"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Schedule CSV",
        data=csv,
        file_name="surgery_schedule.csv",
        mime="text/csv"
    )

    st.info(
        "This is a synthetic-data dashboard using baseline scheduling "
        "rules. DQN/PPO models have not been connected yet."
    )
else:
    st.info("Choose your settings in the sidebar and click Generate Schedule.")
