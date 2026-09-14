"""
Head-to-head telemetry dashboard.

Pick a session and up to 3 drivers; compares their fastest lap in that
session as speed/throttle/brake traces over distance around the lap.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import fastf1
import plotly.graph_objects as go
import streamlit as st

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="F1 Telemetry Compare", layout="wide")
st.title("Head-to-head telemetry")

YEAR = 2026
EVENT = "Spanish Grand Prix"

MAX_DRIVERS = 3
DRIVER_COLORS = ["#e10600", "#1e88e5", "#43a047"]  # kept distinct in both themes


@st.cache_data
def load_session(year, event, session_type):
    session = fastf1.get_session(year, event, session_type)
    session.load()
    return session


session_type = st.sidebar.selectbox(
    "Session", options=["Q", "R"], format_func=lambda s: {"Q": "Qualifying", "R": "Race"}[s]
)
session = load_session(YEAR, EVENT, session_type)

all_drivers = sorted(session.laps["Driver"].unique())
selected_drivers = st.sidebar.multiselect(
    "Drivers (max 3)",
    options=all_drivers,
    default=all_drivers[:2],
    max_selections=MAX_DRIVERS,
)

if not selected_drivers:
    st.info("Select at least one driver from the sidebar.")
    st.stop()

fig_speed = go.Figure()
fig_throttle = go.Figure()
fig_brake = go.Figure()
summary_rows = []

for driver, color in zip(selected_drivers, DRIVER_COLORS):
    lap = session.laps.pick_drivers(driver).pick_fastest()
    telemetry = lap.get_car_data().add_distance()

    fig_speed.add_trace(
        go.Scatter(
            x=telemetry["Distance"], y=telemetry["Speed"],
            name=driver, line=dict(color=color),
        )
    )
    fig_throttle.add_trace(
        go.Scatter(
            x=telemetry["Distance"], y=telemetry["Throttle"],
            name=driver, line=dict(color=color),
        )
    )
    fig_brake.add_trace(
        go.Scatter(
            x=telemetry["Distance"], y=telemetry["Brake"].astype(int),
            name=driver, line=dict(color=color, shape="hv"),
        )
    )
    summary_rows.append(
        {
            "Driver": driver,
            "Lap time": str(lap["LapTime"]).split(" ")[-1][:-3],
            "Top speed (km/h)": int(telemetry["Speed"].max()),
        }
    )

st.subheader(f"{YEAR} {EVENT} — {session_type} — fastest lap comparison")
st.table(summary_rows)

fig_speed.update_layout(title="Speed", xaxis_title="Distance (m)", yaxis_title="km/h", height=350)
fig_throttle.update_layout(title="Throttle", xaxis_title="Distance (m)", yaxis_title="%", height=250)
fig_brake.update_layout(title="Brake", xaxis_title="Distance (m)", yaxis_title="on/off", height=200)

st.plotly_chart(fig_speed, use_container_width=True)
st.plotly_chart(fig_throttle, use_container_width=True)
st.plotly_chart(fig_brake, use_container_width=True)
