"""
Head-to-head telemetry dashboard.

Pick a season, a race weekend, and one of its actual sessions (practice,
qualifying, sprint, race -- whichever that weekend had), then up to 3
drivers; compares their fastest lap in that session.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import fastf1
import numpy as np
import plotly.graph_objects as go
import streamlit as st

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="F1 Telemetry Compare", layout="wide")
st.title("Head-to-head telemetry")

MAX_DRIVERS = 3
DRIVER_COLORS = ["#e10600", "#1e88e5", "#43a047"]  # kept distinct in both themes
CHART_HEIGHT = 380

# FastF1's timing/telemetry data is only reliably complete from 2018 on.
YEARS = list(range(2026, 2017, -1))


@st.cache_data(ttl=3600)
def load_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule["RoundNumber"] > 0]


@st.cache_data
def load_session(year, event, session_name):
    session = fastf1.get_session(year, event, session_name)
    session.load()
    return session


def session_names_for(event_row):
    """The sessions actually run that weekend, in order -- practice/qualifying
    count and naming (Sprint Qualifying, Sprint, ...) depend on the event
    format, so this is read from the schedule instead of assumed fixed."""
    names = []
    for i in range(1, 6):
        name = event_row.get(f"Session{i}")
        if isinstance(name, str) and name:
            names.append(name)
    return names


year = st.sidebar.selectbox("Year", options=YEARS)
schedule = load_schedule(year)

event_name = st.sidebar.selectbox("Grand Prix", options=schedule["EventName"].tolist())
event_row = schedule[schedule["EventName"] == event_name].iloc[0]

session_name = st.sidebar.selectbox("Session", options=session_names_for(event_row))

try:
    session = load_session(year, event_name, session_name)
except Exception as exc:
    st.error(f"Couldn't load this session (it may not have happened yet): {exc}")
    st.stop()

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

laps_by_driver = {d: session.laps.pick_drivers(d).pick_fastest() for d in selected_drivers}
telemetry_by_driver = {d: lap.get_car_data().add_distance() for d, lap in laps_by_driver.items()}
colors = dict(zip(selected_drivers, DRIVER_COLORS))

st.subheader(f"{year} {event_name} — {session_name} — fastest lap comparison")
st.table(
    [
        {
            "Driver": d,
            "Lap time": str(lap["LapTime"]).split(" ")[-1][:-3],
            "Top speed (km/h)": int(telemetry_by_driver[d]["Speed"].max()),
        }
        for d, lap in laps_by_driver.items()
    ]
)


def base_figure(title, yaxis_title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Distance (m)",
        yaxis_title=yaxis_title,
        height=CHART_HEIGHT,
        hovermode="x unified",
        margin=dict(t=40, b=40),
    )
    return fig


fig_speed = base_figure("Speed", "km/h")
fig_throttle = base_figure("Throttle", "%")
fig_brake = base_figure("Brake", "on/off")

for driver in selected_drivers:
    tel = telemetry_by_driver[driver]
    color = colors[driver]
    fig_speed.add_trace(go.Scatter(x=tel["Distance"], y=tel["Speed"], name=driver, line=dict(color=color)))
    fig_throttle.add_trace(go.Scatter(x=tel["Distance"], y=tel["Throttle"], name=driver, line=dict(color=color)))
    fig_brake.add_trace(
        go.Scatter(x=tel["Distance"], y=tel["Brake"].astype(int), name=driver, line=dict(color=color, shape="hv"))
    )

# Delta time: gap to the fastest of the selected laps, over distance. Each
# driver's telemetry has its own Distance grid, so the others are
# interpolated onto the reference driver's grid before subtracting elapsed
# time -- this is the same idea as fastf1.utils.delta_time, done directly
# here since that helper is deprecated and the library's own docs flag it as
# not very accurate.
reference_driver = min(laps_by_driver, key=lambda d: laps_by_driver[d]["LapTime"])
ref_tel = telemetry_by_driver[reference_driver]
ref_elapsed = (ref_tel["Time"] - ref_tel["Time"].iloc[0]).dt.total_seconds().to_numpy()
ref_distance = ref_tel["Distance"].to_numpy()

fig_delta = base_figure("Delta time", f"s (vs. {reference_driver})")
fig_delta.add_hline(y=0, line_dash="dash", line_color="gray")

for driver in selected_drivers:
    if driver == reference_driver:
        continue
    tel = telemetry_by_driver[driver]
    elapsed = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().to_numpy()
    distance = tel["Distance"].to_numpy()
    elapsed_on_ref_grid = np.interp(ref_distance, distance, elapsed)
    delta = elapsed_on_ref_grid - ref_elapsed
    fig_delta.add_trace(
        go.Scatter(x=ref_distance, y=delta, name=f"{driver} vs {reference_driver}", line=dict(color=colors[driver]))
    )

row1_left, row1_right = st.columns(2)
row1_left.plotly_chart(fig_speed, width="stretch")
row1_right.plotly_chart(fig_delta, width="stretch")

row2_left, row2_right = st.columns(2)
row2_left.plotly_chart(fig_throttle, width="stretch")
row2_right.plotly_chart(fig_brake, width="stretch")
