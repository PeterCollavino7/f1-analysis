"""
Head-to-head telemetry dashboard.

Pick a season, a race weekend, and one of its actual sessions (practice,
qualifying, sprint, race -- whichever that weekend had), then up to 3
drivers; compares their fastest lap in that session.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import datetime

import fastf1
import fastf1.plotting
import numpy as np
import plotly.graph_objects as go
import streamlit as st

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="F1 Telemetry Compare", layout="wide", initial_sidebar_state="expanded")

MAX_DRIVERS = 3
CHART_HEIGHT = 340
LINE_STYLES = ["solid", "dash", "dot"]  # cycled when two selected drivers share a team color
KM_TO_MI = 0.621371
M_TO_FT = 3.28084

# FastF1's timing/telemetry data is only reliably complete from 2018 on.
YEARS = list(range(2026, 2017, -1))

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at top left, #1a1f2e 0%, #0e1117 55%); }
    .driver-card {
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        background: rgba(255,255,255,0.04);
        border-left: 5px solid var(--card-color);
    }
    .driver-card .name { font-size: 1.1rem; font-weight: 700; color: var(--card-color); }
    .driver-card .laptime { font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-size: 1.35rem; margin-top: 0.15rem; }
    .driver-card .sub { opacity: 0.65; font-size: 0.8rem; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🏁 Head-to-head telemetry")


@st.cache_data(ttl=3600)
def load_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0]
    # Only weekends that have actually happened -- a future round on the
    # calendar has no session data yet, so it has no business in the
    # dropdown. Session5 is the last session of the weekend (Race, or Sprint
    # weekends still end on Race), so its time is the real "is this done".
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    schedule = schedule[schedule["Session5DateUtc"] <= now_utc]
    return schedule.sort_values("RoundNumber")


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


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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

units = st.sidebar.radio("Units", options=["Metric (km/h, m)", "Imperial (mph, ft)"], horizontal=False)
imperial = units.startswith("Imperial")
speed_unit = "mph" if imperial else "km/h"
dist_unit = "ft" if imperial else "m"

if not selected_drivers:
    st.info("Select at least one driver from the sidebar.")
    st.stop()

laps_by_driver = {d: session.laps.pick_drivers(d).pick_fastest() for d in selected_drivers}
telemetry_by_driver = {d: lap.get_car_data().add_distance() for d, lap in laps_by_driver.items()}

# Real team colors, like FastF1's own plotting module uses -- but two
# selected teammates (e.g. LEC/HAM at Ferrari) then share the exact same
# color, which defeats the point of a head-to-head chart. When that happens,
# later drivers with a repeated color get a dashed/dotted line instead of a
# second solid line of the same color.
driver_style = {}
seen_colors = {}
for driver in selected_drivers:
    color = fastf1.plotting.get_driver_color(driver, session)
    dash = LINE_STYLES[seen_colors.get(color, 0)]
    seen_colors[color] = seen_colors.get(color, 0) + 1
    driver_style[driver] = (color, dash)

st.subheader(f"{year} {event_name} — {session_name} — fastest lap comparison")

cards = st.columns(len(selected_drivers))
for col, driver in zip(cards, selected_drivers):
    lap = laps_by_driver[driver]
    color, dash = driver_style[driver]
    top_speed = telemetry_by_driver[driver]["Speed"].max()
    if imperial:
        top_speed *= KM_TO_MI
    dash_note = f" ({dash} line)" if dash != "solid" else ""
    col.markdown(
        f"""
        <div class="driver-card" style="--card-color:{color}">
            <div class="name">{driver}{dash_note}</div>
            <div class="laptime">{str(lap["LapTime"]).split(" ")[-1][:-3]}</div>
            <div class="sub">Top speed: {top_speed:.0f} {speed_unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


def base_figure(title, yaxis_title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=f"Distance ({dist_unit})",
        yaxis_title=yaxis_title,
        height=CHART_HEIGHT,
        hovermode="x unified",
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


fig_speed = base_figure("Speed", speed_unit)
fig_throttle = base_figure("Throttle", "%")
fig_brake = base_figure("Brake", "")
fig_brake.update_yaxes(tickvals=[0, 1], ticktext=["Off", "On"], range=[-0.15, 1.15])

for driver in selected_drivers:
    tel = telemetry_by_driver[driver]
    color, dash = driver_style[driver]
    distance = tel["Distance"] * (M_TO_FT if imperial else 1)
    speed = tel["Speed"] * (KM_TO_MI if imperial else 1)

    fig_speed.add_trace(
        go.Scatter(
            x=distance, y=speed, name=driver, line=dict(color=color, dash=dash, width=2.5),
            hovertemplate=f"{driver}: %{{y:.0f}} {speed_unit} · %{{x:.0f}} {dist_unit}<extra></extra>",
        )
    )
    fig_throttle.add_trace(
        go.Scatter(
            x=distance, y=tel["Throttle"], name=driver, line=dict(color=color, dash=dash, width=2.5),
            hovertemplate=f"{driver}: %{{y:.0f}}% · %{{x:.0f}} {dist_unit}<extra></extra>",
        )
    )
    brake_state = np.where(tel["Brake"], "On", "Off")
    fig_brake.add_trace(
        go.Scatter(
            x=distance, y=tel["Brake"].astype(int), name=driver, line=dict(color=color, dash=dash, width=2.5, shape="hv"),
            text=brake_state,
            hovertemplate=f"{driver}: " + "%{text}" + f" · %{{x:.0f}} {dist_unit}<extra></extra>",
        )
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
ref_distance_m = ref_tel["Distance"].to_numpy()
ref_distance = ref_distance_m * (M_TO_FT if imperial else 1)

fig_delta = base_figure("Delta time", f"s (vs. {reference_driver})")
fig_delta.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

for driver in selected_drivers:
    if driver == reference_driver:
        continue
    tel = telemetry_by_driver[driver]
    color, dash = driver_style[driver]
    elapsed = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().to_numpy()
    elapsed_on_ref_grid = np.interp(ref_distance_m, tel["Distance"].to_numpy(), elapsed)
    delta = elapsed_on_ref_grid - ref_elapsed
    fig_delta.add_trace(
        go.Scatter(
            x=ref_distance, y=delta, name=f"{driver} vs {reference_driver}",
            line=dict(color=color, dash=dash, width=2.5),
            fill="tozeroy", fillcolor=hex_to_rgba(color, 0.15),
            hovertemplate=f"{driver} vs {reference_driver}: %{{y:+.2f}} s · %{{x:.0f}} {dist_unit}<extra></extra>",
        )
    )

st.plotly_chart(fig_speed, width="stretch")
st.plotly_chart(fig_delta, width="stretch")
st.plotly_chart(fig_throttle, width="stretch")
st.plotly_chart(fig_brake, width="stretch")
