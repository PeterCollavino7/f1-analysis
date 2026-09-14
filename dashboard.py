"""
F1 session dashboard: head-to-head telemetry, and race pace / tyre
degradation, for any past race weekend and any of its actual sessions.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import datetime

import fastf1
import fastf1.plotting
import numpy as np
import plotly.graph_objects as go
import streamlit as st

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="F1 Dashboard", layout="wide", initial_sidebar_state="expanded")

MAX_DRIVERS = 3
CHART_HEIGHT = 340
LINE_STYLES = ["solid", "dash", "dot"]  # cycled when two selected drivers share a team color
KM_TO_MI = 0.621371
M_TO_FT = 3.28084
MIN_LAPS_FOR_TREND = 10  # per compound, for the pooled fuel-correction fit
MIN_LAPS_PER_DRIVER = 6  # per driver+compound, for the per-driver breakdown

COMPOUND_COLORS = {
    "SOFT": "#ff3333",
    "MEDIUM": "#ffd400",
    "HARD": "#f0f0f0",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
}

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
st.title("🏁 F1 Dashboard")


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


def base_figure(title, yaxis_title, xaxis_title, hovermode="x"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=CHART_HEIGHT,
        hovermode=hovermode,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig


# ---------------------------------------------------------------- sidebar --

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

st.subheader(f"{year} {event_name} — {session_name}")

tab_telemetry, tab_pace = st.tabs(["Head-to-head telemetry", "Race pace & tyre degradation"])

# ------------------------------------------------------------- telemetry --

with tab_telemetry:
    all_drivers = sorted(session.laps["Driver"].unique())
    selected_drivers = st.multiselect(
        "Drivers (max 3)",
        options=all_drivers,
        default=all_drivers[:2],
        max_selections=MAX_DRIVERS,
        key="telemetry_drivers",
    )
    units = st.radio("Units", options=["Metric (km/h, m)", "Imperial (mph, ft)"], horizontal=True)
    imperial = units.startswith("Imperial")
    speed_unit = "mph" if imperial else "km/h"
    dist_unit = "ft" if imperial else "m"

    if not selected_drivers:
        st.info("Select at least one driver above.")
        st.stop()

    laps_by_driver = {d: session.laps.pick_drivers(d).pick_fastest() for d in selected_drivers}
    telemetry_by_driver = {d: lap.get_car_data().add_distance() for d, lap in laps_by_driver.items()}

    # Real team colors, like FastF1's own plotting module uses -- but two
    # selected teammates (e.g. LEC/HAM at Ferrari) then share the exact same
    # color, which defeats the point of a head-to-head chart. When that
    # happens, later drivers with a repeated color get a dashed/dotted line
    # instead of a second solid line of the same color.
    driver_style = {}
    seen_colors = {}
    for driver in selected_drivers:
        color = fastf1.plotting.get_driver_color(driver, session)
        dash = LINE_STYLES[seen_colors.get(color, 0)]
        seen_colors[color] = seen_colors.get(color, 0) + 1
        driver_style[driver] = (color, dash)

    cards = st.columns(len(selected_drivers))
    for col, driver in zip(cards, selected_drivers):
        lap = laps_by_driver[driver]
        color, dash = driver_style[driver]
        speed_series = telemetry_by_driver[driver]["Speed"] * (KM_TO_MI if imperial else 1)
        dash_note = f" ({dash} line)" if dash != "solid" else ""
        col.markdown(
            f"""
            <div class="driver-card" style="--card-color:{color}">
                <div class="name">{driver}{dash_note}</div>
                <div class="laptime">{str(lap["LapTime"]).split(" ")[-1][:-3]}</div>
                <div class="sub">Top speed: {speed_series.max():.2f} {speed_unit} · Average: {speed_series.mean():.2f} {speed_unit}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Every driver's telemetry is sampled at its own, slightly different
    # distance points, so hovering used to show each trace's own nearest
    # sample (e.g. ALB at 926m, ANT at 911m) instead of the same point on
    # track. Fixed by resampling every signal for every driver onto one
    # shared distance grid up front, so every trace has a value at exactly
    # the same distance and hovering always compares the same point.
    common_max_distance = min(tel["Distance"].max() for tel in telemetry_by_driver.values())
    common_distance_m = np.linspace(0, common_max_distance, 3000)

    def resample_linear(tel, column):
        return np.interp(common_distance_m, tel["Distance"].to_numpy(), tel[column].to_numpy())

    def resample_step(tel, column):
        # Continuous interpolation would invent fractional gears / brake
        # states between samples; hold the last known value instead.
        x = tel["Distance"].to_numpy()
        y = tel[column].to_numpy()
        idx = np.clip(np.searchsorted(x, common_distance_m, side="right") - 1, 0, len(y) - 1)
        return y[idx]

    resampled = {}
    for driver, tel in telemetry_by_driver.items():
        elapsed = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds()
        elapsed_tel = tel.assign(_Elapsed=elapsed)
        resampled[driver] = {
            "Speed": resample_linear(tel, "Speed"),
            "Throttle": resample_linear(tel, "Throttle"),
            "Brake": resample_step(tel, "Brake").astype(int),
            "nGear": resample_step(tel, "nGear"),
            "Elapsed": resample_linear(elapsed_tel, "_Elapsed"),
        }

    common_distance = common_distance_m * (M_TO_FT if imperial else 1)
    # Pre-formatted "1,346 m" strings used as the x values themselves. Two
    # reasons: (1) Plotly's unified-hover header only accepts a numeric
    # d3-format for the x-axis, with no way to attach a unit to it, so the
    # header showed a bare number; using the label as x makes the header
    # show that exact string, unit included. (2) relying on Plotly's own
    # %{y:+.3f}-style client-side number formatting turned out to be
    # unreliable for the delta chart's single-trace case (it rendered the
    # raw, unrounded float instead), so every value below is pre-formatted
    # in Python with an explicit round and passed as text, never left to
    # the browser to format.
    common_distance_labels = [f"{v:,.0f} {dist_unit}" for v in common_distance]

    dist_title = f"Distance ({dist_unit})"
    fig_speed = base_figure("Speed", speed_unit, dist_title, hovermode="x unified")
    fig_throttle = base_figure("Throttle", "%", dist_title, hovermode="x unified")
    fig_brake = base_figure("Brake", "", dist_title, hovermode="x unified")
    fig_brake.update_yaxes(tickvals=[0, 1], ticktext=["Off", "On"], range=[-0.15, 1.15])
    fig_gear = base_figure("Gear", "", dist_title, hovermode="x unified")
    fig_gear.update_yaxes(tickvals=list(range(1, 9)), range=[0.5, 8.5])
    for fig in (fig_speed, fig_throttle, fig_brake, fig_gear):
        fig.update_xaxes(categoryorder="array", categoryarray=common_distance_labels)

    for driver in selected_drivers:
        color, dash = driver_style[driver]
        speed = resampled[driver]["Speed"] * (KM_TO_MI if imperial else 1)

        fig_speed.add_trace(
            go.Scatter(
                x=common_distance_labels, y=speed, name=driver, line=dict(color=color, dash=dash, width=2.5),
                text=[f"{v:.2f}" for v in speed],
                hovertemplate=f"{driver}: " + "%{text}" + f" {speed_unit}<extra></extra>",
            )
        )
        fig_throttle.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Throttle"], name=driver,
                line=dict(color=color, dash=dash, width=2.5),
                text=[f"{v:.0f}" for v in resampled[driver]["Throttle"]],
                hovertemplate=f"{driver}: " + "%{text}%<extra></extra>",
            )
        )
        brake_state = np.where(resampled[driver]["Brake"], "On", "Off")
        fig_brake.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Brake"], name=driver,
                line=dict(color=color, dash=dash, width=2.5, shape="hv"),
                text=brake_state,
                hovertemplate=f"{driver}: " + "%{text}<extra></extra>",
            )
        )
        fig_gear.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["nGear"], name=driver,
                line=dict(color=color, dash=dash, width=2.5, shape="hv"),
                text=[f"{v:.0f}" for v in resampled[driver]["nGear"]],
                hovertemplate=f"{driver}: gear " + "%{text}<extra></extra>",
            )
        )

    # Delta time: gap to the fastest of the selected laps, over distance --
    # same shared grid, so this is now just a subtraction instead of its own
    # separate interpolation step.
    reference_driver = min(laps_by_driver, key=lambda d: laps_by_driver[d]["LapTime"])
    ref_elapsed = resampled[reference_driver]["Elapsed"]

    fig_delta = base_figure("Delta time", f"s (vs. {reference_driver})", dist_title, hovermode="x unified")
    fig_delta.update_xaxes(categoryorder="array", categoryarray=common_distance_labels)
    fig_delta.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    for driver in selected_drivers:
        if driver == reference_driver:
            continue
        color, dash = driver_style[driver]
        delta = resampled[driver]["Elapsed"] - ref_elapsed
        fig_delta.add_trace(
            go.Scatter(
                x=common_distance_labels, y=delta, name=f"{driver} vs {reference_driver}",
                line=dict(color=color, dash=dash, width=2.5),
                fill="tozeroy", fillcolor=hex_to_rgba(color, 0.15),
                text=[f"{v:+.3f}" for v in delta],
                hovertemplate=f"{driver} vs {reference_driver}: " + "%{text}" + " s<extra></extra>",
            )
        )

    st.plotly_chart(fig_speed, width="stretch")
    st.plotly_chart(fig_delta, width="stretch")
    st.plotly_chart(fig_throttle, width="stretch")
    st.plotly_chart(fig_brake, width="stretch")
    st.plotly_chart(fig_gear, width="stretch")

# ------------------------------------------------------------------ pace --

with tab_pace:
    try:
        pace_laps = session.laps.pick_quicklaps()
        pace_laps = pace_laps[pace_laps["TrackStatus"] == "1"]
    except Exception:
        pace_laps = session.laps.iloc[0:0]

    compounds_with_data = [
        c for c, g in pace_laps.groupby("Compound") if len(g) >= MIN_LAPS_FOR_TREND
    ]

    if not compounds_with_data:
        st.info(
            "Not enough green-flag laps on one compound in this session to fit a degradation "
            "trend (typical for Qualifying, where laps are single push laps rather than a run)."
        )
    else:
        st.caption(
            "Lap time vs. tyre age, corrected for fuel load: a plain fit against tyre age alone "
            "would conflate the tyre wearing in with the car simply getting lighter over the "
            "run, so LapNumber is used as a fuel-burn proxy in a joint fit, and only the "
            "tyre-age part of it is plotted here."
        )

        pace_driver_options = sorted(pace_laps["Driver"].unique())
        pace_drivers = st.multiselect(
            "Show individual laps for", options=pace_driver_options, default=[], key="pace_drivers"
        )

        fig_pace = base_figure("Lap time vs. tyre age", "Lap time (s)", "Tyre life (laps)")

        if pace_drivers:
            for (driver, stint), stint_laps in pace_laps[pace_laps["Driver"].isin(pace_drivers)].groupby(
                ["Driver", "Stint"]
            ):
                if len(stint_laps) < 3:
                    continue
                compound = stint_laps["Compound"].iloc[0]
                color = COMPOUND_COLORS.get(compound, "#999999")
                stint_laps = stint_laps.sort_values("TyreLife")
                lap_times = stint_laps["LapTime"].dt.total_seconds()
                hover_text = [
                    f"{driver}: {t:.3f} s at {tl:.0f} laps" for t, tl in zip(lap_times, stint_laps["TyreLife"])
                ]
                fig_pace.add_trace(
                    go.Scatter(
                        x=stint_laps["TyreLife"],
                        y=lap_times,
                        mode="lines+markers",
                        marker=dict(size=5),
                        line=dict(color=color, width=2),
                        name=f"{driver} ({compound.title()})",
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
        else:
            st.caption("Pick one or more drivers above to see their individual laps under the trend lines.")

        coeffs = {}
        for compound in compounds_with_data:
            compound_laps = pace_laps[pace_laps["Compound"] == compound].dropna(
                subset=["TyreLife", "LapNumber", "LapTime"]
            )
            tyre_life = compound_laps["TyreLife"].to_numpy(dtype=float)
            lap_number = compound_laps["LapNumber"].to_numpy(dtype=float)
            lap_time = compound_laps["LapTime"].dt.total_seconds().to_numpy()
            design = np.column_stack([tyre_life, lap_number, np.ones_like(tyre_life)])
            finite_rows = np.isfinite(design).all(axis=1) & np.isfinite(lap_time)
            design, tyre_life, lap_number, lap_time = (
                design[finite_rows], tyre_life[finite_rows], lap_number[finite_rows], lap_time[finite_rows],
            )
            if len(lap_time) < MIN_LAPS_FOR_TREND:
                continue
            try:
                (tyre_coef, fuel_coef, intercept), *_ = np.linalg.lstsq(design, lap_time, rcond=None)
            except np.linalg.LinAlgError:
                st.warning(f"Couldn't fit a degradation trend for {compound.title()} (bad/degenerate data).")
                continue
            coeffs[compound] = (tyre_coef, fuel_coef)

            mean_lap_number = lap_number.mean()
            x_fit = np.linspace(tyre_life.min(), tyre_life.max(), 2)
            y_fit = tyre_coef * x_fit + fuel_coef * mean_lap_number + intercept
            color = COMPOUND_COLORS.get(compound, "#999999")
            trend_text = [f"{compound.title()}: {t:.3f} s at {tl:.0f} laps" for t, tl in zip(y_fit, x_fit)]
            fig_pace.add_trace(
                go.Scatter(
                    x=x_fit, y=y_fit, mode="lines", line=dict(color=color, width=4),
                    name=f"{compound.title()} ({tyre_coef:+.3f} s/lap)",
                    text=trend_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        st.plotly_chart(fig_pace, width="stretch")

        if not coeffs:
            st.info("No compound had a usable degradation fit in this session.")
            st.stop()

        compound_choice = st.selectbox("Compound (per-driver breakdown)", options=list(coeffs.keys()))
        compound_laps = pace_laps[pace_laps["Compound"] == compound_choice]
        fuel_coef = coeffs[compound_choice][1]

        driver_slopes = {}
        for driver, driver_laps in compound_laps.groupby("Driver"):
            driver_laps = driver_laps.dropna(subset=["TyreLife", "LapNumber", "LapTime"])
            if len(driver_laps) < MIN_LAPS_PER_DRIVER:
                continue
            tyre_life = driver_laps["TyreLife"].to_numpy(dtype=float)
            lap_number = driver_laps["LapNumber"].to_numpy(dtype=float)
            lap_time = driver_laps["LapTime"].dt.total_seconds().to_numpy()
            corrected_time = lap_time - fuel_coef * lap_number
            finite_rows = np.isfinite(tyre_life) & np.isfinite(corrected_time)
            if finite_rows.sum() < MIN_LAPS_PER_DRIVER or np.unique(tyre_life[finite_rows]).size < 2:
                continue
            try:
                tyre_coef, _ = np.polyfit(tyre_life[finite_rows], corrected_time[finite_rows], 1)
            except np.linalg.LinAlgError:
                continue
            driver_slopes[driver] = tyre_coef

        if not driver_slopes:
            st.info(f"No driver ran enough laps on {compound_choice.title()} for a per-driver estimate.")
        else:
            ranked = sorted(driver_slopes.items(), key=lambda kv: kv[1])
            fig_drivers = base_figure(
                f"{compound_choice.title()} — degradation by driver", "Fuel-corrected degradation (s/lap)", ""
            )
            fig_drivers.update_layout(height=max(CHART_HEIGHT, 24 * len(ranked) + 100))
            fig_drivers.add_vline(x=0, line_color="rgba(255,255,255,0.4)")
            fig_drivers.add_trace(
                go.Bar(
                    x=[s for _, s in ranked],
                    y=[d for d, _ in ranked],
                    orientation="h",
                    marker_color=COMPOUND_COLORS.get(compound_choice, "#999999"),
                    text=[f"{d}: {s:+.3f} s/lap" for d, s in ranked],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            fig_drivers.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_drivers, width="stretch")
