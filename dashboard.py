"""
F1 session dashboard: head-to-head telemetry, and race pace / tyre
degradation, for any past race weekend and any of its actual sessions.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import datetime
import re
import urllib.parse

import fastf1
import fastf1.plotting
from fastf1.ergast import Ergast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

fastf1.Cache.enable_cache("cache")

st.set_page_config(page_title="F1 Dashboard", layout="wide", initial_sidebar_state="expanded")

MAX_DRIVERS = 2
CHART_HEIGHT = 340
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

# Every chart was built without a template or font color, so it inherited
# Plotly's default *light-theme* text (near-black) on top of this app's dark
# background -- titles, axis labels and legend text were all barely legible.
# plotly_dark fixes that (light text, matching gridlines/hover boxes) and
# this dict is applied to every figure so none of them drift back out of sync.
DARK_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="'Segoe UI', Inter, -apple-system, sans-serif", color="#d7dbe0", size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)

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
    /* Streamlit's default top-right "running" icon (a boxed bike/runner
       glyph) reads as a stray UI element against this page's own dark
       theme -- hidden rather than restyled, since which icon it is isn't
       under our control, only whether it shows at all. */
    [data-testid="stStatusWidget"] { display: none; }
    /* Same for the spinner Streamlit shows during a slow cached computation
       (e.g. the season-stats crunch) -- its stock icon swapped for a
       slim on-brand ring instead of just hiding the feedback outright,
       since that one's worth keeping (it's the only sign a multi-race
       fetch is actually progressing, not stuck). */
    [data-testid="stSpinner"] svg { display: none; }
    [data-testid="stSpinner"] > div {
        display: flex; align-items: center; gap: 10px;
    }
    [data-testid="stSpinner"] > div::before {
        content: "";
        width: 16px; height: 16px; flex: 0 0 auto;
        border-radius: 50%;
        border: 2.5px solid rgba(91, 140, 255, 0.2);
        border-top-color: #5b8cff;
        animation: f1-spin 0.7s linear infinite;
    }
    @keyframes f1-spin { to { transform: rotate(360deg); } }
    /* st.metric truncates a long value with an ellipsis by default (built
       for short KPI numbers) -- the clinch-round estimate's value includes
       a full Grand Prix name, so this lets it wrap onto a second line
       instead of cutting the name off. */
    [data-testid="stMetricValue"] { white-space: normal; overflow-wrap: break-word; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🏁 F1 Dashboard")


@st.cache_data(ttl=3600)
def load_standings(year, round_number):
    """Championship standings as they stood right after a given round --
    pulled from Ergast (via fastf1.ergast) rather than summed from each
    session's points locally, since Ergast already carries the official,
    round-by-round running totals and doesn't need every prior race of the
    season fetched and parsed just to get today's totals."""
    ergast = Ergast()
    drivers = ergast.get_driver_standings(season=year, round=round_number).content[0]
    constructors = ergast.get_constructor_standings(season=year, round=round_number).content[0]
    return drivers, constructors


@st.cache_data(ttl=3600)
def load_standings_progression(year, up_to_round):
    """Points after each round of the season so far, long-format. Ergast's
    standings endpoint with no round given returns only the latest snapshot
    (season is a running total, not a round-by-round history), so this has
    to ask for one round at a time and stitch the results together."""
    ergast = Ergast()
    driver_rows, constructor_rows = [], []
    for rnd in range(1, up_to_round + 1):
        for _, r in ergast.get_driver_standings(season=year, round=rnd).content[0].iterrows():
            name = r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
            driver_rows.append({"Round": rnd, "Driver": name, "Points": r["points"]})
        for _, r in ergast.get_constructor_standings(season=year, round=rnd).content[0].iterrows():
            constructor_rows.append({"Round": rnd, "Constructor": r["constructorName"], "Points": r["points"]})
    return pd.DataFrame(driver_rows), pd.DataFrame(constructor_rows)


@st.cache_data(ttl=3600)
def total_rounds_in_season(year):
    """Full season length (including rounds not yet run) -- load_schedule
    filters those out for the race picker, but the title-fight odds below
    need to know how many races are actually still left to simulate."""
    schedule = fastf1.get_event_schedule(year)
    return int(schedule[schedule["RoundNumber"] > 0]["RoundNumber"].max())


@st.cache_data(ttl=3600)
def event_name_for_round(year, round_number):
    """Round number -> Grand Prix name, including rounds not yet run (the
    clinch-round estimate below can land on a future race)."""
    schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["RoundNumber"] == round_number]
    return match.iloc[0]["EventName"] if len(match) else f"Round {round_number}"


@st.cache_data(ttl=3600, show_spinner="Crunching stats across every race run so far this season...")
def season_stats(year, event_names):
    """One row per completed race: total overtakes (same rule as the
    per-race chart), retirements, the biggest grid-to-finish recovery, and
    the closest podium fight. Loads every completed race's full Race
    session once (cached after that), so this is the slow one on a cold
    cache."""
    rows = []
    for event_name in event_names:
        try:
            race = fastf1.get_session(year, event_name, "Race")
            race.load()
        except Exception:
            continue
        overtakes_total = sum(count_overtakes(race).values())
        results = race.results
        retirements = int((results["Status"] == "Retired").sum())

        recovery_driver, recovery_places = None, None
        valid = results.dropna(subset=["GridPosition", "Position"])
        if not valid.empty:
            gains = valid["GridPosition"] - valid["Position"]
            best_idx = gains.idxmax()
            if gains.loc[best_idx] > 0:
                recovery_driver, recovery_places = valid.loc[best_idx, "Abbreviation"], int(gains.loc[best_idx])

        closest_gap = None
        runner_up = results[results["Position"] == 2]
        if not runner_up.empty and pd.notna(runner_up.iloc[0]["Time"]):
            closest_gap = runner_up.iloc[0]["Time"].total_seconds()

        rows.append({
            "Event": event_name,
            "Round": int(race.event["RoundNumber"]),
            "Overtakes": overtakes_total,
            "Retirements": retirements,
            "RecoveryDriver": recovery_driver,
            "RecoveryPlaces": recovery_places,
            "ClosestGap": closest_gap,
        })
    return pd.DataFrame(rows)


# 2026's real, announced power unit lineup -- FastF1's results carry the
# team, not the engine, and there's no supplier field to read this from.
ENGINE_SUPPLIERS = {
    "Mercedes": "Mercedes", "McLaren": "Mercedes", "Williams": "Mercedes", "Alpine": "Mercedes",
    "Ferrari": "Ferrari", "Haas F1 Team": "Ferrari", "Cadillac": "Ferrari",
    "Red Bull Racing": "Red Bull Ford", "Racing Bulls": "Red Bull Ford",
    "Aston Martin": "Honda",
    "Audi": "Audi",
}


@st.cache_data(ttl=3600, show_spinner="Checking who took pole in every qualifying session so far...")
def pole_positions(year, event_names):
    """Pole sitter (and their team) for every completed qualifying session
    this season -- results only, no lap/telemetry data needed, so this is
    much lighter than season_stats() despite covering the same races."""
    rows = []
    for event_name in event_names:
        try:
            quali = fastf1.get_session(year, event_name, "Qualifying")
            quali.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception:
            continue
        pole = quali.results.sort_values("Position").iloc[0]
        rows.append({"Event": event_name, "Driver": pole["Abbreviation"], "Team": pole["TeamName"]})
    return pd.DataFrame(rows)


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
    # Descending so the most recently completed round is first -- the
    # Grand Prix selectbox below defaults to whatever's first in the list,
    # and opening on the latest race is more useful than opening on Round 1
    # (Australia) every single time.
    return schedule.sort_values("RoundNumber", ascending=False)


@st.cache_data
def load_session(year, event, session_name):
    session = fastf1.get_session(year, event, session_name)
    session.load()
    return session


def session_names_for(event_row):
    """The sessions actually run that weekend, most recent first -- practice/
    qualifying count and naming (Sprint Qualifying, Sprint, ...) depend on the
    event format, so this is read from the schedule instead of assumed fixed.
    Reversed (like load_schedule's round order) so the session selectbox
    below defaults to the Race instead of Practice 1."""
    names = []
    for i in range(1, 6):
        name = event_row.get(f"Session{i}")
        if isinstance(name, str) and name:
            names.append(name)
    return list(reversed(names))


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def format_lap_time(td):
    if pd.isna(td):
        return "—"
    total = td.total_seconds()
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes}:{seconds:06.3f}" if minutes else f"{seconds:.3f}"


def format_race_gap(row, leader_laps):
    status = row.get("Status")
    if row.get("Position") == 1:
        td = row.get("Time")
        if pd.isna(td):
            return "—"
        total = td.total_seconds()
        h, m, s = int(total // 3600), int((total % 3600) // 60), total % 60
        return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m}:{s:06.3f}"
    if status == "Lapped" and leader_laps is not None and pd.notna(row.get("Laps")):
        laps_down = int(leader_laps - row["Laps"])
        return f"+{laps_down} Lap" + ("s" if laps_down != 1 else "")
    if status == "Finished" and pd.notna(row.get("Time")):
        return f"+{row['Time'].total_seconds():.3f}"
    if pd.notna(row.get("Laps")) and row["Laps"] > 0 and status not in (None, "Finished"):
        return f"{status} ({int(row['Laps'])} laps)"
    return status if pd.notna(status) else "—"


def order_by_classification(session, drivers):
    """Finishing/classification order (P1 first) when available -- falls
    back to alphabetical for sessions with no classification yet, like
    Practice, or for any driver missing from it."""
    try:
        pos = session.results.set_index("Abbreviation")["Position"].dropna()
        ordered = [d for d in pos.sort_values().index if d in drivers]
    except Exception:
        ordered = []
    remaining = sorted(d for d in drivers if d not in ordered)
    return ordered + remaining


def build_driver_styles(drivers, session):
    """Real team colors -- but teammates (e.g. LEC/HAM at Ferrari) would then
    share the exact same color, which defeats a chart that's meant to tell
    drivers apart. A dashed line of the same color turned out too subtle to
    read at a glance (still had to be spelled out in text next to it);
    whoever repeats a color goes plain white instead, which reads instantly."""
    styles = {}
    seen_colors = set()
    for driver in drivers:
        color = fastf1.plotting.get_driver_color(driver, session)
        if color in seen_colors:
            color = "#ffffff"
        else:
            seen_colors.add(color)
        styles[driver] = (color, "solid")
    return styles


def count_overtakes(session):
    """Position gains, lap over lap, net of pit stops (the driver's own and
    any rival's ahead of them), retirements, 3+ place single-lap collapses
    (spin/mechanical, not a queue of individual passes), and any lap race
    control deleted a driver's own laptime for track limits. Shared by the
    per-race overtakes chart and the season-wide stats so the two numbers
    can't drift apart by using two different counting rules."""
    overtake_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if overtake_laps.empty:
        return {}
    overtake_laps = overtake_laps.assign(
        PitFlag=overtake_laps["PitInTime"].notna() | overtake_laps["PitOutTime"].notna()
    )
    track_limit_laps = set()
    try:
        for msg in session.race_control_messages["Message"].astype(str):
            match = re.search(r"CAR \d+ \((\w+)\).*DELETED - TRACK LIMITS AT TURN \d+ LAP (\d+)", msg)
            if match:
                track_limit_laps.add((match.group(1), int(match.group(2))))
    except Exception:
        pass  # race control messages aren't available for every session
    by_lap = {
        lap_num: g.set_index("Driver")[["Position", "PitFlag"]]
        for lap_num, g in overtake_laps.groupby("LapNumber")
    }
    lap_numbers = sorted(by_lap)
    COLLAPSE_THRESHOLD = 3  # places lost in one lap, unpitted, to call it a problem not a pass

    overtake_counts = {d: 0 for d in overtake_laps["Driver"].unique()}
    for prev_lap, lap in zip(lap_numbers, lap_numbers[1:]):
        prev, cur = by_lap[prev_lap], by_lap[lap]
        collapsed = {
            r for r in cur.index
            if r in prev.index and not cur.loc[r, "PitFlag"] and not prev.loc[r, "PitFlag"]
            and cur.loc[r, "Position"] - prev.loc[r, "Position"] >= COLLAPSE_THRESHOLD
        }
        for driver in cur.index:
            if driver not in prev.index:
                continue
            prev_pos, cur_pos = prev.loc[driver, "Position"], cur.loc[driver, "Position"]
            gain = prev_pos - cur_pos
            if gain <= 0:
                continue
            if cur.loc[driver, "PitFlag"] or prev.loc[driver, "PitFlag"]:
                continue  # the driver's own pit lap -- not a real gain either way
            if (driver, int(lap)) in track_limit_laps:
                continue  # race control itself called this lap not clean
            rivals_ahead = prev.index[prev["Position"] < prev_pos]
            free_slots = sum(
                1 for rival in rivals_ahead
                if (rival not in cur.index) or cur.loc[rival, "PitFlag"] or (rival in collapsed)
            )
            overtake_counts[driver] += int(max(0, gain - free_slots))
    return overtake_counts


def base_figure(title, yaxis_title, xaxis_title, hovermode="x"):
    fig = go.Figure()
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=title, font=dict(size=16, color="#f2f4f6")),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=CHART_HEIGHT,
        hovermode=hovermode,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="dot", gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)")
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

tab_telemetry, tab_pace, tab_classification, tab_standings, tab_season_stats = st.tabs(
    ["Head-to-head telemetry", "Race pace & tyre degradation", "Classification", "Standings", "Season stats"]
)

# ------------------------------------------------------------- telemetry --

with tab_telemetry:
    all_drivers = sorted(session.laps["Driver"].unique())
    selected_drivers = st.multiselect(
        "Drivers (max 2)",
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

    # add_distance() zeroes each driver's Distance at their own first
    # telemetry sample -- a point that drifts a little from the true start
    # line depending on where the ~0.2s sample grid happens to land -- and
    # the integrated-speed Distance channel then accumulates further drift
    # over a full lap (two laps of the same physical circuit, run in nearly
    # the same time, were integrating out to totals ~20m apart here). That's
    # enough noise to throw off a fine cross-driver alignment even though
    # the official sector times are precise (they sum to LapTime exactly,
    # checked). This re-anchors each driver's Distance to the shared
    # sector-boundary points implied by those official splits, instead of
    # trusting the raw integrated distance uncorrected for a full lap.
    sector_checkpoints = {}
    try:
        for driver, lap in laps_by_driver.items():
            t1 = lap["Sector1Time"].total_seconds()
            t2 = t1 + lap["Sector2Time"].total_seconds()
            t3 = t2 + lap["Sector3Time"].total_seconds()
            if not np.isfinite([t1, t2, t3]).all():
                raise ValueError("missing sector time")
            sector_checkpoints[driver] = [0.0, t1, t2, t3]

        # Keep only samples genuinely within this lap (0 to LapTime) -- a
        # trailing sample or two from just past the finish line otherwise
        # falls outside the checkpoint table built below and gets clamped
        # to its last entry instead of properly interpolated, which
        # collapsed several end-of-lap points onto the same corrected
        # distance (the flat, wrong tail this chart used to show).
        for driver, tel in list(telemetry_by_driver.items()):
            elapsed = tel["Time"].dt.total_seconds()
            telemetry_by_driver[driver] = tel[
                (elapsed >= 0) & (elapsed <= sector_checkpoints[driver][-1])
            ].copy()

        own_distance_at_checkpoint = {
            driver: [
                np.interp(t, tel["Time"].dt.total_seconds().to_numpy(), tel["Distance"].to_numpy())
                for t in sector_checkpoints[driver]
            ]
            for driver, tel in telemetry_by_driver.items()
        }
        shared_checkpoint_distance = [
            float(np.mean([own_distance_at_checkpoint[d][i] for d in telemetry_by_driver]))
            for i in range(4)
        ]
        for driver, tel in telemetry_by_driver.items():
            tel["Distance"] = np.interp(
                tel["Distance"].to_numpy(), own_distance_at_checkpoint[driver], shared_checkpoint_distance,
            )
    except (KeyError, ValueError, TypeError):
        # A lap missing a sector time (rare -- red flag, first lap of a
        # session, etc.) leaves the raw, uncorrected distance in place
        # rather than breaking the whole tab over one driver's edge case.
        sector_checkpoints = {}

    driver_style = build_driver_styles(selected_drivers, session)

    # st.multiselect only themes its tags with one fixed accent color and
    # exposes no per-tag styling parameter. Confirmed from actual DevTools
    # inspection: the tag isn't BaseWeb's data-baseweb="tag" at all -- it's
    # a plain `<span title="BOR" class="st-emotion-cache-...">BOR</span>`,
    # and the title attribute carries the exact driver code. Turned out that
    # span is only a small inner element, not the visible pill itself (the
    # first version of this only recolored a little square, leaving the red
    # pill it sits inside untouched) -- the pill is its *parent*, reached
    # here with :has() since plain CSS has no other way to select upward.
    tag_rules = "\n".join(
        f'span:has(> span[title="{d}"]) {{ '
        f"background-color: {driver_style[d][0]} !important; "
        f'color: {"#111111" if driver_style[d][0].lower() == "#ffffff" else "#ffffff"} !important; }}\n'
        f'span[title="{d}"] {{ background-color: transparent !important; }}'
        for d in selected_drivers
    )
    st.markdown(f"<style>{tag_rules}</style>", unsafe_allow_html=True)

    cards = st.columns(len(selected_drivers))
    for col, driver in zip(cards, selected_drivers):
        lap = laps_by_driver[driver]
        color, _dash = driver_style[driver]
        speed_series = telemetry_by_driver[driver]["Speed"] * (KM_TO_MI if imperial else 1)
        col.markdown(
            f"""
            <div class="driver-card" style="--card-color:{color}">
                <div class="name">{driver}</div>
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
        # tel["Time"] is already relative to this lap's true start (verified
        # against LapStartTime + Time == SessionTime) -- re-zeroing it to the
        # first telemetry sample, as this used to do, threw that away and
        # replaced it with an arbitrary per-driver offset (the sample grid is
        # ~0.2s coarse and isn't phase-locked to the start line, so that
        # offset differs driver to driver). That was why the delta-time chart
        # didn't converge to the real gap at the end of the lap: it was
        # comparing each driver's *own* start-of-telemetry instant, not the
        # same physical point on track.
        raw_distance = tel["Distance"].to_numpy()
        raw_elapsed = tel["Time"].dt.total_seconds().to_numpy()
        # Even with Distance re-anchored to the shared sector checkpoints
        # above, interpolating raw telemetry between them still drifts --
        # only the two *ends* of that interpolation were pinned to a known-
        # correct value (elapsed 0 at the start line, this driver's own
        # LapTime at the finish), so the middle of the lap was still free to
        # wander off the real sector splits. Inserting the S1/S2 boundary
        # checkpoints too (not just start/finish) forces the curve through
        # all four officially-known points, so only the shape *within* a
        # sector is still telemetry-only -- the sector-to-sector trend is
        # guaranteed to match the real, precise timing splits.
        if sector_checkpoints:
            # A checkpoint distance can land exactly on an existing raw
            # sample's (corrected) distance -- e.g. the very first sample,
            # since that's what the start checkpoint was derived from -- and
            # np.interp's behavior at a duplicate x is undefined. Merging
            # through a dict (checkpoints inserted last, so they win a tie)
            # guarantees a strictly-ordered, duplicate-free table instead.
            table = dict(zip(raw_distance.tolist(), raw_elapsed.tolist()))
            table.update(zip((float(x) for x in shared_checkpoint_distance), sector_checkpoints[driver]))
            points = sorted(table.items())
            raw_distance = np.array([p[0] for p in points])
            raw_elapsed = np.array([p[1] for p in points])
        elapsed_resampled = np.interp(common_distance_m, raw_distance, raw_elapsed)
        resampled[driver] = {
            "Speed": resample_linear(tel, "Speed"),
            "Throttle": resample_linear(tel, "Throttle"),
            "Brake": resample_step(tel, "Brake").astype(int),
            "nGear": resample_step(tel, "nGear"),
            "Elapsed": elapsed_resampled,
        }

    st.caption(
        "Track dominance: color shows which of two drivers is faster at each point on the lap "
        "(a real corner-by-corner read on downforce/drag tradeoffs, not just overall pace)."
    )
    if len(selected_drivers) < 2:
        st.info("Select at least 2 drivers to see the dominance map.")
    else:
        dom_a, dom_b = selected_drivers[0], selected_drivers[1]

        # Reuses the already-resampled, shared-grid speeds from above for the
        # comparison; only X/Y position needs its own resampling here, onto
        # that same grid, using the same resample_linear() helper.
        tel_a = laps_by_driver[dom_a].get_telemetry()
        # get_telemetry() computes its own Distance independently of the
        # car_data-based one corrected above -- same sector-checkpoint
        # re-anchoring, using this channel's own values so it stays
        # self-consistent rather than borrowing the other channel's numbers.
        # Falls back to the raw, uncorrected distance if that correction
        # wasn't available for this driver (see the try/except above).
        try:
            tel_a_checkpoints = [
                np.interp(t, tel_a["Time"].dt.total_seconds().to_numpy(), tel_a["Distance"].to_numpy())
                for t in sector_checkpoints[dom_a]
            ]
            tel_a = tel_a.copy()
            tel_a["Distance"] = np.interp(tel_a["Distance"].to_numpy(), tel_a_checkpoints, shared_checkpoint_distance)
        except (KeyError, NameError):
            pass
        x_on_grid = resample_linear(tel_a, "X")
        y_on_grid = resample_linear(tel_a, "Y")

        speed_a = resampled[dom_a]["Speed"]
        speed_b = resampled[dom_b]["Speed"]
        speed_a_disp = speed_a * (KM_TO_MI if imperial else 1)
        speed_b_disp = speed_b * (KM_TO_MI if imperial else 1)
        color_a, _ = driver_style[dom_a]
        color_b, _ = driver_style[dom_b]

        faster = np.where(speed_a >= speed_b, 0, 1)
        dom_text = [
            f"{dom_a}: {sa:.2f} {speed_unit} · {dom_b}: {sb:.2f} {speed_unit}"
            for sa, sb in zip(speed_a_disp, speed_b_disp)
        ]

        fig_dom = go.Figure()
        fig_dom.add_trace(
            go.Scatter(
                x=x_on_grid, y=y_on_grid, mode="markers",
                marker=dict(
                    size=5, color=faster, cmin=0, cmax=1,
                    colorscale=[[0, color_a], [0.5, color_a], [0.5, color_b], [1, color_b]],
                ),
                text=dom_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
            )
        )
        fig_dom.update_xaxes(visible=False)
        fig_dom.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        fig_dom.update_layout(**DARK_LAYOUT, height=CHART_HEIGHT + 80, margin=dict(t=20, b=10))
        st.plotly_chart(fig_dom, width="stretch")
        st.markdown(
            f'<span style="color:{color_a}">●</span> {dom_a} faster'
            f'&nbsp;&nbsp;&nbsp;<span style="color:{color_b}">●</span> {dom_b} faster',
            unsafe_allow_html=True,
        )

    st.divider()
    # The legend just above this next chart (driver codes, plus "X vs Y" for
    # the delta panel) belongs to it, not to the track dominance map above --
    # without this divider the two ran together with nothing to tell them
    # apart, and that legend read as a stray, unexplained label on the map.
    st.caption("Speed, delta, throttle, brake and gear over the lap -- drag any panel to zoom, all five stay in sync.")

    common_distance = common_distance_m * (M_TO_FT if imperial else 1)
    # Pre-formatted "1,346 m" strings as the x values themselves, so the
    # unified-hover header shows the unit (a plain numeric axis can't carry
    # one). This was tried before and reverted because a category axis
    # defaults to showing far more ticks than a numeric one -- but that's a
    # tick-count problem, not a reason to give up the unit: fixed here by
    # setting tickvals explicitly to a handful of evenly spaced points
    # instead of leaving tick selection to Plotly's category-axis default.
    common_distance_labels = [f"{v:,.0f} {dist_unit}" for v in common_distance]
    tick_idx = np.linspace(0, len(common_distance_labels) - 1, 9).astype(int)
    tick_vals = [common_distance_labels[i] for i in tick_idx]

    dist_title = f"Distance ({dist_unit})"
    # Delta time: gap to the fastest of the selected laps, over distance --
    # same shared grid, so this is just a subtraction, no separate interpolation.
    reference_driver = min(laps_by_driver, key=lambda d: laps_by_driver[d]["LapTime"])
    ref_elapsed = resampled[reference_driver]["Elapsed"]

    # One figure with 5 stacked, x-linked subplots instead of 5 independent
    # charts: dragging a zoom box on any panel zooms all of them together
    # (shared_xaxes wires their x-axes to Plotly's own "matches" mechanism),
    # which is what actually makes "zoom into this corner" useful across
    # speed/throttle/brake/gear/delta at once instead of one panel at a time.
    fig_telemetry = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[1.2, 1.0, 0.8, 0.45, 0.55],
        subplot_titles=("Speed", f"Delta time (vs. {reference_driver})", "Throttle", "Brake", "Gear"),
    )
    fig_telemetry.update_layout(
        **DARK_LAYOUT,
        height=1550, hovermode="x unified", margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    fig_telemetry.update_annotations(font=dict(size=14, color="#f2f4f6"))
    fig_telemetry.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1, spikedash="dot",
        categoryorder="array", categoryarray=common_distance_labels,
        tickmode="array", tickvals=tick_vals, gridcolor="rgba(255,255,255,0.06)",
    )
    fig_telemetry.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    fig_telemetry.update_xaxes(title_text=dist_title, row=5, col=1)
    fig_telemetry.update_yaxes(title_text=speed_unit, row=1, col=1)
    fig_telemetry.update_yaxes(title_text="s", row=2, col=1)
    fig_telemetry.update_yaxes(title_text="%", row=3, col=1)
    fig_telemetry.update_yaxes(tickvals=[0, 1], ticktext=["Off", "On"], range=[-0.15, 1.15], row=4, col=1)
    fig_telemetry.update_yaxes(tickvals=list(range(1, 9)), range=[0.5, 8.5], row=5, col=1)
    fig_telemetry.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)
    if sector_checkpoints:
        for i, sector_end in enumerate(shared_checkpoint_distance[1:3], start=1):
            idx = int(np.argmin(np.abs(common_distance_m - sector_end)))
            fig_telemetry.add_vline(
                x=common_distance_labels[idx], line_dash="dot",
                line_color="rgba(255,255,255,0.35)", line_width=1.5,
                row="all", col=1,
            )
        fig_telemetry.add_annotation(
            text="S1 · S2 · S3 (dotted lines mark the boundaries)", showarrow=False,
            xref="paper", yref="paper", x=0, y=1.06, xanchor="left",
            font=dict(size=11, color="rgba(255,255,255,0.55)"),
        )

    for driver in selected_drivers:
        color, dash = driver_style[driver]
        speed = resampled[driver]["Speed"] * (KM_TO_MI if imperial else 1)

        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=speed, name=driver, line=dict(color=color, dash=dash, width=2.5),
                text=[f"{v:.2f}" for v in speed],
                hovertemplate=f"{driver}: " + "%{text}" + f" {speed_unit}<extra></extra>",
            ),
            row=1, col=1,
        )
        if driver != reference_driver:
            delta = resampled[driver]["Elapsed"] - ref_elapsed
            # Linear interpolation between ~200ms-apart telemetry samples
            # reads as jagged sawtooth noise once plotted at 3000 points, so
            # this is smoothed for display -- but smoothing the raw delta
            # directly pulls the checkpoint-anchored ends away from their
            # verified-correct values. Smoothing the *residual* from a
            # straight-line baseline through the four known-correct
            # checkpoints instead keeps the ends already near-exact by
            # construction. Fading the residual itself to zero over the
            # last/first ~100m (rather than blending the *display value*
            # toward a hard pin, tried first) lets it settle onto the
            # baseline smoothly -- blending the display value briefly
            # pulled it *through* zero on the way, an unreal dip-then-jump
            # right at the line that a hard pin doesn't show either.
            if sector_checkpoints:
                checkpoint_deltas = [
                    sector_checkpoints[driver][i] - sector_checkpoints[reference_driver][i] for i in range(4)
                ]
                baseline = np.interp(common_distance_m, shared_checkpoint_distance, checkpoint_deltas)
                residual = pd.Series(delta - baseline).rolling(window=31, center=True, min_periods=1).mean().to_numpy()
                n = min(60, len(residual) // 4)
                fade_in = np.linspace(0, 1, n)
                residual[:n] *= fade_in
                residual[-n:] *= fade_in[::-1]
                delta_display = baseline + residual
            else:
                delta_display = pd.Series(delta).rolling(window=31, center=True, min_periods=1).mean().to_numpy()
            fig_telemetry.add_trace(
                go.Scatter(
                    x=common_distance_labels, y=delta_display, name=f"{driver} vs {reference_driver}",
                    line=dict(color=color, dash=dash, width=2.5),
                    fill="tozeroy", fillcolor=hex_to_rgba(color, 0.15),
                    # A signed +/- number makes the reader do the "which one
                    # is that relative to" math themselves; naming whoever's
                    # actually ahead at that point removes the step (and
                    # "behind" never appears -- the other driver's row just
                    # says "ahead" instead, from their own side of it).
                    text=[
                        f"{reference_driver} ahead by {abs(v):.3f}" if v >= 0
                        else f"{driver} ahead by {abs(v):.3f}"
                        for v in delta_display
                    ],
                    hovertemplate="%{text} s<extra></extra>",
                ),
                row=2, col=1,
            )
        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Throttle"], name=driver, showlegend=False,
                line=dict(color=color, dash=dash, width=2.5),
                text=[f"{v:.0f}" for v in resampled[driver]["Throttle"]],
                hovertemplate=f"{driver}: " + "%{text}%<extra></extra>",
            ),
            row=3, col=1,
        )
        brake_state = np.where(resampled[driver]["Brake"], "On", "Off")
        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Brake"], name=driver, showlegend=False,
                line=dict(color=color, dash=dash, width=2.5, shape="hv"),
                text=brake_state,
                hovertemplate=f"{driver}: " + "%{text}<extra></extra>",
            ),
            row=4, col=1,
        )
        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["nGear"], name=driver, showlegend=False,
                line=dict(color=color, dash=dash, width=2.5, shape="hv"),
                text=[f"{v:.0f}" for v in resampled[driver]["nGear"]],
                hovertemplate=f"{driver}: gear " + "%{text}<extra></extra>",
            ),
            row=5, col=1,
        )

    st.plotly_chart(fig_telemetry, width="stretch")

# ------------------------------------------------------------------ pace --

def render_pace_tab():
    position_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if position_laps.empty:
        st.info("No lap-by-lap position data available for this session.")
    else:
        st.caption("Position at the end of each lap, for the whole field.")
        field_order = order_by_classification(session, position_laps["Driver"].unique())
        field_style = build_driver_styles(field_order, session)
        max_lap = position_laps["LapNumber"].max()

        fig_position = base_figure("Race position by lap", "Position", "Lap")
        fig_position.update_yaxes(autorange="reversed", dtick=1)
        fig_position.update_xaxes(range=[position_laps["LapNumber"].min() - 1, max_lap + 3])
        fig_position.update_layout(height=max(CHART_HEIGHT, 22 * len(field_order)), showlegend=False)

        for driver in field_order:
            driver_laps = position_laps[position_laps["Driver"] == driver].sort_values("LapNumber")
            color, dash = field_style[driver]
            # Lap number isn't repeated here -- it's already the x-axis value,
            # shown once at the bottom, not per driver.
            hover_text = [f"{driver}: P{p:.0f}" for p in driver_laps["Position"]]
            fig_position.add_trace(
                go.Scatter(
                    x=driver_laps["LapNumber"], y=driver_laps["Position"], mode="lines",
                    line=dict(color=color, dash=dash, width=2), text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            last = driver_laps.iloc[-1]
            fig_position.add_annotation(
                x=last["LapNumber"], y=last["Position"], text=driver, showarrow=False,
                xanchor="left", xshift=6, font=dict(size=10, color=color),
            )

        st.plotly_chart(fig_position, width="stretch")

    st.divider()

    strategy_laps = session.laps.dropna(subset=["Stint", "Compound", "LapNumber"])
    if strategy_laps.empty:
        st.info("No stint data available for a strategy timeline in this session.")
    else:
        st.caption("Tyre strategy: which compound each driver ran, and for how long.")
        strategy_order = order_by_classification(session, strategy_laps["Driver"].unique())

        fig_strategy = base_figure("Tyre strategy", "", "Lap")
        fig_strategy.update_layout(
            height=max(CHART_HEIGHT, 22 * len(strategy_order)), barmode="stack", showlegend=False,
        )
        fig_strategy.update_yaxes(categoryorder="array", categoryarray=list(reversed(strategy_order)))

        shown_compounds = set()
        for driver, driver_laps in strategy_laps.groupby("Driver"):
            for stint, stint_laps in driver_laps.groupby("Stint"):
                compound = stint_laps["Compound"].iloc[0]
                start = stint_laps["LapNumber"].min()
                length = stint_laps["LapNumber"].max() - start + 1
                color = COMPOUND_COLORS.get(compound, "#999999")
                fig_strategy.add_trace(
                    go.Bar(
                        x=[length], y=[driver], base=[start - 1], orientation="h",
                        marker=dict(color=color, line=dict(color="rgba(0,0,0,0.4)", width=1)),
                        name=compound.title(), legendgroup=compound, showlegend=compound not in shown_compounds,
                        text=[f"{driver}: {compound.title()}, laps {start:.0f}-{start + length - 1:.0f}"],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
                shown_compounds.add(compound)

        fig_strategy.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_strategy, width="stretch")

    st.divider()

    overtake_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if overtake_laps.empty:
        st.info("No lap-by-lap position data available to count overtakes in this session.")
    else:
        st.caption(
            "On-track position gains, lap over lap, with a gain credited only for what's left "
            "after excluding four sources of *free* positions: the driver's own pit in/out lap "
            "or a lap where their own time got deleted for track limits (race control's own call "
            "that the lap wasn't clean); however many rivals ahead of them pitted (in or out) "
            "that same lap; and however many rivals ahead of them lost 3+ places themselves that "
            "lap without pitting -- a swing that big in one lap is a spin, contact or mechanical "
            "issue, not a queue of cars each individually passing them. A rival losing 1-2 places "
            "still counts as fair game, since that's within range of a normal on-track pass -- so "
            "this can still include a position gained off a smaller, undetectable mistake, or a "
            "steward's call this data has no way to see. Treat this as an approximation, not an "
            "official count."
        )
        overtake_counts = count_overtakes(session)

        ranked_overtakes = sorted(overtake_counts.items(), key=lambda kv: kv[1], reverse=True)
        ranked_overtakes = [(d, c) for d, c in ranked_overtakes if c > 0]
        if not ranked_overtakes:
            st.info("No on-track position gains detected in this session.")
        else:
            overtake_styles = build_driver_styles([d for d, _ in ranked_overtakes], session)
            fig_overtakes = base_figure("Overtakes by driver", "Position gains", "")
            fig_overtakes.update_layout(height=max(CHART_HEIGHT, 24 * len(ranked_overtakes)), showlegend=False)
            fig_overtakes.add_trace(
                go.Bar(
                    x=[c for _, c in ranked_overtakes],
                    y=[d for d, _ in ranked_overtakes],
                    orientation="h",
                    marker_color=[overtake_styles[d][0] for d, _ in ranked_overtakes],
                    text=[f"{d}: {c}" for d, c in ranked_overtakes],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            fig_overtakes.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_overtakes, width="stretch")

    st.divider()

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
        pace_drivers = st.multiselect("Show individual laps for", options=pace_driver_options, default=[])

        fig_pace = base_figure("Lap time vs. tyre age", "Lap time (s)", "Tyre life (laps)")

        if pace_drivers:
            shown_stints = pace_laps[pace_laps["Driver"].isin(pace_drivers)].groupby(["Driver", "Stint"])
            # A driver can run the same compound in two separate stints (e.g.
            # medium-hard-medium) -- both would otherwise get the same name
            # and color and read as one noisy zigzagging line instead of two
            # clean ones, so count occurrences per driver+compound and number
            # them once there's more than one.
            stint_counts = {}
            for (driver, stint), stint_laps in shown_stints:
                if len(stint_laps) < 3:
                    continue
                compound = stint_laps["Compound"].iloc[0]
                stint_counts[(driver, compound)] = stint_counts.get((driver, compound), 0) + 1

            seen = {}
            for (driver, stint), stint_laps in shown_stints:
                if len(stint_laps) < 3:
                    continue
                compound = stint_laps["Compound"].iloc[0]
                color = COMPOUND_COLORS.get(compound, "#999999")
                stint_laps = stint_laps.sort_values("TyreLife")
                lap_times = stint_laps["LapTime"].dt.total_seconds()
                hover_text = [
                    f"{driver}: {t:.3f} s at {tl:.0f} laps" for t, tl in zip(lap_times, stint_laps["TyreLife"])
                ]
                label = f"{driver} ({compound.title()})"
                if stint_counts[(driver, compound)] > 1:
                    seen[(driver, compound)] = seen.get((driver, compound), 0) + 1
                    label += f" #{seen[(driver, compound)]}"
                fig_pace.add_trace(
                    go.Scatter(
                        x=stint_laps["TyreLife"],
                        y=lap_times,
                        mode="lines+markers",
                        marker=dict(size=5),
                        line=dict(color=color, width=2.5),
                        name=label,
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
            # Fit on the whole field, not just whichever drivers are shown
            # above, so it's often a wider tyre-life range than what's
            # plotted for one driver -- dashed and thinner so it reads as
            # background reference rather than competing with the raw laps.
            fig_pace.add_trace(
                go.Scatter(
                    x=x_fit, y=y_fit, mode="lines", line=dict(color=color, width=2, dash="dot"),
                    name=f"{compound.title()} ({tyre_coef:+.3f} s/lap)",
                    text=trend_text,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig_pace.update_layout(legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
        st.plotly_chart(fig_pace, width="stretch")

        if not coeffs:
            st.info("No compound had a usable degradation fit in this session.")
            return

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


with tab_pace:
    # Every chart here -- position-by-lap, tyre strategy, overtakes,
    # degradation -- is built around a real stint across a race distance.
    # A qualifying lap is a single push lap with nothing to show in any of
    # them, so instead of a wall of "no stint data" / "no strategy data" /
    # etc. info boxes, this just says so once and points at the tab that
    # actually fits a qualifying lap.
    if "Qualifying" in session_name:
        st.caption(
            f"{session_name} is single push laps, not race stints, so there's no pace or tyre "
            "degradation to show here -- for lap comparisons within this session, use the "
            "Head-to-head telemetry tab instead. What does fit a qualifying session is the "
            "season's pole position picture, below."
        )
        poles = pole_positions(year, tuple(schedule["EventName"].tolist()))
        if poles.empty:
            st.info("No completed qualifying sessions to count poles from yet.")
        else:
            def safe_driver_color(code):
                try:
                    return fastf1.plotting.get_driver_color(code, session)
                except Exception:
                    return "#5b8cff"

            col_poles_driver, col_poles_engine = st.columns(2)

            with col_poles_driver:
                by_driver = poles["Driver"].value_counts().sort_values()
                fig_poles_driver = base_figure("Pole positions by driver", "", "Poles")
                fig_poles_driver.update_layout(showlegend=False, height=max(CHART_HEIGHT, 24 * len(by_driver)))
                fig_poles_driver.add_trace(
                    go.Bar(
                        x=by_driver.to_numpy(), y=by_driver.index, orientation="h",
                        marker_color=[safe_driver_color(d) for d in by_driver.index],
                        text=by_driver.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                    )
                )
                st.plotly_chart(fig_poles_driver, width="stretch")

            with col_poles_engine:
                engines = poles["Team"].map(ENGINE_SUPPLIERS).fillna(poles["Team"])
                by_engine = engines.value_counts().sort_values()
                fig_poles_engine = base_figure("Pole positions by engine manufacturer", "", "Poles")
                fig_poles_engine.update_layout(showlegend=False, height=max(CHART_HEIGHT, 24 * len(by_engine)))
                fig_poles_engine.add_trace(
                    go.Bar(
                        x=by_engine.to_numpy(), y=by_engine.index, orientation="h",
                        marker_color="#5b8cff",
                        text=by_engine.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                    )
                )
                st.plotly_chart(fig_poles_engine, width="stretch")
    else:
        render_pace_tab()

# --------------------------------------------------------- classification --

with tab_classification:
    results = session.results

    if results.empty or results["Position"].isna().all():
        # Practice has no official classification -- rank by fastest lap
        # instead, the closest equivalent to what timing screens show live.
        st.caption(
            f"{session_name} has no official classification -- ranked by each driver's "
            "fastest lap instead."
        )
        fastest = session.laps.groupby("Driver")["LapTime"].min().dropna().sort_values()
        team_by_driver = results.set_index("Abbreviation")["TeamName"] if not results.empty else {}
        rows = [
            {
                "Pos": rank,
                "Driver": driver,
                "Team": team_by_driver.get(driver, ""),
                "Best lap": format_lap_time(lap_time),
                "Gap": "—" if rank == 1 else f"+{(lap_time - fastest.iloc[0]).total_seconds():.3f}",
            }
            for rank, (driver, lap_time) in enumerate(fastest.items(), start=1)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        has_quali_times = results[["Q1", "Q2", "Q3"]].notna().any().any()
        classified = results.sort_values("Position")
        leader_row = classified[classified["Position"] == 1]
        leader_laps = leader_row["Laps"].iloc[0] if len(leader_row) and pd.notna(leader_row["Laps"].iloc[0]) else None

        rows = []
        for _, r in classified.iterrows():
            row = {
                "Pos": int(r["Position"]) if pd.notna(r["Position"]) else "—",
                "Driver": r["Abbreviation"],
                "Team": r["TeamName"],
            }
            if has_quali_times:
                row["Q1"] = format_lap_time(r["Q1"])
                row["Q2"] = format_lap_time(r["Q2"])
                row["Q3"] = format_lap_time(r["Q3"])
            else:
                row["Grid"] = int(r["GridPosition"]) if pd.notna(r["GridPosition"]) else "—"
                row["Gap"] = format_race_gap(r, leader_laps)
                row["Pts"] = r["Points"] if pd.notna(r["Points"]) else 0
                row["Status"] = r["Status"] if pd.notna(r["Status"]) else "—"
            rows.append(row)

        st.dataframe(rows, hide_index=True, width="stretch")

# ------------------------------------------------------------- standings --

with tab_standings:
    round_number = int(event_row["RoundNumber"])
    st.caption(f"Championship standings as they stood right after Round {round_number} ({event_name}).")

    try:
        driver_standings, constructor_standings = load_standings(year, round_number)
        driver_progress, constructor_progress = load_standings_progression(year, round_number)
    except Exception as exc:
        st.error(f"Couldn't load standings for this round: {exc}")
    else:
        def pos_label(pos):
            # Ergast can list a driver with no numeric position at all (seen
            # for a scoreless entry after round 1) rather than just a high
            # number, so this can't assume int() always succeeds.
            if pd.isna(pos):
                return "—"
            pos = int(pos)
            return {1: "🥇 1", 2: "🥈 2", 3: "🥉 3"}.get(pos, str(pos))

        def team_color(name):
            try:
                return fastf1.plotting.get_team_color(name, session)
            except Exception:
                return "#999999"

        def team_badge(name):
            # No official team logos ship with FastF1 (or anywhere else
            # local to this app), and pulling real ones in from the web
            # means trusting an outside source to have the right, current
            # marks -- this draws a small badge instead: team color, three
            # letters, generated on the fly as an inline SVG data URI, so
            # there's no external file and no license/accuracy risk.
            color = team_color(name)
            text_color = "#111111" if color.lower() == "#ffffff" else "#ffffff"
            initials = name[:3].upper()
            svg = (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
                f'<circle cx="20" cy="20" r="18" fill="{color}" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>'
                f'<text x="20" y="25" font-family="Arial, sans-serif" font-size="13" font-weight="700" '
                f'fill="{text_color}" text-anchor="middle">{initials}</text></svg>'
            )
            return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

        def standings_table(rows, points_col="Points", team_col=None):
            # Row 0 is the leader (rows come pre-sorted by position) -- a
            # highlighted background makes that visible at a glance instead
            # of making the reader scan the Pos column for "1".
            df = pd.DataFrame(rows)
            column_config = {
                points_col: st.column_config.ProgressColumn(
                    points_col, format="%.0f", min_value=0, max_value=float(df[points_col].max()) or 1.0,
                ),
            }
            if team_col:
                df.insert(0, "", df[team_col].map(team_badge))
                column_config[""] = st.column_config.ImageColumn("", width="small")
            styled = df.style.apply(
                lambda row: ["background-color: rgba(91,140,255,0.16)" if row.name == 0 else "" for _ in row],
                axis=1,
            )
            st.dataframe(styled, hide_index=True, width="stretch", column_config=column_config)

        def driver_color(code):
            # Unlike the telemetry tab, this covers the whole season, so it
            # can include a driver who isn't in *this* session's entry list
            # (a mid-season replacement, a one-off stand-in) -- get_driver_color
            # raises for those instead of falling back, so every lookup here
            # needs its own guard rather than reusing build_driver_styles.
            try:
                return fastf1.plotting.get_driver_color(code, session)
            except Exception:
                return "#999999"

        st.subheader("Drivers' Championship")

        ordered_drivers = driver_standings.sort_values("position")
        if len(ordered_drivers) >= 2:
            leader_row, second_row = ordered_drivers.iloc[0], ordered_drivers.iloc[1]
            leader_name = leader_row["driverCode"] if pd.notna(leader_row["driverCode"]) else f"{leader_row['givenName']} {leader_row['familyName']}"
            second_name = second_row["driverCode"] if pd.notna(second_row["driverCode"]) else f"{second_row['givenName']} {second_row['familyName']}"
            st.metric(
                "Championship lead",
                f"{leader_name} by {leader_row['points'] - second_row['points']:.0f} pts",
                f"over {second_name} ({second_row['points']:.0f} pts)",
                delta_color="off",
            )

        if round_number > 1:
            # "x unified" hover and a top-right legend both fall apart with a
            # full 20+-driver field -- the unified box turned into a single
            # tooltip taller than the chart itself, and the legend wrapped
            # into rows that collided with the title. Same fix the race
            # position chart already uses for the same problem: no legend,
            # closest-point hover, driver code labeled at the end of its own line.
            fig_driver_progress = base_figure("Points progression", "Points", "Round", hovermode="closest")
            fig_driver_progress.update_layout(height=420, showlegend=False, margin=dict(t=40, b=40, r=50))
            driver_order = [
                r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
                for _, r in driver_standings.sort_values("position").iterrows()
            ]
            # Only the top 10 get an end-of-line label -- past that, drivers
            # bunch up near zero points and their labels just pile on top of
            # each other; the line and its hover are still there for anyone
            # further back, they just aren't individually labeled.
            for rank, driver in enumerate(driver_order):
                driver_pts = driver_progress[driver_progress["Driver"] == driver].sort_values("Round")
                color = driver_color(driver)
                fig_driver_progress.add_trace(
                    go.Scatter(
                        x=driver_pts["Round"], y=driver_pts["Points"], mode="lines",
                        line=dict(color=color, width=2),
                        hovertemplate=f"{driver}: " + "%{y:.0f} pts (round %{x})<extra></extra>",
                    )
                )
                if rank >= 10:
                    continue
                last = driver_pts.iloc[-1]
                fig_driver_progress.add_annotation(
                    x=last["Round"], y=last["Points"], text=driver, showarrow=False,
                    xanchor="left", xshift=6, font=dict(size=9, color=color),
                )
            fig_driver_progress.update_xaxes(dtick=1)
            st.plotly_chart(fig_driver_progress, width="stretch")

        driver_rows = [
            {
                "Pos": pos_label(r["position"]),
                "Driver": r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}",
                "Team": r["constructorNames"][0] if len(r["constructorNames"]) else "—",
                "Points": r["points"],
                "Wins": int(r["wins"]),
            }
            for _, r in driver_standings.iterrows()
        ]
        standings_table(driver_rows, team_col="Team")

        st.divider()

        st.subheader("Constructors' Championship")

        ordered_constructors = constructor_standings.sort_values("position")
        if len(ordered_constructors) >= 2:
            leader_row_c, second_row_c = ordered_constructors.iloc[0], ordered_constructors.iloc[1]
            st.metric(
                "Championship lead",
                f"{leader_row_c['constructorName']} by {leader_row_c['points'] - second_row_c['points']:.0f} pts",
                f"over {second_row_c['constructorName']} ({second_row_c['points']:.0f} pts)",
                delta_color="off",
            )

        if round_number > 1:
            fig_constructor_progress = base_figure("Points progression", "Points", "Round", hovermode="closest")
            fig_constructor_progress.update_layout(height=380, showlegend=False, margin=dict(t=40, b=40, r=90))
            constructor_order = constructor_standings.sort_values("position")["constructorName"].tolist()
            for constructor in constructor_order:
                team_pts = constructor_progress[constructor_progress["Constructor"] == constructor].sort_values("Round")
                color = team_color(constructor)
                fig_constructor_progress.add_trace(
                    go.Scatter(
                        x=team_pts["Round"], y=team_pts["Points"], mode="lines",
                        line=dict(color=color, width=2),
                        hovertemplate=f"{constructor}: " + "%{y:.0f} pts (round %{x})<extra></extra>",
                    )
                )
                last = team_pts.iloc[-1]
                fig_constructor_progress.add_annotation(
                    x=last["Round"], y=last["Points"], text=constructor, showarrow=False,
                    xanchor="left", xshift=6, font=dict(size=9, color=color),
                )
            fig_constructor_progress.update_xaxes(dtick=1)
            st.plotly_chart(fig_constructor_progress, width="stretch")

        constructor_rows = [
            {
                "Pos": pos_label(r["position"]),
                "Constructor": r["constructorName"],
                "Points": r["points"],
                "Wins": int(r["wins"]),
            }
            for _, r in constructor_standings.iterrows()
        ]
        standings_table(constructor_rows, team_col="Constructor")

        st.divider()

        st.subheader("Title fight: win probability (top 3)")
        total_rounds = total_rounds_in_season(year)
        remaining_rounds = total_rounds - round_number
        st.caption(
            f"{remaining_rounds} race{'s' if remaining_rounds != 1 else ''} left this season. "
            "A simplified Monte Carlo estimate limited to the current top 3: each remaining race "
            "is resampled from that driver's/team's own points scored per race so far, "
            "independently of who else is on track (so a driver's simulated score doesn't take "
            "points away from a rival the way a real race would) and ignoring anyone outside the "
            "top 3. Treat this as a feel for how close the fight is, not a real forecast."
        )

        def race_history(progress_df, col, name):
            pts = progress_df[progress_df[col] == name].sort_values("Round")["Points"].to_numpy()
            return np.diff(pts, prepend=0.0)

        def simulate_top3_odds(names, current_pts, histories, remaining, trials=8000, seed=42):
            if remaining <= 0:
                leader = names[int(np.argmax(current_pts))]
                return {n: (1.0 if n == leader else 0.0) for n in names}
            rng = np.random.default_rng(seed)
            sim_totals = np.tile(np.array(current_pts, dtype=float)[:, None], trials)
            for i, hist in enumerate(histories):
                sim_totals[i] += rng.choice(hist, size=(remaining, trials)).sum(axis=0)
            winners = np.argmax(sim_totals, axis=0)
            counts = np.bincount(winners, minlength=len(names))
            return {n: counts[i] / trials for i, n in enumerate(names)}

        def odds_chart(title, names, odds, colors):
            # A bar chart invites reading the axis as an absolute scale; these
            # three numbers only mean something relative to each other (they
            # sum to 100% by construction, since the sim always picks exactly
            # one of the three as champion) -- a donut makes that "shares of
            # one whole" relationship the shape itself, not something to infer.
            ranked = sorted(zip(names, odds, colors), key=lambda t: t[1], reverse=True)
            fig = go.Figure(
                go.Pie(
                    labels=[n for n, _, _ in ranked], values=[o * 100 for _, o, _ in ranked],
                    marker=dict(colors=[c for _, _, c in ranked], line=dict(color="#0e1117", width=2)),
                    hole=0.55, sort=False, textinfo="label+percent",
                    textfont=dict(size=13, color="#f2f4f6"),
                    hovertemplate="%{label}: %{percent}<extra></extra>",
                )
            )
            fig.update_layout(
                **DARK_LAYOUT,
                title=dict(text=title, font=dict(size=16, color="#f2f4f6")),
                height=260, showlegend=False, margin=dict(t=40, b=20, l=10, r=10),
            )
            fig.add_annotation(
                text=f"{ranked[0][0]}<br>{ranked[0][1] * 100:.0f}%", showarrow=False,
                font=dict(size=14, color=ranked[0][2]), x=0.5, y=0.5,
            )
            return fig

        col_drivers_odds, col_constructors_odds = st.columns(2)

        with col_drivers_odds:
            top3_drivers = driver_standings.sort_values("position").head(3)
            names = [
                r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
                for _, r in top3_drivers.iterrows()
            ]
            histories = [race_history(driver_progress, "Driver", n) for n in names]
            odds = simulate_top3_odds(names, top3_drivers["points"].tolist(), histories, remaining_rounds)
            colors = [driver_color(n) for n in names]
            st.plotly_chart(odds_chart("Drivers", names, [odds[n] for n in names], colors), width="stretch")

        with col_constructors_odds:
            top3_constructors = constructor_standings.sort_values("position").head(3)
            names_c = top3_constructors["constructorName"].tolist()
            histories_c = [race_history(constructor_progress, "Constructor", n) for n in names_c]
            odds_c = simulate_top3_odds(names_c, top3_constructors["points"].tolist(), histories_c, remaining_rounds)
            colors_c = [team_color(n) for n in names_c]
            st.plotly_chart(odds_chart("Constructors", names_c, [odds_c[n] for n in names_c], colors_c), width="stretch")

        st.divider()

        st.subheader("When could it be decided?")
        st.caption(
            "Same resampling, but round by round: for each simulated run, the earliest future "
            "round at which the current runner-up could no longer catch the current leader even "
            "by scoring the most points anyone has scored in a single race this season, every "
            "remaining round. A mathematical best-case for the leader, not a real prediction."
        )

        def max_points_per_race(progress_df, entity_col):
            per_entity_max = [
                np.diff(g.sort_values("Round")["Points"].to_numpy(), prepend=0.0).max()
                for _, g in progress_df.groupby(entity_col)
            ]
            return max(per_entity_max) if per_entity_max else 25.0

        def clinch_round_distribution(leader_pts, second_pts, leader_hist, second_hist, max_per_race, trials=6000, seed=7):
            if remaining_rounds <= 0:
                return np.array([float(round_number)])
            rng = np.random.default_rng(seed)
            leader_cum = leader_pts + np.cumsum(rng.choice(leader_hist, size=(remaining_rounds, trials)), axis=0)
            second_cum = second_pts + np.cumsum(rng.choice(second_hist, size=(remaining_rounds, trials)), axis=0)
            races_left_after = (remaining_rounds - np.arange(1, remaining_rounds + 1))[:, None]
            clinched = (leader_cum - second_cum) > races_left_after * max_per_race
            rounds = round_number + np.arange(1, remaining_rounds + 1)
            return np.where(clinched.any(axis=0), rounds[clinched.argmax(axis=0)], total_rounds).astype(float)

        col_clinch_drivers, col_clinch_constructors = st.columns(2)

        with col_clinch_drivers:
            if len(top3_drivers) >= 2:
                clinches = clinch_round_distribution(
                    top3_drivers["points"].iloc[0], top3_drivers["points"].iloc[1],
                    histories[0], histories[1], max_points_per_race(driver_progress, "Driver"),
                )
                median_round = int(round(np.median(clinches)))
                # st.metric's value is built for a short number and silently
                # truncates anything wider than the card with "..." -- a CSS
                # override aimed at that was tried and, even after several
                # goes, still showed the same ellipsis, so this sidesteps
                # the widget's own truncation instead of fighting it: the
                # short part stays the metric value, the race name (the part
                # that was actually getting cut off) goes in a caption below,
                # which wraps normally rather than truncating.
                st.metric(
                    f"Drivers' -- {names[0]} leading",
                    f"Round {median_round} of {total_rounds}",
                    f"{(clinches < total_rounds).mean() * 100:.0f}% chance it's sealed before the finale",
                    delta_color="off",
                )
                st.caption(f"📍 {event_name_for_round(year, median_round)}")
            else:
                st.info("Need at least 2 drivers with points to estimate this.")

        with col_clinch_constructors:
            if len(top3_constructors) >= 2:
                clinches_c = clinch_round_distribution(
                    top3_constructors["points"].iloc[0], top3_constructors["points"].iloc[1],
                    histories_c[0], histories_c[1], max_points_per_race(constructor_progress, "Constructor"),
                )
                median_round_c = int(round(np.median(clinches_c)))
                st.metric(
                    f"Constructors' -- {names_c[0]} leading",
                    f"Round {median_round_c} of {total_rounds}",
                    f"{(clinches_c < total_rounds).mean() * 100:.0f}% chance it's sealed before the finale",
                    delta_color="off",
                )
                st.caption(f"📍 {event_name_for_round(year, median_round_c)}")
            else:
                st.info("Need at least 2 constructors with points to estimate this.")

# ----------------------------------------------------------- season stats --

with tab_season_stats:
    st.caption(
        f"Computed across all {len(schedule)} completed races of the {year} season so far -- "
        "first load is slow (every race's full session gets fetched once), instant after that."
    )
    stats_df = season_stats(year, tuple(schedule["EventName"].tolist()))

    if stats_df.empty:
        st.info("No completed races to compute stats from yet.")
    else:
        st.caption(
            "Overtakes here use the same net-of-pit-stops, net-of-incidents rule as the "
            "Overtakes chart on the pace tab -- an approximation, not an official count."
        )
        # st.metric truncates a value that doesn't fit its column with an
        # ellipsis rather than wrapping, no matter how it's shortened -- so
        # the number goes in the metric value (always short) and the race
        # name that was getting cut off moves to a caption underneath,
        # which wraps normally instead of truncating.
        most_ot = stats_df.loc[stats_df["Overtakes"].idxmax()]
        fewest_ot = stats_df.loc[stats_df["Overtakes"].idxmin()]
        most_dnf = stats_df.loc[stats_df["Retirements"].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("Most overtakes", f"{int(most_ot['Overtakes'])}")
        col1.caption(f"📍 {most_ot['Event']}")
        col2.metric("Fewest overtakes", f"{int(fewest_ot['Overtakes'])}")
        col2.caption(f"📍 {fewest_ot['Event']}")
        col3.metric("Most retirements", f"{int(most_dnf['Retirements'])}")
        col3.caption(f"📍 {most_dnf['Event']}")

        col4, col5 = st.columns(2)
        recovery_df = stats_df.dropna(subset=["RecoveryPlaces"])
        if not recovery_df.empty:
            best_recovery = recovery_df.loc[recovery_df["RecoveryPlaces"].idxmax()]
            col4.metric("Best recovery drive", f"+{int(best_recovery['RecoveryPlaces'])} places")
            col4.caption(f"📍 {best_recovery['RecoveryDriver']} -- {best_recovery['Event']}")
        closest_df = stats_df.dropna(subset=["ClosestGap"])
        if not closest_df.empty:
            tightest = closest_df.loc[closest_df["ClosestGap"].idxmin()]
            col5.metric("Closest finish", f"{tightest['ClosestGap']:.3f} s")
            col5.caption(f"📍 {tightest['Event']} (P1 to P2)")

        st.divider()

        st.subheader("Overtakes per race")
        # Vertical bars with 20+ rotated race names along the bottom were
        # unreadable at anything but full-screen -- horizontal, like the
        # per-driver overtakes chart, so every name reads flat and the
        # chart just grows downward instead of squeezing sideways. Kept in
        # season order (top to bottom = round 1 to the latest) rather than
        # sorted by count, since the point is the season's shape over time,
        # not a leaderboard -- the color gradient carries the ranking.
        ordered_races = stats_df.sort_values("Round", ascending=False)
        fig_overtakes_season = base_figure("Overtakes by race", "", "Overtakes")
        fig_overtakes_season.update_layout(
            showlegend=False, height=max(CHART_HEIGHT, 26 * len(ordered_races)),
        )
        fig_overtakes_season.add_trace(
            go.Bar(
                x=ordered_races["Overtakes"], y=ordered_races["Event"].str.replace(" Grand Prix", ""),
                orientation="h",
                marker=dict(
                    color=ordered_races["Overtakes"], colorscale=[[0, "#2a2f45"], [1, "#5b8cff"]],
                    line=dict(color="rgba(255,255,255,0.15)", width=1),
                ),
                text=ordered_races["Overtakes"].astype(int),
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}: %{x} overtakes<extra></extra>",
            )
        )
        st.plotly_chart(fig_overtakes_season, width="stretch")
