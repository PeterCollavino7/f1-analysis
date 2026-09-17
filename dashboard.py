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
# Plotly's hover toolbar (camera / zoom / pan / reset) floats over the
# top-right of every chart, overlapping titles and adding controls none of
# these charts need -- the zoom that matters (drag on the telemetry panels)
# works without it.
PLOTLY_CONFIG = {"displayModeBar": False}
MIN_LAPS_FOR_TREND = 10  # per compound, for the pooled fuel-correction fit
MIN_LAPS_PER_DRIVER = 6  # per driver+compound, for the per-driver breakdown
TRACK_SECTOR_COUNT = 20  # mini-sectors for the track dominance map, not the official S1/S2/S3

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
    /* Sidebar. Restyled with background, border, spacing and typography
       only -- nothing here hides a Streamlit element, because the internal
       DOM isn't ours and a selector that stops matching after an upgrade
       would silently take a control with it. If a rule below misses, the
       control still renders, just unstyled. */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #181d29 0%, #101320 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.68rem;
        font-weight: 600;
        color: rgba(215, 219, 224, 0.55);
    }
    /* The section picker reads as a nav rather than a form control: each
       option gets its own row, and the chosen one carries the accent. */
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        text-transform: none;
        letter-spacing: 0.01em;
        font-size: 0.95rem;
        font-weight: 600;
        color: #e8ebef;
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.25rem;
        border-radius: 9px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(255, 255, 255, 0.02);
        transition: background 0.12s ease, border-color 0.12s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.12);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: rgba(225, 6, 0, 0.14);
        border-color: rgba(225, 6, 0, 0.55);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.1rem 0 0.9rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }
    .sidebar-brand .flag {
        width: 22px; height: 22px; flex: 0 0 auto; border-radius: 4px;
        background-image:
            linear-gradient(45deg, #e8ebef 25%, transparent 25%, transparent 75%, #e8ebef 75%),
            linear-gradient(45deg, #e8ebef 25%, #12151d 25%, #12151d 75%, #e8ebef 75%);
        background-size: 11px 11px;
        background-position: 0 0, 5.5px 5.5px;
    }
    .sidebar-brand .word {
        font-size: 1.05rem; font-weight: 800; letter-spacing: 0.06em; color: #f2f4f6;
    }
    .sidebar-brand .word em {
        font-style: italic; color: #e10600; font-weight: 900; margin-right: 0.15rem;
    }
    /* Header. Stock photography of an F1 car would be someone else's
       copyright, so the artwork is the circuit itself, traced from the
       session's own position telemetry -- it changes with the race being
       looked at, which a stock photo wouldn't. */
    .hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        overflow: hidden;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.1rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.07);
        background:
            radial-gradient(120% 180% at 0% 0%, rgba(225, 6, 0, 0.22) 0%, transparent 55%),
            linear-gradient(135deg, #1b2130 0%, #11141d 60%);
    }
    /* Speed lines: a few skewed streaks fading out across the panel. */
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
            100deg,
            transparent 0 26px,
            rgba(255, 255, 255, 0.045) 26px 28px
        );
        mask-image: linear-gradient(90deg, transparent 0%, #000 45%, transparent 100%);
        pointer-events: none;
    }
    .hero-text { position: relative; z-index: 1; min-width: 0; }
    .hero-kicker {
        display: inline-flex; align-items: center; gap: 0.5rem;
        text-transform: uppercase; letter-spacing: 0.14em;
        font-size: 0.66rem; font-weight: 700;
        color: rgba(255, 255, 255, 0.55);
        margin-bottom: 0.45rem;
    }
    .hero-kicker .pill {
        padding: 0.15rem 0.5rem; border-radius: 999px;
        background: #e10600; color: #fff; letter-spacing: 0.08em;
    }
    .hero-title {
        font-size: 1.85rem; font-weight: 800; line-height: 1.1; color: #f4f6f8;
        letter-spacing: -0.01em;
    }
    .hero-sub {
        margin-top: 0.35rem; font-size: 0.92rem; color: rgba(215, 219, 224, 0.62);
    }
    .hero-track { position: relative; z-index: 1; flex: 0 0 auto; opacity: 0.75; }
    .hero-track svg { display: block; height: 96px; width: auto; }
    /* The loading indicator is styled as a panel, not replaced: an earlier
       version hid Streamlit's own glyph and drew a ring in its place, but
       the selector stopped matching and the two ended up side by side. The
       wording is fixed where it actually comes from -- every cached
       function below passes its own show_spinner text, since the default
       prints the function name at the reader ("Running load_session(...)"). */
    [data-testid="stSpinner"] {
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(22, 27, 39, 0.92);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        width: fit-content;
    }
    [data-testid="stSpinner"] p {
        font-size: 0.9rem;
        color: rgba(232, 235, 239, 0.9);
        margin: 0;
    }
    /* st.metric truncates a long value with an ellipsis by default (built
       for short KPI numbers) -- the clinch-round estimate's value includes
       a full Grand Prix name, so this lets it wrap onto a second line
       instead of cutting the name off. */
    [data-testid="stMetricValue"] {
        white-space: normal;
        overflow-wrap: break-word;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }
    /* Every table here is hand-rolled HTML rather than st.dataframe: the
       built-in grid draws a border around every cell, fixes its own row
       height and scrolls inside itself, which reads as a spreadsheet dump.
       This wants a timing screen -- no vertical rules, a hairline between
       rows, the leader carrying a colored edge, and the points bar tinted
       per entrant (st.column_config.ProgressColumn is one fixed color for
       the whole column). tabular-nums is what makes the numeric columns
       line up: proportional digits leave them ragged, since a "1" is
       narrower than an "8". */
    table.standings {
        width: 100%;
        border-collapse: collapse;
        font-family: "Segoe UI Variable Text", "Segoe UI", Inter, -apple-system, sans-serif;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }
    table.standings th {
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-size: 0.68rem;
        font-weight: 600;
        color: rgba(215, 219, 224, 0.55);
        padding: 0 0.7rem 0.55rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        white-space: nowrap;
    }
    table.standings td {
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.95rem;
        color: #e8ebef;
        vertical-align: middle;
    }
    table.standings tr:last-child td { border-bottom: none; }
    table.standings tr:hover td { background: rgba(255, 255, 255, 0.035); }
    table.standings td.pos {
        width: 2.6rem;
        font-weight: 700;
        color: rgba(215, 219, 224, 0.75);
        border-left: 3px solid var(--row-accent, transparent);
    }
    table.standings tr.leader td { background: rgba(91, 140, 255, 0.10); }
    table.standings tr.leader td.name { font-weight: 700; }
    table.standings td.badge { width: 3.6rem; padding-right: 0; }
    table.standings td.badge img { display: block; height: 30px; }
    table.standings td.name { font-weight: 600; letter-spacing: 0.01em; }
    table.standings td.team { color: rgba(215, 219, 224, 0.62); font-size: 0.88rem; }
    table.standings td.num { text-align: right; width: 3.4rem; }
    table.standings td.points { width: 42%; }
    table.standings td.mono {
        font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
        font-size: 0.88rem;
        color: rgba(232, 235, 239, 0.9);
    }
    table.standings td.status { color: rgba(215, 219, 224, 0.6); font-size: 0.85rem; }
    /* Track map: the whole point of drawing it as SVG instead of a Plotly
       figure is this :hover rule -- the browser lights the mini-sector under
       the cursor with no callback, no rerun and no JS. */
    .trackmap { display: flex; justify-content: center; padding: 0.5rem 0 0.2rem; }
    .trackmap svg { width: 100%; max-width: 100%; overflow: visible; }
    .trackmap polyline.sector {
        fill: none;
        stroke-width: 9;
        stroke-linecap: round;
        stroke-linejoin: round;
        transition: stroke-width 0.12s ease, filter 0.12s ease;
    }
    /* No dimming of the other sectors on hover: the svg's hover area is its
       bounding box, which includes the whole infield, so the cursor merely
       being near the track would dim everything. */
    .trackmap polyline.sector:hover {
        stroke-width: 16;
        filter: drop-shadow(0 0 8px currentColor);
    }
    /* The hover card. pointer-events stays off so the card can never sit
       between the cursor and the sector that summoned it, which would make
       it flicker as the two fight over the hover. */
    .trackmap .tip { opacity: 0; pointer-events: none; transition: opacity 0.12s ease; }
    .trackmap .tip rect {
        fill: #161b27;
        stroke: rgba(255, 255, 255, 0.16);
        stroke-width: 1.5;
        filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.55));
    }
    .trackmap .tip line { stroke: rgba(255, 255, 255, 0.14); stroke-width: 1.5; }
    .trackmap .tip text {
        font-family: "Segoe UI Variable Text", "Segoe UI", Inter, sans-serif;
        font-variant-numeric: tabular-nums;
    }
    /* No font-size here on purpose: it's set per element in the markup,
       converted from pixels into viewBox units, since one unit is only a
       fraction of a pixel on a portrait circuit. */
    .trackmap .tip-head { font-weight: 700; fill: #f2f4f6; }
    .trackmap .tip-code { font-weight: 700; }
    .trackmap .tip-time {
        font-weight: 600; fill: #e8ebef;
        font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
    }
    .trackmap .tip-gap { fill: rgba(215, 219, 224, 0.6); }
    .points-wrap { display: flex; align-items: center; gap: 0.6rem; }
    .points-bar {
        flex: 1;
        height: 7px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.07);
        overflow: hidden;
    }
    .points-bar > span { display: block; height: 100%; border-radius: 999px; }
    .points-val { min-width: 2.9rem; text-align: right; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner="Loading the championship standings...")
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


@st.cache_data(ttl=3600, show_spinner="Rebuilding the points table round by round...")
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


@st.cache_data(ttl=3600, show_spinner=False)
def total_rounds_in_season(year):
    """Full season length (including rounds not yet run) -- load_schedule
    filters those out for the race picker, but the title-fight odds below
    need to know how many races are actually still left to simulate."""
    schedule = fastf1.get_event_schedule(year)
    return int(schedule[schedule["RoundNumber"] > 0]["RoundNumber"].max())


@st.cache_data(ttl=3600, show_spinner=False)
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
    driver_overtakes = {}
    for event_name in event_names:
        try:
            race = fastf1.get_session(year, event_name, "Race")
            race.load()
        except Exception:
            continue
        race_overtakes = count_overtakes(race)
        for driver, count in race_overtakes.items():
            driver_overtakes[driver] = driver_overtakes.get(driver, 0) + count
        overtakes_total = sum(race_overtakes.values())
        results = race.results
        retirements = int((results["Status"] == "Retired").sum())

        recovery_driver, recovery_places = None, None
        valid = results.dropna(subset=["GridPosition", "Position"])
        if not valid.empty:
            gains = valid["GridPosition"] - valid["Position"]
            best_idx = gains.idxmax()
            if gains.loc[best_idx] > 0:
                recovery_driver, recovery_places = valid.loc[best_idx, "Abbreviation"], int(gains.loc[best_idx])

        rows.append({
            "Event": event_name,
            "Round": int(race.event["RoundNumber"]),
            "Overtakes": overtakes_total,
            "Retirements": retirements,
            "RecoveryDriver": recovery_driver,
            "RecoveryPlaces": recovery_places,
        })
    return pd.DataFrame(rows), driver_overtakes


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


@st.cache_data(ttl=86400, show_spinner="Pulling every result in F1 history for this one...")
def all_time_finishes(results_position=None, grid_position=None):
    """Every race result in F1 history at one finishing (or starting) slot.

    Ergast has no all-time totals endpoint, and summing season standings
    would mean a request per season since 1950. Filtering results to a
    single position instead returns roughly one row per race ever held --
    about 1160 -- which paging fetches in two requests rather than
    seventy-odd. Row per race, so the caller groups it however it likes.
    """
    ergast = Ergast(result_type="pandas", auto_cast=True)
    # The API caps a page at 100 however large a limit is asked for, and the
    # offset counts results, so the loop has to step by the page size it
    # actually gets rather than the one it requested (stepping by a larger
    # requested limit silently skips most of the history).
    page_size = 100
    rows, offset = [], 0
    while True:
        response = ergast.get_race_results(
            results_position=results_position, grid_position=grid_position,
            limit=page_size, offset=offset,
        )
        frames = response.content
        if not frames:
            break
        # The season and race name live on the response's description (one
        # row per race), the driver and constructor on its content (one
        # frame per race), so a usable row needs both halves.
        descriptions = response.description
        for i, frame in enumerate(frames):
            if not len(frame):
                continue
            row = frame.iloc[0].to_dict()
            for field in ("season", "round", "raceName", "raceDate"):
                row[field] = descriptions.iloc[i][field]
            rows.append(row)
        offset += page_size
        if offset >= response.total_results:
            break
    return pd.DataFrame(rows)


def full_name(row):
    return f"{row['givenName']} {row['familyName']}"


@st.cache_data(ttl=3600, show_spinner="Loading the race calendar...")
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


@st.cache_data(show_spinner="Loading timing and telemetry for this session...")
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


def interp_extrapolate(t, xp, fp):
    """np.interp, but linearly extrapolating past the ends instead of
    clamping to the nearest sample -- used for "distance at elapsed=0" of a
    lap, which is always a query *before* the first telemetry sample (a
    driver's first sample lags their true lap-start instant by anything
    from ~2ms to over 100ms, and that lag differs enough driver to driver
    that clamping silently baked several extra meters of "distance" into
    whichever driver's telemetry started up slower -- worth ~0.1s once that
    fed into the delta-time chart as a fake early-lap gap)."""
    if t <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return fp[0] + slope * (t - xp[0])
    if t >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return fp[-1] + slope * (t - xp[-1])
    return float(np.interp(t, xp, fp))


def safe_driver_color(code, session):
    """get_driver_color raises for a code that isn't in this session's entry
    list (a mid-season replacement, a one-off stand-in, or any driver from a
    season-wide view), so every lookup outside the telemetry tab needs its
    own guard rather than reusing build_driver_styles."""
    try:
        return fastf1.plotting.get_driver_color(code, session)
    except Exception:
        return "#999999"


def driver_number_badge(number, color):
    # No official driver-number graphics ship with FastF1, and pulling real
    # ones off the web means trusting an outside source for accuracy and
    # license -- this draws one instead, styled after a real race number
    # rather than a plain label: heavy italic numerals, a thick dark outline
    # (paint-order puts the stroke behind the fill so it reads as an outline,
    # not a smudged edge) and the driver's color as a stripe across the top
    # of the plate. Two-digit numbers get a smaller face -- at one size they
    # ran past the plate edges (checked rendered, "63" and "81" were worst).
    font_size = 26 if len(str(number)) <= 1 else 22
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="56" height="40" viewBox="0 0 56 40">'
        '<defs><clipPath id="c"><rect x="2" y="2" width="52" height="36" rx="8"/></clipPath></defs>'
        '<g clip-path="url(#c)">'
        '<rect x="2" y="2" width="52" height="36" fill="#f4f6f8"/>'
        f'<rect x="2" y="2" width="52" height="8" fill="{color}"/>'
        '</g>'
        '<rect x="2" y="2" width="52" height="36" rx="8" fill="none" stroke="#0b0d12" stroke-width="3"/>'
        f'<text x="28" y="31" font-family="Arial Black, Arial, Helvetica, sans-serif" font-size="{font_size}" '
        'font-weight="900" font-style="italic" text-anchor="middle" fill="#f4f6f8" '
        f'stroke="#0b0d12" stroke-width="3" paint-order="stroke">{number}</text></svg>'
    )
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def size_horizontal_bars(fig, values, row_px=34, extra_px=110, bar_px=20, pad_frac=0.18):
    """Size a horizontal bar chart by its bar count instead of a fixed floor.

    A fixed height with only a handful of bars makes Plotly stretch each one
    to fill its slot, so four entries render as four slabs. Height follows the
    count, and the bar width (a *fraction of the category slot*, not pixels)
    is solved back from the slot height so a bar is about bar_px thick
    whatever the count. The x range gets headroom too, so the outside value
    labels aren't clipped against the plot edge.
    """
    count = max(1, len(values))
    height = max(180, row_px * count + extra_px)
    slot_px = max(1.0, (height - extra_px) / count)
    fig.update_layout(height=height)
    fig.update_traces(width=min(0.75, bar_px / slot_px), selector=dict(type="bar"))
    if not len(values):
        return
    low, high = float(min(values)), float(max(values))
    # Bars that run both ways (degradation slopes) need the negative side
    # kept, not clipped to a zero-anchored range.
    pad = (high - low) * pad_frac or abs(high) * pad_frac or 1.0
    fig.update_xaxes(range=[min(0.0, low) - (pad if low < 0 else 0.0), high + pad])


def render_table(rows, columns):
    """A results table as hand-rolled HTML (see the table.standings CSS).

    Each row is a dict of cell values, optionally carrying "badge" (an image
    URI, rendered in a leading column) and "color" (the accent stripe on the
    first cell). Columns are (key, header, css class) triples.
    """
    has_badges = any(row.get("badge") for row in rows)
    html = ["<table class='standings'><thead><tr>"]
    if has_badges:
        html.append("<th></th>")
    html += [f"<th>{header}</th>" for _, header, _ in columns]
    html.append("</tr></thead><tbody>")
    for rank, row in enumerate(rows):
        html.append("<tr class='leader'>" if rank == 0 else "<tr>")
        if has_badges:
            badge = row.get("badge")
            html.append(f"<td class='badge'>{f'<img src=\"{badge}\">' if badge else ''}</td>")
        for index, (key, _, css_class) in enumerate(columns):
            accent = f" style='--row-accent:{row.get('color', 'transparent')}'" if index == 0 else ""
            html.append(f"<td class='{css_class}'{accent}>{row.get(key, '')}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)


def circuit_outline(session, stroke=5):
    """The circuit traced as a small SVG, for the page header.

    Returns an empty string rather than raising if the session has no usable
    position data -- the header is decoration, and a practice session nobody
    completed a lap in shouldn't take the page down with it.
    """
    try:
        lap = session.laps.pick_fastest()
        telemetry = lap.get_telemetry()
        x = telemetry["X"].to_numpy(dtype=float)[::8]
        y = -telemetry["Y"].to_numpy(dtype=float)[::8]
    except Exception:
        return ""
    if len(x) < 10:
        return ""
    pad = stroke
    span_x = max(x.max() - x.min(), 1e-6)
    span_y = max(y.max() - y.min(), 1e-6)
    scale = (300 - 2 * pad) / span_x
    view_height = span_y * scale + 2 * pad
    points = " ".join(
        f"{(px - x.min()) * scale + pad:.1f},{(py - y.min()) * scale + pad:.1f}"
        for px, py in zip(x, y)
    )
    return (
        f'<svg viewBox="0 0 300 {view_height:.0f}" preserveAspectRatio="xMidYMid meet">'
        f'<polyline points="{points}" fill="none" stroke="rgba(255,255,255,0.75)" '
        f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def render_hero(kicker_pill, kicker_text, title, subtitle, artwork=""):
    st.markdown(
        '<div class="hero"><div class="hero-text">'
        f'<div class="hero-kicker"><span class="pill">{kicker_pill}</span>{kicker_text}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>'
        f'</div><div class="hero-track">{artwork}</div></div>',
        unsafe_allow_html=True,
    )


def render_track_map(x, y, sectors, height=620, stroke_room=16):
    """The lap as inline SVG, one path per mini-sector.

    Drawn by hand rather than with Plotly because the highlight has to
    follow the mouse: st.plotly_chart has no hover callback into Python, and
    wiring plotly_hover through a raw-JS component (tried with a CDN script
    tag, then with plotly.js inlined) froze the whole app both times. An SVG
    path needs no JS at all -- the browser's own :hover does it, and the
    payload is a few hundred coordinates instead of a re-embedded plotting
    library.
    """
    # SVG's y axis grows downward, so the track comes out mirrored unless
    # it's flipped back here.
    x = np.asarray(x, dtype=float)
    y = -np.asarray(y, dtype=float)
    # Room for the widest the stroke gets (the hover state) plus its glow,
    # so a lit-up sector at the edge of the lap isn't clipped.
    pad = stroke_room * 1.6
    span_x = max(x.max() - x.min(), 1e-6)
    span_y = max(y.max() - y.min(), 1e-6)
    scale = (1000 - 2 * pad) / span_x
    view_height = span_y * scale + 2 * pad

    def to_view(xs, ys):
        return (xs - x.min()) * scale + pad, (ys - y.min()) * scale + pad

    paths, tips, rules = [], [], []
    # Card geometry is a share of the viewBox, not a pixel size converted
    # through an assumed render height. The svg is scaled to fit whatever
    # width the column happens to be, so any pixel assumption is wrong the
    # moment the window changes size -- converting from an assumed 430px
    # tall render is what made this card come out nearly as wide as the
    # whole lap. As a fraction of the drawing it stays proportionate at any
    # size: roughly a third of the lap's width.
    box_w, box_h = 480.0, 186.0
    pad_x = 30.0
    head_y, rule_y, row_a_y, row_b_y = 52.0, 74.0, 122.0, 164.0
    font_head, font_code, font_time, font_gap = 42.0, 38.0, 38.0, 29.0
    for index, sector in enumerate(sectors):
        # Every third sample: at full resolution the path data runs to tens
        # of thousands of coordinates per lap for a curve that looks
        # identical either way.
        seg_x, seg_y = to_view(x[sector["slice"]][::3], y[sector["slice"]][::3])
        points = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(seg_x, seg_y))
        # color is set alongside stroke so the hover drop-shadow can pick the
        # sector's own color up through currentColor.
        paths.append(
            f'<polyline id="tds{index}" class="sector" points="{points}" '
            f'stroke="{sector["color"]}" style="color:{sector["color"]}"/>'
        )

        # The card is drawn in SVG rather than HTML so it can live in the
        # same coordinate space as the track and still need no JS. Every
        # tooltip is emitted *after* every polyline so it paints on top of
        # the whole lap; that puts it out of reach of a sibling selector, so
        # each one is tied to its sector through :has() instead.
        mid_x, mid_y = seg_x[len(seg_x) // 2], seg_y[len(seg_y) // 2]
        # Flip the card below the track when the sector sits near the top,
        # so it doesn't hang off the edge of the figure.
        offset = 22.0
        above = mid_y > box_h + offset
        top = mid_y - box_h - offset if above else mid_y + offset
        left = min(max(mid_x - box_w / 2, 4), 1000 - box_w - 4)
        gap_text = f"{'+' if sector['time_a'] >= sector['time_b'] else '-'}{sector['gap']:.3f}s"
        tips.append(
            f'<g id="tdt{index}" class="tip" transform="translate({left:.1f},{top:.1f})">'
            f'<rect width="{box_w:.1f}" height="{box_h:.1f}" rx="14"/>'
            f'<text class="tip-head" x="{box_w / 2:.1f}" y="{head_y:.1f}" '
            f'font-size="{font_head:.1f}" text-anchor="middle">Mini-sector {sector["number"]}</text>'
            f'<line x1="{pad_x:.1f}" y1="{rule_y:.1f}" x2="{box_w - pad_x:.1f}" y2="{rule_y:.1f}"/>'
            f'<text class="tip-code" x="{pad_x:.1f}" y="{row_a_y:.1f}" font-size="{font_code:.1f}" '
            f'fill="{sector["color_a"]}">{sector["driver_a"]}</text>'
            f'<text class="tip-time" x="{box_w - pad_x:.1f}" y="{row_a_y:.1f}" '
            f'font-size="{font_time:.1f}" text-anchor="end">{sector["time_a"]:.3f}s</text>'
            f'<text class="tip-code" x="{pad_x:.1f}" y="{row_b_y:.1f}" font-size="{font_code:.1f}" '
            f'fill="{sector["color_b"]}">{sector["driver_b"]}</text>'
            f'<text class="tip-time" x="{box_w - pad_x:.1f}" y="{row_b_y:.1f}" '
            f'font-size="{font_time:.1f}" text-anchor="end">{sector["time_b"]:.3f}s'
            f'<tspan class="tip-gap" font-size="{font_gap:.1f}"> {gap_text}</tspan></text>'
            f"</g>"
        )
        rules.append(f".trackmap svg:has(#tds{index}:hover) #tdt{index}{{opacity:1}}")

    st.markdown(
        f"<style>{''.join(rules)}</style>"
        f'<div class="trackmap"><svg viewBox="0 0 1000 {view_height:.0f}" '
        f'style="height:{height}px" preserveAspectRatio="xMidYMid meet">'
        f'{"".join(paths)}{"".join(tips)}</svg></div>',
        unsafe_allow_html=True,
    )


def points_cell(points, leader_points, color):
    """Points as a bar tinted with the entrant's own color, plus the value.

    st.column_config.ProgressColumn can only paint one color for a whole
    column, which loses the per-driver identity this table is built around.
    """
    pct = max(1.5, 100 * points / (leader_points or 1))
    return (
        "<div class='points-wrap'>"
        f"<div class='points-bar'><span style='width:{pct:.1f}%;background:{color}'></span></div>"
        f"<div class='points-val'>{points:.0f}</div>"
        "</div>"
    )


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

# Three different scopes used to share one tab strip under a single "2026
# Spanish Grand Prix -- Race" heading, which made no sense for the two that
# don't belong to a session: the standings are a season's, the records are
# the sport's. Picking the scope first also means each one only asks for the
# selectors it actually uses -- and the all-time view loads no session at
# all, instead of fetching a full race weekend it never reads.
SECTION_WEEKEND = "Race weekend"
SECTION_SEASON = "Season"
SECTION_ALL_TIME = "All-time records"

st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="flag"></div>'
    '<div class="word"><em>F1</em> DASHBOARD</div></div>',
    unsafe_allow_html=True,
)
section = st.sidebar.radio("Section", [SECTION_WEEKEND, SECTION_SEASON, SECTION_ALL_TIME])

session = None
if section in (SECTION_WEEKEND, SECTION_SEASON):
    st.sidebar.divider()
    year = st.sidebar.selectbox("Year", options=YEARS)
    schedule = load_schedule(year)
    event_name = st.sidebar.selectbox(
        "Grand Prix" if section == SECTION_WEEKEND else "Standings after",
        options=schedule["EventName"].tolist(),
    )
    event_row = schedule[schedule["EventName"] == event_name].iloc[0]
    # The season views don't compare laps, but they do color drivers and
    # teams, and fastf1.plotting needs a loaded session to do that -- the
    # weekend's race is the one that's always there.
    session_name = (
        st.sidebar.selectbox("Session", options=session_names_for(event_row))
        if section == SECTION_WEEKEND else "Race"
    )
    try:
        session = load_session(year, event_name, session_name)
    except Exception as exc:
        st.error(f"Couldn't load this session (it may not have happened yet): {exc}")
        st.stop()

if section == SECTION_WEEKEND:
    render_hero(
        f"Round {int(event_row['RoundNumber'])}",
        f"{year} · {event_row['Country']}",
        event_name,
        f"{session_name} · {event_row['Location']} · "
        f"{pd.to_datetime(event_row['EventDate']).strftime('%d %B %Y')}",
        circuit_outline(session),
    )
    tab_telemetry, tab_pace, tab_classification = st.tabs(
        ["Head-to-head telemetry", "Race pace & tyre degradation", "Classification"]
    )
elif section == SECTION_SEASON:
    render_hero(
        f"{year}", f"{len(schedule)} races run",
        f"{year} season",
        f"Standings and stats as they stood after {event_name}",
        circuit_outline(session),
    )
    tab_standings, tab_season_stats = st.tabs(["Standings", "Season stats"])
else:
    render_hero(
        "1950 — today", "World championship",
        "All-time records",
        "Every world-championship race ever run, ranked",
    )
    (tab_all_time,) = st.tabs(["Every race since 1950"])

# ------------------------------------------------------------- telemetry --

if section == SECTION_WEEKEND:
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

        # pick_fastest() returns None for a driver with no timed lap at all --
        # routine in practice, and it happens in a race weekend whenever someone
        # never gets out. Dropping them here rather than letting the telemetry
        # call fail: this block runs at module level, so one such driver used to
        # take the whole app down, every tab with it.
        laps_by_driver = {}
        for driver in selected_drivers:
            lap = session.laps.pick_drivers(driver).pick_fastest()
            if lap is not None and pd.notna(lap["LapTime"]):
                laps_by_driver[driver] = lap
        missing = [d for d in selected_drivers if d not in laps_by_driver]
        if missing:
            st.warning(f"No timed lap for {', '.join(missing)} in this session -- left out.")
        selected_drivers = [d for d in selected_drivers if d in laps_by_driver]
        if not selected_drivers:
            st.info("None of the selected drivers set a timed lap in this session.")
            st.stop()

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
                    interp_extrapolate(t, tel["Time"].dt.total_seconds().to_numpy(), tel["Distance"].to_numpy())
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
            f"Track dominance: the lap is split into {TRACK_SECTOR_COUNT} mini-sectors (their length "
            "follows this track's own lap distance, so every circuit gets its own boundaries), each "
            "colored by whichever driver actually spent less time crossing it -- a real corner-by-corner "
            "read on downforce/drag tradeoffs, not just overall pace. Hover a stretch of track for its "
            "mini-sector number and both drivers' times."
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
                    interp_extrapolate(t, tel_a["Time"].dt.total_seconds().to_numpy(), tel_a["Distance"].to_numpy())
                    for t in sector_checkpoints[dom_a]
                ]
                tel_a = tel_a.copy()
                tel_a["Distance"] = np.interp(tel_a["Distance"].to_numpy(), tel_a_checkpoints, shared_checkpoint_distance)
            except (KeyError, NameError):
                pass
            x_on_grid = resample_linear(tel_a, "X")
            y_on_grid = resample_linear(tel_a, "Y")

            color_a, _ = driver_style[dom_a]
            color_b, _ = driver_style[dom_b]

            # Real corner-by-corner dominance instead of a per-point speed
            # comparison: whichever instant a speed trace happens to be higher
            # at can flip mid-corner without saying who was actually quicker end
            # to end through that stretch. Reusing the elapsed-time channel
            # already resampled onto the shared distance grid above (the same
            # one the delta-time panel below is built from) and comparing how
            # long each driver actually took to cross each mini-sector answers
            # that properly.
            elapsed_a = resampled[dom_a]["Elapsed"]
            elapsed_b = resampled[dom_b]["Elapsed"]
            n_points = len(common_distance_m)
            sector_edges = [round(i * (n_points - 1) / TRACK_SECTOR_COUNT) for i in range(TRACK_SECTOR_COUNT + 1)]

            # Real hover highlighting needs a plotly_hover/plotly_unhover JS
            # listener wired straight into the chart's own div -- st.plotly_chart
            # has no hover callback into Python. Tried twice as a raw-JS HTML
            # component (CDN <script src>, then a fully inlined plotly.js) and
            # both froze the whole app. Click selection is different: Streamlit
            # has native support for it (on_select="rerun"), reusing the same
            # plotly runtime it already ships instead of re-embedding one, so a
            # click -- not a hover -- persists a highlighted sector across the
            # rerun it triggers, via session_state.
            # Teammates resolve to the same team color, and build_driver_styles
            # turns the repeat into plain white -- fine on a line chart, drab as
            # half of a two-color map. A distinct second color keeps the two
            # sides of the comparison legible whoever the pair is.
            if color_b.lower() in ("#ffffff", color_a.lower()):
                color_b = "#2ee86e"

            sector_paths = []
            for i in range(TRACK_SECTOR_COUNT):
                start, end = sector_edges[i], sector_edges[i + 1]
                seg_time_a = elapsed_a[end] - elapsed_a[start]
                seg_time_b = elapsed_b[end] - elapsed_b[start]
                sector_paths.append({
                    "number": i + 1,
                    "color": color_a if seg_time_a <= seg_time_b else color_b,
                    # One point of overlap into the next sector, so consecutive
                    # stretches meet instead of leaving a hairline of background
                    # showing at every boundary.
                    "slice": slice(start, min(end + 2, n_points)),
                    "driver_a": dom_a, "color_a": color_a, "time_a": seg_time_a,
                    "driver_b": dom_b, "color_b": color_b, "time_b": seg_time_b,
                    "gap": abs(seg_time_a - seg_time_b),
                })
            render_track_map(x_on_grid, y_on_grid, sector_paths)
            st.caption("Hover a stretch of track to light up its mini-sector.")
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
                        showlegend=False,
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

        st.plotly_chart(fig_telemetry, width="stretch", config=PLOTLY_CONFIG)

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

        st.plotly_chart(fig_position, width="stretch", config=PLOTLY_CONFIG)

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
        st.plotly_chart(fig_strategy, width="stretch", config=PLOTLY_CONFIG)

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
            fig_overtakes.update_layout(showlegend=False)
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
            size_horizontal_bars(fig_overtakes, [c for _, c in ranked_overtakes])
            st.plotly_chart(fig_overtakes, width="stretch", config=PLOTLY_CONFIG)

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
        st.plotly_chart(fig_pace, width="stretch", config=PLOTLY_CONFIG)

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
            size_horizontal_bars(fig_drivers, [s for _, s in ranked])
            st.plotly_chart(fig_drivers, width="stretch", config=PLOTLY_CONFIG)


if section == SECTION_WEEKEND:
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
                col_poles_driver, col_poles_engine = st.columns(2)

                with col_poles_driver:
                    by_driver = poles["Driver"].value_counts().sort_values()
                    fig_poles_driver = base_figure("Pole positions by driver", "", "Poles")
                    fig_poles_driver.update_layout(showlegend=False)
                    fig_poles_driver.add_trace(
                        go.Bar(
                            x=by_driver.to_numpy(), y=by_driver.index, orientation="h",
                            marker_color=[safe_driver_color(d, session) for d in by_driver.index],
                            text=by_driver.to_numpy(), textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                        )
                    )
                    size_horizontal_bars(fig_poles_driver, by_driver.to_numpy())
                    st.plotly_chart(fig_poles_driver, width="stretch", config=PLOTLY_CONFIG)

                with col_poles_engine:
                    engines = poles["Team"].map(ENGINE_SUPPLIERS).fillna(poles["Team"])
                    by_engine = engines.value_counts().sort_values()
                    fig_poles_engine = base_figure("Pole positions by engine manufacturer", "", "Poles")
                    fig_poles_engine.update_layout(showlegend=False)
                    fig_poles_engine.add_trace(
                        go.Bar(
                            x=by_engine.to_numpy(), y=by_engine.index, orientation="h",
                            marker_color="#5b8cff",
                            text=by_engine.to_numpy(), textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                        )
                    )
                    size_horizontal_bars(fig_poles_engine, by_engine.to_numpy())
                    st.plotly_chart(fig_poles_engine, width="stretch", config=PLOTLY_CONFIG)
        else:
            render_pace_tab()

# --------------------------------------------------------- classification --

if section == SECTION_WEEKEND:
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
            number_by_driver = results.set_index("Abbreviation")["DriverNumber"] if not results.empty else {}
            rows = [
                {
                    "badge": driver_number_badge(
                        number_by_driver.get(driver, "—"), safe_driver_color(driver, session)
                    ),
                    "color": safe_driver_color(driver, session),
                    "pos": rank,
                    "driver": driver,
                    "team": team_by_driver.get(driver, ""),
                    "best": format_lap_time(lap_time),
                    "gap": "—" if rank == 1 else f"+{(lap_time - fastest.iloc[0]).total_seconds():.3f}",
                }
                for rank, (driver, lap_time) in enumerate(fastest.items(), start=1)
            ]
            render_table(rows, [
                ("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team"),
                ("best", "Best lap", "mono"), ("gap", "Gap", "mono"),
            ])
        else:
            has_quali_times = results[["Q1", "Q2", "Q3"]].notna().any().any()
            classified = results.sort_values("Position")
            leader_row = classified[classified["Position"] == 1]
            leader_laps = leader_row["Laps"].iloc[0] if len(leader_row) and pd.notna(leader_row["Laps"].iloc[0]) else None

            rows = []
            for _, r in classified.iterrows():
                color = safe_driver_color(r["Abbreviation"], session)
                row = {
                    "badge": driver_number_badge(
                        r["DriverNumber"] if pd.notna(r["DriverNumber"]) else "—", color,
                    ),
                    "color": color,
                    "pos": str(int(r["Position"])) if pd.notna(r["Position"]) else "—",
                    "driver": r["Abbreviation"],
                    "team": r["TeamName"],
                }
                if has_quali_times:
                    row["q1"] = format_lap_time(r["Q1"])
                    row["q2"] = format_lap_time(r["Q2"])
                    row["q3"] = format_lap_time(r["Q3"])
                else:
                    row["grid"] = str(int(r["GridPosition"])) if pd.notna(r["GridPosition"]) else "—"
                    row["gap"] = format_race_gap(r, leader_laps)
                    row["pts"] = f"{r['Points']:.0f}" if pd.notna(r["Points"]) else "0"
                    row["status"] = r["Status"] if pd.notna(r["Status"]) else "—"
                rows.append(row)

            columns = [("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team")]
            if has_quali_times:
                columns += [("q1", "Q1", "mono"), ("q2", "Q2", "mono"), ("q3", "Q3", "mono")]
            else:
                columns += [
                    ("grid", "Grid", "num"), ("gap", "Gap", "mono"),
                    ("pts", "Pts", "num"), ("status", "Status", "status"),
                ]
            render_table(rows, columns)

# ------------------------------------------------------------- standings --

if section == SECTION_SEASON:
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
                # No medal emoji on the top three: at the size a table cell
                # renders them they're noise next to the number, and the leader
                # row already carries a highlighted background.
                return "—" if pd.isna(pos) else str(int(pos))

            def team_color(name):
                try:
                    return fastf1.plotting.get_team_color(name, session)
                except Exception:
                    return "#999999"

            def driver_color(code):
                return safe_driver_color(code, session)

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
                st.plotly_chart(fig_driver_progress, width="stretch", config=PLOTLY_CONFIG)

            leader_points = driver_standings["points"].max()
            driver_rows = []
            for _, r in driver_standings.iterrows():
                code = r["driverCode"] if pd.notna(r["driverCode"]) else ""
                color = driver_color(code)
                driver_rows.append({
                    "badge": driver_number_badge(
                        int(r["driverNumber"]) if pd.notna(r["driverNumber"]) else "—", color,
                    ),
                    "pos": pos_label(r["position"]),
                    "name": code or f"{r['givenName']} {r['familyName']}",
                    "team": r["constructorNames"][0] if len(r["constructorNames"]) else "—",
                    "points": points_cell(r["points"], leader_points, color),
                    "wins": int(r["wins"]),
                    "color": color,
                })
            render_table(driver_rows, [
                ("pos", "Pos", "pos"), ("name", "Driver", "name"), ("team", "Team", "team"),
                ("points", "Points", "points"), ("wins", "Wins", "num"),
            ])

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
                st.plotly_chart(fig_constructor_progress, width="stretch", config=PLOTLY_CONFIG)

            leader_team_points = constructor_standings["points"].max()
            constructor_rows = [
                {
                    "pos": pos_label(r["position"]),
                    "name": r["constructorName"],
                    "points": points_cell(r["points"], leader_team_points, team_color(r["constructorName"])),
                    "wins": int(r["wins"]),
                    "color": team_color(r["constructorName"]),
                }
                for _, r in constructor_standings.iterrows()
            ]
            render_table(constructor_rows, [
                ("pos", "Pos", "pos"), ("name", "Constructor", "name"),
                ("points", "Points", "points"), ("wins", "Wins", "num"),
            ])

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
                st.plotly_chart(
                    odds_chart("Drivers", names, [odds[n] for n in names], colors),
                    width="stretch", config=PLOTLY_CONFIG,
                )

            with col_constructors_odds:
                top3_constructors = constructor_standings.sort_values("position").head(3)
                names_c = top3_constructors["constructorName"].tolist()
                histories_c = [race_history(constructor_progress, "Constructor", n) for n in names_c]
                odds_c = simulate_top3_odds(names_c, top3_constructors["points"].tolist(), histories_c, remaining_rounds)
                colors_c = [team_color(n) for n in names_c]
                st.plotly_chart(
                    odds_chart("Constructors", names_c, [odds_c[n] for n in names_c], colors_c),
                    width="stretch", config=PLOTLY_CONFIG,
                )

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

if section == SECTION_SEASON:
    with tab_season_stats:
        st.caption(
            f"Computed across all {len(schedule)} completed races of the {year} season so far -- "
            "first load is slow (every race's full session gets fetched once), instant after that."
        )
        stats_df, driver_overtakes = season_stats(year, tuple(schedule["EventName"].tolist()))

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

            recovery_df = stats_df.dropna(subset=["RecoveryPlaces"])
            if not recovery_df.empty:
                best_recovery = recovery_df.loc[recovery_df["RecoveryPlaces"].idxmax()]
                st.metric("Best recovery drive", f"+{int(best_recovery['RecoveryPlaces'])} places")
                st.caption(f"📍 {best_recovery['RecoveryDriver']} -- {best_recovery['Event']}")

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
            fig_overtakes_season.update_layout(showlegend=False)
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
            size_horizontal_bars(fig_overtakes_season, ordered_races["Overtakes"].to_numpy())
            st.plotly_chart(fig_overtakes_season, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Overtakes by driver (season)")
            ranked_driver_overtakes = sorted(driver_overtakes.items(), key=lambda kv: kv[1], reverse=True)
            ranked_driver_overtakes = [(d, c) for d, c in ranked_driver_overtakes if c > 0]
            if not ranked_driver_overtakes:
                st.info("No on-track position gains detected across this season's races yet.")
            else:
                fig_driver_overtakes = base_figure("Overtakes by driver", "Position gains", "")
                fig_driver_overtakes.update_layout(showlegend=False)
                fig_driver_overtakes.add_trace(
                    go.Bar(
                        x=[c for _, c in ranked_driver_overtakes],
                        y=[d for d, _ in ranked_driver_overtakes],
                        orientation="h",
                        marker_color=[safe_driver_color(d, session) for d, _ in ranked_driver_overtakes],
                        text=[f"{d}: {c}" for d, c in ranked_driver_overtakes],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
                fig_driver_overtakes.update_yaxes(autorange="reversed")
                size_horizontal_bars(fig_driver_overtakes, [c for _, c in ranked_driver_overtakes])
                st.plotly_chart(fig_driver_overtakes, width="stretch", config=PLOTLY_CONFIG)

# ------------------------------------------------------------ all-time --

if section == SECTION_ALL_TIME:
    with tab_all_time:
        st.caption(
            "Every world-championship race since 1950. First load is slow (the whole history gets "
            "fetched once), instant after that."
        )

        try:
            wins = all_time_finishes(results_position=1)
            poles = all_time_finishes(grid_position=1)
        except Exception as exc:
            st.error(f"Couldn't load the historical results: {exc}")
        else:
            wins["Driver"] = wins.apply(full_name, axis=1)
            poles["Driver"] = poles.apply(full_name, axis=1)
            wins_by_driver = wins["Driver"].value_counts()
            poles_by_driver = poles["Driver"].value_counts()
            wins_by_team = wins["constructorName"].value_counts()

            # Everything below comes out of the same two downloads: every race
            # winner and every pole sitter already carry their grid slot, date
            # of birth and race date, so the streaks, the season counts and the
            # age records need no further requests.
            chronological = wins.sort_values(["season", "round"]).reset_index(drop=True)
            ages = (
                pd.to_datetime(chronological["raceDate"]) - pd.to_datetime(chronological["dateOfBirth"])
            ).dt.days / 365.25
            youngest = chronological.loc[ages.idxmin()]
            oldest = chronological.loc[ages.idxmax()]
            from_grid = chronological[chronological["grid"] > 0]
            comeback = from_grid.loc[from_grid["grid"].idxmax()]
            pole_wins = int((chronological["grid"] == 1).sum())

            col1, col2, col3 = st.columns(3)
            col1.metric("Most wins", f"{int(wins_by_driver.iloc[0])}")
            col1.caption(f"🏆 {wins_by_driver.index[0]}")
            col2.metric("Most poles", f"{int(poles_by_driver.iloc[0])}")
            col2.caption(f"🏁 {poles_by_driver.index[0]}")
            col3.metric("Most wins, constructor", f"{int(wins_by_team.iloc[0])}")
            col3.caption(f"🔧 {wins_by_team.index[0]}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Youngest winner", f"{ages.min():.1f} years")
            col4.caption(f"👶 {youngest['Driver']} — {int(youngest['season'])} {youngest['raceName']}")
            col5.metric("Oldest winner", f"{ages.max():.1f} years")
            col5.caption(f"🎩 {oldest['Driver']} — {int(oldest['season'])} {oldest['raceName']}")
            col6.metric("Win from furthest back", f"P{int(comeback['grid'])}")
            col6.caption(f"🚀 {comeback['Driver']} — {int(comeback['season'])} {comeback['raceName']}")

            st.metric("Wins started from pole", f"{100 * pole_wins / len(chronological):.0f}%")
            st.caption(f"🎯 {pole_wins} of {len(chronological)} races — pole is worth well under a coin flip")

            # Career points totals are deliberately absent: F1 has rescored
            # itself repeatedly (8 points for a win until 1960, 9, then 10, then
            # 25 from 2010, half points for short races, best-N-results dropped
            # scores for decades), so an all-time points table ranks eras rather
            # than drivers. Counts of wins and poles mean the same thing in
            # every season.
            st.caption(
                "Ranked by wins and poles rather than career points: the points system has been "
                "rewritten often enough (8 for a win in the fifties, 25 today, dropped scores for "
                "decades) that an all-time points table would rank eras, not drivers. Poles are "
                "counted as starts from P1, so a grid penalty can move one."
            )

            st.divider()

            st.subheader("Most race wins")
            top_wins = wins_by_driver.head(15).sort_values()
            fig_wins = base_figure("Race wins (all-time top 15)", "", "Wins")
            fig_wins.update_layout(showlegend=False)
            fig_wins.add_trace(
                go.Bar(
                    x=top_wins.to_numpy(), y=top_wins.index, orientation="h",
                    marker=dict(
                        color=top_wins.to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#5b8cff"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=top_wins.to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} wins<extra></extra>",
                )
            )
            size_horizontal_bars(fig_wins, top_wins.to_numpy())
            st.plotly_chart(fig_wins, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Most pole positions")
            top_poles = poles_by_driver.head(15).sort_values()
            fig_poles_all = base_figure("Pole positions (all-time top 15)", "", "Poles")
            fig_poles_all.update_layout(showlegend=False)
            fig_poles_all.add_trace(
                go.Bar(
                    x=top_poles.to_numpy(), y=top_poles.index, orientation="h",
                    marker=dict(
                        color=top_poles.to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#b57bff"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=top_poles.to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} poles<extra></extra>",
                )
            )
            size_horizontal_bars(fig_poles_all, top_poles.to_numpy())
            st.plotly_chart(fig_poles_all, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Most wins by constructor")
            top_team_wins = wins_by_team.head(15).sort_values()
            fig_team_wins = base_figure("Constructor wins (all-time top 15)", "", "Wins")
            fig_team_wins.update_layout(showlegend=False)
            fig_team_wins.add_trace(
                go.Bar(
                    x=top_team_wins.to_numpy(), y=top_team_wins.index, orientation="h",
                    marker=dict(
                        color=top_team_wins.to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#2ee86e"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=top_team_wins.to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} wins<extra></extra>",
                )
            )
            size_horizontal_bars(fig_team_wins, top_team_wins.to_numpy())
            st.plotly_chart(fig_team_wins, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Most wins in a single season")
            st.caption(
                "With the share of that season's races alongside the count -- the calendar has "
                "grown from 7 races to 24, so 13 wins from 18 starts is a bigger season than the "
                "raw number makes it look next to 19 from 22."
            )
            races_per_season = chronological.groupby("season").size()
            season_wins = chronological.groupby(["season", "Driver"]).size().sort_values(ascending=False)
            top_season_wins = season_wins.head(12).sort_values()
            season_labels = [f"{driver} · {int(season)}" for season, driver in top_season_wins.index]
            season_shares = [
                100 * count / races_per_season[season]
                for (season, _), count in top_season_wins.items()
            ]
            fig_season_wins = base_figure("Wins in one season (top 12)", "", "Wins")
            fig_season_wins.update_layout(showlegend=False)
            fig_season_wins.add_trace(
                go.Bar(
                    x=top_season_wins.to_numpy(), y=season_labels, orientation="h",
                    marker=dict(
                        color=top_season_wins.to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#ffb340"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=[
                        f"{count}  ·  {share:.0f}%"
                        for count, share in zip(top_season_wins.to_numpy(), season_shares)
                    ],
                    textposition="outside", cliponaxis=False,
                    customdata=[
                        [races_per_season[season], share]
                        for (season, _), share in zip(top_season_wins.index, season_shares)
                    ],
                    hovertemplate="%{y}: %{x} of %{customdata[0]} races (%{customdata[1]:.0f}%)<extra></extra>",
                )
            )
            # Wider headroom than the default: this chart's bar labels carry the
            # share as well as the count, so they need more room to the right.
            size_horizontal_bars(fig_season_wins, top_season_wins.to_numpy(), pad_frac=0.3)
            st.plotly_chart(fig_season_wins, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Longest winning streaks")
            st.caption(
                "Consecutive races won, counted straight down the calendar and across season "
                "boundaries -- a streak doesn't reset in January."
            )
            # A new streak starts wherever the winner differs from the previous
            # race's, so a running count of those changes labels each streak.
            # The grouping key has to be renamed: it inherits the name "Driver"
            # from the column it's derived from, which collides with grouping by
            # the column itself.
            streak_id = (chronological["Driver"] != chronological["Driver"].shift()).cumsum().rename("streak")
            streaks = (
                chronological.groupby([streak_id, chronological["Driver"]])
                .size().rename("races").reset_index()
                .sort_values("races", ascending=False)
            )
            top_streaks = streaks.head(10).iloc[::-1]
            streak_labels = []
            for _, streak in top_streaks.iterrows():
                seasons = chronological.loc[streak_id == streak["streak"], "season"]
                span = f"{int(seasons.min())}" if seasons.min() == seasons.max() else f"{int(seasons.min())}–{int(seasons.max())}"
                streak_labels.append(f"{streak['Driver']} · {span}")
            fig_streaks = base_figure("Consecutive wins (top 10)", "", "Races in a row")
            fig_streaks.update_layout(showlegend=False)
            fig_streaks.add_trace(
                go.Bar(
                    x=top_streaks["races"].to_numpy(), y=streak_labels, orientation="h",
                    marker=dict(
                        color=top_streaks["races"].to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#ff5c8a"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=top_streaks["races"].to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} in a row<extra></extra>",
                )
            )
            size_horizontal_bars(fig_streaks, top_streaks["races"].to_numpy())
            st.plotly_chart(fig_streaks, width="stretch", config=PLOTLY_CONFIG)

            st.divider()

            st.subheader("Most poles by constructor")
            poles_by_team = poles["constructorName"].value_counts().head(12).sort_values()
            fig_team_poles = base_figure("Constructor poles (all-time top 12)", "", "Poles")
            fig_team_poles.update_layout(showlegend=False)
            fig_team_poles.add_trace(
                go.Bar(
                    x=poles_by_team.to_numpy(), y=poles_by_team.index, orientation="h",
                    marker=dict(
                        color=poles_by_team.to_numpy(), colorscale=[[0, "#2a2f45"], [1, "#b57bff"]],
                        line=dict(color="rgba(255,255,255,0.15)", width=1),
                    ),
                    text=poles_by_team.to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} poles<extra></extra>",
                )
            )
            size_horizontal_bars(fig_team_poles, poles_by_team.to_numpy())
            st.plotly_chart(fig_team_poles, width="stretch", config=PLOTLY_CONFIG)
