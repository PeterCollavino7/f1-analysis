"""
F1 session dashboard: head-to-head telemetry, and race pace / tyre
degradation, for any past race weekend and any of its actual sessions.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import colorsys
import datetime
import functools
import io
import logging
import os
import re
import shutil
import tempfile
import threading
import urllib.parse
import zipfile
from contextlib import contextmanager

import streamlit as st
import streamlit.components.v1 as components

# "auto": open on a desktop, closed on a phone -- forced open, the sidebar
# covered four fifths of a phone screen and the charts peeked out beside it.
# Tab icon: the sidebar's checkered flag on a red pole (assets/icon.svg is the
# source; the PNG is what browsers get). Without it the tab shows Streamlit's own.
st.set_page_config(
    page_title="F1 Dashboard",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png"),
    layout="wide",
    initial_sidebar_state="auto",
)
# Streamlit's top-right "running" icon (a boxed bike/runner glyph) reads as a
# stray UI element against this page's own dark theme -- hidden rather than
# restyled, since which icon it is isn't under our control, only whether it
# shows at all. This one rule is sent before the heavy imports below, not with
# the main stylesheet: the icon appears the moment a run starts, and on the
# hosted app importing fastf1/pandas/plotly takes long enough that it sat
# there, visible, on every page open until the stylesheet finally arrived.
st.markdown('<style>[data-testid="stStatusWidget"] { display: none; }</style>', unsafe_allow_html=True)

import fastf1
import fastf1.plotting
from fastf1.ergast import Ergast
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ghost_lap import ghost_lap_html, ghost_lap_payload

# FastF1 refuses to start on a cache directory that doesn't exist, and on a
# fresh deploy (Streamlit Community Cloud) nothing has created it yet -- the
# folder is gitignored, and the disk it lives on is wiped on every restart.
os.makedirs("cache", exist_ok=True)
fastf1.Cache.enable_cache("cache")


# FastF1 keeps its HTTP cache in one SQLite database, and Streamlit runs every
# visitor -- and every rerun -- on its own thread. Two of them loading at once
# (two tabs, or a page opened while another was still loading) hit that
# database from two threads, and on the hosted app that killed the whole
# process with a segmentation fault. Every call into FastF1 or Ergast that
# can touch the network or the cache goes through this one process-wide
# lock. It wraps those calls only, never a Streamlit-cached function: those
# have locks of their own, and holding this one while waiting on one of
# them could deadlock two threads against each other.
@st.cache_resource
def _fastf1_lock():
    return threading.RLock()


FF1_LOCK = _fastf1_lock()

MAX_DRIVERS = 2
CHART_HEIGHT = 360
KM_TO_MI = 0.621371
M_TO_FT = 3.28084
# Plotly's hover toolbar (camera / zoom / pan / reset) floats over the
# top-right of every chart, overlapping titles and adding controls none of
# these charts need -- the zoom that matters (drag on the telemetry panels)
# works without it.
PLOTLY_CONFIG = {"displayModeBar": False}


def on_phone():
    try:
        agent = st.context.headers.get("User-Agent", "")
    except Exception:
        return False
    return "Mobi" in agent or "Android" in agent


def plotly_chart(fig, **kwargs):
    """st.plotly_chart, fixed up for phones. A legend down the right-hand side
    (and the wide right margin kept for it) took half of a 390 px screen and
    squeezed the plot into a sliver; on a phone it goes under the chart."""
    if on_phone():
        legend = fig.layout.legend
        if fig.layout.showlegend is not False and legend.orientation != "h":
            fig.update_layout(legend=dict(orientation="h", x=0, xanchor="left", y=-0.18, yanchor="top"))
            if (fig.layout.margin.r or 0) > 60:
                fig.update_layout(margin=dict(r=16))
            fig.update_layout(margin=dict(b=max(fig.layout.margin.b or 0, 90)))
    return st.plotly_chart(fig, **kwargs)
# Bumped whenever detect_passes() changes: it's an argument of season_stats(),
# so a new counting rule can't be served from a day-old cache of the old one
# (a code push doesn't clear Streamlit's cache on the hosted app).
OVERTAKE_RULES = 5
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
# The letter in a tyre ring. Up here with the colours rather than beside the
# classification: the Pace tab's long-run table draws rings too, and it runs
# earlier in the script than the function that used to define this.
COMPOUND_LETTER = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}

# The same values as the CSS custom properties in the stylesheet above --
# Plotly draws its figures from Python and never sees the page's CSS, so the
# palette has to exist on both sides. Anything that changes here changes in
# :root too, or a chart and the card around it stop matching.
PALETTE = {
    "red": "#e10600",
    "blue": "#5b8cff",
    "teal": "#2ee86e",
    "amber": "#ffb340",
    "violet": "#b57bff",
    "pink": "#ff5c8a",
    "ink": "#eef1f5",
    "ink_dim": "rgba(226,232,240,0.62)",
    "ink_faint": "rgba(226,232,240,0.40)",
    "grid": "rgba(255,255,255,0.055)",
    "axis": "rgba(255,255,255,0.14)",
    "track": "#2a2f45",  # the unfilled end of every sequential bar scale
}

# Sequential scales for the ranking charts: every one runs from the same dark
# "track" color up to one accent, so a chart's identity is its hue and the
# length of a bar always means the same thing.
def bar_scale(color):
    return [[0, PALETTE["track"]], [1, color]]


# Every chart was built without a template or font color, so it inherited
# Plotly's default *light-theme* text (near-black) on top of this app's dark
# background -- titles, axis labels and legend text were all barely legible.
# plotly_dark fixes that (light text, matching gridlines/hover boxes) and
# this dict is applied to every figure so none of them drift back out of sync.
DARK_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="'Titillium Web', 'Segoe UI', Inter, -apple-system, sans-serif",
              color="rgba(226,232,240,0.78)", size=12),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    # Plotly's stock hover box is a light-bordered slab with the trace color
    # bleeding into it; this matches the cards the charts sit in -- dark
    # panel, hairline border, the app's own typeface.
    hoverlabel=dict(
        bgcolor="rgba(13,16,24,0.95)",
        bordercolor="rgba(255,255,255,0.16)",
        font=dict(family="'Titillium Web', 'Segoe UI', sans-serif", size=12.5, color="#eef1f5"),
        align="left",
    ),
    colorway=["#5b8cff", "#2ee86e", "#ffb340", "#b57bff", "#ff5c8a", "#e10600"],
)

# FastF1's timing/telemetry data is only reliably complete from 2018 on. The
# top of the range follows the calendar rather than being written in, so a new
# season shows up on its own -- see available_years() for when.
FIRST_YEAR = 2018

st.markdown(
    """
    <style>
    /* Titillium Web is the closest freely-licensed match to Formula 1's own
       display face (which is proprietary and can't ship here); JetBrains Mono
       carries every lap time, gap and sector split, where fixed-width digits
       are what keeps a column of times from dancing. Both degrade to the
       system stack if the font request fails (offline, blocked CDN) -- nothing
       below depends on them loading. */
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:ital,wght@0,400;0,600;0,700;0,900;1,900&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* One place for every color the app uses. Plotly can't read CSS custom
       properties (its figures are drawn from Python, not styled by the page),
       so the same values are mirrored in the PALETTE dict below -- a color
       that changes here has to change there too. */
    :root {
        --f1-red: #e10600;
        --f1-blue: #5b8cff;
        --f1-teal: #2ee86e;
        --f1-amber: #ffb340;
        --ink: #eef1f5;
        --ink-dim: rgba(226, 232, 240, 0.62);
        --ink-faint: rgba(226, 232, 240, 0.40);
        --line: rgba(255, 255, 255, 0.075);
        --line-strong: rgba(255, 255, 255, 0.14);
        --display: 'Titillium Web', 'Segoe UI', Inter, -apple-system, sans-serif;
        --mono: 'JetBrains Mono', ui-monospace, 'SFMono-Regular', Consolas, monospace;
    }

    /* The page background is three layers, not one flat color: a wide red
       glow anchored top-left (the accent, kept far from the content so it
       tints rather than competes), a cool counter-glow bottom-right so the
       canvas doesn't read as one muddy gradient, and a faint 40px grid over
       the top that gives large empty areas some texture without ever being
       legible as lines behind a chart. */
    .stApp {
        background:
            radial-gradient(90% 60% at 0% -10%, rgba(225, 6, 0, 0.16) 0%, transparent 60%),
            radial-gradient(70% 50% at 100% 100%, rgba(91, 140, 255, 0.10) 0%, transparent 65%),
            linear-gradient(180deg, #0c0f17 0%, #090b11 100%);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            linear-gradient(rgba(255, 255, 255, 0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.018) 1px, transparent 1px);
        background-size: 40px 40px;
        mask-image: radial-gradient(75% 60% at 50% 0%, #000 0%, transparent 100%);
    }
    html, body, [class*="st-"], button, input, select, textarea { font-family: var(--display); }
    /* ...but not the icons. Streamlit draws its chevrons, arrows and status
       glyphs as ligatures in a Material Symbols font, and the broad rule above
       matches them too (they carry an st-emotion-cache class), which replaced
       every icon in the app with its own name spelled out -- the expander
       arrow rendered as the literal text "keyboard_arrow_right". */
    [data-testid="stIconMaterial"], .material-icons, [class*="material-symbols"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }
    .block-container { padding-top: 2.4rem; max-width: 1500px; }
    h1, h2, h3, h4 { font-family: var(--display); letter-spacing: -0.01em; }

    /* The stock header bar is an empty translucent strip above the hero: on a
       local run it holds no menu and no deploy button, and clearing its
       background lets the hero sit at the very top of the page. */
    [data-testid="stHeader"] { background: transparent; }
    .table-scroll { width: 100%; overflow-x: auto; }

    /* The Deploy button is Streamlit Cloud's call to action; this app runs on
       a laptop and has nowhere to deploy to, so it's dead weight sitting over
       the hero. Same reasoning as the status widget (hidden at the top of
       this file) -- and as there,
       only its visibility is touched, so a selector that stops matching costs
       a stray button, nothing more. The hamburger menu next to it stays:
       rerun, settings and "clear cache" are all genuinely useful here. */
    [data-testid="stAppDeployButton"] { display: none; }

    /* ------------------------------------------------------------ sidebar */
    /* Restyled with background, border, spacing and typography, selected by
       test id and ARIA attributes. The one thing hidden is the radio dot on
       the navigation, which is decoration (the row itself is the control);
       if that selector ever stops matching, the dot simply comes back. */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(130% 35% at 0% 0%, rgba(225, 6, 0, 0.10) 0%, transparent 70%),
            linear-gradient(180deg, #131824 0%, #0b0e16 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.75rem; }
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        text-transform: uppercase; letter-spacing: 0.13em;
        font-size: 0.6rem; font-weight: 700; color: var(--ink-faint);
    }

    /* Navigation: full-width rows with an icon, a name and one line on
       what's there. The icons are masks filled with the text color, so the
       chosen row's icon turns red with it. */
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stRadio"]),
    [data-testid="stSidebar"] [data-testid="stRadio"],
    [data-testid="stSidebar"] [role="radiogroup"] { width: 100% !important; }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        position: relative; width: 100%; margin: 0;
        padding: 0.55rem 0.75rem 0.6rem 2.85rem;
        border-radius: 12px; border: 1px solid transparent;
        text-transform: none; letter-spacing: normal;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div > div:first-child:not([data-testid]) {
        display: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]::before {
        content: ""; position: absolute; left: 0.85rem; top: 0.72rem; width: 19px; height: 19px;
        background: var(--ink-faint);
        -webkit-mask: var(--icon) center / contain no-repeat; mask: var(--icon) center / contain no-repeat;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(1) {
        --icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M5 21V4'/><path d='M5 4h13l-2.5 4.5L18 13H5'/></svg>");
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(2) {
        --icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M3 20h18'/><path d='M6 20v-6'/><path d='M11 20V8'/><path d='M16 20v-9'/><path d='M21 20V4'/></svg>");
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:nth-of-type(3) {
        --icon: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M8 21h8'/><path d='M12 17v4'/><path d='M7 4h10v5a5 5 0 0 1-10 0z'/><path d='M17 6h3v1.5A3.5 3.5 0 0 1 16.5 11'/><path d='M7 6H4v1.5A3.5 3.5 0 0 0 7.5 11'/></svg>");
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {
        margin: 0; font-size: 0.93rem; font-weight: 600; letter-spacing: 0.01em; color: var(--ink-dim);
        text-transform: none;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stCaptionContainer"] {
        margin: 0.05rem 0 0; font-size: 0.72rem; line-height: 1.35; color: var(--ink-faint);
        text-transform: none; letter-spacing: normal; font-weight: 400;
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: rgba(255, 255, 255, 0.045); }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover::before { background: var(--ink-dim); }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover [data-testid="stMarkdownContainer"] p { color: var(--ink); }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:is([data-selected="true"], :has(input:checked)) {
        background: linear-gradient(90deg, rgba(225, 6, 0, 0.20), rgba(225, 6, 0, 0.03) 80%);
        border-color: rgba(225, 6, 0, 0.32);
        box-shadow: inset 3px 0 0 var(--f1-red);
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:is([data-selected="true"], :has(input:checked))::before {
        background: var(--f1-red);
    }
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:is([data-selected="true"], :has(input:checked)) [data-testid="stMarkdownContainer"] p {
        color: #fff;
    }

    /* The Year / Grand Prix fields. react-aria draws them now (1.63), so the
       old baseweb selector here matched nothing and they rendered stock. */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
        background: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        border-radius: 10px !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"]:hover { border-color: rgba(255, 255, 255, 0.24) !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"]:focus-within {
        border-color: rgba(225, 6, 0, 0.7) !important; box-shadow: 0 0 0 3px rgba(225, 6, 0, 0.16);
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] input { font-weight: 600; font-size: 0.9rem; min-width: 0; }
    .data-status { padding-left: 0.2rem; }

    /* Session buttons: one connected strip, the chosen one filled red. */
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button {
        min-height: 36px; padding: 0 0.2rem;
        background: rgba(255, 255, 255, 0.03); border-color: rgba(255, 255, 255, 0.10);
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button p {
        font-family: var(--display); font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em; color: var(--ink-dim);
    }
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button:hover { background: rgba(255, 255, 255, 0.07); }
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button:is([aria-checked="true"], [aria-pressed="true"]) {
        background: linear-gradient(180deg, #ef1a12, #c80500); border-color: #e10600;
        box-shadow: 0 6px 16px -8px rgba(225, 6, 0, 0.9);
    }
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button:is([aria-checked="true"], [aria-pressed="true"]) p { color: #fff; }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.1rem 0.15rem 1rem;
        margin-bottom: 0.2rem;
        border-bottom: 1px solid var(--line);
    }
    .sidebar-brand .flag {
        width: 26px; height: 26px; flex: 0 0 auto; border-radius: 6px;
        background-image:
            linear-gradient(45deg, #e8ebef 25%, transparent 25%, transparent 75%, #e8ebef 75%),
            linear-gradient(45deg, #e8ebef 25%, #12151d 25%, #12151d 75%, #e8ebef 75%);
        background-size: 13px 13px;
        background-position: 0 0, 6.5px 6.5px;
        box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.12), 0 4px 14px rgba(0, 0, 0, 0.55);
    }
    .sidebar-brand .word {
        font-size: 1.12rem; font-weight: 700; letter-spacing: 0.08em; color: #f2f4f6; line-height: 1.15;
    }
    .sidebar-brand .word em {
        font-style: italic; color: var(--f1-red); font-weight: 900; margin-right: 0.2rem;
    }
    .sidebar-brand .word span {
        /* Sized to fit the sidebar's width on one line -- at the first size
           tried it wrapped, and a two-line kicker under a one-line wordmark
           reads as a mistake. */
        display: block; font-size: 0.46rem; letter-spacing: 0.15em;
        color: var(--ink-faint); font-weight: 600; white-space: nowrap;
    }

    /* What's next on the calendar, and how fresh the data is. */
    .next-race {
        position: relative; overflow: hidden; margin-top: 0.35rem;
        padding: 0.75rem 0.85rem 0.8rem 1rem; border-radius: 12px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.015));
    }
    .next-race::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
        background: linear-gradient(180deg, var(--f1-red), rgba(225, 6, 0, 0.15));
    }
    .next-race .kicker {
        display: flex; align-items: center; gap: 0.45rem;
        font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--ink-faint);
    }
    .next-race .kicker i { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-faint); flex: none; }
    .next-race.live .kicker { color: #ff7a73; }
    .next-race.live .kicker i { background: var(--f1-red); animation: live-pulse 1.8s ease-out infinite; }
    @keyframes live-pulse {
        0% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0.6); }
        70% { box-shadow: 0 0 0 7px rgba(225, 6, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0); }
    }
    @media (prefers-reduced-motion: reduce) { .next-race.live .kicker i { animation: none; } }
    .next-race .name { margin-top: 0.3rem; font-size: 1rem; font-weight: 700; line-height: 1.2; color: #f4f6f8; }
    .next-race .detail { margin-top: 0.2rem; font-size: 0.76rem; color: var(--ink-dim); }
    .data-status {
        display: flex; align-items: center; gap: 0.5rem; margin-top: 0.7rem;
        font-size: 0.72rem; color: var(--ink-faint);
    }
    .data-status i {
        width: 7px; height: 7px; border-radius: 50%; flex: none;
        background: var(--f1-teal); box-shadow: 0 0 8px var(--f1-teal);
    }
    .data-status b { color: var(--ink-dim); font-weight: 600; }
    .data-status.late { margin-top: 0.35rem; color: var(--f1-amber); }
    .data-status.late i { background: var(--f1-amber); box-shadow: 0 0 8px var(--f1-amber); }
    [data-testid="stSidebar"] [data-testid="stExpander"] { background: transparent; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: 0.74rem; }
    .sidebar-foot { font-size: 0.72rem; line-height: 1.75; color: var(--ink-faint); }
    .sidebar-foot b { color: var(--ink-dim); font-weight: 600; }

    /* --------------------------------------------------------------- hero */
    /* Stock photography of an F1 car would be someone else's copyright, so
       the artwork is the circuit itself, traced from the session's own
       position telemetry -- it changes with the race being looked at, which
       a stock photo wouldn't. */
    .hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        overflow: hidden;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.2rem;
        border-radius: 18px;
        border: 1px solid var(--line);
        background:
            radial-gradient(120% 180% at 0% 0%, rgba(225, 6, 0, 0.24) 0%, transparent 55%),
            linear-gradient(135deg, #191f2d 0%, #0f121b 60%);
        box-shadow: 0 18px 40px -24px rgba(0, 0, 0, 0.9);
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
    /* The red edge down one side that every F1 broadcast graphic has. */
    .hero::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0; width: 4px;
        background: linear-gradient(180deg, var(--f1-red), rgba(225, 6, 0, 0.08));
    }
    .hero-text { position: relative; z-index: 1; min-width: 0; }
    .hero-kicker {
        display: inline-flex; align-items: center; gap: 0.55rem;
        text-transform: uppercase; letter-spacing: 0.16em;
        font-size: 0.66rem; font-weight: 700;
        color: rgba(255, 255, 255, 0.55);
        margin-bottom: 0.5rem;
    }
    .hero-kicker .pill {
        padding: 0.18rem 0.55rem; border-radius: 999px;
        background: var(--f1-red); color: #fff; letter-spacing: 0.1em;
        box-shadow: 0 4px 14px -4px rgba(225, 6, 0, 0.9);
    }
    .hero-title {
        font-size: 2.25rem; font-weight: 900; line-height: 1.05; color: #f6f8fa;
        letter-spacing: -0.02em; text-transform: uppercase;
    }
    .hero-sub { margin-top: 0.4rem; font-size: 0.92rem; color: var(--ink-dim); }
    .hero-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }
    .hero-chips span {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.28rem 0.65rem; border-radius: 999px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--line);
        font-size: 0.74rem; font-weight: 600; color: var(--ink-dim);
    }
    .hero-track { position: relative; z-index: 1; flex: 0 0 auto; }
    .hero-track svg {
        display: block; height: 140px; width: auto; max-width: 300px; overflow: visible;
        filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.12));
    }
    .hero-track .draw {
        stroke-dasharray: 1; stroke-dashoffset: 1;
        animation: hero-draw 1.8s cubic-bezier(0.6, 0.1, 0.2, 1) 0.15s forwards;
    }
    @keyframes hero-draw { to { stroke-dashoffset: 0; } }
    .hero-track .car {
        fill: #fff; filter: drop-shadow(0 0 5px #fff);
        opacity: 0; animation: hero-car 0.4s ease 2s forwards;
    }
    @keyframes hero-car { to { opacity: 1; } }
    .hero-track .speed-key {
        display: flex; align-items: center; justify-content: flex-end; gap: 0.4rem; margin-top: 0.55rem;
        font: 500 0.6rem var(--mono); color: var(--ink-faint);
    }
    .hero-track .speed-key i {
        width: 4.5rem; height: 4px; border-radius: 2px;
        background: linear-gradient(90deg, #3d6bff, #19c6d8, #ffd23f, #ff3b30);
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-track .draw { animation: none; stroke-dashoffset: 0; }
        .hero-track .car { display: none; }
    }

    /* --------------------------------------------------------------- tabs */
    /* Streamlit's own test ids and ARIA roles, not the emotion class names
       (st-emotion-cache-csawen and friends are content hashes that change on
       every Streamlit build) and not data-baseweb either -- 1.63 builds its
       tabs on react-aria, which emits data-testid="stTab" plus the standard
       role/aria-selected pair. Both were checked against the rendered DOM. */
    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.3rem;
        background: transparent;
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.5rem;
    }
    [data-testid="stTab"] {
        height: auto;
        padding: 0.55rem 1.1rem;
        border-radius: 10px 10px 0 0;
        color: var(--ink-dim);
        font-weight: 700;
        font-size: 0.84rem;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        transition: background 0.14s ease, color 0.14s ease;
    }
    /* The label is a markdown block inside the tab, and it carries its own
       color, so the rule has to reach it rather than stopping at the tab. */
    [data-testid="stTab"] [data-testid="stMarkdownContainer"] { color: inherit; font-weight: 700; }
    [data-testid="stTab"]:hover { background: rgba(255, 255, 255, 0.05); color: var(--ink); }
    [data-testid="stTab"][aria-selected="true"] {
        color: #fff;
        background: linear-gradient(180deg, rgba(225, 6, 0, 0.22), rgba(225, 6, 0, 0.01));
    }

    /* ------------------------------------------------------------- panels */
    /* Every chart sits in a card: a header strip (title, plus a one-line
       hint) and the figure under it. The wrapper is Streamlit's own bordered
       container, matched through :has(.panel-head) so only the containers
       that actually carry one of these headers get restyled -- the test id
       alone is on every vertical block, bordered or not. */
    /* st.container(border=True) renders as a bordered [data-testid=
       "stVerticalBlock"] -- there is no separate border-wrapper element in
       1.63, which is what an earlier version of this rule was aiming at. The
       :has() keeps it to the containers that carry one of these headers: the
       test id itself is on every vertical block on the page. The child
       combinator matters -- without it, an outer block containing a panel
       would match too and draw a card around the card. */
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .panel-head) {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.042), rgba(255, 255, 255, 0.012));
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 0.3rem 0.45rem 0.15rem;
        box-shadow: 0 14px 34px -26px rgba(0, 0, 0, 0.9);
    }
    .panel-head {
        display: flex; align-items: baseline; gap: 0.7rem; flex-wrap: wrap;
        padding: 0.55rem 0.7rem 0.65rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 0.25rem;
    }
    .panel-head .title {
        font-size: 1.0rem; font-weight: 700; color: #f2f4f6; letter-spacing: 0.01em;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .panel-head .title::before {
        content: ""; width: 3px; height: 16px; border-radius: 2px;
        background: var(--accent, var(--f1-red));
    }
    .panel-head .hint { font-size: 0.78rem; color: var(--ink-faint); font-weight: 400; }

    /* Section heading: the same accent rule as a panel, one size up, for the
       bands of a page that group several panels together. */
    .section-head { margin: 0.5rem 0 0.9rem; }
    .section-head .eyebrow {
        text-transform: uppercase; letter-spacing: 0.18em;
        font-size: 0.63rem; font-weight: 700; color: var(--ink-faint);
        margin-bottom: 0.25rem;
    }
    .section-head .heading {
        font-size: 1.35rem; font-weight: 700; color: #f4f6f8; letter-spacing: -0.01em;
        display: flex; align-items: center; gap: 0.6rem;
    }
    .section-head .heading::before {
        content: ""; width: 4px; height: 22px; border-radius: 2px;
        background: linear-gradient(180deg, var(--accent, var(--f1-red)), transparent);
    }
    .section-head .lede { font-size: 0.85rem; color: var(--ink-dim); margin-top: 0.35rem; max-width: 95ch; }

    /* ---------------------------------------------------------- stat cards */
    /* Hand-rolled rather than st.metric: the metric widget truncates any
       value wider than its column with an ellipsis instead of wrapping (a
       Grand Prix name never fit), which had forced every stat on the page
       into a number-plus-caption pair. A card that owns its own layout
       carries the label, the number and the context line together. */
    .stat-card {
        position: relative; overflow: hidden; height: 100%;
        padding: 0.85rem 1rem 0.9rem;
        border-radius: 14px;
        border: 1px solid var(--line);
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.012));
    }
    .stat-card::before {
        content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
        background: var(--card-accent, var(--f1-red));
    }
    .stat-card .label {
        text-transform: uppercase; letter-spacing: 0.12em;
        font-size: 0.62rem; font-weight: 700; color: var(--ink-faint);
    }
    .stat-card .value {
        font-size: 1.5rem; font-weight: 700; color: #f4f6f8; line-height: 1.15;
        margin-top: 0.3rem; font-variant-numeric: tabular-nums;
    }
    .stat-card .value small { font-size: 0.78rem; font-weight: 600; color: var(--ink-dim); }
    .stat-card .sub { margin-top: 0.35rem; font-size: 0.78rem; color: var(--ink-dim); line-height: 1.45; }
    .stat-card .sub b { color: var(--card-accent, var(--ink)); font-weight: 700; }

    /* ------------------------------------------------------ driver pills */
    /* Every pill picker whose key starts with "pick_" (driver_picker, the
       compound choice). Each pill's --pill color is set per button, in option
       order, by pill_colors(); everything else lives here. Selected is read
       from aria-pressed (multi-select pills) or aria-checked (single-select,
       which react-aria renders as a radio group). */
    [class*="st-key-pick_"] [role="toolbar"] { gap: 6px; }
    [class*="st-key-pick_"] button[data-variant] {
        display: inline-flex; align-items: center;
        min-height: 0; padding: 5px 13px 5px 10px;
        border-radius: 999px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--line-strong) !important;
        transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.08s ease;
    }
    [class*="st-key-pick_"] button[data-variant]::before {
        content: ""; flex: none;
        width: 8px; height: 8px; margin-right: 8px; border-radius: 50%;
        background: var(--pill, #999999);
        box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.35);
    }
    /* Titillium, the app's display face, rather than the mono used for
       timing figures: bold mono made three-letter codes read like terminal
       output, and these are names, not numbers. */
    [class*="st-key-pick_"] button[data-variant] p {
        font-family: var(--display); font-size: 0.86rem; font-weight: 600;
        letter-spacing: 0.07em; line-height: 1; color: var(--ink-dim);
    }
    [class*="st-key-pick_"] button[data-variant]:hover {
        border-color: var(--pill, #999999) !important;
        background: color-mix(in srgb, var(--pill, #999999) 10%, transparent) !important;
    }
    [class*="st-key-pick_"] button[data-variant]:hover p { color: var(--ink); }
    [class*="st-key-pick_"] button[data-variant]:active { transform: scale(0.96); }
    [class*="st-key-pick_"] button:is([aria-pressed="true"], [aria-checked="true"]) {
        border-color: var(--pill) !important;
        background: color-mix(in srgb, var(--pill) 26%, transparent) !important;
        box-shadow: 0 0 14px -4px var(--pill);
    }
    [class*="st-key-pick_"] button:is([aria-pressed="true"], [aria-checked="true"]) p { color: #ffffff; font-weight: 700; }
    @media (prefers-reduced-motion: reduce) {
        [class*="st-key-pick_"] button[data-variant] { transition: none; }
        [class*="st-key-pick_"] button[data-variant]:active { transform: none; }
    }

    /* ------------------------------------------------------- empty states */
    /* empty_state(): what a chart shows when there's nothing to draw. It
       replaced Streamlit's st.info, whose saturated blue box was the loudest
       thing on any page it appeared on -- for a message that is either "pick
       something" or "no data here", neither of which should outshout the
       charts. A prompt (the page is waiting on the reader) is a step
       brighter and points up at the picker; plain "no data" recedes. */
    .empty-state {
        display: flex; align-items: center; justify-content: center; gap: 0.6rem;
        padding: 1.5rem 1rem; margin: 0.25rem 0 0.5rem;
        border: 1px dashed var(--line-strong); border-radius: 12px;
        color: var(--ink-faint); font-size: 0.92rem; text-align: center;
    }
    .empty-state svg { flex: none; width: 17px; height: 17px; opacity: 0.8; }
    .empty-state.prompt { color: var(--ink-dim); border-color: rgba(255, 255, 255, 0.2); }
    .empty-state.note {
        justify-content: flex-start; padding: 0.55rem 0.9rem; border-style: solid;
        border-color: rgba(255, 179, 64, 0.25); color: rgba(255, 179, 64, 0.85); font-size: 0.86rem;
    }

    /* ---------------------------------------------------- teammate table */
    /* The Teammates overview: one row per pair, each battle a split bar with
       its two counts outside it, so nothing has to fit inside a sliver of
       bar or share a line with a label (the chart version it replaced
       overprinted both at anything below full width). */
    /* Fixed layout with proportional columns: with a min-width on every
       split bar, the table outgrew its panel on anything narrower than a
       wide desktop and the last column hung over the edge. The wrapper
       scrolls sideways instead on a phone. */
    .tm-scroll { width: 100%; overflow-x: auto; }
    .tm-table { width: 100%; min-width: 640px; table-layout: fixed; border-collapse: collapse; font-size: 0.9rem; }
    .tm-table col.c-team { width: 15%; }
    .tm-table col.c-pair { width: 14%; }
    .tm-table col.c-split { width: 19%; }
    .tm-table col.c-gap { width: 14%; }
    .tm-table th {
        text-align: left; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--ink-faint); padding: 0 10px 8px;
    }
    .tm-table td { padding: 9px 10px; border-top: 1px solid var(--line); vertical-align: middle; overflow: hidden; }
    .tm-team { display: flex; align-items: center; gap: 9px; color: var(--ink-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .tm-dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
    .tm-pair { font-family: var(--display); font-weight: 700; letter-spacing: 0.04em; white-space: nowrap; }
    .tm-pair span { color: var(--ink-faint); font-weight: 400; margin: 0 5px; }
    .tm-split { display: flex; align-items: center; gap: 6px; min-width: 0; }
    .tm-split b { min-width: 2.2em; text-align: center; white-space: nowrap; font-family: var(--mono); font-weight: 500; }
    .tm-bar { flex: 1; min-width: 24px; display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--line); }
    .tm-bar i { display: block; height: 100%; }
    .tm-gap { font-family: var(--mono); font-size: 0.82rem; color: var(--ink-dim); white-space: nowrap; }

    /* ------------------------------------------------------- pit stop list */
    /* DHL's stationary times, as a ranked list: the time is the thing to
       read, so it's the biggest element on the row. */
    .pit-list { display: flex; flex-direction: column; }
    .pit-row {
        display: grid; grid-template-columns: 2.6rem 5.2rem 4px 1fr auto; align-items: center;
        gap: 0.8rem; padding: 0.55rem 0.2rem; border-top: 1px solid var(--line);
    }
    .pit-row:first-child { border-top: none; }
    .pit-rank { font-family: var(--mono); font-size: 0.82rem; color: var(--ink-faint); text-align: right; }
    .pit-time { font-family: var(--mono); font-size: 1.25rem; font-weight: 700; color: var(--ink); }
    .pit-time small { font-size: 0.75rem; font-weight: 400; color: var(--ink-faint); margin-left: 2px; }
    .pit-bar { width: 4px; height: 1.6rem; border-radius: 2px; }
    .pit-who { min-width: 0; overflow: hidden; }
    .pit-who b { display: block; font-family: var(--display); letter-spacing: 0.02em; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis; }
    .pit-who span { display: block; font-size: 0.8rem; color: var(--ink-faint); }
    .pit-extra { font-size: 0.8rem; color: var(--ink-dim); text-align: right; white-space: nowrap;
                 max-width: 14rem; overflow: hidden; text-overflow: ellipsis; }
    .pit-row.best .pit-time { color: var(--f1-amber); }
    .pit-source { font-size: 0.78rem; color: var(--ink-faint); margin-top: 0.4rem; }
    .pit-source a { color: var(--ink-dim); }

    /* ------------------------------------------------------- driver cards */
    .driver-card {
        position: relative; overflow: hidden;
        border-radius: 14px;
        padding: 0.95rem 1.15rem;
        background: linear-gradient(120deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.015));
        border: 1px solid var(--line);
        border-left: 4px solid var(--card-color);
    }
    /* A wash of the driver's own color fading out across the card -- enough
       to tell two cards apart at a glance without tinting the text. */
    .driver-card::after {
        content: ""; position: absolute; inset: 0; pointer-events: none;
        background: linear-gradient(100deg, var(--card-color) -60%, transparent 55%);
        opacity: 0.16;
    }
    .driver-card .name {
        font-size: 1.3rem; font-weight: 900; color: var(--card-color);
        letter-spacing: 0.04em; text-transform: uppercase;
    }
    .driver-card .laptime {
        font-family: var(--mono); font-size: 1.45rem; font-weight: 700;
        margin-top: 0.1rem; color: #f4f6f8; letter-spacing: -0.02em;
    }
    .driver-card .sub { color: var(--ink-dim); font-size: 0.78rem; margin-top: 0.3rem; }

    /* -------------------------------------------------------------- notes */
    /* The long methodology text (how an overtake is counted, what the Monte
       Carlo actually does) is real and has to stay, but as a paragraph of
       body copy above a chart it was the loudest thing on the page. It now
       lives in a collapsed expander styled as a quiet note strip. */
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.022);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary { font-size: 0.8rem; color: var(--ink-dim); font-weight: 600; }
    [data-testid="stExpander"] summary:hover { color: var(--ink); }
    [data-testid="stExpander"] p { font-size: 0.82rem; color: var(--ink-dim); line-height: 1.65; }
    [data-testid="stCaptionContainer"] p { color: var(--ink-faint); }

    /* Legend chips under a chart (compound colors, who's faster) -- the same
       shape as the hero chips, so the page has one vocabulary for "small
       labelled token". */
    .chip-row { display: flex; flex-wrap: wrap; gap: 0.45rem; padding: 0.1rem 0.7rem 0.55rem; }
    /* Streamlit hangs a -16px bottom margin on every markdown container, to
       swallow the bottom margin of the <p> it normally holds. A chip row is a
       bare flex div with no such margin, so that -16px came straight off the
       height its card is measured from and the chips hung out past the border.
       Zeroing it on the containers that hold one puts the height back -- and
       stays right if Streamlit ever changes the value. */
    [data-testid="stMarkdownContainer"]:has(> .chip-row) { margin-bottom: 0 !important; }
    .chip {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.24rem 0.65rem; border-radius: 999px;
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid var(--line);
        font-size: 0.74rem; font-weight: 600; color: var(--ink-dim);
    }
    .chip i {
        width: 9px; height: 9px; border-radius: 50%; display: inline-block;
        background: var(--chip-color); box-shadow: 0 0 8px -1px var(--chip-color);
    }

    /* The loading indicator is styled as a panel, not replaced: an earlier
       version hid Streamlit's own glyph and drew a ring in its place, but
       the selector stopped matching and the two ended up side by side. The
       wording is fixed where it actually comes from -- every cached
       function below passes its own show_spinner text, since the default
       prints the function name at the reader ("Running load_session(...)"). */
    [data-testid="stSpinner"] {
        padding: 0.6rem 0.9rem;
        border-radius: 10px;
        border: 1px solid var(--line-strong);
        background: rgba(20, 24, 36, 0.92);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.45);
        width: fit-content;
    }
    [data-testid="stSpinner"] p { font-size: 0.9rem; color: rgba(232, 235, 239, 0.9); margin: 0; }

    /* ------------------------------------------------------------- tables */
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
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }
    table.standings th {
        text-align: left;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.64rem;
        font-weight: 700;
        color: var(--ink-faint);
        padding: 0 0.7rem 0.6rem;
        border-bottom: 1px solid var(--line-strong);
        white-space: nowrap;
    }
    table.standings td {
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.045);
        font-size: 0.95rem;
        color: var(--ink);
        vertical-align: middle;
    }
    table.standings tr:last-child td { border-bottom: none; }
    table.standings tr:hover td { background: rgba(255, 255, 255, 0.04); }
    table.standings td.pos {
        width: 2.8rem;
        font-weight: 700;
        font-size: 1rem;
        color: var(--ink-dim);
        border-left: 3px solid var(--row-accent, transparent);
    }
    table.standings tr.leader td { background: rgba(225, 6, 0, 0.09); }
    table.standings tr.leader td.name { font-weight: 700; }
    table.standings tr.leader td.pos { color: #fff; }
    table.standings td.badge { width: 3.6rem; padding-right: 0; }
    table.standings td.badge img { display: block; height: 30px; }
    table.standings td.name { font-weight: 700; letter-spacing: 0.02em; }
    table.standings td.team { color: var(--ink-dim); font-size: 0.86rem; }
    table.standings td.num { text-align: right; width: 3.4rem; }
    table.standings td.points { width: 42%; }
    table.standings td.mono {
        font-family: var(--mono);
        font-size: 0.85rem;
        color: rgba(232, 235, 239, 0.92);
    }
    table.standings td.status { color: var(--ink-faint); font-size: 0.82rem; }
    /* Qualifying sector map: each sector gap is a tile whose strength says
       how close it was to the session's best sector -- one blue ramp, bright
       near the best and fading out -- with the best itself in the timing
       screens' purple. */
    table.standings td.sector { text-align: center; padding-left: 0.3rem; padding-right: 0.3rem; }
    .sectile {
        display: inline-block; min-width: 4.9em; padding: 0.22rem 0.4rem; border-radius: 5px;
        font-family: var(--mono); font-size: 0.8rem; color: #eef1f5; background: var(--heat);
    }
    .sectile.best { background: #8e44ff; font-weight: 700; }
    /* Places gained/lost on the results table: a signed number that carries
       its own color instead of needing a legend. */
    table.standings td.delta { text-align: right; width: 4.6rem; font-weight: 700; font-size: 0.85rem; }
    table.standings td.delta .up { color: var(--f1-teal); }
    table.standings td.delta .down { color: #ff6b6b; }
    table.standings td.delta .flat { color: var(--ink-faint); }

    .points-wrap { display: flex; align-items: center; gap: 0.6rem; }
    .points-bar {
        flex: 1;
        height: 8px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        overflow: hidden;
    }
    .points-bar > span { display: block; height: 100%; border-radius: 999px; }
    .points-val { min-width: 2.9rem; text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }

    /* Podium positions in podium colors -- after the leader rule, which
       would otherwise paint P1's number plain white. */
    table.standings tr.p1 td.pos { color: #ffcf4a; }
    table.standings tr.p2 td.pos { color: #d9dfe7; }
    table.standings tr.p3 td.pos { color: #e3a06b; }
    /* A labelled rule between two groups of rows: the points line on a race
       result, the knockout zones in qualifying. Dashed and faint so it reads
       as a boundary, not as a row. */
    table.standings tr.divider td {
        padding: 0.45rem 0.7rem 0.25rem; border-bottom: none; background: none !important;
    }
    table.standings tr.divider td span {
        display: flex; align-items: center; gap: 0.7rem;
        font-size: 0.6rem; letter-spacing: 0.16em; text-transform: uppercase;
        font-weight: 700; color: var(--ink-faint);
    }
    table.standings tr.divider td span::after {
        content: ""; flex: 1; border-top: 1px dashed rgba(255, 255, 255, 0.16);
    }
    /* Tyre stints as the broadcast draws them: one ring per stint in the
       compound's color, its initial inside, in the order they were run. */
    .tyres { display: inline-flex; align-items: center; gap: 4px; white-space: nowrap; }
    .tyre {
        width: 19px; height: 19px; border-radius: 50%; flex: none;
        display: inline-grid; place-items: center;
        border: 2.5px solid var(--tc); background: rgba(0, 0, 0, 0.35);
        font: 700 0.58rem/1 var(--display); color: var(--tc);
    }
    /* The race's fastest lap in timing-screen purple. */
    .purple-lap {
        color: #e2cbff; background: rgba(181, 123, 255, 0.2);
        border-radius: 6px; padding: 0.1rem 0.4rem; margin-left: -0.4rem;
    }
    /* DNF / DSQ / DNS, and the lap it happened on, as one small red tag in
       the gap column -- the separate Status column that used to repeat it
       ("DNF" under Gap, "Retired" under Status) is gone. */
    .out {
        display: inline-flex; align-items: baseline; gap: 0.35rem;
        padding: 0.08rem 0.45rem; border-radius: 6px;
        background: rgba(255, 92, 92, 0.13); color: #ff8080;
        font: 700 0.74rem/1.5 var(--display); letter-spacing: 0.06em;
    }
    .out small { font-weight: 600; color: rgba(255, 128, 128, 0.7); letter-spacing: 0; }
    /* Gap to the fastest, with a bar beside the number: twenty "+0.4xx"
       figures read as noise, their lengths show where the field splits. */
    .gapcell { display: flex; align-items: center; gap: 0.7rem; }
    .gapcell span { min-width: 4.3rem; }
    .gapcell b { flex: 1; max-width: 8rem; height: 5px; border-radius: 3px; background: rgba(255, 255, 255, 0.05); }
    .gapcell i { display: block; height: 100%; border-radius: 3px; background: rgba(255, 255, 255, 0.34); }
    /* Last five rounds as a tiny bar chart in a standings row: form, which a
       season total hides. */
    .formbars { display: inline-flex; align-items: flex-end; gap: 3px; height: 22px; vertical-align: middle; }
    .formbars i { display: block; width: 6px; min-height: 2px; border-radius: 2px 2px 1px 1px; background: var(--fc); }
    .formbars i.zero { opacity: 0.22; }
    td.behind { color: var(--ink-faint); font-family: var(--mono); font-size: 0.82rem; text-align: right; }
    .nil { color: var(--ink-faint); font-weight: 400; }
    /* ▲2 / ▼1 beside a standings position: places moved since the round
       before. */
    .moved { font-size: 0.66rem; font-weight: 700; margin-left: 0.3rem; vertical-align: 2px; }
    .moved.up { color: var(--f1-teal); }
    .moved.down { color: #ff6b6b; }

    /* ------------------------------------------------------- race story */
    /* The race's length as a bar: neutralised stretches painted on it (solid
       amber for a safety car, dashed for a VSC, red for a red flag) and a
       dot for every change of lead, retirement and the finish. */
    .story-strip { position: relative; height: 38px; margin: 0.7rem 0.9rem 0.3rem; }
    .story-strip .track, .story-strip .seg { position: absolute; top: 10px; height: 6px; border-radius: 3px; }
    .story-strip .track { left: 0; right: 0; background: rgba(255, 255, 255, 0.07); }
    .story-strip .seg.sc { background: var(--f1-amber); }
    .story-strip .seg.vsc { background: repeating-linear-gradient(90deg, var(--f1-amber) 0 5px, transparent 5px 8px); }
    .story-strip .seg.red { background: var(--f1-red); }
    .story-strip .mark {
        position: absolute; top: 7px; width: 12px; height: 12px; margin-left: -6px; border-radius: 50%;
        background: var(--mc); border: 2px solid #0e1119; box-shadow: 0 0 10px -2px var(--mc);
    }
    .story-strip .tick {
        position: absolute; top: 22px; transform: translateX(-50%);
        font: 500 0.64rem var(--mono); color: var(--ink-faint);
    }
    .story-list { columns: 2 24rem; column-gap: 2.2rem; padding: 0.2rem 0.9rem 0.7rem; }
    .story-item {
        break-inside: avoid; display: grid; grid-template-columns: 2.6rem 0.8rem 1fr; gap: 0.45rem;
        align-items: start; padding: 0.45rem 0; border-top: 1px solid var(--line);
        font-size: 0.88rem; line-height: 1.45; color: var(--ink-dim);
    }
    .story-lap { font: 500 0.74rem/1.9 var(--mono); color: var(--ink-faint); }
    .story-icon {
        width: 9px; height: 9px; margin-top: 0.42rem; border-radius: 50%;
        background: var(--ic); box-shadow: 0 0 8px -1px var(--ic);
    }
    .story-item b { font-weight: 700; letter-spacing: 0.03em; }

    /* ---------------------------------------------------------- track map */
    /* The whole point of drawing it as SVG instead of a Plotly figure is the
       :hover rule -- the browser lights the mini-sector under the cursor
       with no callback, no rerun and no JS. */
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
        font-family: 'Titillium Web', "Segoe UI", Inter, sans-serif;
        font-variant-numeric: tabular-nums;
    }
    /* No font-size here on purpose: it's set per element in the markup,
       converted from pixels into viewBox units, since one unit is only a
       fraction of a pixel on a portrait circuit. */
    .trackmap .tip-head { font-weight: 700; fill: #f2f4f6; }
    .trackmap .tip-code { font-weight: 700; }
    .trackmap .tip-time {
        font-weight: 600; fill: #e8ebef;
        font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
    }
    .trackmap .tip-gap { fill: var(--ink-dim); }

    /* st.divider draws a full-width hairline that cuts the page in two; the
       bands here are already separated by their own cards, so the rule is
       faded out at both ends to read as a breath rather than a wall. */
    hr { border: none; height: 1px; background: linear-gradient(90deg, transparent, var(--line-strong), transparent); }
    /* ---------------------------------------------------------------- phones */
    /* Checked at 390 px (an iPhone) in headless Chrome. Kept last in the
       stylesheet so it overrides the desktop rules above it. */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; padding-top: 3.2rem; }
        /* No room for the gap bars on a phone: the number stays. */
        .gapcell b { display: none; }
        .gapcell span { min-width: 0; }
        /* The header stacks: beside the text, the circuit squeezed a Grand
           Prix name onto three lines. */
        .hero { flex-direction: column; align-items: stretch; padding: 1.2rem 1.2rem 1rem; }
        .hero-title { font-size: 1.8rem; }
        .hero-track { align-self: center; }
        .hero-track svg { height: 112px; }
        /* The header bar is see-through on a desktop; on a phone its menu
           buttons sat on top of the tab strip once the page scrolled. */
        [data-testid="stHeader"] { background: rgba(11, 14, 21, 0.92); backdrop-filter: blur(6px); }
        /* Results tables keep what a phone reader wants -- position, driver,
           the numbers -- and drop the car badge, the team and the status
           column, whose content the gap column already carries. */
        table.standings th.badge, table.standings td.badge,
        table.standings th.team, table.standings td.team,
        table.standings th.status, table.standings td.status,
        table.standings th.tyrecol, table.standings td.tyrecol,
        table.standings th.best, table.standings td.best,
        table.standings th.formcol, table.standings td.formcol { display: none; }
        table.standings th { padding: 0 0.35rem 0.5rem; }
        table.standings td { padding: 0.45rem 0.35rem; font-size: 0.86rem; }
        table.standings td.mono { font-size: 0.8rem; }
        table.standings td.num { width: auto; }
        table.standings td.delta { width: auto; white-space: nowrap; font-size: 0.78rem; }
        /* Pit stop lists: the lap / points / circuit line drops under the
           name instead of squeezing it to "George R...". */
        .pit-row { grid-template-columns: 1.8rem 4.2rem 4px 1fr; row-gap: 0.1rem; }
        .pit-time { font-size: 1.05rem; }
        .pit-extra { grid-column: 4; text-align: left; max-width: none; margin-top: -0.2rem; }
        .panel-head .hint { display: block; margin-top: 0.2rem; }
        /* Teammates overview: the pair's color already says which team it
           is, and the median gap is in the pairing detail below -- without
           those two columns the table fits a phone instead of scrolling. */
        /* (.tm-scroll prefix: these rules sit above the base .tm-table
           ones in the stylesheet, so they need the extra specificity.) */
        .tm-scroll .tm-table { min-width: 0; table-layout: auto; font-size: 0.82rem; }
        .tm-scroll .tm-table col.c-team, .tm-scroll .tm-table col.c-gap,
        .tm-scroll .tm-table th:first-child, .tm-scroll .tm-table td:first-child,
        .tm-scroll .tm-table th:last-child, .tm-scroll .tm-table td:last-child { display: none; }
        .tm-scroll .tm-table th, .tm-scroll .tm-table td { padding: 8px 4px; }
        .tm-scroll .tm-split { gap: 4px; }
        .tm-scroll .tm-split b { min-width: 1.6em; }
        .tm-scroll .tm-bar { min-width: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# A day for both standings loaders: the round is part of the cache key, so a
# new race is picked up at once anyway, and the Ergast mirror (jolpica)
# rate-limits by the hour -- the progression alone is two requests per round
# already run, and answering "Too Many Requests" is what a short ttl bought.
@st.cache_data(ttl=86400, show_spinner="Loading the championship standings...")
def load_standings(year, round_number):
    """Championship standings as they stood right after a given round --
    pulled from Ergast (via fastf1.ergast) rather than summed from each
    session's points locally, since Ergast already carries the official,
    round-by-round running totals and doesn't need every prior race of the
    season fetched and parsed just to get today's totals."""
    with FF1_LOCK:
        ergast = Ergast()
        drivers = ergast.get_driver_standings(season=year, round=round_number).content[0]
        constructors = ergast.get_constructor_standings(season=year, round=round_number).content[0]
    return drivers, constructors


@st.cache_data(ttl=86400, show_spinner="Rebuilding the points table round by round...")
def load_standings_progression(year, up_to_round):
    """Points after each round of the season so far, long-format. Ergast's
    standings endpoint with no round given returns only the latest snapshot
    (season is a running total, not a round-by-round history), so this has
    to ask for one round at a time and stitch the results together."""
    ergast = Ergast()
    driver_rows, constructor_rows = [], []
    for rnd in range(1, up_to_round + 1):
        with FF1_LOCK:
            driver_table = ergast.get_driver_standings(season=year, round=rnd).content[0]
            constructor_table = ergast.get_constructor_standings(season=year, round=rnd).content[0]
        for _, r in driver_table.iterrows():
            name = r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
            driver_rows.append({"Round": rnd, "Driver": name, "Points": r["points"], "Position": r["position"]})
        for _, r in constructor_table.iterrows():
            constructor_rows.append({
                "Round": rnd, "Constructor": r["constructorName"], "Points": r["points"], "Position": r["position"],
            })
    return pd.DataFrame(driver_rows), pd.DataFrame(constructor_rows)


@st.cache_data(ttl=3600, show_spinner=False)
def total_rounds_in_season(year):
    """Full season length (including rounds not yet run) -- load_schedule
    filters those out for the race picker, but the title-fight odds below
    need to know how many races are actually still left to simulate."""
    with FF1_LOCK:
        schedule = fastf1.get_event_schedule(year)
    return int(schedule[schedule["RoundNumber"] > 0]["RoundNumber"].max())


@st.cache_data(ttl=3600, show_spinner=False)
def sprint_rounds_in_season(year):
    """Round numbers of the season's sprint weekends, which put up points
    twice -- the title maths has to count them."""
    with FF1_LOCK:
        schedule = fastf1.get_event_schedule(year)
    sprints = schedule[schedule["EventFormat"].astype(str).str.contains("sprint")]
    return {int(r) for r in sprints["RoundNumber"] if r > 0}


@st.cache_data(ttl=3600, show_spinner=False)
def event_name_for_round(year, round_number):
    """Round number -> Grand Prix name, including rounds not yet run (the
    clinch-round estimate below can land on a future race)."""
    with FF1_LOCK:
        schedule = fastf1.get_event_schedule(year)
    match = schedule[schedule["RoundNumber"] == round_number]
    return match.iloc[0]["EventName"] if len(match) else f"Round {round_number}"


# A day, not an hour: a newly finished race already changes event_names, and
# with it the cache key, so a shorter ttl buys no freshness -- it only makes
# some visitor wait for the whole season to be recomputed.
@st.cache_data(ttl=86400, show_spinner="Crunching stats across every race run so far this season...")
def season_stats(year, event_names, rules=OVERTAKE_RULES):
    """One row per completed race: total overtakes (same rule as the
    per-race chart), retirements, the biggest grid-to-finish recovery, and
    the closest podium fight. Loads every completed race's full Race
    session once (cached after that), so this is the slow one on a cold
    cache."""
    rows = []
    driver_overtakes = {}
    for event_name in event_names:
        try:
            with FF1_LOCK:
                race = fastf1.get_session(year, event_name, "Race")
            fetch_published(race, year, telemetry=False)
            # Laps and race control messages are all count_overtakes() and the
            # results columns read. Telemetry is most of a session's download
            # and memory, and across a whole season it's what pushed this past
            # the hosted app's ~1 GB.
            with FF1_LOCK:
                race.load(telemetry=False, weather=False)
            _ = race.laps  # raises if they didn't load, so the race is skipped
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


# Real power unit lineups, by season -- FastF1's results carry the team, not
# the engine, and there's no supplier field to read this from. Only seasons
# listed here get the "Poles by engine" chart: teams change suppliers between
# years, so reusing one year's table for another would credit the wrong
# manufacturer. A new season needs its own entry.
ENGINE_SUPPLIERS = {2026: {
    "Mercedes": "Mercedes", "McLaren": "Mercedes", "Williams": "Mercedes", "Alpine": "Mercedes",
    "Ferrari": "Ferrari", "Haas F1 Team": "Ferrari", "Cadillac": "Ferrari",
    "Red Bull Racing": "Red Bull Ford", "Racing Bulls": "Red Bull Ford",
    "Aston Martin": "Honda",
    "Audi": "Audi",
}}


@st.cache_data(ttl=3600, show_spinner="Checking who took pole in every qualifying session so far...")
def pole_positions(year, event_names):
    """Pole sitter (and their team) for every completed qualifying session
    this season -- results only, no lap/telemetry data needed, so this is
    much lighter than season_stats() despite covering the same races."""
    rows = []
    for event_name in event_names:
        try:
            with FF1_LOCK:
                quali = fastf1.get_session(year, event_name, "Qualifying")
            fetch_published(quali, year, telemetry=False)
            with FF1_LOCK:
                quali.load(laps=False, telemetry=False, weather=False, messages=False)
        except Exception:
            continue
        pole = quali.results.sort_values("Position").iloc[0]
        rows.append({"Event": event_name, "Driver": pole["Abbreviation"], "Team": pole["TeamName"]})
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner="Comparing every pair of teammates, race by race...")
def teammate_battles(year, event_names):
    """One row per team pair per session (qualifying and race) of the season
    so far, with both drivers' positions, race points and whether they were
    classified, and in qualifying the gap in the last part both drivers took
    part in (Q3, else Q2, else Q1) -- the usual way the qualifying gap
    between teammates is measured. A and B are the pair in alphabetical
    order, so a pair keeps the same row all season."""
    rows = []
    for event_name in event_names:
        for kind in ("Qualifying", "Race"):
            try:
                with FF1_LOCK:
                    s = fastf1.get_session(year, event_name, kind)
                fetch_published(s, year, telemetry=False)
                # Laps for qualifying: Q1/Q2/Q3 times are worked out from them.
                with FF1_LOCK:
                    s.load(laps=kind == "Qualifying", telemetry=False, weather=False, messages=False)
                results = s.results
                round_number = int(s.event["RoundNumber"])
            except Exception:
                continue
            for team, g in results.groupby("TeamName"):
                if len(g) != 2:
                    continue
                g = g.set_index("Abbreviation").sort_index()
                (a, ra), (b, rb) = list(g.iterrows())
                if pd.isna(ra["Position"]) or pd.isna(rb["Position"]):
                    continue
                gap = np.nan
                if kind == "Qualifying":
                    for q in ("Q3", "Q2", "Q1"):
                        if pd.notna(ra.get(q)) and pd.notna(rb.get(q)):
                            gap = (rb[q] - ra[q]).total_seconds()  # positive: A quicker
                            break
                classified = lambda r: str(r.get("ClassifiedPosition", "")).isdigit()
                rows.append({
                    "Event": event_name, "Round": round_number, "Kind": kind, "Team": team,
                    "A": a, "B": b, "PosA": int(ra["Position"]), "PosB": int(rb["Position"]),
                    "PtsA": float(ra.get("Points") or 0), "PtsB": float(rb.get("Points") or 0),
                    "FinA": classified(ra), "FinB": classified(rb),
                    "AAhead": ra["Position"] < rb["Position"], "Gap": gap,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- DHL pit stops
# Stationary pit stop times -- the 2-second numbers -- aren't in the timing
# feed or in Ergast; they're published by DHL for its Fastest Pit Stop Award.
# Its pages load three public JSON feeds (the per-race ranking, the award
# standings plus the season's fastest stops, and team averages per race),
# and robots.txt only disallows /admin/. Seasons from 2023 have their own
# page; earlier ones redirect to the current season, which dhl_feeds()
# detects. DHL answers datacenter IPs, so the hosted app reads it directly.
DHL = "https://inmotion.dhl"
DHL_HEADERS = {"User-Agent": "Mozilla/5.0 (pitwall-pc F1 dashboard)"}
DHL_FIRST_SEASON = 2023


def dhl_json(element_id, event=None):
    response = requests.get(
        f"{DHL}/api/f1-award-element-data/{element_id}",
        params={"event": event} if event else None, headers=DHL_HEADERS, timeout=20,
    )
    response.raise_for_status()
    return response.json()["data"]


@st.cache_data(ttl=6 * 3600, show_spinner="Fetching DHL's pit stop times...")
def dhl_season(year):
    """The season's DHL data -- events, award standings, fastest stops, and
    the feed ids -- or None when DHL has nothing for that year."""
    url = (f"{DHL}/en/formula-1/fastest-pit-stop-award" if year == datetime.date.today().year
           else f"{DHL}/formula-1/fastest-pit-stop-award-{year}")
    try:
        html = requests.get(url, headers=DHL_HEADERS, timeout=20).text
        feeds = re.findall(r'data-url="/api/f1-award-element-data/(\d+)"', html)[:3]
        if len(feeds) < 3:
            return None
        events = dhl_json(feeds[2])["chart"]["events"]
        # A season DHL has no page for redirects to the current one.
        if not events or str(year) not in events[0]["title"]:
            return None
        summary = dhl_json(feeds[1])["chart"]
    except Exception:
        return None
    for e in events:
        e["day"] = e["date"]["date"][:10]
    return {"feeds": feeds, "events": events, "standings": summary.get("standings", []),
            "fastest": summary.get("season_fastest", [])}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def dhl_event_stops(year, event_day):
    """(DHL event, its ranked stops) for the race held on event_day
    (YYYY-MM-DD) -- matched by date, which both calendars agree on, rather
    than by name ("Spanish" vs "Madrid", sponsor names in DHL's titles)."""
    season = dhl_season(year)
    if not season:
        return None, []
    event = next((e for e in season["events"] if e["day"] == event_day), None)
    if event is None:
        return None, []
    try:
        stops = dhl_json(season["feeds"][0], event=event["id"]).get("chart") or []
    except Exception:
        return event, []
    return event, sorted(stops, key=lambda r: r["duration"])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def dhl_all_time_fastest():
    """Every season's ten fastest stops since DHL's records start, pooled."""
    rows = []
    for season_year in range(DHL_FIRST_SEASON, datetime.date.today().year + 1):
        season = dhl_season(season_year)
        for r in (season or {}).get("fastest", []):
            rows.append({**r, "year": season_year})
    return sorted(rows, key=lambda r: r["duration"])


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
        with FF1_LOCK:
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
def load_schedule(year, include_in_progress=False):
    with FF1_LOCK:
        schedule = fastf1.get_event_schedule(year)
    schedule = schedule[schedule["RoundNumber"] > 0]
    # Only weekends that have actually happened -- a future round on the
    # calendar has no session data yet, so it has no business in the
    # dropdown. Session5 is the last session of the weekend (Race, or Sprint
    # weekends still end on Race), so its time is the real "is this done".
    # include_in_progress tests the *first* session instead, which is what the
    # weekend view wants: on a Friday evening practice has run and is worth
    # looking at, and waiting for Sunday to list the weekend at all meant
    # published practice data sat in the release, invisible. The season views
    # keep the stricter test -- they read the weekend's race for standings and
    # for driver colors, and a weekend in progress hasn't got one.
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    first_or_last = "Session1DateUtc" if include_in_progress else "Session5DateUtc"
    schedule = schedule[schedule[first_or_last] <= now_utc]
    # Descending so the most recently completed round is first -- the
    # Grand Prix selectbox below defaults to whatever's first in the list,
    # and opening on the latest race is more useful than opening on Round 1
    # (Australia) every single time.
    return schedule.sort_values("RoundNumber", ascending=False)


# ------------------------------------------------------ published data --
# The official timing feed answers 403 to datacenter IPs, so on the hosted
# app FastF1 can't download a session itself. publish_data.py, run from a
# home connection, uploads each session's FastF1 cache folder as release
# assets of this private repo; the functions below download and unpack one
# into cache/ right before FastF1 opens it, and FastF1 then reads it from
# disk without ever calling the feed. With no token configured (a local run)
# all of this is skipped and FastF1 downloads from the feed as it always did.
DATA_REPO = "PeterCollavino7/f1-data"


def data_store_token():
    try:
        return st.secrets.get("F1_DATA_TOKEN")
    except Exception:
        return None  # no secrets file at all -- a local run


def github_headers(accept="application/vnd.github+json"):
    return {"Authorization": f"Bearer {data_store_token()}", "Accept": accept}


@st.cache_data(ttl=900, show_spinner=False)
def release_assets(year):
    """Every asset of the season's release, as GitHub lists them (name, API
    url, upload time); empty if there's no release for that year."""
    release = requests.get(
        f"https://api.github.com/repos/{DATA_REPO}/releases/tags/{year}", headers=github_headers(), timeout=20,
    )
    if release.status_code == 404:
        return []
    release.raise_for_status()
    assets, page = [], 1
    while True:
        # The release object's own asset list isn't the place to read a full
        # season from; this endpoint pages through all of them.
        batch = requests.get(
            f"https://api.github.com/repos/{DATA_REPO}/releases/{release.json()['id']}/assets",
            headers=github_headers(), params={"per_page": 100, "page": page}, timeout=20,
        ).json()
        assets += [{"name": a["name"], "url": a["url"], "updated": a["updated_at"]} for a in batch]
        if len(batch) < 100:
            return assets
        page += 1


def published_assets(year):
    """{asset name: API url} for the season's release."""
    return {a["name"]: a["url"] for a in release_assets(year)}


def freshness_line():
    """"Updated 3 h ago" in the sidebar, from the newest upload in this
    season's release -- and a warning when a session that should have been
    published by now isn't. The hosted app only knows what the home PC has
    uploaded, so a weekend with that PC switched off used to look exactly
    like a weekend with no racing: nothing new, and no way to tell why."""
    if data_store_token() is None:
        return ""
    this_year = datetime.datetime.now(datetime.timezone.utc).year
    try:
        assets = release_assets(this_year)
    except Exception:
        return ""
    if not assets:
        return ""
    newest = max(pd.Timestamp(a["updated"]) for a in assets)
    age = pd.Timestamp.now(tz="UTC") - newest
    hours = age.total_seconds() / 3600
    ago = (f"{max(1, round(age.total_seconds() / 60))} min ago" if hours < 1
           else f"{hours:.0f} h ago" if hours < 36 else f"{age.days} days ago")
    line = (
        f'<div class="data-status" title="{newest:%d %b %Y, %H:%M} UTC"><i></i>'
        f"<span>Data updated <b>{ago}</b></span></div>"
    )

    # A session is late when it started more than 32 hours ago -- two hours
    # to run, the six publish_data.py waits for the feed to settle, and a day
    # for the daily task to come round -- and still isn't in the release.
    try:
        published = published_sessions(this_year)
        now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        late = []
        for _, event in load_schedule(this_year, include_in_progress=True).iterrows():
            for i in range(1, 6):
                name, start = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
                if (isinstance(name, str) and pd.notna(start)
                        and now_utc - datetime.timedelta(days=10) < start < now_utc - datetime.timedelta(hours=32)
                        and name not in published.get(event["EventName"], [])):
                    late.append(f"{event['EventName'].replace(' Grand Prix', ' GP')} {name}")
    except Exception:
        late = []
    if late:
        line += f'<div class="data-status late"><i></i><span>Not published yet · {", ".join(late[-2:])}</span></div>'
    return line


@st.cache_data(ttl=3600, show_spinner=False)
def season_calendar(year):
    """The whole season's schedule, rounds not yet run included, as plain
    rows: name, place, date and every session's start (UTC)."""
    with FF1_LOCK:
        schedule = fastf1.get_event_schedule(year)
    rows = []
    for _, e in schedule[schedule["RoundNumber"] > 0].iterrows():
        sessions = [
            (e[f"Session{i}"], e[f"Session{i}DateUtc"]) for i in range(1, 6)
            if isinstance(e.get(f"Session{i}"), str) and e.get(f"Session{i}") and pd.notna(e.get(f"Session{i}DateUtc"))
        ]
        rows.append({"name": e["EventName"], "location": e["Location"], "round": int(e["RoundNumber"]),
                     "date": pd.Timestamp(e["EventDate"]), "sessions": sessions})
    return rows


def until(delta):
    """"in 40 min" / "in 22 h" / "in 5 days" -- relative, so no time zone to
    get wrong for a reader anywhere."""
    minutes = delta.total_seconds() / 60
    if minutes < 60:
        return f"in {max(1, round(minutes))} min"
    if minutes < 48 * 60:
        return f"in {minutes / 60:.0f} h"
    return f"in {round(minutes / 1440)} days"


def next_race_card():
    """What's next on the calendar, at the top of the sidebar's foot: the
    weekend that's on now and its next session, or the next Grand Prix and
    how far away it is. Gives the page a sense of where the season is, and
    of when the next data will land."""
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    try:
        calendar = season_calendar(now.year)
    except Exception:
        return ""
    for event in calendar:
        if not event["sessions"]:
            continue
        first, last = event["sessions"][0][1], event["sessions"][-1][1]
        if last + datetime.timedelta(hours=3) < now:
            continue
        name = event["name"]
        if first <= now:
            upcoming = [(s, t) for s, t in event["sessions"] if t > now]
            detail = (f"{upcoming[0][0]} {until(upcoming[0][1] - now)}" if upcoming
                      else f"{event['sessions'][-1][0]} under way")
            kicker, live = "This weekend", True
        else:
            detail = f"{event['location']} · {event['date']:%a %d %b} · {until(first - now)}"
            kicker, live = f"Next race · round {event['round']}", False
        return (
            f'<div class="next-race{" live" if live else ""}">'
            f'<div class="kicker"><i></i>{kicker}</div>'
            f'<div class="name">{name}</div><div class="detail">{detail}</div></div>'
        )
    return ""


def session_key(session):
    # Same naming as publish_data.py: the event and session folders of the
    # session's api_path, which is also its folder under the FastF1 cache.
    _, _, event_dir, session_dir = session.api_path.strip("/").split("/")
    return f"{event_dir}__{session_dir}"


@st.cache_data(ttl=900, show_spinner=False)
def published_sessions(year):
    """{event name: [session names]} that have been published for the
    season -- the hosted app lists only these, since anything else would fail
    to load."""
    published = published_assets(year)
    available = {}
    if not published:
        return available
    for _, event in load_schedule(year, include_in_progress=True).iterrows():
        for name in session_names_for(event):
            with FF1_LOCK:
                key = session_key(fastf1.get_session(year, event["EventName"], name))
            if f"{key}.base.zip" in published:
                available.setdefault(event["EventName"], []).append(name)
    return available


def fetch_published(session, year, telemetry):
    """Unpack the session's published cache into cache/ if it isn't there
    yet. Telemetry is a separate, much bigger asset, fetched only when asked
    for, so the season views never pay for it."""
    if data_store_token() is None:
        return
    key = session_key(session)
    folder = os.path.join("cache", *session.api_path.strip("/").split("/")[1:])
    for part in ("base", "telemetry") if telemetry else ("base",):
        marker = os.path.join(folder, f".{part}-unpacked")
        url = published_assets(year).get(f"{key}.{part}.zip")
        if os.path.exists(marker) or url is None:
            continue
        # requests drops the Authorization header on the redirect to GitHub's
        # storage host, which is what that host requires.
        response = requests.get(url, headers=github_headers("application/octet-stream"), timeout=180)
        response.raise_for_status()
        # Unpacked beside the target and moved in file by file, so another
        # thread opening the same session never reads a half-written file.
        os.makedirs(folder, exist_ok=True)
        staging = tempfile.mkdtemp(dir=folder)
        try:
            zipfile.ZipFile(io.BytesIO(response.content)).extractall(staging)
            for name in os.listdir(staging):
                os.replace(os.path.join(staging, name), os.path.join(folder, name))
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        open(marker, "w").close()


def available_years():
    """Seasons with at least one completed round, newest first. The current
    year only joins the list once its first race is over: between New Year and
    the season opener it has nothing to show, and an empty Grand Prix dropdown
    would leave the rest of the page with no event to load. On the hosted app
    a season is listed once it has published data, for the same reason."""
    this_year = datetime.datetime.now(datetime.timezone.utc).year
    years = list(range(this_year, FIRST_YEAR - 1, -1))
    if data_store_token() is not None:
        return [y for y in years if published_assets(y)]
    try:
        if load_schedule(this_year).empty:
            years.remove(this_year)
    except Exception:
        # No calendar published yet (or the API is down) -- same answer.
        years.remove(this_year)
    return years


# ttl so a race loaded in the first minutes after the flag, before the timing
# feed is complete, gets fetched again later instead of staying half-empty for
# as long as the server runs. cache_resource, not cache_data: cache_data
# pickles what it stores and hands every rerun its own unpickled copy, so a
# session with telemetry -- a few hundred MB -- sat in memory once in the
# cache and again per run, and with three of them cached the hosted app
# (about 1 GB) ran out of memory and crashed outright. One shared object per
# session, and at most two of them; nothing in this file writes to a session.
@st.cache_resource(ttl=6 * 3600, max_entries=2, show_spinner="Loading timing and telemetry for this session...")
def load_session(year, event, session_name, telemetry=True):
    with FF1_LOCK:
        session = fastf1.get_session(year, event, session_name)
    fetch_published(session, year, telemetry)
    # load() doesn't raise when the timing feed fails -- it logs, returns, and
    # leaves session.laps unset, so the first chart to touch the laps crashed
    # with a raw traceback. The laps are touched here instead, which turns
    # that into the plain error message below (and st.cache_data never caches
    # an exception, so the next visit tries the download again). What FastF1
    # logged along the way is kept and attached, since the real cause is only
    # ever in the log, never in the exception.
    problems = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = lambda record: problems.append(
        record.getMessage() + (f" ({record.exc_info[1]!r})"[:300] if record.exc_info else "")
    )
    fastf1_logger = logging.getLogger("fastf1")
    fastf1_logger.addHandler(handler)
    try:
        with FF1_LOCK:
            session.load(telemetry=telemetry, weather=telemetry)
    finally:
        fastf1_logger.removeHandler(handler)
    try:
        # Assigned, not a bare expression: Streamlit "magic" renders any bare
        # expression in the script, and this one put the raw laps table on top
        # of the page.
        _ = session.laps
    except fastf1.core.DataNotLoadedError:
        failures = [m for m in problems if "fail" in m.lower() or "error" in m.lower()]
        raise RuntimeError("the lap timing data didn't download. " + " | ".join((failures or problems)[:3]))
    return session


def session_names_for(event_row):
    """The sessions actually run that weekend, most recent first -- practice/
    qualifying count and naming (Sprint Qualifying, Sprint, ...) depend on the
    event format, so this is read from the schedule instead of assumed fixed.
    Reversed (like load_schedule's round order) so the session selectbox
    below defaults to the Race instead of Practice 1. A session that hasn't
    started is left out: with a weekend in progress now listed from its first
    session, the rest of it is still in the future."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    names = []
    for i in range(1, 6):
        name, start = event_row.get(f"Session{i}"), event_row.get(f"Session{i}DateUtc")
        if isinstance(name, str) and name and pd.notna(start) and start <= now_utc:
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


# Team colors are the teams' own, picked for white TV graphics -- on this
# app's near-black panels three of the 2026 eleven all but disappeared: Red
# Bull's #0600ef (1.9:1 against the panel), Cadillac's #444444 (1.8:1) and
# Aston Martin's #00665f (2.6:1). VER's line on the race pace chart was a dark
# blue scribble. readable() lifts a color's lightness, hue and saturation
# kept, until it clears 3:1 -- the WCAG floor for lines and marks -- so Red
# Bull is still unmistakably Red Bull blue, just one you can see. Colors that
# already pass come back unchanged.
PANEL_BG = "#141824"
MIN_CONTRAST = 3.2


def _luminance(hex_color):
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


@functools.lru_cache(maxsize=512)
def readable(color):
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return color
    if _contrast(color, PANEL_BG) >= MIN_CONTRAST:
        return color
    h, l, s = colorsys.rgb_to_hls(*(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)))
    while l < 0.95:
        l = min(0.95, l + 0.02)
        lifted = "#" + "".join(f"{round(c * 255):02x}" for c in colorsys.hls_to_rgb(h, l, s))
        if _contrast(lifted, PANEL_BG) >= MIN_CONTRAST:
            return lifted
    return lifted


def safe_driver_color(code, session):
    """get_driver_color raises for a code that isn't in this session's entry
    list (a mid-season replacement, a one-off stand-in, or any driver from a
    season-wide view), so every lookup outside the telemetry tab needs its
    own guard rather than reusing build_driver_styles."""
    try:
        return readable(fastf1.plotting.get_driver_color(code, session))
    except Exception:
        return "#999999"


def team_color(name):
    """A team's color, made readable -- grey for a name FastF1 can't place
    (DHL's short names, past teams)."""
    try:
        return readable(fastf1.plotting.get_team_color(name, session))
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

    Two extras for the timing-screen feel: a row can carry "divider", a
    label drawn as a thin rule *above* it (the points line, the qualifying
    knockout zones), and the top three rows are marked p1/p2/p3 so their
    position numbers take the podium colors.
    """
    has_badges = any(row.get("badge") for row in rows)
    span = len(columns) + (1 if has_badges else 0)
    html = ["<table class='standings'><thead><tr>"]
    if has_badges:
        html.append("<th class='badge'></th>")
    html += [f"<th class='{css_class}'>{header}</th>" for _, header, css_class in columns]
    html.append("</tr></thead><tbody>")
    for rank, row in enumerate(rows):
        if row.get("divider"):
            html.append(f"<tr class='divider'><td colspan='{span}'><span>{row['divider']}</span></td></tr>")
        classes = (["leader"] if rank == 0 else []) + ([f"p{rank + 1}"] if rank < 3 and row.get("podium", True) else [])
        html.append(f"<tr class='{' '.join(classes)}'>")
        if has_badges:
            badge = row.get("badge")
            html.append(f"<td class='badge'>{f'<img src=\"{badge}\">' if badge else ''}</td>")
        for index, (key, _, css_class) in enumerate(columns):
            accent = f" style='--row-accent:{row.get('color', 'transparent')}'" if index == 0 else ""
            html.append(f"<td class='{css_class}'{accent}>{row.get(key, '')}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("<div class='table-scroll'>" + "".join(html) + "</div>", unsafe_allow_html=True)


# Slow to fast, for the speed-colored circuit in the header: the ramp speed
# maps usually use (cool corners, hot straights), in four stops so the middle
# doesn't wash out to grey the way a straight blue-to-red blend does.
SPEED_RAMP = ["#3d6bff", "#19c6d8", "#ffd23f", "#ff3b30"]


def ramp_color(t):
    t = min(max(float(t), 0.0), 1.0) * (len(SPEED_RAMP) - 1)
    i = min(int(t), len(SPEED_RAMP) - 2)
    a, b = SPEED_RAMP[i], SPEED_RAMP[i + 1]
    mix = [int(a[k : k + 2], 16) + (int(b[k : k + 2], 16) - int(a[k : k + 2], 16)) * (t - i) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(c):02x}" for c in mix)


def circuit_outline(session, stroke=5):
    """The circuit for the page header, traced from the session's fastest
    lap and colored by how fast that lap was at every point -- the braking
    zones cool, the straights hot -- so the artwork says something about the
    track instead of only showing its shape. It draws itself in when the
    page opens, and a dot then laps it on a loop at the real lap's own
    rhythm (fast on the straights, slow through the corners -- the timing
    comes from the telemetry, compressed to a few seconds). Both are off
    under prefers-reduced-motion.

    Returns an empty string rather than raising if the session has no usable
    position data -- the header is decoration, and a practice session nobody
    completed a lap in shouldn't take the page down with it.
    """
    try:
        lap = session.laps.pick_fastest()
        telemetry = lap.get_telemetry()
        step = 6
        x = telemetry["X"].to_numpy(dtype=float)[::step]
        y = -telemetry["Y"].to_numpy(dtype=float)[::step]
        speed = telemetry["Speed"].to_numpy(dtype=float)[::step]
        elapsed = telemetry["Time"].dt.total_seconds().to_numpy(dtype=float)[::step]
    except Exception:
        return ""
    if len(x) < 10 or not np.isfinite(speed).any():
        return ""
    pad = stroke
    span_x = max(x.max() - x.min(), 1e-6)
    span_y = max(y.max() - y.min(), 1e-6)
    scale = (300 - 2 * pad) / span_x
    view_height = span_y * scale + 2 * pad
    px = (x - x.min()) * scale + pad
    py = (y - y.min()) * scale + pad
    low, high = np.nanmin(speed), np.nanmax(speed)
    segments = "".join(
        f'<line x1="{px[i]:.1f}" y1="{py[i]:.1f}" x2="{px[i + 1]:.1f}" y2="{py[i + 1]:.1f}" '
        f'stroke="{ramp_color((speed[i] - low) / max(high - low, 1))}"/>'
        for i in range(len(px) - 1)
    )
    path = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))

    # The dot's timing: where along the path it is (keyPoints, a share of
    # the path's length) at each moment of the lap (keyTimes, a share of the
    # lap time). Forty samples is plenty for the eye and keeps the markup
    # short.
    lengths = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    duration = elapsed[-1] - elapsed[0]
    motion = ""
    if lengths[-1] > 0 and duration > 0:
        picks = np.linspace(0, len(px) - 1, 40).astype(int)
        key_times = (elapsed[picks] - elapsed[0]) / duration
        key_points = np.maximum.accumulate(lengths[picks] / lengths[-1])
        key_times[0], key_times[-1], key_points[0], key_points[-1] = 0.0, 1.0, 0.0, 1.0
        seconds = max(4.0, duration / 12)
        motion = (
            f'<circle class="car" r="{stroke * 1.15:.1f}">'
            f'<animateMotion dur="{seconds:.1f}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{";".join(f"{v:.4f}" for v in key_times)}" '
            f'keyPoints="{";".join(f"{v:.4f}" for v in key_points)}">'
            f'<mpath href="#hero-lap"/></animateMotion></circle>'
        )
    return (
        f'<svg viewBox="-4 -4 308 {view_height + 8:.0f}" preserveAspectRatio="xMidYMid meet">'
        f'<defs><path id="hero-lap" d="{path}"/>'
        f'<mask id="hero-draw" maskUnits="userSpaceOnUse">'
        f'<path class="draw" d="{path}" pathLength="1" fill="none" stroke="#fff" '
        f'stroke-width="{stroke * 3}" stroke-linecap="round" stroke-linejoin="round"/></mask></defs>'
        f'<g mask="url(#hero-draw)" stroke-width="{stroke}" stroke-linecap="round">{segments}</g>'
        f"{motion}</svg>"
        f'<div class="speed-key"><span>{low:.0f}</span><i></i><span>{high:.0f} km/h</span></div>'
    )


def render_hero(kicker_pill, kicker_text, title, subtitle, artwork="", chip_items=()):
    """The page header: kicker, title, one line of context, and a row of
    chips for the facts that used to be crammed into that one line
    (circuit, date, session, race count) separated by middle dots."""
    chips_html = (
        '<div class="hero-chips">'
        # A chip that brings its own <span> (one with a tooltip) goes in as
        # is; a None is a fact the session didn't have, and is skipped.
        + "".join(c if c.startswith("<span") else f"<span>{c}</span>" for c in chip_items if c)
        + "</div>"
        if chip_items else ""
    )
    st.markdown(
        '<div class="hero"><div class="hero-text">'
        f'<div class="hero-kicker"><span class="pill">{kicker_pill}</span>{kicker_text}</div>'
        f'<div class="hero-title">{title}</div>'
        f'<div class="hero-sub">{subtitle}</div>'
        f'{chips_html}'
        f'</div><div class="hero-track">{artwork}</div></div>',
        unsafe_allow_html=True,
    )


def track_temperature_chip(session):
    """The header chip for the track temperature: the session average, from
    the weather feed the timing service samples about once a minute. The
    range and the air temperature ride in the tooltip -- one number fits a
    chip, three don't. None when the session has no weather data, so the
    header simply goes without the chip."""
    try:
        weather = session.weather_data
        track = weather["TrackTemp"].dropna()
    except Exception:
        return None
    if track.empty:
        return None
    tip = f"Track temperature over the session: {track.min():.0f}–{track.max():.0f} °C"
    air = weather["AirTemp"].dropna() if "AirTemp" in weather else pd.Series(dtype=float)
    if not air.empty:
        tip += f". Air {air.mean():.0f} °C"
    if "Rainfall" in weather and weather["Rainfall"].fillna(False).astype(bool).any():
        tip += ". Rain at some point"
    return f'<span title="{tip}.">\U0001F321️ Track {track.mean():.0f} °C</span>'


def weekend_headline_cards(session, session_name):
    """The handful of facts worth knowing before any chart on the weekend
    pages: who won (or took pole, or was quickest in practice), the fastest
    lap, the speed trap, and one session-appropriate fourth.

    Each card is built in its own try block: a practice session with no
    classification, a sprint weekend with no speed-trap column, or a red-
    flagged session with no complete lap should cost its own card, not the
    whole strip.
    """
    cards = []
    results = session.results if session.results is not None else pd.DataFrame()
    classified = (
        results.dropna(subset=["Position"]).sort_values("Position")
        if not results.empty and "Position" in results else pd.DataFrame()
    )
    is_race = session_name in ("Race", "Sprint")
    is_quali = "Qualifying" in session_name

    try:
        if not classified.empty:
            top = classified.iloc[0]
            color = safe_driver_color(top["Abbreviation"], session)
            label = "Winner" if is_race else ("Pole position" if is_quali else "Quickest")
            cards.append((label, top["Abbreviation"], f"<b>{top['TeamName']}</b>", color))
    except Exception:
        pass

    try:
        fastest = session.laps.pick_fastest()
        if fastest is not None and pd.notna(fastest["LapTime"]):
            color = safe_driver_color(fastest["Driver"], session)
            cards.append((
                "Fastest lap",
                format_lap_time(fastest["LapTime"]),
                f"<b>{fastest['Driver']}</b> · lap {int(fastest['LapNumber'])}",
                color,
            ))
    except Exception:
        pass

    try:
        # SpeedST is the speed-trap reading carried on every lap row, so this
        # costs nothing extra -- the alternative (max of every driver's Speed
        # channel) would mean downloading full telemetry for the whole field.
        trap = session.laps.dropna(subset=["SpeedST"])
        if not trap.empty:
            row = trap.loc[trap["SpeedST"].idxmax()]
            cards.append((
                "Speed trap",
                f"{row['SpeedST']:.0f} <small>km/h</small>",
                f"<b>{row['Driver']}</b>",
                safe_driver_color(row["Driver"], session),
            ))
    except Exception:
        pass

    try:
        if is_race and not classified.empty:
            movers = classified.dropna(subset=["GridPosition"])
            movers = movers[movers["GridPosition"] > 0]
            gains = movers["GridPosition"] - movers["Position"]
            if len(gains) and gains.max() > 0:
                best = movers.loc[gains.idxmax()]
                cards.append((
                    "Biggest mover",
                    f"+{int(gains.max())} <small>places</small>",
                    f"<b>{best['Abbreviation']}</b> · P{int(best['GridPosition'])} → P{int(best['Position'])}",
                    PALETTE["teal"],
                ))
        elif is_quali and len(classified) >= 2:
            # Pole margin: the gap between the two quickest laps actually set
            # in the session's final segment.
            best_times = classified[["Q3", "Q2", "Q1"]].bfill(axis=1).iloc[:, 0] if "Q3" in classified else None
            if best_times is not None and best_times.notna().sum() >= 2:
                ordered = best_times.dropna().sort_values()
                margin = (ordered.iloc[1] - ordered.iloc[0]).total_seconds()
                cards.append((
                    "Pole margin",
                    f"{margin:.3f} <small>s</small>",
                    f"over <b>{classified.loc[ordered.index[1], 'Abbreviation']}</b>",
                    PALETTE["blue"],
                ))
        else:
            laps_done = int(session.laps["LapNumber"].count())
            cards.append(("Laps completed", f"{laps_done}", "across the whole field", PALETTE["blue"]))
    except Exception:
        pass

    return cards


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
    #
    # ...up to a point. A third of the lap is right for a tall circuit, whose
    # drawing is held to the 560px height and comes out narrow, but a wide
    # one (Baku, Jeddah) fills the whole column, and a third of that was a
    # card 370px across with 27px type -- bigger than the stretch of track it
    # was describing. So the card shrinks when the drawing renders large: k
    # is 1 for a tall circuit (the card as it was) and drops towards ~0.45
    # for the widest, from how many pixels one unit of the viewBox is going
    # to take. That only needs the column width to be in the right range,
    # not exact -- a phone's (~360px) or a desktop's (~1000px).
    px_per_unit = min((360 if on_phone() else 1000) / 1000, height / view_height)
    k = min(1.0, 0.44 / px_per_unit)
    box_w, box_h = 480.0 * k, 186.0 * k
    pad_x = 30.0 * k
    head_y, rule_y, row_a_y, row_b_y = 52.0 * k, 74.0 * k, 122.0 * k, 164.0 * k
    font_head, font_code, font_time, font_gap = 42.0 * k, 38.0 * k, 38.0 * k, 29.0 * k
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
        offset = 22.0 * k
        above = mid_y > box_h + offset
        top = mid_y - box_h - offset if above else mid_y + offset
        left = min(max(mid_x - box_w / 2, 4), 1000 - box_w - 4)
        gap_text = f"{'+' if sector['time_a'] >= sector['time_b'] else '-'}{sector['gap']:.3f}s"
        tips.append(
            f'<g id="tdt{index}" class="tip" transform="translate({left:.1f},{top:.1f})">'
            f'<rect width="{box_w:.1f}" height="{box_h:.1f}" rx="{14 * k:.1f}"/>'
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


def format_odds(percent):
    """A championship share, rounded to what the simulation can actually
    support. 8000 trials can't tell 0.0125% from zero, and printing four
    decimal places claims it can."""
    if percent >= 1:
        return f"{percent:.0f}%"
    if percent >= 0.1:
        return f"{percent:.1f}%"
    return "&lt;0.1%" if percent > 0 else "0%"


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


def form_cell(scores, color, scale):
    """A standings row's last five rounds as a tiny bar chart (see .formbars).
    scores: [(round, points)], oldest first. One fixed scale for every row
    -- a full-points weekend -- so the bars compare down the table, not
    only along a row."""
    bars = "".join(
        f'<i class="{"zero" if pts <= 0 else ""}" '
        f'style="height:{max(2.0, 22 * min(pts, scale) / scale):.0f}px;--fc:{color}" '
        f'title="Round {rnd}: {pts:.0f} pts"></i>'
        for rnd, pts in scores[-5:]
    )
    return f'<span class="formbars">{bars}</span>'


def standings_history(progress, entity_col, round_number):
    """From the running totals round by round: each entrant's points per
    round, and their official position after the round before this one
    (Ergast's, so a tie on points is settled by count-back as it should be,
    not by whichever order a sort happens to leave it in)."""
    running = progress.pivot_table(index=entity_col, columns="Round", values="Points", aggfunc="last")
    running = running.reindex(columns=sorted(running.columns)).ffill(axis=1).fillna(0)
    per_round = running.diff(axis=1)
    per_round[running.columns[0]] = running[running.columns[0]]
    scores = {name: list(zip(per_round.columns, row)) for name, row in per_round.iterrows()}
    before = progress[progress["Round"] == round_number - 1].dropna(subset=["Position"])
    previous = dict(zip(before[entity_col], before["Position"].astype(int)))
    return scores, previous


def moved_badge(previous_rank, rank):
    """▲2 / ▼1 beside a standings position: places gained since the round
    before. Nothing when it didn't change."""
    if previous_rank is None or pd.isna(rank):
        return ""
    moved = int(previous_rank - rank)
    if moved == 0:
        return ""
    return f'<span class="moved {"up" if moved > 0 else "down"}">{"▲" if moved > 0 else "▼"}{abs(moved)}</span>'


def format_lap_time(td):
    if pd.isna(td):
        return "—"
    total = td.total_seconds()
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes}:{seconds:06.3f}" if minutes else f"{seconds:.3f}"


# Short codes for everyone the classification doesn't place, read from
# FastF1's ClassifiedPosition letter: a table cell says DNF, not "Retired
# (43 laps)" or the driver's name and "did not finish".
NOT_CLASSIFIED = {"R": "DNF", "D": "DSQ", "W": "DNS", "N": "NC", "E": "EXC", "F": "DNQ"}


def result_code(row):
    """DNF / DSQ / DNS / ... for an unclassified driver, else None."""
    return NOT_CLASSIFIED.get(str(row.get("ClassifiedPosition", "")).strip())


def format_race_gap(row, leader_laps):
    status = row.get("Status")
    if result_code(row):
        return result_code(row)
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
    """Finishing/classification order (P1 first) when available. Practice
    has no classification, so there -- and for any driver missing from it --
    the order is by best lap, as a timing screen ranks a practice session.
    It used to fall back to alphabetical, which made every chart that opens
    on "the top two" open on Albon against Alonso."""
    try:
        pos = session.results.set_index("Abbreviation")["Position"].dropna()
        ordered = [d for d in pos.sort_values().index if d in drivers]
    except Exception:
        ordered = []
    remaining = [d for d in drivers if d not in ordered]
    try:
        best = session.laps.groupby("Driver")["LapTime"].min()
    except Exception:
        best = pd.Series(dtype="timedelta64[ns]")
    remaining.sort(key=lambda d: (pd.isna(best.get(d)), best.get(d) if pd.notna(best.get(d)) else pd.Timedelta(0), d))
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
        color = readable(fastf1.plotting.get_driver_color(driver, session))
        if color in seen_colors:
            color = "#ffffff"
        else:
            seen_colors.add(color)
        styles[driver] = (color, "solid")
    return styles


NEUTRALISED = set("4567")  # safety car, red flag, VSC deployed, VSC ending
SLOW_LAP = 1.06  # a lap this much slower than the field's, same lap, is an incident
OPENING_LAPS = 4  # the field still sorting itself out after the start
# The rule set was checked against published per-race totals: Spain 2026 had
# 4 overtakes and Monaco 2026 had 10, and these rules give exactly those, move
# for move. Lap 1 alone, or no own-out-lap rule, gave 11 in Spain; a 5% slow
# threshold dropped LIN on ALB at Monaco (ALB 5.1% off the field that lap).


def detect_passes(session):
    """Every on-track pass of the race as (lap, passer, passed).

    Pair by pair rather than by net places gained: A passed B on lap N if A
    was behind B at the end of lap N-1 and ahead at the end of lap N. Each
    pass is counted once, with who made it and who suffered it, and several
    passes in one lap all count. What's thrown out is every way a pair can
    swap order without an overtake:

    - either car in the pit lane on lap N (in-lap or out-lap);
    - the passed car retiring during lap N (it has no lap N);
    - the passed car having a problem: a lap N more than 5% slower than the
      field's median for that same lap is a spin, damage or a failure, not a
      pass. Same lap, not the driver's own usual pace: under a yellow or in
      the opening laps the whole field is slower, and measured against their
      normal pace a car limping home (HAM, Spain 2026: +8% on the field while
      seven cars went by) sat under any threshold that still let a car
      simply losing a battle count as passed;
    - a lap run entirely under neutralisation. The track status of a lap is
      chronological, so one that *ends* in safety car / VSC / red flag is
      neutralised, while one that starts under the safety car and ends green
      is a restart -- and restart passes are real ones;
    - the opening laps (OPENING_LAPS), after the start and after a red-flag
      restart: the field sorting itself out, not passing;
    - the passer's first lap after its own pit stop: fresh tyres against
      old, an undercut completing, which published counts leave out --
      unless that lap is a restart (a stop made under the safety car, then
      a pass at the green flag is a real one);
    - a pass race control made the driver hand back ("advantage" / "give
      back"): both the pass and the handing back are dropped.

    Lapping doesn't need a rule: race order doesn't change when a leader
    laps a backmarker. A deleted lap time for track limits isn't a reason to
    drop a pass either -- it's about the lap time, not the move. What this
    can't see: a pass and a re-pass inside the same lap, since positions
    exist only at the line."""
    laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if laps.empty:
        return []
    laps = laps.assign(
        Pitting=laps["PitInTime"].notna() | laps["PitOutTime"].notna(),
        Seconds=laps["LapTime"].dt.total_seconds(),
        Status=laps["TrackStatus"].fillna("1").astype(str),
    )
    field = laps[~laps["Pitting"]].groupby("LapNumber")["Seconds"].median()

    # Laps that are starts: lap 1, and the first lap after one with a red flag.
    red_laps = set(laps.loc[laps["Status"].str.contains("5"), "LapNumber"].astype(int))
    # A red-flag restart is a start too (a standing one since 2021), so the
    # same opening laps after it are left out: Monza 2026, stopped on laps
    # 2-3, otherwise scored 21 "overtakes" on lap 6 as the field resorted.
    start_laps = set(range(1, OPENING_LAPS + 1))
    # Only a standing restart, though: Monaco 2026, stopped on lap 67,
    # restarted behind the safety car, and the passes at the green flag there
    # are real ones (and part of its published 10). Rolling restart = the
    # leader's first laps after the stoppage include safety-car running.
    if red_laps:
        restart = max(red_laps)
        leader_status = (
            laps[laps["Position"] == 1].drop_duplicates("LapNumber").set_index("LapNumber")["Status"]
        )
        after = "".join(str(leader_status.get(n, "")) for n in (restart + 1, restart + 2))
        if "4" not in after:
            start_laps |= set(range(restart + 1, restart + 1 + OPENING_LAPS))

    # Race control telling a driver to give a place back.
    handed_back = set()
    try:
        for msg in session.race_control_messages["Message"].astype(str):
            m = re.search(r"CAR \d+ \((\w{3})\).*(GAINING AN ADVANTAGE|GIVE (?:THE )?POSITION BACK|POSITION.*RETURNED)", msg)
            if m:
                handed_back.add(m.group(1))
    except Exception:
        pass

    by_lap = {int(n): g.set_index("Driver") for n, g in laps.groupby("LapNumber")}
    passes = []
    for n in sorted(by_lap):
        if n in start_laps or (n - 1) not in by_lap:
            continue
        prev, cur = by_lap[n - 1], by_lap[n]
        # A lap that ends neutralised: no passing allowed, any swap is a pit
        # stop or a penalty. Read off the leader's lap, which the whole field
        # shares for track status.
        leader = cur["Position"].idxmin()
        if cur.loc[leader, "Status"][-1] in NEUTRALISED:
            continue
        # Started under a safety car or red flag, ended green: a restart.
        restart = any(ch in cur.loc[leader, "Status"][:-1] for ch in "45")
        for a in cur.index:
            if a not in prev.index or cur.loc[a, "Pitting"]:
                continue
            if prev.loc[a, "Pitting"] and not restart:
                continue
            for b in cur.index:
                if b == a or b not in prev.index or cur.loc[b, "Pitting"]:
                    continue
                if not (prev.loc[a, "Position"] > prev.loc[b, "Position"]
                        and cur.loc[a, "Position"] < cur.loc[b, "Position"]):
                    continue
                slow = cur.loc[b, "Seconds"]
                if pd.notna(slow) and pd.notna(field.get(n)) and slow > SLOW_LAP * field[n]:
                    continue  # b had a problem this lap
                passes.append((n, a, b))

    # Handed back: a pass by a driver race control named, undone by the same
    # rival within the next three laps, cancels out -- both moves go.
    if handed_back:
        dropped = set()
        for i, (n, a, b) in enumerate(passes):
            if a not in handed_back or i in dropped:
                continue
            for j, (m, a2, b2) in enumerate(passes):
                if j not in dropped and (a2, b2) == (b, a) and n < m <= n + 3:
                    dropped |= {i, j}
                    break
        passes = [p for i, p in enumerate(passes) if i not in dropped]
    return passes


def count_overtakes(session):
    """Passes made per driver (see detect_passes). Shared by the per-race
    chart and the season stats so the two can't count differently."""
    counts = {d: 0 for d in session.laps["Driver"].dropna().unique()}
    for _, passer, _ in detect_passes(session):
        counts[passer] = counts.get(passer, 0) + 1
    return counts


def base_figure(title="", yaxis_title="", xaxis_title="", hovermode="x"):
    """A themed, empty figure.

    Titles are normally left empty now: every chart sits in a panel whose own
    header carries the title (see chart_panel), and drawing it twice -- once
    in HTML, once inside the plot area -- was both redundant and misaligned,
    since the in-figure title indents with the plot margin and the card's
    doesn't. Passing a title still works for the rare chart that isn't in a
    panel; the top margin follows automatically.
    """
    fig = go.Figure()
    # The title key is left out entirely when there's no title, not set to
    # None: plotly.js renders a null title as the literal string "undefined"
    # in the top-left of the plot area (seen on every panel-titled chart).
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15, color="#f2f4f6"), x=0, xanchor="left"))
    fig.update_layout(
        **DARK_LAYOUT,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=CHART_HEIGHT,
        hovermode=hovermode,
        margin=dict(t=42 if title else 16, b=44, l=8, r=18),
        bargap=0.32,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11.5),
        ),
    )
    # Axis titles read as small caps labels rather than sentence-case prose,
    # so they sit closer to the tick labels than to the data.
    axis_title = dict(font=dict(size=11, color="rgba(226,232,240,0.45)"), standoff=10)
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1, spikedash="dot",
        spikecolor="rgba(255,255,255,0.25)",
        gridcolor=PALETTE["grid"], zeroline=False,
        linecolor=PALETTE["axis"], ticks="outside", tickcolor="rgba(255,255,255,0.10)",
        ticklen=4, tickfont=dict(size=11.5), title=axis_title,
    )
    fig.update_yaxes(
        gridcolor=PALETTE["grid"], zeroline=False,
        linecolor="rgba(0,0,0,0)", tickfont=dict(size=11.5), title=axis_title,
    )
    return fig


def spread_labels(values, plot_height_px, value_span, min_gap_px=17.0):
    """Pixel offsets that stop a column of end-of-line labels overlapping.

    A label sits at its line's final value, and lines that finish close
    together produce labels drawn on top of each other -- unreadable exactly
    where a championship is tightest. This walks the labels from the top down
    and pushes each one just far enough below the previous to clear it,
    returning the offset (in pixels, positive = down) for each input value in
    its original order. The label keeps a leader line back to its real point,
    so the nudge never misrepresents where the line actually ended.
    """
    if not len(values) or value_span <= 0:
        return [0.0] * len(values)
    # Data units -> pixels down from the top of the plot area.
    def to_px(value):
        return plot_height_px * (1 - value / value_span)

    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    wanted = {i: to_px(values[i]) for i in order}

    # Pass one, top down: push each label just far enough below the previous
    # one to clear it.
    placed = {}
    previous = None
    for i in order:
        position = wanted[i] if previous is None else max(wanted[i], previous + min_gap_px)
        placed[i] = position
        previous = position

    # Pass two, bottom up: the first pass can only push *down*, so a tight
    # cluster near the bottom of the chart -- the four teams on single-figure
    # points -- ended up below the plot area entirely, and the axis grew a
    # -100 gridline to accommodate the labels. This pulls the overflow back up
    # inside the frame; between the two passes every label lands in
    # [0, plot_height] whenever the labels can physically fit.
    bottom_limit = plot_height_px
    previous = None
    for i in reversed(order):
        position = min(placed[i], bottom_limit if previous is None else previous - min_gap_px)
        placed[i] = max(position, 0.0)
        previous = placed[i]

    return [placed[i] - wanted[i] for i in range(len(values))]


def style_bars(fig, radius=5):
    """Rounded bar ends, applied after the traces are added.

    Every bar chart in the app goes through this, so none of them can drift
    back to Plotly's stock square corners on their own.
    """
    fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
    fig.update_traces(
        textfont=dict(size=11.5, color="rgba(226,232,240,0.75)"), selector=dict(type="bar"),
    )


# ------------------------------------------------------------ page furniture --

@contextmanager
def chart_panel(title, hint="", accent=None):
    """A chart in a card: header strip (title + optional one-line hint), then
    whatever the caller draws inside.

    Streamlit can't wrap an arbitrary widget in custom HTML, so the card is
    its own bordered container and the header is a markdown block inside it;
    the stylesheet finds the pair through :has(.panel-head).
    """
    with st.container(border=True):
        st.markdown(
            f'<div class="panel-head" style="--accent:{accent or PALETTE["red"]}">'
            f'<div class="title">{title}</div>'
            + (f'<div class="hint">{hint}</div>' if hint else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        yield


def section_head(heading, eyebrow="", lede="", accent=None):
    st.markdown(
        f'<div class="section-head" style="--accent:{accent or PALETTE["red"]}">'
        + (f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else "")
        + f'<div class="heading">{heading}</div>'
        + (f'<div class="lede">{lede}</div>' if lede else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def stat_cards(cards, per_row=None):
    """A row of stat cards: (label, value, sub, accent) tuples.

    Replaces st.metric + st.caption pairs throughout -- see the .stat-card CSS
    for why the built-in widget couldn't carry these values.
    """
    if not cards:
        return
    per_row = per_row or len(cards)
    for start in range(0, len(cards), per_row):
        chunk = cards[start : start + per_row]
        columns = st.columns(per_row)
        for col, (label, value, sub, accent) in zip(columns, chunk):
            col.markdown(
                f'<div class="stat-card" style="--card-accent:{accent}">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                + (f'<div class="sub">{sub}</div>' if sub else "")
                + "</div>",
                unsafe_allow_html=True,
            )


def chips(items):
    """A row of colored legend chips -- (color, text) pairs."""
    st.markdown(
        '<div class="chip-row">'
        + "".join(
            f'<span class="chip" style="--chip-color:{color}"><i></i>{text}</span>'
            for color, text in items
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def method_note(text, label="How this is worked out"):
    """Methodology text, collapsed.

    These notes are the difference between a number that can be trusted and
    one that can't, so none of them were cut -- but several ran to a full
    paragraph directly above the chart they described, which made the page
    read as documentation with figures in it rather than a dashboard.
    """
    with st.expander(label):
        st.markdown(text)


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


def tab_slug(label):
    return label.lower().replace(" ", "-")


def url_tabs(labels, key):
    """st.tabs that remember which one is open -- in the URL (?tab=pace), so
    a shared link lands on the tab it was copied from -- and that run only the
    open tab's code. Plain st.tabs runs every tab on every rerun, whatever is
    on screen: picking a driver on Head-to-head recomputed the whole Pace tab
    underneath it, and opening Season loaded every race of the year for the
    Race by race tab before showing the standings. on_change="rerun" is what
    gives each tab its .open flag; showing() reads it.

    default= is always the tab that's open already. Streamlit counts it as
    part of the tabs' identity, so a default that changed between reruns --
    the URL's tab on the first run, nothing after it -- made them a new
    widget, back on its first tab: picking a driver on Head-to-head bounced
    the page to Results. Moving to a session with other tabs keeps the
    reader on the same one, or its counterpart (Pace <-> Stats)."""
    counterpart = {"Pace": "Stats", "Stats": "Pace"}
    open_now = st.session_state.get(key)
    wanted = st.query_params.get("tab")
    # Links shared before two tabs were renamed (2026-09-25) keep working.
    wanted = {"head-to-head": "telemetry", "race-by-race": "stats"}.get(wanted, wanted)
    from_url = next((label for label in labels if tab_slug(label) == wanted), None)
    default = next(
        (label for label in (open_now, counterpart.get(open_now), from_url) if label in labels), None,
    )
    if default == labels[0]:
        default = None
    tabs = st.tabs(labels, default=default, key=key, on_change="rerun")
    current = st.session_state.get(key) or labels[0]
    if current == labels[0]:
        st.query_params.pop("tab", None)
    else:
        st.query_params["tab"] = tab_slug(current)
    if tab_slug(current) not in ("pace", "telemetry"):
        st.query_params.pop("drivers", None)
    return tabs


def showing(tab):
    """Whether a tab's content should run this time: it's the open one, or
    the tabs don't track state (open is None) and everything runs."""
    return tab is not None and tab.open is not False


def keep_valid(key, options):
    """Drop a remembered choice the new options no longer contain (a Grand
    Prix after the year changed, Sprint after moving to a weekend without
    one), so the widget falls back to its default -- the latest round, the
    race -- rather than raising over a value it can't show."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="flag"></div>'
    '<div class="word"><em>F1</em> DASHBOARD'
    '<span>TELEMETRY · STANDINGS · RECORDS</span></div></div>',
    unsafe_allow_html=True,
)
# Every choice in the sidebar is bound to the URL (bind="query-params"), so
# what's on screen can be shared as a link -- ?gp=Spanish&session=Quali&tab=
# head-to-head&drivers=NOR,ANT opens exactly that, and ?view=Season the
# season (a bound widget writes the label it shows, which is why the Grand
# Prix and session labels are the short ones). Before
# this, any link to the app opened on whatever the latest session was, and
# "look at this" meant describing four dropdowns in a message. Values equal
# to the default drop out of the URL on their own, so the bare address stays
# bare.
#
# The section picker is the sidebar's navigation, so it's styled as one (see
# the sidebar CSS): full-width rows with an icon and a line saying what's
# there, no radio circles, no "Section" label over it.
section = st.sidebar.radio(
    "Section", [SECTION_WEEKEND, SECTION_SEASON, SECTION_ALL_TIME], key="view", bind="query-params",
    captions=["Timing, telemetry, strategy", "Standings, form, title fight", "Every race since 1950"],
    label_visibility="collapsed",
)

# Short labels for the session buttons -- five of them have to share one row
# of a 300px sidebar. They're also what goes in the URL (?session=Quali).
SESSION_SHORT = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Quali",
    "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ", "Sprint": "Sprint", "Race": "Race",
}


def short_gp(name):
    return name.replace(" Grand Prix", "")


session = None
if section in (SECTION_WEEKEND, SECTION_SEASON):
    years = available_years()
    keep_valid("year", years)
    # Year and Grand Prix share a row: the year is four digits and was
    # taking a full-width field, a label and a gap of its own.
    col_year, col_gp = st.sidebar.columns([1.25, 2], gap="small")
    with col_year:
        year = st.selectbox("Year", options=years, key="year", bind="query-params")
    schedule = load_schedule(year, include_in_progress=section == SECTION_WEEKEND)
    published = published_sessions(year) if data_store_token() is not None else None
    if published is not None:
        schedule = schedule[schedule["EventName"].isin(published)]
    event_options = schedule["EventName"].tolist()
    keep_valid("gp", event_options)
    with col_gp:
        event_name = st.selectbox(
            "Grand Prix" if section == SECTION_WEEKEND else "Standings after",
            options=event_options, format_func=short_gp, key="gp", bind="query-params",
        )
    event_row = schedule[schedule["EventName"] == event_name].iloc[0]
    # The season views don't compare laps, but they do color drivers and
    # teams, and fastf1.plotting needs a loaded session to do that -- the
    # weekend's race is the one that's always there.
    if section == SECTION_WEEKEND:
        # One button per session, in the order the weekend runs, rather than
        # a dropdown: every session is in view and one tap away. Opens on the
        # latest one.
        session_options = [
            n for n in reversed(session_names_for(event_row)) if published is None or n in published[event_name]
        ]
        keep_valid("session", session_options)
        session_name = st.sidebar.segmented_control(
            "Session", session_options, format_func=lambda n: SESSION_SHORT.get(n, n),
            default=session_options[-1], required=True, key="session", bind="query-params", width="stretch",
        ) or session_options[-1]
    else:
        session_name = "Race"
    try:
        # The season views only need the session for driver and team colors,
        # so they skip its telemetry.
        session = load_session(year, event_name, session_name, telemetry=section == SECTION_WEEKEND)
    except Exception as exc:
        st.error(f"Couldn't load this session (it may not have happened yet): {exc}")
        st.stop()

if section == SECTION_WEEKEND:
    render_hero(
        f"Round {int(event_row['RoundNumber'])}",
        f"{year} season",
        event_name,
        f"{event_row['OfficialEventName'].title() if 'OfficialEventName' in event_row and pd.notna(event_row['OfficialEventName']) else event_name}",
        circuit_outline(session),
        # No flag emoji on the country chip: Windows ships no glyphs for the
        # regional-indicator pairs, so every flag renders as the two letters of
        # its country code -- "ES Spain" -- which is worse than no flag at all.
        chip_items=[
            f"\U0001F30D {event_row['Country']}",
            f"\U0001F4CD {event_row['Location']}",
            f"\U0001F4C5 {pd.to_datetime(event_row['EventDate']).strftime('%d %b %Y')}",
            f"\U0001F3AC {session_name}",
            track_temperature_chip(session),
        ],
    )
    # The headline numbers sit above the tabs, not inside one of them: they
    # describe the session itself, and a reader shouldn't have to guess which
    # tab hides "who won".
    stat_cards(weekend_headline_cards(session, session_name))
    st.write("")
    # One tab per question, in the order a weekend is read: what happened
    # (Results: classification, positions, overtakes), how it was run
    # (Strategy: tyres and pit stops -- races and sprints only), who was quick
    # (Pace; "Stats" for qualifying: gap to pole, ideal lap, car
    # characteristics), then the lap-level forensics (Telemetry -- called
    # Head-to-head until 2026-09-25, a name it shared with a Pace chart and
    # the Teammates table). Charts
    # used to pile up in one "Race pace" tab whatever they were about.
    is_race_like = session_name in ("Race", "Sprint")
    if "Qualifying" in session_name:
        tab_labels = ["Results", "Stats", "Telemetry"]
    elif is_race_like:
        tab_labels = ["Results", "Strategy", "Pace", "Telemetry"]
    else:
        tab_labels = ["Results", "Pace", "Telemetry"]
    weekend_tabs = url_tabs(tab_labels, "tabs_weekend")
    tab_classification, tab_pace, tab_telemetry = weekend_tabs[0], weekend_tabs[-2], weekend_tabs[-1]
    tab_strategy = weekend_tabs[1] if is_race_like else None
elif section == SECTION_SEASON:
    render_hero(
        f"{year}", "World championship",
        f"{year} season",
        f"Standings and stats as they stood after the {event_name}",
        circuit_outline(session),
        chip_items=[
            f"\U0001F3C1 {len(schedule)} races run",
            f"\U0001F4CD latest: {event_name.replace(' Grand Prix', '')}",
            f"\U0001F4CA round {int(event_row['RoundNumber'])}",
        ],
    )
    tab_standings, tab_season_stats, tab_teammates, tab_pits = url_tabs(
        ["Championship", "Stats", "Teammates", "Pit stops"], "tabs_season"
    )
else:
    render_hero(
        "1950 — today", "World championship",
        "All-time records",
        "Every world-championship race ever run, ranked",
        chip_items=["\U0001F3C6 wins", "\U0001F3C1 poles", "\U0001F4C8 streaks", f"\U0001F551 {datetime.date.today().year - 1949} seasons"],
    )
    for stale in ("tab", "drivers"):
        st.query_params.pop(stale, None)

st.sidebar.markdown(next_race_card() + freshness_line(), unsafe_allow_html=True)
# The small print, folded away: it was four lines at the foot of every page
# that nobody needs twice.
with st.sidebar.expander("About the data"):
    st.markdown(
        '<div class="sidebar-foot">'
        '<b>Data</b> · FastF1 (official timing &amp; telemetry), Ergast, DHL pit stop times<br>'
        '<b>Cache</b> · every session is fetched once, then read from disk<br>'
        '<b>Note</b> · overtake counts and title odds are estimates<br>'
        '<b>Unofficial</b> · not affiliated with Formula 1, the FIA or any team'
        '</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------- telemetry --

EMPTY_ICONS = {
    # Arrow up to the picker above; a dashed circle for "nothing here"; a
    # small warning mark for a partial result.
    "prompt": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>',
    "empty": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
             'stroke-dasharray="3 3"><circle cx="12" cy="12" r="9"/></svg>',
    "note": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round"><path d="M12 8v5M12 16.5v.5"/><circle cx="12" cy="12" r="9"/></svg>',
}


def empty_state(text, kind="empty"):
    """What a chart shows when there is nothing to draw -- see the .empty-state
    styles. kind: "prompt" (waiting on a choice above), "empty" (no data for
    this session), "note" (a partial result worth flagging). Real failures
    stay on st.error: those should stand out."""
    st.markdown(
        f'<div class="empty-state {kind}">{EMPTY_ICONS[kind]}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def pill_colors(key, colors):
    """Give each pill of a "pick_" picker its own --pill color, by position:
    the buttons carry no value attribute to match on, only their order."""
    rules = [f".st-key-{key} button:nth-of-type({i}) {{ --pill: {c}; }}" for i, c in enumerate(colors, start=1)]
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)


def session_slug():
    return re.sub(r"\W+", "_", f"{year}_{event_name}_{session_name}")


def picker_key(name):
    # The "pick_" prefix is what the pill stylesheet matches on; the session
    # in the key makes a new race or session start from nothing.
    return f"pick_drivers_{name}_{session_slug()}"


def driver_picker(name, label, help_text, eligible, on_pick=None, limit=MAX_DRIVERS, default=(), share=False):
    """A two-driver choice: one pill per driver, teammates side by side, each
    marked with its team color. Used by the Head-to-head tab and by the race
    pace comparison; `name` keeps their selections apart, `eligible` is who
    can be picked there. Replaced a multiselect dropdown,
    which took an open-scroll-pick-reopen round per driver, offered a "Select
    all" that made no sense with a limit of two, and simply refused a third
    pick. Here every driver is in view, one click toggles, and a third pick
    replaces the older of the two.

    Never starts empty: with no `default` it opens on the top of the session
    (P1 and P2, or P1 alone for a one-driver picker), so every chart behind
    a picker is already drawn when the tab opens. It used to start empty on
    principle ("a choice, not whichever two come first"), which left the
    ghost lap -- the app's best view -- behind a blank prompt; Peter reversed
    that on 2026-09-24. The widget key
    carries the session, but with share=True the pair also lives in the URL
    (?drivers=VER,NOR), so it follows the reader to another session or race
    -- qualifying, then the race, same two drivers -- minus anyone the new
    session doesn't have.
    """
    results = session.results.sort_values("Position")
    with_laps = set(eligible)
    # Teams in order of their best finisher, teammates adjacent.
    teams = list(dict.fromkeys(results["TeamName"]))
    order = [
        d for team in teams
        for d in results.loc[results["TeamName"] == team, "Abbreviation"] if d in with_laps
    ]
    order += sorted(with_laps - set(order))  # anyone the results don't list

    key = picker_key(name)
    history_key = key + "_order"
    if not default:
        default = order_by_classification(session, list(eligible))[:limit]

    def keep_newest_two():
        # st.pills has no max_selections, and it reports the selection in
        # option order, not click order -- so the click order is tracked
        # here, both to know which pick is the oldest and to keep the first
        # driver chosen as the reference lap.
        current = st.session_state.get(key) or []
        previous = st.session_state.get(history_key, [])
        ordered = [d for d in previous if d in current] + [d for d in current if d not in previous]
        ordered = ordered[-limit:]
        st.session_state[key] = ordered
        st.session_state[history_key] = ordered
        if on_pick:
            on_pick()

    # A starting selection goes in through session state, not the widget's
    # default=: the callback above writes st.session_state[key], and a widget
    # given both a default and a state-set value warns on screen. What it
    # starts from, in order: the pick made here before (the click-order list
    # is plain session state, which outlives the widget -- the widget's own
    # state is dropped whenever its tab isn't the one open, and with only the
    # open tab running now, that was every tab switch); the pair in the URL,
    # for a shared link; and only then the caller's default.
    if key not in st.session_state:
        from_url = [d for d in st.query_params.get("drivers", "").split(",") if d in order] if share else []
        # A shared pair missing a driver the default has (the pace chart
        # opens on two) isn't a pair any more; the default takes over.
        if len(from_url) < len(list(default)[:limit]):
            from_url = []
        remembered = st.session_state.get(history_key) or from_url or list(default)
        initial = [d for d in remembered if d in order][-limit:]
        st.session_state[key] = initial
        st.session_state[history_key] = initial
    st.pills(
        label, options=order, selection_mode="multi", key=key,
        on_change=keep_newest_two, help=help_text,
    )
    selected = [d for d in st.session_state.get(history_key, []) if d in order]
    if share:
        if selected:
            st.query_params["drivers"] = ",".join(selected)
        else:
            st.query_params.pop("drivers", None)

    # Team color per pill, via nth-of-type on the buttons in option order.
    # A selected pill takes the color its line will have on the charts, which
    # for the second driver of a team is white (see build_driver_styles).
    line_color = {d: c for d, (c, _) in build_driver_styles(selected, session).items()}
    pill_colors(key, [line_color.get(d) or safe_driver_color(d, session) for d in order])
    return selected


def render_telemetry_tab():
    """Head-to-head telemetry for the two selected drivers.

    A function rather than inline tab code so that a session with no
    usable lap can bail out with a return: st.stop() would take the
    sibling tabs down with it, since Streamlit runs the whole script
    top to bottom on every rerun whatever tab is on screen.
    """
    selected_drivers = driver_picker(
        "telemetry", "Choose two drivers to compare",
        "The first driver you pick is the reference lap. Picking a third replaces the older of the two.",
        session.laps["Driver"].unique(), share=True,
    )
    # st.segmented_control rather than a radio: this is a two-way toggle
    # between unit systems, not a list of choices, and the segmented
    # control renders it as one -- it can also return None (nothing
    # selected), hence the fallback.
    units = st.segmented_control(
        "Units", options=["Metric", "Imperial"], default="Metric", key="telemetry_units",
    )
    imperial = units == "Imperial"
    speed_unit = "mph" if imperial else "km/h"
    dist_unit = "ft" if imperial else "m"

    if not selected_drivers:
        empty_state("Choose one or two drivers above to compare their fastest laps", "prompt")
        return

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
        empty_state(f"No timed lap for {', '.join(missing)} in this session, so left out", "note")
    selected_drivers = [d for d in selected_drivers if d in laps_by_driver]
    if not selected_drivers:
        empty_state("None of the selected drivers set a timed lap in this session")
        return

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

    cards = st.columns(len(selected_drivers))
    for col, driver in zip(cards, selected_drivers):
        lap = laps_by_driver[driver]
        color, _dash = driver_style[driver]
        speed_series = telemetry_by_driver[driver]["Speed"] * (KM_TO_MI if imperial else 1)
        col.markdown(
            f"""
            <div class="driver-card" style="--card-color:{color}">
                <div class="name">{driver}</div>
                <div class="laptime">{format_lap_time(lap["LapTime"])}</div>
                <div class="sub">Top speed {speed_series.max():.0f} {speed_unit} · average {speed_series.mean():.0f} {speed_unit}</div>
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

    if len(selected_drivers) < 2:
        empty_state("Pick a second driver to see where on the lap each one is faster", "prompt")
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

        # The two laps raced against each other, before the charts that take
        # them apart: the lap trace below says where the time went, this
        # shows it happening -- who brakes later, who gets on the power
        # first, and the gap opening and closing as they go.
        with chart_panel(
            "Ghost lap",
            f"{dom_a} and {dom_b}'s fastest laps on the same track at the same time · "
            "speed, gear, pedals and the live gap",
            accent=color_a,
        ):
            components.html(
                ghost_lap_html(
                    ghost_lap_payload(
                        common_distance_m, x_on_grid, y_on_grid,
                        [
                            {
                                "code": d, "color": c,
                                "lap_time": laps_by_driver[d]["LapTime"].total_seconds(),
                                "elapsed": resampled[d]["Elapsed"], "speed": resampled[d]["Speed"],
                                "throttle": resampled[d]["Throttle"], "brake": resampled[d]["Brake"],
                                "gear": resampled[d]["nGear"],
                            }
                            for d, c in ((dom_a, color_a), (dom_b, color_b))
                        ],
                        sector_distances=shared_checkpoint_distance[1:3] if sector_checkpoints else (),
                        speed_unit=speed_unit, speed_factor=KM_TO_MI if imperial else 1.0,
                    ),
                    height=600,
                ),
                height=610,
            )

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
        with chart_panel(
            "Track dominance",
            f"Hover any stretch of track for its mini-sector times · {dom_a} vs {dom_b}",
            accent=color_a,
        ):
            render_track_map(x_on_grid, y_on_grid, sector_paths, height=560)
            chips([(color_a, f"{dom_a} faster"), (color_b, f"{dom_b} faster")])

        method_note(
            f"The lap is split into **{TRACK_SECTOR_COUNT} mini-sectors** whose length follows this "
            "track's own lap distance, so every circuit gets its own boundaries. Each one is colored "
            "by whichever driver actually spent less time crossing it -- a corner-by-corner read on "
            "downforce/drag tradeoffs rather than overall pace. The comparison uses elapsed time "
            "across the sector, not instantaneous speed: which trace happens to be higher at one "
            "instant can flip mid-corner without saying who was quicker end to end.",
            "What the mini-sector colors mean",
        )

        # Official sector splits alongside the mini-sector map: the map
        # shows *where* the lap was won, these three rows show by how much,
        # against the timing that the sport itself recognises.
        sector_rows = []
        for index, field in enumerate(("Sector1Time", "Sector2Time", "Sector3Time"), start=1):
            time_a = laps_by_driver[dom_a][field]
            time_b = laps_by_driver[dom_b][field]
            if pd.isna(time_a) or pd.isna(time_b):
                continue
            sector_rows.append((f"Sector {index}", time_a.total_seconds(), time_b.total_seconds()))
        lap_a, lap_b = laps_by_driver[dom_a]["LapTime"], laps_by_driver[dom_b]["LapTime"]
        if pd.notna(lap_a) and pd.notna(lap_b):
            sector_rows.append(("Full lap", lap_a.total_seconds(), lap_b.total_seconds()))

        if sector_rows:
            with chart_panel(
                "Official sector splits",
                "Bar length is the gap; it points to whoever was quicker",
                accent=color_b,
            ):
                # A diverging bar around a zero line rather than two bars per
                # sector: the quantity that matters is the *gap*, and a pair
                # of 30-second bars differing by 0.08s shows nothing at all.
                fig_sectors = base_figure("", "", "Gap (s)", hovermode="closest")
                deltas = [b - a for _, a, b in sector_rows]  # >0 means dom_a quicker
                fig_sectors.add_trace(
                    go.Bar(
                        x=deltas,
                        y=[label for label, _, _ in sector_rows],
                        orientation="h",
                        marker=dict(
                            color=[color_a if d > 0 else color_b for d in deltas],
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=[
                            f"{dom_a if d > 0 else dom_b} by {abs(d):.3f}s"
                            for d in deltas
                        ],
                        textposition="outside", cliponaxis=False,
                        customdata=[[a, b] for _, a, b in sector_rows],
                        hovertemplate=(
                            f"%{{y}}<br>{dom_a}: %{{customdata[0]:.3f}}s"
                            f"<br>{dom_b}: %{{customdata[1]:.3f}}s<extra></extra>"
                        ),
                    )
                )
                fig_sectors.update_yaxes(autorange="reversed")
                fig_sectors.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1)
                size_horizontal_bars(fig_sectors, deltas, row_px=42, extra_px=90, pad_frac=0.45)
                style_bars(fig_sectors, radius=4)
                plotly_chart(fig_sectors, width="stretch", config=PLOTLY_CONFIG)

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
    # Headroom above the top speed: the S1/S2/S3 labels sit inside the top of
    # this panel, and on autorange the fastest stretch of the lap ran straight
    # through them. Room is made in the axis rather than the labels moved up,
    # because above the plot area S2 lands on the centred "Speed" title.
    all_speeds = np.concatenate(
        [np.asarray(resampled[d]["Speed"], dtype=float) for d in selected_drivers]
    ) * (KM_TO_MI if imperial else 1)
    speed_lo, speed_hi = np.nanmin(all_speeds), np.nanmax(all_speeds)
    speed_span = speed_hi - speed_lo
    fig_telemetry.update_yaxes(
        range=[speed_lo - 0.05 * speed_span, speed_hi + 0.2 * speed_span], row=1, col=1,
    )
    fig_telemetry.update_yaxes(title_text="s", row=2, col=1)
    fig_telemetry.update_yaxes(title_text="%", row=3, col=1)
    fig_telemetry.update_yaxes(tickvals=[0, 1], ticktext=["Off", "On"], range=[-0.15, 1.15], row=4, col=1)
    fig_telemetry.update_yaxes(tickvals=list(range(1, 9)), range=[0.5, 8.5], row=5, col=1)
    fig_telemetry.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=2, col=1)
    if sector_checkpoints:
        band_edges = [0.0] + list(shared_checkpoint_distance[1:3]) + [float(common_distance_m[-1])]
        edge_index = [int(np.argmin(np.abs(common_distance_m - edge))) for edge in band_edges]
        # One thin solid line at the end of S1 and of S2, through all five
        # panels. Tried before it: dotted lines at half opacity, which broke up
        # into noise across the dense throttle and brake traces, and a tinted
        # band over sector 2, which Peter found heavy -- a faint continuous
        # line marks the boundary without competing with the data.
        # Shapes take the *index* of the category, not its label. The x axis
        # here is a category axis (the tick labels carry the unit, which a
        # numeric axis can't), and on one of those a shape is positioned on the
        # underlying 0..n-1 scale -- passing the label string drew nothing at
        # all. Annotations are the exception: they do accept the label, which
        # is why the S1/S2/S3 markers showed up while the lines didn't.
        # exclude_empty_subplots=False: with row="all", Plotly skips every
        # subplot that has no trace yet, and the driver traces are only added
        # further down -- by default the lines were silently dropped from all
        # five panels.
        for idx in edge_index[1:3]:
            fig_telemetry.add_vline(
                x=idx, line_color="rgba(255,255,255,0.25)", line_width=1,
                row="all", col=1, exclude_empty_subplots=False,
            )
        # The labels sit *inside* the top of the speed panel, not above it: at
        # y=1 they landed in the same strip as the "Speed" subplot title and
        # the two overprinted each other.
        for i in range(3):
            mid = (band_edges[i] + band_edges[i + 1]) / 2
            idx = int(np.argmin(np.abs(common_distance_m - mid)))
            fig_telemetry.add_annotation(
                x=common_distance_labels[idx], y=0.97, yref="y domain", row=1, col=1,
                text=f"S{i + 1}", showarrow=False, yanchor="top",
                font=dict(size=10.5, color="rgba(226,232,240,0.6)"),
                bgcolor="rgba(11,14,21,0.6)", borderpad=3,
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
        # Throttle and brake are filled to the axis, not left as bare
        # lines: both are "how much of the time is this pedal down"
        # signals, and an area reads as coverage where a line reads as a
        # value to be traced point by point. The throttle fill is kept
        # very light so two drivers' areas can overlap without turning
        # into a single muddy block; brake is a near-binary trace, so its
        # fill can afford to be solid enough to spot at a glance.
        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Throttle"], name=driver, showlegend=False,
                line=dict(color=color, dash=dash, width=2),
                fill="tozeroy", fillcolor=hex_to_rgba(color, 0.08),
                text=[f"{v:.0f}" for v in resampled[driver]["Throttle"]],
                hovertemplate=f"{driver}: " + "%{text}%<extra></extra>",
            ),
            row=3, col=1,
        )
        brake_state = np.where(resampled[driver]["Brake"], "On", "Off")
        fig_telemetry.add_trace(
            go.Scatter(
                x=common_distance_labels, y=resampled[driver]["Brake"], name=driver, showlegend=False,
                line=dict(color=color, dash=dash, width=2, shape="hv"),
                fill="tozeroy", fillcolor=hex_to_rgba(color, 0.22),
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

    with chart_panel(
        "Lap trace",
        "Speed, delta, throttle, brake and gear -- drag any panel to zoom, all five stay in sync",
        accent=PALETTE["blue"],
    ):
        plotly_chart(fig_telemetry, width="stretch", config=PLOTLY_CONFIG)


if section == SECTION_WEEKEND and showing(tab_telemetry):
    with tab_telemetry:
        render_telemetry_tab()

def render_poles():
    """Poles by driver and by engine, season so far -- a season statistic,
    so on the Season view rather than on one qualifying session."""
    poles = pole_positions(year, tuple(schedule["EventName"].tolist()))
    if poles.empty:
        empty_state("No completed qualifying sessions to count poles from yet")
    else:
        col_poles_driver, col_poles_engine = st.columns(2)

        with col_poles_driver:
            with chart_panel("Poles by driver", f"{year} season so far", accent=PALETTE["violet"]):
                by_driver = poles["Driver"].value_counts().sort_values()
                fig_poles_driver = base_figure("", "", "Poles")
                fig_poles_driver.update_layout(showlegend=False)
                fig_poles_driver.add_trace(
                    go.Bar(
                        x=by_driver.to_numpy(), y=by_driver.index, orientation="h",
                        marker=dict(
                            color=[safe_driver_color(d, session) for d in by_driver.index],
                            line=dict(color="rgba(255,255,255,0.10)", width=1),
                        ),
                        text=by_driver.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_poles_driver, by_driver.to_numpy())
                style_bars(fig_poles_driver)
                plotly_chart(fig_poles_driver, width="stretch", config=PLOTLY_CONFIG)

        with col_poles_engine:
            with chart_panel("Poles by engine", "Power unit behind each pole", accent=PALETTE["blue"]):
                if year not in ENGINE_SUPPLIERS:
                    st.caption(f"No power unit table for {year} yet.")
                else:
                    engines = poles["Team"].map(ENGINE_SUPPLIERS[year]).fillna(poles["Team"])
                    by_engine = engines.value_counts().sort_values()
                    fig_poles_engine = base_figure("", "", "Poles")
                    fig_poles_engine.update_layout(showlegend=False)
                    fig_poles_engine.add_trace(
                        go.Bar(
                            x=by_engine.to_numpy(), y=by_engine.index, orientation="h",
                            marker=dict(
                                color=by_engine.to_numpy(), colorscale=bar_scale(PALETTE["blue"]),
                                line=dict(color="rgba(255,255,255,0.10)", width=1),
                            ),
                            text=by_engine.to_numpy(), textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}: %{x} pole(s)<extra></extra>",
                        )
                    )
                    size_horizontal_bars(fig_poles_engine, by_engine.to_numpy())
                    style_bars(fig_poles_engine)
                    plotly_chart(fig_poles_engine, width="stretch", config=PLOTLY_CONFIG)


# ------------------------------------------------------------------ pace --

def render_position_chart():
    """Every driver's position lap by lap -- the first chart on the Results
    tab. It replaced a grid-to-finish slope chart there: that one joined only
    each driver's start and finish, and this shows the same two ends plus the
    whole race in between, so the slope chart had nothing left to add."""
    position_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if position_laps.empty:
        return
    else:
        field_order = order_by_classification(session, position_laps["Driver"].unique())
        field_style = build_driver_styles(field_order, session)
        max_lap = position_laps["LapNumber"].max()

        with chart_panel(
            "Race position by lap",
            "Every driver's classified position at the end of each lap",
            accent=PALETTE["amber"],
        ):
            fig_position = base_figure("", "Position", "Lap")
            # An explicit range instead of autorange: Plotly padded a reversed
            # axis out to a P0 at the top and a P23 below the last finisher,
            # neither of which is a position anyone can be in.
            last_position = position_laps["Position"].max()
            fig_position.update_yaxes(autorange=False, range=[last_position + 0.6, 0.4], dtick=1)
            fig_position.update_xaxes(range=[position_laps["LapNumber"].min() - 1, max_lap + 3])
            fig_position.update_layout(height=max(CHART_HEIGHT, 23 * len(field_order)), showlegend=False)
            # A tinted band across the top three places: with twenty lines on
            # one chart, the thing a reader is actually looking for is who was
            # in the podium slots and when that changed, and a band answers it
            # without anyone counting gridlines down from the top.
            fig_position.add_hrect(
                y0=0.5, y1=3.5, fillcolor="rgba(255,179,64,0.07)", line_width=0, layer="below",
            )
            # Safety car, VSC and red-flag laps shaded, as on the pace chart:
            # a field that bunches up and stops swapping places makes sense
            # once you can see why. Read off the leader's laps, which carry
            # the track status the whole field shares.
            for _, lap_row in position_laps[position_laps["Position"] == 1].drop_duplicates("LapNumber").iterrows():
                status = str(lap_row["TrackStatus"]) if pd.notna(lap_row["TrackStatus"]) else ""
                if any(ch in status for ch in "4567"):
                    fig_position.add_vrect(
                        x0=lap_row["LapNumber"] - 1, x1=lap_row["LapNumber"],
                        fillcolor="rgba(255,179,64,0.08)", line_width=0, layer="below",
                    )
            fig_position.add_annotation(
                x=position_laps["LapNumber"].min() - 0.8, y=3.5, text="PODIUM", showarrow=False,
                xanchor="left", yanchor="bottom", textangle=0,
                font=dict(size=9, color="rgba(255,179,64,0.6)"),
            )

            for driver in field_order:
                driver_laps = position_laps[position_laps["Driver"] == driver].sort_values("LapNumber")
                color, dash = field_style[driver]
                # The three drivers who finished on the podium carry a heavier
                # line: at twenty traces, equal weight everywhere means the
                # front of the race is as hard to follow as the back of it.
                podium = field_order.index(driver) < 3
                # Lap number isn't repeated here -- it's already the x-axis value,
                # shown once at the bottom, not per driver.
                hover_text = [f"{driver}: P{p:.0f}" for p in driver_laps["Position"]]
                fig_position.add_trace(
                    go.Scatter(
                        x=driver_laps["LapNumber"], y=driver_laps["Position"], mode="lines",
                        line=dict(color=color, dash=dash, width=3 if podium else 1.7, shape="spline", smoothing=0.5),
                        opacity=1.0 if podium else 0.85,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )
                last = driver_laps.iloc[-1]
                # The end-of-line labels used to be bare text, which ran into
                # each other wherever two cars finished on consecutive laps;
                # a small plate gives each one its own background.
                fig_position.add_annotation(
                    x=last["LapNumber"], y=last["Position"], text=f"<b>{driver}</b>", showarrow=False,
                    xanchor="left", xshift=8, font=dict(size=10, color=color),
                    bgcolor="rgba(10,13,20,0.75)", bordercolor=color, borderwidth=1, borderpad=2,
                )

            plotly_chart(fig_position, width="stretch", config=PLOTLY_CONFIG)


def sector_tile(gap, widest):
    """One sector cell of the qualifying sector map: the gap to the
    session's best sector on a blue tile that fades the further it is from
    that best (the best itself is drawn by the caller, in purple)."""
    if pd.isna(gap):
        return "—"
    # Scaled to 0.8 s at most: one slow driver (a lap aborted in Q1, say)
    # would otherwise stretch the ramp so far that the front of the field all
    # read the same bright blue.
    scale = min(widest, 0.8) if widest else 0.8
    strength = 1 - min(1.0, gap / scale)
    alpha = 0.05 + 0.75 * strength ** 2.2
    return f'<span class="sectile" style="--heat:rgba(91,140,255,{alpha:.2f})">+{gap:.3f}</span>'


def render_quali_stats():
    """The sector map, then gap to pole and ideal lap, the two numbers a
    qualifying session is read by. A Q1 -> Q2 -> Q3 dot chart and an
    ideal-lap slope chart were tried in place of the two bar charts on
    2026-09-25 and removed the same day -- Peter preferred the bars and kept
    only the sector map. Deleted laps (track limits) are left out of all
    three: a lap that didn't count for the grid shouldn't count here either."""
    laps = session.laps
    if "Deleted" in laps.columns:
        laps = laps[laps["Deleted"] != True]  # noqa: E712 -- the column holds NaN too
    timed = laps.dropna(subset=["LapTime"])
    if timed.empty:
        empty_state("No timed laps in this session")
        return
    best = timed.groupby("Driver")["LapTime"].min().dt.total_seconds().sort_values()
    order = list(reversed(best.index))  # fastest at the top of a horizontal bar chart
    colors = [safe_driver_color(d, session) for d in order]

    sector_cols = ["Sector1Time", "Sector2Time", "Sector3Time"]
    sectors = timed.groupby("Driver")[sector_cols].min().apply(lambda c: c.dt.total_seconds())
    sectors = sectors.reindex(best.index)

    # Sectors only: the gap to pole and the time left on the table were
    # columns here too, the same numbers as the two bar charts below. The
    # table says where on the lap the time was, the bars say how much.
    with chart_panel(
        "Sector map",
        "Each driver's best time in each sector against the session's best · purple is the fastest",
        accent=PALETTE["violet"],
    ):
        session_best = sectors.min()
        widest = {c: (sectors[c] - session_best[c]).max() for c in sector_cols}
        rows = []
        for rank, driver in enumerate(best.index, start=1):
            row = {"pos": str(rank), "driver": driver, "color": safe_driver_color(driver, session)}
            for i, col in enumerate(sector_cols, start=1):
                value = sectors.at[driver, col]
                if pd.notna(value) and value == session_best[col]:
                    row[f"s{i}"] = f'<span class="sectile best">{value:.3f}</span>'
                else:
                    row[f"s{i}"] = sector_tile(value - session_best[col], widest[col])
            rows.append(row)
        render_table(rows, [
            ("pos", "Pos", "pos"), ("driver", "Driver", "name"),
            ("s1", "Sector 1", "sector"), ("s2", "Sector 2", "sector"), ("s3", "Sector 3", "sector"),
        ])

    st.write("")
    col_gap, col_ideal = st.columns(2)
    with col_gap:
        with chart_panel("Gap to pole", "Each driver's best lap, behind the fastest", accent=PALETTE["red"]):
            gap = (best - best.iloc[0])[order]
            fig_gap = base_figure("", "", "Seconds behind")
            fig_gap.update_layout(showlegend=False)
            fig_gap.update_yaxes(tickfont=dict(size=11))
            fig_gap.add_trace(
                go.Bar(
                    x=gap.to_numpy(), y=order, orientation="h",
                    marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.10)", width=1)),
                    text=["POLE" if g == 0 else f"+{g:.3f}" for g in gap], textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: +%{x:.3f} s<extra></extra>",
                )
            )
            size_horizontal_bars(fig_gap, gap.to_numpy(), row_px=26, bar_px=16)
            style_bars(fig_gap)
            plotly_chart(fig_gap, width="stretch", config=PLOTLY_CONFIG)

    with col_ideal:
        with chart_panel(
            "Time left on the table", "Best lap minus the sum of the driver's best sectors",
            accent=PALETTE["amber"],
        ):
            sectors = timed.groupby("Driver")[["Sector1Time", "Sector2Time", "Sector3Time"]].min()
            ideal = sectors.sum(axis=1, min_count=3).dt.total_seconds()
            lost = (best - ideal).dropna().clip(lower=0)
            lost = lost.reindex([d for d in order if d in lost.index])
            fig_ideal = base_figure("", "", "Seconds")
            fig_ideal.update_layout(showlegend=False)
            fig_ideal.update_yaxes(tickfont=dict(size=11))
            fig_ideal.add_trace(
                go.Bar(
                    x=lost.to_numpy(), y=list(lost.index), orientation="h",
                    marker=dict(
                        color=[safe_driver_color(d, session) for d in lost.index],
                        line=dict(color="rgba(255,255,255,0.10)", width=1),
                    ),
                    text=[f"{v:.3f}" for v in lost], textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x:.3f} s off their ideal lap<extra></extra>",
                )
            )
            size_horizontal_bars(fig_ideal, lost.to_numpy(), row_px=26, bar_px=16)
            style_bars(fig_ideal)
            plotly_chart(fig_ideal, width="stretch", config=PLOTLY_CONFIG)
    method_note(
        "**Ideal lap** is the sum of a driver's three best sector times, wherever in the "
        "session each was set. The bar is how much slower their best actual lap was than that "
        "-- a driver who strung their best sectors together on one lap scores zero. Drivers "
        "stay in order of their best lap, as in the gap chart beside it, so the two read row by row."
    )


def dhl_team_color(team):
    """DHL's team names are short ("Red Bull", "Haas") and include past
    teams; FastF1 matches loosely, and anything it can't place goes grey."""
    return team_color(team)


def pit_list(rows, best_duration=None):
    """rows: dicts with duration, name, team, an 'extra' string and, where
    the list isn't a ranking, a 'rank' label to show instead of 1, 2, 3."""
    items = []
    for i, r in enumerate(rows, start=1):
        best = best_duration is not None and abs(r["duration"] - best_duration) < 1e-9
        items.append(
            f'<div class="pit-row{" best" if best else ""}">'
            f'<div class="pit-rank">{r.get("rank", i)}</div>'
            f'<div class="pit-time">{r["duration"]:.2f}<small>s</small></div>'
            f'<div class="pit-bar" style="background:{dhl_team_color(r["team"])}"></div>'
            f'<div class="pit-who"><b>{r["name"]}</b><span>{r["team"]}</span></div>'
            f'<div class="pit-extra">{r.get("extra", "")}</div></div>'
        )
    st.markdown(f'<div class="pit-list">{"".join(items)}</div>', unsafe_allow_html=True)


DHL_SOURCE = (
    '<div class="pit-source">Stationary time, from the car stopping to it moving off -- published by '
    f'<a href="{DHL}/en/formula-1/fastest-pit-stop-award" target="_blank">DHL for its Fastest Pit Stop '
    "Award</a>. Records since 2023, the first season DHL still publishes.</div>"
)


def pit_record_cards(this_event=None):
    """Fastest stop here (if given), this season's and the all-time one --
    and on the season view, where there's no race, the award leader."""
    cards = []
    if this_event:
        r = this_event
        cards.append(("Fastest stop here", f"{r['duration']:.2f} s",
                      f"{r['tla']} · {r['team']} · lap {r['lap']}", dhl_team_color(r["team"])))
    season = dhl_season(year)
    if season and season["fastest"]:
        r = min(season["fastest"], key=lambda x: x["duration"])
        cards.append((f"{year} season record", f"{r['duration']:.2f} s",
                      f"{r['firstName'][0]}. {r['lastName']} · {r['shortTitle']}", dhl_team_color(r["team"])))
    everything = dhl_all_time_fastest()
    if everything:
        r = everything[0]
        cards.append(("All-time record", f"{r['duration']:.2f} s",
                      f"{r['firstName'][0]}. {r['lastName']} · {r['team']} · {r['year']}", PALETTE["amber"]))
    if this_event is None and season and season["standings"]:
        leader = season["standings"][0]
        cards.append(("DHL award leader", leader["team"], f"{leader['points']} points",
                      dhl_team_color(leader["team"])))
    stat_cards(cards)


def render_pit_stops():
    """The race's fastest stops by stationary time (DHL), with the season's
    and the all-time record beside the winner. Replaced a chart of pit lane
    times by team -- the only pit data in the timing feed, entry line to exit
    line, 20-odd seconds that mostly measure the length of the lane."""
    if session_name != "Race" or year < DHL_FIRST_SEASON:
        return
    event, stops = dhl_event_stops(year, pd.Timestamp(event_row["EventDate"]).strftime("%Y-%m-%d"))
    if not stops:
        return
    section_head("Pit stops", "DHL stationary times", "", accent=PALETTE["amber"])
    pit_record_cards(stops[0])
    st.write("")
    with chart_panel("Fastest stops of the race", "Ranked by stationary time · DHL award points on the right",
                     accent=PALETTE["amber"]):
        pit_list(
            [{"duration": r["duration"], "name": f"{r['firstName']} {r['lastName']}", "team": r["team"],
              "extra": f"lap {r['lap']} · {r['points']} pts" + (" · irregular" if r.get("irregular") else "")}
             for r in stops[:10]],
            best_duration=stops[0]["duration"],
        )
        st.markdown(DHL_SOURCE, unsafe_allow_html=True)
    st.write("")


def corner_apex_speeds(distance, speed, window_m=120, min_drop=25):
    """Minimum speed at each corner of a lap: a point that is the slowest
    within window_m either side and sits at least min_drop km/h under the
    fastest point of that stretch. Read off the telemetry itself rather than
    FastF1's circuit corner list, which comes from a separate web service."""
    apexes = []
    for i in range(len(speed)):
        near = (distance >= distance[i] - window_m) & (distance <= distance[i] + window_m)
        wide = (distance >= distance[i] - 3 * window_m) & (distance <= distance[i] + 3 * window_m)
        if speed[i] <= speed[near].min() and speed[wide].max() - speed[i] >= min_drop:
            if not apexes or distance[i] - apexes[-1][0] > window_m:
                apexes.append((distance[i], speed[i]))
    return [v for _, v in apexes]


def render_car_characteristics():
    """Top speed against average corner speed, one point per team, from each
    team's fastest lap of the session. The only chart here about the car
    rather than the driver: low drag shows up as top speed, downforce as
    speed through the corners, and a team's set-up choice as where it sits
    between the two."""
    results = session.results
    rows = []
    for team, entries in results.groupby("TeamName"):
        try:
            lap = session.laps.pick_drivers(list(entries["Abbreviation"])).pick_fastest()
            if lap is None:
                continue
            tel = lap.get_car_data().add_distance()
        except Exception:
            continue
        speed = tel["Speed"].to_numpy(dtype=float)
        distance = tel["Distance"].to_numpy(dtype=float)
        apexes = corner_apex_speeds(distance, speed)
        if len(apexes) < 4:
            continue
        rows.append({"Team": team, "Driver": lap["Driver"], "Top": np.nanmax(speed),
                     "Corner": float(np.mean(apexes)), "Corners": len(apexes)})
    if len(rows) < 3:
        return
    df = pd.DataFrame(rows)
    df["Label"] = df["Team"].replace({"Red Bull Racing": "Red Bull", "Haas F1 Team": "Haas"})

    # Label placement: above the point, unless that would collide with a
    # label already placed above a nearby point -- then below, then to the
    # right. Points closer than ~6% of the axis span in x and ~9% in y count
    # as nearby (a label is wide and short).
    span_x = max(df["Corner"].max() - df["Corner"].min(), 1)
    span_y = max(df["Top"].max() - df["Top"].min(), 1)
    taken = []
    positions = []
    for r in df.sort_values("Top", ascending=False).itertuples():
        for position in ("top center", "bottom center", "middle right", "middle left"):
            if not any(p == position and abs(r.Corner - x) < 0.12 * span_x and abs(r.Top - y) < 0.09 * span_y
                       for x, y, p in taken):
                break
        taken.append((r.Corner, r.Top, position))
        positions.append((r.Index, position))
    df["Position"] = pd.Series(dict(positions))

    with chart_panel(
        "Straight-line speed vs. cornering",
        "Each team's fastest lap · right is quicker through the corners, up is quicker on the straights",
        accent=PALETTE["violet"],
    ):
        fig = base_figure("", "Top speed (km/h)", "Average minimum corner speed (km/h)", hovermode="closest")
        fig.update_layout(height=460, showlegend=False)
        fig.add_hline(y=df["Top"].median(), line_color="rgba(255,255,255,0.15)", line_dash="dot")
        fig.add_vline(x=df["Corner"].median(), line_color="rgba(255,255,255,0.15)", line_dash="dot")
        fig.add_trace(go.Scatter(
            x=df["Corner"], y=df["Top"], mode="markers+text",
            marker=dict(size=15, color=[team_color(t) for t in df["Team"]],
                        line=dict(color="#0b0e15", width=2)),
            text=df["Label"], textposition=list(df["Position"]),
            textfont=dict(size=11, color="rgba(226,232,240,0.8)"),
            customdata=df[["Team", "Driver", "Corners"]],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>top %{y:.0f} km/h · corners %{x:.1f} km/h"
                          " over %{customdata[2]} corners<extra></extra>",
        ))
        pad_x = (df["Corner"].max() - df["Corner"].min()) * 0.15 + 1
        pad_y = (df["Top"].max() - df["Top"].min()) * 0.15 + 1
        fig.update_xaxes(range=[df["Corner"].min() - pad_x, df["Corner"].max() + pad_x])
        fig.update_yaxes(range=[df["Top"].min() - pad_y, df["Top"].max() + 2 * pad_y])
        plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    method_note(
        "One lap per team, the quickest either driver set in this session. **Top speed** is that "
        "lap's highest reading. **Corner speed** is the average of its minimum speeds, one per "
        "corner, found in the telemetry as a speed dip of at least 25 km/h. A team up and to the "
        "left runs less wing (quick on the straights, slower in the corners); down and to the right "
        "runs more. The dotted lines are the field's medians. It's one lap each, so a tow on the "
        "straight or a scruffy corner moves a point: read it as a tendency, not a measurement.",
        "How the two speeds are taken",
    )


def render_strategy_tab():
    """Race and sprint only: how each team ran its race -- the tyres, and
    the stops that changed them."""
    strategy_laps = session.laps.dropna(subset=["Stint", "Compound", "LapNumber"])
    if strategy_laps.empty:
        empty_state("No stint data available for a strategy timeline in this session")
    else:
        strategy_order = order_by_classification(session, strategy_laps["Driver"].unique())

        with chart_panel(
            "Tyre strategy",
            "Which compound each driver ran, and for how long",
            accent=COMPOUND_COLORS["MEDIUM"],
        ):
            fig_strategy = base_figure("", "", "Lap")
            fig_strategy.update_layout(
                height=max(CHART_HEIGHT, 23 * len(strategy_order)), barmode="stack", showlegend=False,
            )
            fig_strategy.update_yaxes(categoryorder="array", categoryarray=list(reversed(strategy_order)))

            used_compounds = []
            for driver, driver_laps in strategy_laps.groupby("Driver"):
                for stint, stint_laps in driver_laps.groupby("Stint"):
                    compound = stint_laps["Compound"].iloc[0]
                    start = stint_laps["LapNumber"].min()
                    length = stint_laps["LapNumber"].max() - start + 1
                    color = COMPOUND_COLORS.get(compound, "#999999")
                    if compound not in used_compounds:
                        used_compounds.append(compound)
                    fig_strategy.add_trace(
                        go.Bar(
                            x=[length], y=[driver], base=[start - 1], orientation="h",
                            # The separator is drawn in the page's own background
                            # color rather than translucent black: against a light
                            # compound (hard is near-white) a black hairline read
                            # as a drawn border, this reads as a gap.
                            marker=dict(color=color, line=dict(color="#0b0e15", width=2), cornerradius=3),
                            # The stint length, printed in the stint itself --
                            # "how many laps was that middle stint" was otherwise
                            # a question only the hover could answer.
                            text=[f"{length:.0f}" if length >= 4 else ""],
                            textposition="inside", insidetextanchor="middle",
                            textangle=0, constraintext="none", cliponaxis=False,
                            textfont=dict(
                                size=10.5, family="'JetBrains Mono', monospace",
                                color="#0b0e15" if compound in ("HARD", "MEDIUM") else "#ffffff",
                            ),
                            customdata=[[driver, compound.title(), start, start + length - 1, length]],
                            hovertemplate=(
                                "%{customdata[0]} · %{customdata[1]}<br>"
                                "laps %{customdata[2]:.0f}–%{customdata[3]:.0f} "
                                "(%{customdata[4]:.0f} laps)<extra></extra>"
                            ),
                        )
                    )

            plotly_chart(fig_strategy, width="stretch", config=PLOTLY_CONFIG)
            # Compound colors as chips instead of a Plotly legend: the legend
            # entry for a stacked bar chart is one swatch per trace, and there
            # is one trace per stint -- roughly sixty of them.
            chips([
                (COMPOUND_COLORS.get(c, "#999999"), c.title())
                for c in sorted(used_compounds, key=lambda c: list(COMPOUND_COLORS).index(c) if c in COMPOUND_COLORS else 9)
            ])

    st.write("")

    render_pit_stops()


def render_overtakes():
    """The race's overtakes, on the Results tab: part of what happened in the
    race rather than of how fast anyone was."""
    overtake_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if overtake_laps.empty:
        empty_state("No lap-by-lap position data available to count overtakes in this session")
    else:
        method_note(
            "Each overtake is one car getting ahead of another between two crossings of the "
            "line, found pair by pair in the official lap-end positions -- so three cars passed "
            "in one lap are three overtakes. Not counted: moves where either car was in the pit "
            "lane that lap; a pass made on the passer's first lap after its own pit stop (fresh "
            "tyres completing an undercut -- except at a restart); a car that retired, or had a "
            "problem (a lap more than 6% slower than the rest of the field on that same lap: a "
            "spin, damage, a failure); laps that end under the safety car, VSC or a red flag "
            "(restart laps do count); the first four laps and the lap after a red flag, while "
            "the field sorts itself out after a start; and a place race control made the driver give back, together with the "
            "handing back. Lapping doesn't change race order, so it never counts. The one blind "
            "spot: a pass and a re-pass within the same lap, since positions only exist at the line.",
            "How an overtake is counted here",
        )
        race_passes = detect_passes(session)
        overtake_counts = {}
        passed_counts = {}
        for _, passer, passed in race_passes:
            overtake_counts[passer] = overtake_counts.get(passer, 0) + 1
            passed_counts[passed] = passed_counts.get(passed, 0) + 1

        ranked_overtakes = sorted(overtake_counts.items(), key=lambda kv: kv[1], reverse=True)
        ranked_overtakes = [(d, c) for d, c in ranked_overtakes if c > 0]
        if not ranked_overtakes:
            empty_state("No overtakes detected in this session")
        else:
            overtake_styles = build_driver_styles([d for d, _ in ranked_overtakes], session)
            with chart_panel(
                "Overtakes by driver",
                f"{len(race_passes)} on-track passes, pit stops, incidents and starts left out · "
                "hover for who they passed",
                accent=PALETTE["teal"],
            ):
                fig_overtakes = base_figure("", "", "Overtakes", hovermode="closest")
                fig_overtakes.update_layout(showlegend=False)
                fig_overtakes.add_trace(
                    go.Bar(
                        x=[c for _, c in ranked_overtakes],
                        y=[d for d, _ in ranked_overtakes],
                        orientation="h",
                        marker=dict(
                            color=[overtake_styles[d][0] for d, _ in ranked_overtakes],
                            line=dict(color="rgba(255,255,255,0.10)", width=1),
                        ),
                        text=[c for _, c in ranked_overtakes],
                        textposition="outside", cliponaxis=False,
                        customdata=[
                            [", ".join(f"{b} (lap {n})" for n, a, b in race_passes if a == d),
                             passed_counts.get(d, 0)]
                            for d, _ in ranked_overtakes
                        ],
                        hovertemplate="<b>%{y}</b>: %{x} overtakes, passed %{customdata[1]} times"
                                      "<br>%{customdata[0]}<extra></extra>",
                    )
                )
                fig_overtakes.update_yaxes(autorange="reversed")
                size_horizontal_bars(fig_overtakes, [c for _, c in ranked_overtakes])
                style_bars(fig_overtakes)
                plotly_chart(fig_overtakes, width="stretch", config=PLOTLY_CONFIG)

    st.write("")


# A long run: at least this many consecutive laps on one set of tyres, none
# under safety car, VSC or red flag, none an in- or out-lap, none slower than LONG_RUN_WINDOW times
# the driver's own best. Five is what a team calls a race simulation at the
# shortest; 107% keeps race-pace laps and drops the cool-down laps practice
# is full of, and because the laps must be consecutive, one cool-down lap
# ends a run instead of hiding inside it.
LONG_RUN_MIN_LAPS = 5
LONG_RUN_WINDOW = 1.07


def find_long_runs(laps):
    """[(driver, laps of one run)] for every long run in the session."""
    laps = laps.dropna(subset=["LapTime", "LapNumber", "Stint"])
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]
    runs = []
    for driver, driver_laps in laps.groupby("Driver"):
        best = driver_laps["LapTime"].min()
        # Neutralised laps are out (4 safety car, 5 red flag, 6/7 VSC); a
        # yellow (2) is not -- practice is full of local yellows, and
        # requiring pure green broke Leclerc's Baku FP2 run in half.
        neutralised = driver_laps["TrackStatus"].astype(str).str.contains("[4567]", regex=True)
        usable = driver_laps[(driver_laps["LapTime"] <= best * LONG_RUN_WINDOW) & ~neutralised]
        for _, stint in usable.groupby("Stint"):
            stint = stint.sort_values("LapNumber")
            block = (stint["LapNumber"].diff() != 1).cumsum()
            for _, run in stint.groupby(block):
                if len(run) >= LONG_RUN_MIN_LAPS:
                    runs.append((driver, run))
    return runs


def render_long_runs():
    """Practice's answer to "who has race pace": each driver's longest run,
    ranked by its typical lap. It replaces the lap-by-lap race-pace chart in
    practice, where every other lap is a cool-down and the line zigzagged
    between 1:50 and 2:25 -- the run programme, not pace. On a Friday this is
    the number the paddock argues about."""
    runs = find_long_runs(session.laps)
    with chart_panel(
        "Long runs",
        f"Each driver's longest run: {LONG_RUN_MIN_LAPS}+ consecutive laps on one set of tyres, "
        "cool-down and neutralised laps excluded",
        accent=PALETTE["blue"],
    ):
        if not runs:
            empty_state(
                f"No long runs in this session -- nobody did {LONG_RUN_MIN_LAPS} clean laps "
                "in a row on one set of tyres"
            )
            return
        longest = {}
        for driver, run in runs:
            seconds = run["LapTime"].dt.total_seconds()
            candidate = (len(run), -seconds.median(), run)
            if driver not in longest or candidate[:2] > longest[driver][:2]:
                longest[driver] = candidate
        results = session.results
        team_by_driver = results.set_index("Abbreviation")["TeamName"] if not results.empty else {}
        number_by_driver = results.set_index("Abbreviation")["DriverNumber"] if not results.empty else {}
        summary = []
        for driver, (_, _, run) in longest.items():
            seconds = run["LapTime"].dt.total_seconds().to_numpy()
            # Slope of a straight line through the run: tyre wear pushes it
            # up, fuel burning off pulls it down, and on a Friday nobody
            # outside the team knows the fuel load, so it's shown as it is.
            trend = float(np.polyfit(np.arange(len(seconds)), seconds, 1)[0])
            known = run["Compound"].dropna()
            summary.append({
                "driver": driver,
                "median": float(np.median(seconds)),
                "laps": len(seconds),
                "trend": trend,
                "compound": str(known.iloc[0]) if len(known) else "UNKNOWN",
                "first": int(run["LapNumber"].min()),
                "last": int(run["LapNumber"].max()),
            })
        summary.sort(key=lambda r: r["median"])
        fastest = summary[0]["median"]
        rows = []
        for rank, r in enumerate(summary, start=1):
            color = safe_driver_color(r["driver"], session)
            compound = r["compound"]
            rows.append({
                "badge": driver_number_badge(number_by_driver.get(r["driver"], "—"), color),
                "color": color,
                "pos": rank,
                "driver": r["driver"],
                "team": team_by_driver.get(r["driver"], ""),
                "tyre": (f'<span class="tyres"><span class="tyre" style="--tc:'
                         f'{COMPOUND_COLORS.get(compound, "#8a8f98")}" '
                         f'title="{compound.title()} · laps {r["first"]}–{r["last"]}">'
                         f'{COMPOUND_LETTER.get(compound, "?")}</span></span>'),
                "median": format_lap_time(pd.Timedelta(seconds=r["median"])),
                "gap": "—" if rank == 1 else f"+{r['median'] - fastest:.3f}",
                "laps": str(r["laps"]),
                "trend": f"{r['trend']:+.2f} s/lap",
            })
        render_table(rows, [
            ("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team"),
            ("tyre", "Tyre", "tyrecol"), ("median", "Typical lap", "mono"), ("gap", "Gap", "mono"),
            ("laps", "Laps", "num"), ("trend", "Trend", "mono"),
        ])
    method_note(
        f"A long run is at least {LONG_RUN_MIN_LAPS} consecutive laps on one set of tyres, none "
        "under safety car, VSC or red flag (local yellows are fine), none an in- or out-lap, and none slower than 107% of the driver's own "
        "best lap -- which is what drops the cool-down laps between pushes. Each driver's longest "
        "run is shown; the typical lap is its median, so one lap in traffic doesn't move it. "
        "**Compounds aren't like for like**: the ring is the tyre the run was on, and a medium "
        "run a few tenths behind a soft one may be the better pace. **Trend** is the slope of "
        "the run's lap times: tyre wear pushes it up, fuel burning off pulls it down, and the "
        "fuel load is the team's secret, so it's shown uncorrected.",
        "How long runs are found",
    )


def render_pace_tab():
    """Who was quick: two drivers lap by lap, the field's spread of lap
    times, and how fast each compound wore."""
    # Head-to-head race pace: two drivers' lap times side by side, on the tyre
    # each was actually running. The degradation fits further down answer "how
    # fast does this compound wear for the whole field"; they can't answer "was
    # he quicker than the car he was racing, and when" -- which is the question
    # a race is argued over, and the one the tyre-age chart kept being asked to
    # do and couldn't (its x axis is tyre life, so two drivers on different
    # strategies get laid on top of each other out of sequence).
    h2h_laps = session.laps.dropna(subset=["LapTime", "LapNumber"])
    if session_name.startswith("Practice"):
        # Practice isn't raced: lap by lap it's pushes and cool-downs, so the
        # comparison there is long runs instead.
        render_long_runs()
        st.write("")
    elif not h2h_laps.empty and h2h_laps["Driver"].nunique() >= 2:
        # Opens on the winner against the runner-up, so the chart is there
        # from the start rather than behind a choice; the pills live in a
        # popover inside the panel and close themselves on each pick (a new
        # driver replaces the older of the two). The title needs the pair
        # before the picker is drawn, so it's read from the order the picker
        # keeps in session state, falling back to that same default pair.
        h2h_eligible = h2h_laps["Driver"].unique()
        default_pair = order_by_classification(session, h2h_eligible)[:2]
        url_pair = [d for d in st.query_params.get("drivers", "").split(",") if d in h2h_eligible]
        picked = [
            d for d in (st.session_state.get(picker_key("pace") + "_order")
                        or (url_pair if len(url_pair) == 2 else default_pair))
            if d in h2h_eligible
        ]
        with chart_panel(
            "Race pace head-to-head" + (f" · {picked[0]} vs {picked[1]}" if len(picked) == 2 else ""),
            "Every lap, with the compound it was run on · lap 1, pit laps and "
            "laps off the scale dropped, safety-car laps shaded",
            accent=build_driver_styles(picked, session)[picked[0]][0] if picked else PALETTE["blue"],
        ):
            pace_popover = f"popover_pace_{session_slug()}"

            def close_pace_popover():
                st.session_state[pace_popover] = False

            with st.popover(
                "Change drivers" if len(picked) == 2 else "Pick drivers",
                icon=":material/group:", key=pace_popover, on_change="rerun",
            ):
                h2h_drivers = driver_picker(
                    "pace", "Drivers",
                    "The first driver is the one the gap is measured from. "
                    "Picking a new one replaces the older of the two.",
                    h2h_eligible, on_pick=close_pace_popover, default=default_pair, share=True,
                )
            if len(h2h_drivers) < 2:
                empty_state("Pick two drivers to see their pace lap by lap", "prompt")
            else:
                first_driver, second_driver = h2h_drivers
                h2h_styles = build_driver_styles([first_driver, second_driver], session)
                fig_h2h = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                    row_heights=[1.0, 0.5],
                    subplot_titles=(
                        "Lap time",
                        f"Cumulative gap — {second_driver} relative to {first_driver}",
                    ),
                )

                compounds_seen = []
                clean_times = []
                on_track_by_driver = {}
                for driver in (first_driver, second_driver):
                    driver_laps = h2h_laps[h2h_laps["Driver"] == driver].sort_values("LapNumber")
                    # In- and out-laps are a pit stop, not pace: left in, they
                    # add a 25-second spike per stop that flattens the whole
                    # scale. Dropping them also breaks the line at each stop,
                    # which is exactly where it should break.
                    on_track = driver_laps[driver_laps["PitInTime"].isna() & driver_laps["PitOutTime"].isna()]
                    # Lap 1 is a standing start, a first corner and whatever
                    # happened in it -- several seconds slower than anything
                    # that follows, and not race pace by any reading. Left in,
                    # it either owns the top of the scale or gets clipped by it
                    # and draws a vertical stripe off the top of the panel.
                    on_track = on_track[on_track["LapNumber"] > 1]
                    on_track_by_driver[driver] = on_track
                    clean_times.extend(on_track["LapTime"].dt.total_seconds().tolist())

                # The scale is decided before any trace is added, because the
                # traces are drawn against it: a lap slower than the ceiling is
                # cut from the line rather than clipped by the axis. Clipping
                # leaves the segment running to the top of the panel, which
                # reads as a spike with no top to it -- two of them showed up
                # in the opening laps of every race with an early safety car.
                floor = min(clean_times) - 0.6 if clean_times else 0
                ceiling = float(np.quantile(clean_times, 0.95)) + 1.5 if clean_times else 1

                for driver in (first_driver, second_driver):
                    color = h2h_styles[driver][0]
                    on_track = on_track_by_driver[driver]
                    first_stint = True
                    for _, stint_laps in on_track.groupby("Stint"):
                        compounds = stint_laps["Compound"].fillna("UNKNOWN")
                        for compound in compounds.unique():
                            if compound not in compounds_seen:
                                compounds_seen.append(compound)
                        fig_h2h.add_trace(
                            go.Scatter(
                                x=stint_laps["LapNumber"],
                                y=stint_laps["LapTime"].dt.total_seconds().where(
                                    lambda seconds: seconds <= ceiling
                                ),
                                mode="lines+markers",
                                line=dict(color=color, width=2.4),
                                # The line is the driver, the marker fill is the
                                # tyre: one chart answering "who was quicker"
                                # and "on what" at the same time, without a
                                # second chart to cross-reference.
                                marker=dict(
                                    size=6,
                                    color=[COMPOUND_COLORS.get(c, "#999999") for c in compounds],
                                    line=dict(color=color, width=1.2),
                                ),
                                name=driver, legendgroup=driver, showlegend=first_stint,
                                customdata=[
                                    [driver, str(c).title(), life]
                                    for c, life in zip(compounds, stint_laps["TyreLife"].fillna(0))
                                ],
                                hovertemplate=(
                                    "%{customdata[0]}: %{y:.3f}s"
                                    "<br>%{customdata[1]}, %{customdata[2]:.0f} laps old<extra></extra>"
                                ),
                            ),
                            row=1, col=1,
                        )
                        first_stint = False

                # Safety car and virtual safety car laps, shaded across both
                # panels: without them the slow laps look like someone lifting.
                try:
                    status_laps = h2h_laps[h2h_laps["Driver"] == first_driver][["LapNumber", "TrackStatus"]]
                    neutralised = sorted(
                        int(row["LapNumber"]) for _, row in status_laps.iterrows()
                        if isinstance(row["TrackStatus"], str)
                        and ("4" in row["TrackStatus"] or "6" in row["TrackStatus"])
                    )
                except Exception:
                    neutralised = []
                for lap_number in neutralised:
                    fig_h2h.add_vrect(
                        x0=lap_number - 0.5, x1=lap_number + 0.5,
                        fillcolor="rgba(255,179,64,0.07)", line_width=0, layer="below",
                        row="all", col=1,
                    )

                # Cumulative gap, computed on *every* lap including the pit
                # laps dropped above: a stop is the single biggest thing that
                # happens to a gap, and leaving it out would draw an undercut
                # as if it never occurred.
                seconds = {
                    driver: h2h_laps[h2h_laps["Driver"] == driver]
                    .set_index("LapNumber")["LapTime"].dt.total_seconds()
                    for driver in (first_driver, second_driver)
                }
                shared_laps = seconds[first_driver].index.intersection(seconds[second_driver].index).sort_values()
                if len(shared_laps) > 1:
                    gap = (
                        seconds[second_driver].loc[shared_laps].cumsum()
                        - seconds[first_driver].loc[shared_laps].cumsum()
                    )
                    gap_color = h2h_styles[second_driver][0]
                    fig_h2h.add_trace(
                        go.Scatter(
                            x=shared_laps, y=gap, mode="lines", name="gap", showlegend=False,
                            line=dict(color=gap_color, width=2.2),
                            fill="tozeroy",
                            fillcolor=hex_to_rgba(gap_color, 0.14) if gap_color.startswith("#")
                            else "rgba(255,255,255,0.08)",
                            text=[
                                f"{second_driver} ahead by {abs(v):.1f}s" if v < 0
                                else f"{first_driver} ahead by {abs(v):.1f}s"
                                for v in gap
                            ],
                            hovertemplate="%{text}<extra></extra>",
                        ),
                        row=2, col=1,
                    )
                    fig_h2h.add_hline(
                        y=0, line_dash="dash", line_color="rgba(255,255,255,0.35)", row=2, col=1,
                    )

                fig_h2h.update_layout(
                    **DARK_LAYOUT,
                    height=560, hovermode="x unified", margin=dict(t=46, b=44, l=8, r=18),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                fig_h2h.update_annotations(font=dict(size=13, color="#f2f4f6"))
                fig_h2h.update_xaxes(
                    gridcolor=PALETTE["grid"], zeroline=False, tickfont=dict(size=11.5),
                    linecolor=PALETTE["axis"],
                )
                fig_h2h.update_yaxes(
                    gridcolor=PALETTE["grid"], zeroline=False, tickfont=dict(size=11.5),
                )
                if clean_times:
                    # A robust top to the scale rather than the slowest lap:
                    # one lap stuck behind a backmarker is worth ten seconds,
                    # and letting it set the range squashes the tenths the
                    # rest of the chart is about.
                    fig_h2h.update_yaxes(range=[floor, ceiling], row=1, col=1)
                fig_h2h.update_yaxes(title_text="Lap time (s)", row=1, col=1)
                fig_h2h.update_yaxes(title_text="Gap (s)", row=2, col=1)
                fig_h2h.update_xaxes(title_text="Lap", row=2, col=1)
                plotly_chart(fig_h2h, width="stretch", config=PLOTLY_CONFIG)
                chips(
                    [
                        (COMPOUND_COLORS.get(c, "#999999"), str(c).title())
                        for c in compounds_seen
                    ]
                    + ([("rgba(255,179,64,0.5)", "Safety car / VSC")] if neutralised else [])
                )

        st.write("")

    try:
        pace_laps = session.laps.pick_quicklaps()
        pace_laps = pace_laps[pace_laps["TrackStatus"] == "1"]
    except Exception:
        pace_laps = session.laps.iloc[0:0]

    # New here: the pace *spread*, not just its trend. A driver's race is a
    # distribution -- a median, a tight or loose middle 50%, and a tail of
    # traffic and out-laps -- and the degradation fit below deliberately
    # throws all of that away to fit one slope. A box per driver, ordered by
    # median, is the standard way that comparison is read in the paddock, and
    # it needs no model at all: only green-flag quick laps, which is what
    # pace_laps already is.
    pace_by_driver = [
        (driver, laps["LapTime"].dt.total_seconds().dropna())
        for driver, laps in pace_laps.groupby("Driver")
    ]
    pace_by_driver = [(d, s) for d, s in pace_by_driver if len(s) >= 5]
    if pace_by_driver:
        pace_by_driver.sort(key=lambda item: item[1].median(), reverse=True)
        with chart_panel(
            "Race pace spread",
            "Green-flag laps only · box = middle 50%, line = median, dots = outliers",
            accent=PALETTE["blue"],
        ):
            fig_spread = base_figure("", "", "Lap time (s)", hovermode="closest")
            fig_spread.update_layout(
                height=max(CHART_HEIGHT, 30 * len(pace_by_driver) + 90), showlegend=False,
            )
            fastest_median = min(series.median() for _, series in pace_by_driver)
            for driver, series in pace_by_driver:
                color = safe_driver_color(driver, session)
                fig_spread.add_trace(
                    go.Box(
                        x=series, name=driver, orientation="h",
                        marker=dict(color=color, size=4, opacity=0.55),
                        line=dict(color=color, width=1.6),
                        fillcolor=hex_to_rgba(color, 0.18) if color.startswith("#") else "rgba(255,255,255,0.1)",
                        boxpoints="outliers", hoverlabel=dict(namelength=-1),
                    )
                )
            # The reference line is the quickest median in the session, so the
            # chart reads as "how far off the best race pace was everyone",
            # which is the question -- not the absolute lap time, which means
            # nothing without knowing the circuit.
            fig_spread.add_vline(
                x=fastest_median, line_dash="dot", line_color="rgba(255,255,255,0.35)", line_width=1.5,
                annotation_text="best median", annotation_position="top",
                annotation_font=dict(size=10, color="rgba(226,232,240,0.5)"),
            )
            plotly_chart(fig_spread, width="stretch", config=PLOTLY_CONFIG)

        st.write("")

    compounds_with_data = [
        c for c, g in pace_laps.groupby("Compound") if len(g) >= MIN_LAPS_FOR_TREND
    ]

    if not compounds_with_data:
        empty_state(
            "Not enough green-flag laps on one compound in this session to fit a degradation "
            "trend (typical for Qualifying, where laps are single push laps rather than a run)"
        )
    else:
        method_note(
            "Lap time vs. tyre age, **corrected for fuel load**: a plain fit against tyre age alone "
            "would conflate the tyre wearing in with the car simply getting lighter over the "
            "run, so `LapNumber` is used as a fuel-burn proxy in a joint fit, and only the "
            "tyre-age part of it is plotted here. The per-driver breakdown further down doesn't "
            "refit both terms -- for a driver who ran a compound in a single stint the two climb "
            "together 1-for-1 and the fit is collinear -- it subtracts the pooled fuel slope and "
            "fits tyre age alone.",
            "How the degradation slope is fitted",
        )

        pace_driver_options = sorted(pace_laps["Driver"].unique())
        # Chosen in a popover inside the chart's panel (below), not in a
        # widget that stays open above it: few people use it, and a third
        # permanent driver picker on this tab was one too many. The figure is
        # built before that popover is drawn, so the choice is read straight
        # from session state -- a widget's new value is already there when
        # the rerun it triggered starts. Before anyone has picked, it is the
        # picker's own default -- the session's leader -- so the laps are on
        # the chart from the first render, not only after a click.
        laps_default = order_by_classification(session, pace_driver_options)[:1]
        pace_drivers = [
            d for d in st.session_state.get(picker_key("laps") + "_order", laps_default) if d in pace_driver_options
        ]

        fig_pace = base_figure("", "Lap time (s), fuel-corrected", "Tyre life (laps)")

        # The fits run before anything is drawn, because the individual laps
        # need them. The trend lines were always fuel-corrected -- each is
        # evaluated at its compound's own mean lap number -- while the laps
        # plotted under them were raw, so a long Hard stint's dots fell three
        # seconds down the chart beneath a trend line that was nearly flat.
        # The chart contradicted itself. Correcting the dots the same way puts
        # them back on the axis the line claims to fit.
        fits = {}
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
                empty_state(f"Couldn't fit a degradation trend for {compound.title()} (bad or degenerate data)", "note")
                continue
            fits[compound] = {
                "tyre": tyre_coef, "fuel": fuel_coef, "intercept": intercept,
                "mean_lap": lap_number.mean(),
                "life_range": (tyre_life.min(), tyre_life.max()),
            }
        # The per-driver breakdown below reads the fuel slope from here.
        coeffs = {compound: (fit["tyre"], fit["fuel"]) for compound, fit in fits.items()}

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
                fit = fits.get(compound)
                if fit is not None:
                    # Every lap pulled to the fuel load the trend line is drawn
                    # at, so a stint late in the race isn't simply lower on the
                    # chart than the same stint run early.
                    lap_times = lap_times - fit["fuel"] * (stint_laps["LapNumber"] - fit["mean_lap"])
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
                        marker=dict(size=6, line=dict(color="rgba(11,14,21,0.8)", width=1)),
                        line=dict(color=color, width=2.5),
                        name=label,
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        for compound, fit in fits.items():
            x_fit = np.linspace(*fit["life_range"], 2)
            y_fit = fit["tyre"] * x_fit + fit["fuel"] * fit["mean_lap"] + fit["intercept"]
            tyre_coef = fit["tyre"]
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

        fig_pace.update_layout(
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            margin=dict(r=140),
        )
        with chart_panel(
            "Lap time vs. tyre age",
            "Fuel-corrected · dotted lines are the fitted degradation trend per compound",
            accent=COMPOUND_COLORS["SOFT"],
        ):
            laps_label = f"Selected driver: {pace_drivers[0]}" if pace_drivers else "Select driver"
            # Keyed and stateful so the pick can close it: every pick is a
            # finished choice here, and the chart it changes is right under
            # the popover, so the popover gets out of the way at once instead
            # of waiting for a click outside it. One driver, as the name
            # says: the laps are colored by compound, not driver, so with two
            # their Hard stints were indistinguishable anyway.
            laps_popover = f"popover_laps_{session_slug()}"

            def close_laps_popover():
                st.session_state[laps_popover] = False

            with st.popover(laps_label, icon=":material/person_search:", key=laps_popover, on_change="rerun"):
                driver_picker(
                    "laps", "Show each lap for",
                    "Every lap of the chosen drivers' stints, fuel-corrected like the trend lines, "
                    "plotted under them. One driver at a time: picking another replaces them, "
                    "and clicking the same one again clears the chart.",
                    pace_driver_options, on_pick=close_laps_popover, limit=1,
                )
            plotly_chart(fig_pace, width="stretch", config=PLOTLY_CONFIG)

        if not coeffs:
            empty_state("No compound had a usable degradation fit in this session")
            return

        # The compound is chosen with pills inside the chart's own panel (see
        # below), so it's read from session state here, before they're drawn.
        compound_options = list(coeffs.keys())
        compound_key = f"pick_compound_{session_slug()}"
        compound_choice = st.session_state.get(compound_key)
        if compound_choice not in compound_options:
            compound_choice = compound_options[0]
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

        with chart_panel(
            f"{compound_choice.title()} — degradation by driver",
            "Lower is better · a negative slope means the car got quicker as the stint went on",
            accent=COMPOUND_COLORS.get(compound_choice, "#999999"),
        ):
            st.pills(
                "Compound", compound_options, selection_mode="single", default=compound_choice,
                required=True, format_func=str.title, key=compound_key,
            )
            pill_colors(compound_key, [COMPOUND_COLORS.get(c, "#999999") for c in compound_options])
            if not driver_slopes:
                empty_state(f"No driver ran enough laps on {compound_choice.title()} for a per-driver estimate")
            else:
                ranked = sorted(driver_slopes.items(), key=lambda kv: kv[1])
                fig_drivers = base_figure("", "", "Fuel-corrected degradation (s/lap)")
                fig_drivers.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=1)
                # Driver colors rather than one flat compound color: this chart
                # is a comparison *between drivers* on the same tyre, so the
                # compound is a constant here and carrying it as the only color
                # said nothing -- the panel header names it anyway.
                fig_drivers.add_trace(
                    go.Bar(
                        x=[s for _, s in ranked],
                        y=[d for d, _ in ranked],
                        orientation="h",
                        marker=dict(
                            color=[safe_driver_color(d, session) for d, _ in ranked],
                            line=dict(color="rgba(255,255,255,0.10)", width=1),
                        ),
                        text=[f"{s:+.3f}" for _, s in ranked],
                        textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x:+.3f} s/lap<extra></extra>",
                    )
                )
                fig_drivers.update_yaxes(autorange="reversed")
                size_horizontal_bars(fig_drivers, [s for _, s in ranked])
                style_bars(fig_drivers)
                plotly_chart(fig_drivers, width="stretch", config=PLOTLY_CONFIG)


if section == SECTION_WEEKEND and showing(tab_pace):
    with tab_pace:
        # Qualifying gets its own Stats here: pace and degradation are built
        # around stints over a race distance, and a qualifying lap is a single
        # push lap.
        if "Qualifying" in session_name:
            render_quali_stats()
            st.write("")
            render_car_characteristics()
        else:
            render_pace_tab()

if section == SECTION_WEEKEND and showing(tab_strategy):
    with tab_strategy:
        render_strategy_tab()

# --------------------------------------------------------- classification --

def render_race_story():
    """The race told in its moments, built from the data rather than
    written: the start, every change of lead (on track, or through the
    pits), safety cars and red flags, penalties, retirements, the fastest
    lap and the finish. A strip across the top places them on the race's
    own length. The charts on this tab show what happened to everyone; this
    is the version a reader who wasn't watching can follow."""
    laps = session.laps.dropna(subset=["LapNumber"])
    if laps.empty or laps["Position"].isna().all():
        return
    total = int(laps["LapNumber"].max())
    results = session.results
    color = lambda code: safe_driver_color(code, session)
    who = lambda code: f'<b style="color:{color(code)}">{code}</b>'
    events = []  # (lap, order within the lap, kind, html)

    # The start: who led lap one, and from where.
    leaders = (
        laps[laps["Position"] == 1].drop_duplicates("LapNumber").set_index("LapNumber")["Driver"].sort_index()
    )
    grid = results.set_index("Abbreviation")["GridPosition"] if "GridPosition" in results else pd.Series(dtype=float)
    if 1 in leaders.index:
        first = leaders.loc[1]
        start_slot = grid.get(first)
        if pd.notna(start_slot) and start_slot == 1:
            events.append((1, 0, "start", f"{who(first)} converts pole into the lead"))
        elif pd.notna(start_slot) and start_slot > 0:
            events.append((1, 0, "start", f"{who(first)} leads from P{int(start_slot)} on the grid"))
        else:
            events.append((1, 0, "start", f"{who(first)} leads the opening lap"))
        lap_one = laps[laps["LapNumber"] == 1].set_index("Driver")["Position"].dropna()
        gains = {d: grid.get(d) - p for d, p in lap_one.items() if pd.notna(grid.get(d)) and grid.get(d) > 0}
        if gains:
            best = max(gains, key=gains.get)
            if gains[best] >= 3:
                events.append((1, 1, "start", f"{who(best)} gains {gains[best]:.0f} places on the opening lap"))

    # Changes of lead, and how they happened.
    passes = {(n, a, b) for n, a, b in detect_passes(session)}
    pitting = laps[laps["PitInTime"].notna()].groupby("Driver")["LapNumber"].apply(set)
    previous = leaders.iloc[0] if len(leaders) else None
    for lap_number, leader in leaders.items():
        if lap_number == 1 or leader == previous:
            previous = leader
            continue
        if (lap_number, leader, previous) in passes:
            text = f"{who(leader)} passes {who(previous)} for the lead"
        elif lap_number in pitting.get(previous, set()) or (lap_number - 1) in pitting.get(previous, set()):
            text = f"{who(leader)} inherits the lead as {who(previous)} pits"
        else:
            text = f"{who(leader)} takes the lead from {who(previous)}"
        events.append((int(lap_number), 2, "lead", text))
        previous = leader

    # Neutralisations, from race control: deployed ... ending, as a span.
    spans = []
    try:
        messages = session.race_control_messages
    except Exception:
        messages = None  # not loaded: the story goes on without the flags
    if messages is None:
        messages = pd.DataFrame()
    open_span = None
    for _, m in messages.iterrows():
        text, lap_number = str(m.get("Message", "")), m.get("Lap")
        if pd.isna(lap_number):
            continue
        lap_number = int(lap_number)
        if text in ("SAFETY CAR DEPLOYED", "VSC DEPLOYED", "VIRTUAL SAFETY CAR DEPLOYED"):
            open_span = ["vsc" if "V" in text.split()[0] else "sc", lap_number]
        elif open_span and text in ("SAFETY CAR IN THIS LAP", "VSC ENDING", "VIRTUAL SAFETY CAR ENDING"):
            spans.append((open_span[0], open_span[1], lap_number))
            open_span = None
        elif m.get("Flag") == "RED":
            spans.append(("red", lap_number, lap_number))
        penalty = re.search(
            r"(\d+ SECOND TIME PENALTY|DRIVE THROUGH PENALTY|STOP AND GO PENALTY|\d+ PLACE GRID PENALTY)"
            r" FOR CAR \d+ \((\w{3})\)(?:\s*-\s*([^(]+))?", text,
        )
        if penalty and "SERVED" not in text:
            what = penalty.group(1).replace(" SECOND ", " s ").lower().replace(" penalty", " penalty")
            why = f" for {penalty.group(3).strip().lower()}" if penalty.group(3) else ""
            events.append((lap_number, 5, "penalty", f"{who(penalty.group(2))} gets a {what}{why}"))
    if open_span:
        spans.append((open_span[0], open_span[1], total))
    names = {"sc": "Safety car", "vsc": "Virtual safety car", "red": "Red flag"}
    for kind, start, end in spans:
        span_text = f"laps {start}–{end}" if end > start else f"lap {start}"
        events.append((start, 3, kind, f"{names[kind]} · {span_text}"))

    # Retirements.
    for _, r in results.iterrows():
        code_out = result_code(r)
        if code_out not in ("DNF", "DSQ"):
            continue
        done = laps[laps["Driver"] == r["Abbreviation"]]["LapNumber"].max()
        reason = str(r["Status"]) if pd.notna(r.get("Status")) and r["Status"] not in ("Retired", "Did not finish") else ""
        verb = "is disqualified" if code_out == "DSQ" else "retires"
        lap_out = int(done) + 1 if pd.notna(done) else 1
        events.append((min(lap_out, total), 4, "out", f"{who(r['Abbreviation'])} {verb}" + (f" · {reason.lower()}" if reason else "")))

    # Fastest lap and the finish.
    try:
        fastest = laps.pick_fastest()
        if fastest is not None and pd.notna(fastest["LapTime"]):
            events.append((int(fastest["LapNumber"]), 6, "fastest",
                           f"{who(fastest['Driver'])} sets the fastest lap, {format_lap_time(fastest['LapTime'])}"))
    except Exception:
        pass
    order = results.sort_values("Position")
    if len(order) >= 2 and pd.notna(order.iloc[0]["Position"]):
        winner, second = order.iloc[0], order.iloc[1]
        margin = second["Time"].total_seconds() if pd.notna(second.get("Time")) else None
        by = f" by {margin:.3f} s" if margin else ""
        events.append((total, 9, "finish", f"{who(winner['Abbreviation'])} wins{by} from {who(second['Abbreviation'])}"))

    if len(events) < 3:
        return
    events.sort(key=lambda e: (e[0], e[1]))

    icon = {"start": PALETTE["teal"], "lead": "#ffcf4a", "sc": PALETTE["amber"], "vsc": PALETTE["amber"],
            "red": PALETTE["red"], "penalty": PALETTE["pink"], "out": "#ff6b6b",
            "fastest": PALETTE["violet"], "finish": "#f4f6f8"}
    pct = lambda lap_number: 100 * (lap_number - 1) / max(total - 1, 1)
    strip = ['<div class="story-strip"><div class="track"></div>']
    for kind, start, end in spans:
        strip.append(
            f'<div class="seg {kind}" style="left:{pct(start):.2f}%;width:{max(pct(end + 1) - pct(start), 0.8):.2f}%" '
            f'title="{names[kind]} · lap {start}{"–" + str(end) if end > start else ""}"></div>'
        )
    for lap_number, _, kind, html in events:
        if kind in ("lead", "out", "finish") or (kind == "start" and lap_number == 1):
            strip.append(
                f'<div class="mark" style="left:{min(pct(lap_number), 100):.2f}%;--mc:{icon[kind]}" '
                f'title="Lap {lap_number}: {re.sub("<[^>]+>", "", html)}"></div>'
            )
    for tick in sorted({1, *range(10, total, 10), total}):
        strip.append(f'<div class="tick" style="left:{pct(tick):.2f}%">{tick}</div>')
    strip.append("</div>")

    items = "".join(
        f'<div class="story-item"><span class="story-lap">L{lap_number}</span>'
        f'<span class="story-icon" style="--ic:{icon[kind]}"></span><span>{html}</span></div>'
        for lap_number, _, kind, html in events
    )
    with chart_panel("How the race unfolded", f"{total} laps · built from the timing data and race control",
                     accent="#ffcf4a"):
        st.markdown("".join(strip) + f'<div class="story-list">{items}</div>', unsafe_allow_html=True)
    st.write("")


def tyre_rings(driver_laps):
    """A driver's stints as compound rings, in the order run (see .tyre)."""
    rings = []
    for _, stint in driver_laps.dropna(subset=["Stint"]).groupby("Stint"):
        known = stint["Compound"].dropna()
        compound = str(known.iloc[0]) if len(known) else "UNKNOWN"
        first, last = int(stint["LapNumber"].min()), int(stint["LapNumber"].max())
        rings.append(
            f'<span class="tyre" style="--tc:{COMPOUND_COLORS.get(compound, "#8a8f98")}" '
            f'title="{compound.title()} · laps {first}–{last}">{COMPOUND_LETTER.get(compound, "?")}</span>'
        )
    return f'<span class="tyres">{"".join(rings)}</span>' if rings else "—"


def best_lap_cell(lap_time, overall_best):
    """A driver's best lap; the session's fastest in purple, as on a timing
    screen."""
    if pd.isna(lap_time):
        return "—"
    text = format_lap_time(lap_time)
    return f'<span class="purple-lap">{text}</span>' if lap_time == overall_best else text


def gap_bar_cell(text, gap_seconds, widest):
    # The bar's length is the information -- how the field spreads out
    # behind P1, which a column of numbers doesn't show at a glance. It is
    # one neutral grey: in team colours it repeated the stripe and the number
    # badge on the same row and turned the column into a rainbow.
    width = 0 if not widest else min(100, 100 * gap_seconds / widest)
    return f'<div class="gapcell"><span>{text}</span><b><i style="width:{width:.1f}%"></i></b></div>'


def render_classification():
    results = session.results
    laps_all = session.laps
    best_by_driver = laps_all.groupby("Driver")["LapTime"].min()
    overall_best = best_by_driver.min()

    if results.empty or results["Position"].isna().all():
        # Practice has no official classification -- rank by fastest lap
        # instead, the closest equivalent to what timing screens show live.
        section_head(
            "Fastest laps",
            session_name,
            f"{session_name} has no official classification -- this ranks every driver by "
            "their best lap of the session instead.",
            accent=PALETTE["blue"],
        )
        fastest = best_by_driver.dropna().sort_values()
        # Laps completed, beside the time -- a practice ranking without it
        # says who was quick but not who was working: a 1:45.4 on lap 3 of
        # 28 and the same time as a driver's only flying lap are different
        # sessions. Every row of session.laps is a lap the driver crossed
        # the line on, in and out laps included, which is the same count a
        # timing screen shows.
        laps_run = laps_all.groupby("Driver").size()
        team_by_driver = results.set_index("Abbreviation")["TeamName"] if not results.empty else {}
        number_by_driver = results.set_index("Abbreviation")["DriverNumber"] if not results.empty else {}
        # The compound each driver's best lap was set on: in practice that is
        # half the story of the time -- a medium-tyre long run near the top is
        # worth more than a soft-tyre glory lap just ahead of it.
        best_rows = laps_all.dropna(subset=["LapTime"]).sort_values("LapTime").drop_duplicates("Driver")
        best_compound = best_rows.set_index("Driver")["Compound"]
        widest = (fastest.iloc[-1] - fastest.iloc[0]).total_seconds() if len(fastest) else 0
        rows = []
        for rank, (driver, lap_time) in enumerate(fastest.items(), start=1):
            color = safe_driver_color(driver, session)
            compound = str(best_compound.get(driver, "UNKNOWN"))
            gap = (lap_time - fastest.iloc[0]).total_seconds()
            rows.append({
                "badge": driver_number_badge(number_by_driver.get(driver, "—"), color),
                "color": color,
                "pos": rank,
                "driver": driver,
                "team": team_by_driver.get(driver, ""),
                "best": best_lap_cell(lap_time, overall_best),
                "tyre": (f'<span class="tyres"><span class="tyre" style="--tc:'
                         f'{COMPOUND_COLORS.get(compound, "#8a8f98")}" title="{compound.title()}">'
                         f'{COMPOUND_LETTER.get(compound, "?")}</span></span>'),
                "gap": gap_bar_cell("—" if rank == 1 else f"+{gap:.3f}", gap, widest),
                "laps": str(int(laps_run.get(driver, 0))),
            })
        with chart_panel("Session ranking", "By best lap · the ring is the tyre it was set on",
                         accent=PALETTE["blue"]):
            render_table(rows, [
                ("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team"),
                ("best", "Best lap", "mono"), ("tyre", "Tyre", "tyrecol"), ("gap", "Gap", "mono"),
                ("laps", "Laps", "num"),
            ])
        return

    has_quali_times = results[["Q1", "Q2", "Q3"]].notna().any().any()
    classified = results.sort_values("Position")
    leader_row = classified[classified["Position"] == 1]
    leader_laps = leader_row["Laps"].iloc[0] if len(leader_row) and pd.notna(leader_row["Laps"].iloc[0]) else None
    laps_run = laps_all.groupby("Driver")["LapNumber"].max()
    # The best time of each part of qualifying, for the purple.
    segment_best = {q: classified[q].min() for q in ("Q1", "Q2", "Q3")} if has_quali_times else {}
    points_places = 8 if session_name == "Sprint" else 10

    rows = []
    for index, (_, r) in enumerate(classified.iterrows()):
        code = r["Abbreviation"]
        color = safe_driver_color(code, session)
        row = {
            "badge": driver_number_badge(r["DriverNumber"] if pd.notna(r["DriverNumber"]) else "—", color),
            "color": color,
            "pos": str(int(r["Position"])) if pd.notna(r["Position"]) else "—",
            "driver": code,
            "team": r["TeamName"],
        }
        if has_quali_times:
            for q in ("Q1", "Q2", "Q3"):
                row[q.lower()] = best_lap_cell(r[q], segment_best[q])
            # Gap within the part a driver last ran, to that part's fastest:
            # a Q1 time against a pole lap set an hour later on a rubbered-in
            # track would mostly measure the track.
            last = next((q for q in ("Q3", "Q2", "Q1") if pd.notna(r[q])), None)
            if last is None:
                row["gap"] = "—"
            else:
                gap = (r[last] - segment_best[last]).total_seconds()
                row["gap"] = "—" if gap == 0 else f"+{gap:.3f}"
            # Knockout zones: the first driver who didn't reach Q3 (then Q2)
            # opens a band of their own, labelled.
            if index and pd.isna(r["Q3"]) and pd.notna(classified.iloc[index - 1]["Q3"]):
                row["divider"] = "Knocked out in Q2" if pd.notna(r["Q2"]) else "Knocked out in Q1"
            elif index and pd.isna(r["Q2"]) and pd.notna(classified.iloc[index - 1]["Q2"]):
                row["divider"] = "Knocked out in Q1"
        else:
            row["grid"] = (
                "PL" if pd.notna(r["GridPosition"]) and r["GridPosition"] == 0
                else str(int(r["GridPosition"])) if pd.notna(r["GridPosition"]) else "—"
            )
            code_out = result_code(r)
            if code_out:
                # DNF and the lap it came on; the reason, when the results
                # carry one ("Engine", "Collision"), on hover.
                completed = r["Laps"] if pd.notna(r.get("Laps")) else laps_run.get(code)
                reason = r["Status"] if pd.notna(r.get("Status")) else ""
                row["gap"] = (
                    f'<span class="out" title="{reason}">{code_out}'
                    + (f"<small>L{int(completed)}</small>" if pd.notna(completed) and completed else "")
                    + "</span>"
                )
            else:
                row["gap"] = format_race_gap(r, leader_laps)
            points = r["Points"] if pd.notna(r["Points"]) else 0
            row["pts"] = f"{points:.0f}" if points else '<span class="nil">0</span>'
            row["tyres"] = tyre_rings(laps_all.pick_drivers(code))
            row["best"] = best_lap_cell(best_by_driver.get(code), overall_best)
            # Places gained or lost, as its own column. It was derivable
            # from Grid and Pos side by side, but only by subtracting
            # two numbers in your head for every one of twenty rows --
            # and it's the single most-asked question of a results
            # table. A grid value of 0 means a pit-lane start, which
            # has no meaningful "places gained" to report.
            if pd.notna(r["GridPosition"]) and pd.notna(r["Position"]) and r["GridPosition"] > 0:
                moved = int(r["GridPosition"] - r["Position"])
                css = "up" if moved > 0 else ("down" if moved < 0 else "flat")
                arrow = "▲" if moved > 0 else ("▼" if moved < 0 else "·")
                row["delta"] = f"<span class='{css}'>{arrow} {abs(moved) if moved else ''}</span>"
            else:
                row["delta"] = "<span class='flat'>—</span>"
            if index == points_places and len(classified) > points_places:
                row["divider"] = "Outside the points"
        rows.append(row)

    columns = [("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team")]
    if has_quali_times:
        columns += [("q1", "Q1", "mono"), ("q2", "Q2", "mono"), ("q3", "Q3", "mono"), ("gap", "Gap", "mono")]
        hint = f"{event_name} · {session_name} · purple is the fastest time of each part"
    else:
        columns += [
            ("grid", "Grid", "num"), ("delta", "+/−", "delta"), ("tyres", "Tyres", "tyrecol"),
            ("best", "Best lap", "best mono"), ("gap", "Gap", "mono"), ("pts", "Pts", "num"),
        ]
        hint = f"{event_name} · {session_name} · tyres in the order run, fastest lap in purple"
    panel_title = "Qualifying classification" if has_quali_times else f"{session_name} classification"
    with chart_panel(panel_title, hint, accent=PALETTE["red"]):
        render_table(rows, columns)

    if not has_quali_times:
        st.write("")
        render_race_story()
        render_position_chart()
        st.write("")
        render_overtakes()


if section == SECTION_WEEKEND and showing(tab_classification):
    with tab_classification:
        render_classification()

# ------------------------------------------------------------- standings --

if section == SECTION_SEASON and showing(tab_standings):
    with tab_standings:
        round_number = int(event_row["RoundNumber"])

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

            def driver_color(code):
                return safe_driver_color(code, session)

            section_head(
                "Drivers' Championship",
                f"after round {round_number}",
                f"Standings as they stood right after the {event_name}.",
            )

            ordered_drivers = driver_standings.sort_values("position")
            if len(ordered_drivers) >= 2:
                leader_row, second_row = ordered_drivers.iloc[0], ordered_drivers.iloc[1]
                leader_name = leader_row["driverCode"] if pd.notna(leader_row["driverCode"]) else f"{leader_row['givenName']} {leader_row['familyName']}"
                second_name = second_row["driverCode"] if pd.notna(second_row["driverCode"]) else f"{second_row['givenName']} {second_row['familyName']}"
                stat_cards([
                    (
                        "Championship leader", leader_name,
                        f"<b>{leader_row['points']:.0f} pts</b> · {int(leader_row['wins'])} wins",
                        driver_color(leader_name),
                    ),
                    (
                        "Lead", f"{leader_row['points'] - second_row['points']:.0f} <small>pts</small>",
                        f"over <b>{second_name}</b> ({second_row['points']:.0f} pts)",
                        PALETTE["amber"],
                    ),
                    (
                        "Race winners so far",
                        f"{int((driver_standings['wins'] > 0).sum())}",
                        "different drivers have won a race",
                        PALETTE["blue"],
                    ),
                ])
                st.write("")

            if round_number > 1:
                # "x unified" hover and a top-right legend both fall apart with a
                # full 20+-driver field -- the unified box turned into a single
                # tooltip taller than the chart itself, and the legend wrapped
                # into rows that collided with the title. Same fix the race
                # position chart already uses for the same problem: no legend,
                # closest-point hover, driver code labeled at the end of its own line.
                fig_driver_progress = base_figure("", "Points", "Round", hovermode="closest")
                fig_driver_progress.update_layout(height=460, showlegend=False, margin=dict(t=24, b=44, r=96, l=8))
                driver_order = [
                    r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
                    for _, r in driver_standings.sort_values("position").iterrows()
                ]
                # Only the top 10 get an end-of-line label -- past that, drivers
                # bunch up near zero points and their labels just pile on top of
                # each other; the line and its hover are still there for anyone
                # further back, they just aren't individually labeled.
                # Where each labelled line ends, worked out before anything is
                # drawn, so the labels can be de-clumped against each other.
                driver_finals = {
                    d: driver_progress[driver_progress["Driver"] == d].sort_values("Round")["Points"].iloc[-1]
                    for d in driver_order
                }
                labelled_drivers = driver_order[:10]
                # The axis range is pinned rather than left to autorange: an
                # annotation anchored to a data point counts towards the range
                # Plotly picks, so nudged labels dragged the axis out to fit
                # themselves (a points chart with a -100 gridline on it).
                driver_points_top = (max(driver_finals.values()) or 1) * 1.06
                driver_offsets = dict(zip(
                    labelled_drivers,
                    spread_labels(
                        [driver_finals[d] for d in labelled_drivers],
                        plot_height_px=460 - 24 - 44,  # figure height less its margins
                        value_span=driver_points_top,
                    ),
                ))
                for rank, driver in enumerate(driver_order):
                    driver_pts = driver_progress[driver_progress["Driver"] == driver].sort_values("Round")
                    color = driver_color(driver)
                    # The championship leader's line is filled to the axis and
                    # drawn heaviest: on a chart of twenty lines the one that
                    # matters is the one everyone else is being measured
                    # against, and equal weight hid it in the bundle.
                    leading = rank == 0
                    fig_driver_progress.add_trace(
                        go.Scatter(
                            x=driver_pts["Round"], y=driver_pts["Points"], mode="lines",
                            line=dict(color=color, width=3.2 if leading else (2.2 if rank < 3 else 1.5)),
                            opacity=1.0 if rank < 3 else 0.75,
                            fill="tozeroy" if leading else None,
                            fillcolor=hex_to_rgba(color, 0.10) if leading and color.startswith("#") else None,
                            hovertemplate=f"{driver}: " + "%{y:.0f} pts (round %{x})<extra></extra>",
                        )
                    )
                    if rank >= 10:
                        continue
                    last = driver_pts.iloc[-1]
                    # showarrow=True is what allows the pixel nudge (ax/ay): the
                    # label moves, the anchor doesn't, and the thin leader line
                    # keeps it tied to the point it belongs to.
                    fig_driver_progress.add_annotation(
                        x=last["Round"], y=last["Points"], text=f"<b>{driver}</b>",
                        showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=hex_to_rgba(color, 0.45)
                        if color.startswith("#") else "rgba(255,255,255,0.35)",
                        ax=26, ay=driver_offsets.get(driver, 0.0), xanchor="left", yanchor="middle",
                        font=dict(size=9.5, color=color),
                        bgcolor="rgba(10,13,20,0.82)", bordercolor=color, borderwidth=1, borderpad=2,
                    )
                fig_driver_progress.update_xaxes(
                    dtick=1, range=[driver_progress["Round"].min() - 0.15, round_number + 0.15],
                )
                fig_driver_progress.update_yaxes(range=[0, driver_points_top])
                with chart_panel(
                    "Points progression",
                    "Running total after every round · top 10 labelled",
                    accent=driver_color(driver_order[0]) if driver_order else None,
                ):
                    plotly_chart(fig_driver_progress, width="stretch", config=PLOTLY_CONFIG)
                st.write("")

            leader_points = driver_standings["points"].max()
            driver_scores, driver_before = standings_history(driver_progress, "Driver", round_number)
            driver_rows = []
            for _, r in driver_standings.sort_values("position").iterrows():
                code = r["driverCode"] if pd.notna(r["driverCode"]) else ""
                name = code or f"{r['givenName']} {r['familyName']}"
                color = driver_color(code)
                driver_rows.append({
                    "badge": driver_number_badge(
                        int(r["driverNumber"]) if pd.notna(r["driverNumber"]) else "—", color,
                    ),
                    "pos": pos_label(r["position"]) + moved_badge(driver_before.get(name), r["position"]),
                    "name": name,
                    "team": r["constructorNames"][0] if len(r["constructorNames"]) else "—",
                    "points": points_cell(r["points"], leader_points, color),
                    "behind": "—" if r["points"] == leader_points else f"−{leader_points - r['points']:.0f}",
                    # A race and a sprint in one weekend is the most anyone
                    # can score in a round: 25 + 8.
                    "form": form_cell(driver_scores.get(name, []), color, 33),
                    "wins": int(r["wins"]),
                    "color": color,
                })
            with chart_panel("Drivers' standings",
                             f"after round {round_number} · ▲▼ places moved since the round before · "
                             "bars: points in each of the last five rounds"):
                render_table(driver_rows, [
                    ("pos", "Pos", "pos"), ("name", "Driver", "name"), ("team", "Team", "team"),
                    ("points", "Points", "points"), ("behind", "Gap", "behind"),
                    ("form", "Last 5", "formcol"), ("wins", "Wins", "num"),
                ])

            st.write("")

            section_head(
                "Constructors' Championship",
                f"after round {round_number}",
                accent=PALETTE["blue"],
            )

            ordered_constructors = constructor_standings.sort_values("position")
            if len(ordered_constructors) >= 2:
                leader_row_c, second_row_c = ordered_constructors.iloc[0], ordered_constructors.iloc[1]
                stat_cards([
                    (
                        "Leading constructor", leader_row_c["constructorName"],
                        f"<b>{leader_row_c['points']:.0f} pts</b> · {int(leader_row_c['wins'])} wins",
                        team_color(leader_row_c["constructorName"]),
                    ),
                    (
                        "Lead", f"{leader_row_c['points'] - second_row_c['points']:.0f} <small>pts</small>",
                        f"over <b>{second_row_c['constructorName']}</b> ({second_row_c['points']:.0f} pts)",
                        PALETTE["amber"],
                    ),
                ])
                st.write("")

            if round_number > 1:
                fig_constructor_progress = base_figure("", "Points", "Round", hovermode="closest")
                fig_constructor_progress.update_layout(height=400, showlegend=False, margin=dict(t=24, b=44, r=150, l=8))
                constructor_order = constructor_standings.sort_values("position")["constructorName"].tolist()
                constructor_finals = {
                    c: constructor_progress[constructor_progress["Constructor"] == c]
                    .sort_values("Round")["Points"].iloc[-1]
                    for c in constructor_order
                }
                constructor_points_top = (max(constructor_finals.values()) or 1) * 1.06
                constructor_offsets = dict(zip(
                    constructor_order,
                    spread_labels(
                        [constructor_finals[c] for c in constructor_order],
                        plot_height_px=400 - 24 - 44,
                        value_span=constructor_points_top,
                    ),
                ))
                for rank_c, constructor in enumerate(constructor_order):
                    team_pts = constructor_progress[constructor_progress["Constructor"] == constructor].sort_values("Round")
                    color = team_color(constructor)
                    leading = rank_c == 0
                    fig_constructor_progress.add_trace(
                        go.Scatter(
                            x=team_pts["Round"], y=team_pts["Points"], mode="lines",
                            line=dict(color=color, width=3.2 if leading else 2),
                            opacity=1.0 if rank_c < 3 else 0.8,
                            fill="tozeroy" if leading else None,
                            fillcolor=hex_to_rgba(color, 0.10) if leading and color.startswith("#") else None,
                            hovertemplate=f"{constructor}: " + "%{y:.0f} pts (round %{x})<extra></extra>",
                        )
                    )
                    last = team_pts.iloc[-1]
                    # "Alpine F1 Team" and "Cadillac F1 Team" are the same team
                    # as "Alpine" and "Cadillac"; the suffix only made the
                    # widest labels wider. The full name stays in the hover.
                    short_name = constructor.replace(" F1 Team", "").replace(" Racing", "")
                    fig_constructor_progress.add_annotation(
                        x=last["Round"], y=last["Points"], text=f"<b>{short_name}</b>",
                        showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=hex_to_rgba(color, 0.45)
                        if color.startswith("#") else "rgba(255,255,255,0.35)",
                        ax=26, ay=constructor_offsets.get(constructor, 0.0),
                        xanchor="left", yanchor="middle",
                        font=dict(size=9.5, color=color),
                        bgcolor="rgba(10,13,20,0.82)", bordercolor=color, borderwidth=1, borderpad=2,
                    )
                fig_constructor_progress.update_xaxes(
                    dtick=1, range=[constructor_progress["Round"].min() - 0.15, round_number + 0.15],
                )
                fig_constructor_progress.update_yaxes(range=[0, constructor_points_top])
                with chart_panel(
                    "Points progression",
                    "Running total after every round",
                    accent=team_color(constructor_order[0]) if constructor_order else None,
                ):
                    plotly_chart(fig_constructor_progress, width="stretch", config=PLOTLY_CONFIG)
                st.write("")

            leader_team_points = constructor_standings["points"].max()
            team_scores, team_before = standings_history(constructor_progress, "Constructor", round_number)
            constructor_rows = [
                {
                    "pos": pos_label(r["position"]) + moved_badge(team_before.get(r["constructorName"]), r["position"]),
                    "name": r["constructorName"],
                    "points": points_cell(r["points"], leader_team_points, team_color(r["constructorName"])),
                    "behind": "—" if r["points"] == leader_team_points else f"−{leader_team_points - r['points']:.0f}",
                    # Both cars first and second, in the race and the sprint.
                    "form": form_cell(team_scores.get(r["constructorName"], []), team_color(r["constructorName"]), 58),
                    "wins": int(r["wins"]),
                    "color": team_color(r["constructorName"]),
                }
                for _, r in constructor_standings.sort_values("position").iterrows()
            ]
            with chart_panel("Constructors' standings", f"after round {round_number}", accent=PALETTE["blue"]):
                render_table(constructor_rows, [
                    ("pos", "Pos", "pos"), ("name", "Constructor", "name"),
                    ("points", "Points", "points"), ("behind", "Gap", "behind"),
                    ("form", "Last 5", "formcol"), ("wins", "Wins", "num"),
                ])

            st.write("")

            total_rounds = total_rounds_in_season(year)
            remaining_rounds = total_rounds - round_number
            section_head(
                "Title fight",
                f"{remaining_rounds} race{'s' if remaining_rounds != 1 else ''} left",
                "Can anyone still catch the leader? Each line is the lead over one rival, round by round; "
                "the shaded area is every point still to play for. A lead that climbs out of it can't be "
                "overturned any more.",
                accent=PALETTE["amber"],
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

            sprint_rounds = sprint_rounds_in_season(year)

            def still_available(after_round, per_race, per_sprint):
                """Points still to be won once after_round rounds are done."""
                return sum(
                    per_race + (per_sprint if rnd in sprint_rounds else 0)
                    for rnd in range(after_round + 1, total_rounds + 1)
                )

            def title_fight(progress, entity_col, names, colors, per_race, per_sprint, label):
                """The leader's margin over each of the next two, against the
                points still available -- the arithmetic of a title, drawn.
                It replaced two donuts of simulated odds, which with a big
                lead came out as a solid ring reading "100%": a chart with
                nothing on it, and a certainty the resampling can't claim.
                This one has no model in it at all."""
                running = progress.pivot_table(index="Round", columns=entity_col, values="Points", aggfunc="last")
                running = running.sort_index().ffill().fillna(0)
                leader = names[0]
                rounds_all = list(range(0, total_rounds + 1))
                available = [still_available(r, per_race, per_sprint) for r in rounds_all]
                margin_now = running[leader].iloc[-1] - running[names[1]].iloc[-1] if len(names) > 1 else 0
                available_now = still_available(round_number, per_race, per_sprint)

                fig = base_figure("", "Points", "Round", hovermode="x unified")
                fig.update_layout(height=380, showlegend=False, margin=dict(t=16, b=44, l=8, r=70))
                fig.add_trace(go.Scatter(
                    x=rounds_all, y=available, mode="lines", line=dict(color="rgba(255,179,64,0.55)", width=1.5, shape="hv"),
                    fill="tozeroy", fillcolor="rgba(255,179,64,0.07)",
                    name="Still to play for", hovertemplate="%{y:.0f} pts still to play for<extra></extra>",
                ))
                rivals = list(zip(names, colors))[1:]
                ends = [running[leader].iloc[-1] - running[rival].iloc[-1] for rival, _ in rivals]
                # The two end labels are pushed apart when the two leads
                # finish close together, as they do when P2 and P3 are level.
                offsets = spread_labels(ends, plot_height_px=380 - 16 - 44, value_span=max(available[0], 1))
                for (rival, color), offset in zip(rivals, offsets):
                    margin = [0.0] + list(running[leader] - running[rival])
                    xs = [0] + list(running.index)
                    fig.add_trace(go.Scatter(
                        x=xs, y=margin, mode="lines+markers", name=rival,
                        line=dict(color=color, width=2.6), marker=dict(size=5, color=color),
                        hovertemplate=f"{leader} ahead of {rival} by " + "%{y:.0f}<extra></extra>",
                    ))
                    fig.add_annotation(
                        x=xs[-1], y=margin[-1], text=f"<b>vs {rival.replace(' F1 Team', '')}</b>",
                        showarrow=False, xanchor="left", xshift=8, yshift=-offset, font=dict(size=10.5, color=color),
                    )
                fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1)
                fig.add_vline(x=round_number, line_color="rgba(255,255,255,0.3)", line_dash="dot", line_width=1)
                fig.update_xaxes(dtick=2, range=[-0.3, total_rounds + 0.3])
                fig.update_yaxes(rangemode="tozero")

                short = lambda name: name.replace(" F1 Team", "")
                if len(names) > 1 and margin_now > available_now:
                    headline = ("Decided", f"<b>{short(leader)}</b> can't be caught any more", colors[0])
                elif len(names) > 1:
                    # The magic number: how much the lead over the runner-up
                    # still has to grow before nobody can close it.
                    need = available_now - margin_now + 1
                    headline = (
                        f"{need:.0f} <small>pts</small>",
                        f"on top of the {margin_now:.0f}-point lead <b>{short(leader)}</b> has over "
                        f"{short(names[1])} · {available_now:.0f} still to play for",
                        colors[0],
                    )
                else:
                    headline = ("—", "", colors[0])
                stat_cards([(f"{label} · magic number", *headline)])
                st.write("")
                with chart_panel(
                    f"Can anyone catch {leader.replace(' F1 Team', '')}?",
                    "Lead over each rival vs. points still available · dotted line: now",
                    accent=colors[0],
                ):
                    plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
                    histories_here = [race_history(progress, entity_col, n) for n in names]
                    current = [running[n].iloc[-1] for n in names]
                    odds = simulate_top3_odds(names, current, histories_here, remaining_rounds)
                    chips(
                        [(PALETTE["ink_faint"], "Simulated odds")]
                        + [(c, f"{n.replace(' F1 Team', '')} {format_odds(odds[n] * 100)}")
                           for n, c in sorted(zip(names, colors), key=lambda p: odds[p[0]], reverse=True)]
                    )

            top3_drivers = driver_standings.sort_values("position").head(3)
            names = [
                r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
                for _, r in top3_drivers.iterrows()
            ]
            histories = [race_history(driver_progress, "Driver", n) for n in names]
            top3_constructors = constructor_standings.sort_values("position").head(3)
            names_c = top3_constructors["constructorName"].tolist()
            histories_c = [race_history(constructor_progress, "Constructor", n) for n in names_c]

            col_drivers_fight, col_constructors_fight = st.columns(2)
            with col_drivers_fight:
                # 25 for a win, 8 for a sprint win -- no fastest-lap point
                # since 2025.
                title_fight(driver_progress, "Driver", names, [driver_color(n) for n in names], 25, 8, "Drivers'")
            with col_constructors_fight:
                # A one-two: 25 + 18, and 8 + 7 in a sprint.
                title_fight(constructor_progress, "Constructor", names_c, [team_color(n) for n in names_c],
                            43, 15, "Constructors'")
            method_note(
                "**The chart is exact**: the shaded area is the most points anyone could still score -- "
                "a win in every remaining race, plus every sprint -- and each line is the leader's current "
                "advantage over one rival. While a line is inside the area, that rival can still mathematically "
                "win; once it's above, they can't. Ties on points (settled by count-back) are ignored.\n\n"
                "**The odds are not**: a simplified Monte Carlo limited to the current top 3, each remaining "
                "race resampled from that driver's (or team's) own points per race so far, independently of "
                "who else is on track -- so a simulated score never takes points away from a rival the way a "
                "real race would. Treat them as a feel for how close the fight is, not a forecast.",
                "How the title maths and the odds are worked out",
            )

            st.write("")

            section_head(
                "When could it be decided?",
                "clinch estimate",
                accent=PALETTE["violet"],
            )
            method_note(
                "Same resampling, but round by round: for each simulated run, the earliest future "
                "round at which the current runner-up could no longer catch the current leader even "
                "by scoring the most points anyone has scored in a single race this season, every "
                "remaining round. **A mathematical best case for the leader, not a prediction.**",
                "How the clinch round is estimated",
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
                    # A stat card rather than st.metric: the widget truncates any
                    # value wider than its column, and the race name is exactly
                    # the part that was being cut off.
                    stat_cards([(
                        f"Drivers' — {names[0]} leading",
                        f"Round {median_round} <small>of {total_rounds}</small>",
                        f"\U0001F4CD <b>{event_name_for_round(year, median_round)}</b><br>"
                        f"{(clinches < total_rounds).mean() * 100:.0f}% chance it's sealed before the finale",
                        driver_color(names[0]),
                    )])
                else:
                    empty_state("Need at least 2 drivers with points to estimate this")

            with col_clinch_constructors:
                if len(top3_constructors) >= 2:
                    clinches_c = clinch_round_distribution(
                        top3_constructors["points"].iloc[0], top3_constructors["points"].iloc[1],
                        histories_c[0], histories_c[1], max_points_per_race(constructor_progress, "Constructor"),
                    )
                    median_round_c = int(round(np.median(clinches_c)))
                    stat_cards([(
                        f"Constructors' — {names_c[0]} leading",
                        f"Round {median_round_c} <small>of {total_rounds}</small>",
                        f"\U0001F4CD <b>{event_name_for_round(year, median_round_c)}</b><br>"
                        f"{(clinches_c < total_rounds).mean() * 100:.0f}% chance it's sealed before the finale",
                        team_color(names_c[0]),
                    )])
                else:
                    empty_state("Need at least 2 constructors with points to estimate this")

# ----------------------------------------------------------- season stats --

if section == SECTION_SEASON and showing(tab_season_stats):
    with tab_season_stats:
        section_head(
            "The season so far",
            f"{len(schedule)} races",
            f"Computed across every completed race of the {year} season. The first load is slow "
            "(each race's full session is fetched once), instant after that.",
            accent=PALETTE["teal"],
        )
        # Poles first: the tab used to be "Race by race" with the poles
        # tacked on at the bottom, where the name said they didn't belong.
        render_poles()
        st.write("")
        stats_df, driver_overtakes = season_stats(year, tuple(schedule["EventName"].tolist()), OVERTAKE_RULES)

        if stats_df.empty:
            empty_state("No completed races to compute stats from yet")
        else:
            most_ot = stats_df.loc[stats_df["Overtakes"].idxmax()]
            fewest_ot = stats_df.loc[stats_df["Overtakes"].idxmin()]
            most_dnf = stats_df.loc[stats_df["Retirements"].idxmax()]

            # Four stat cards rather than st.metric plus a caption each: the
            # widget truncated any value long enough to carry a Grand Prix
            # name, which is exactly the context these numbers need.
            season_cards = [
                ("Most overtakes", f"{int(most_ot['Overtakes'])}",
                 f"\U0001F4CD <b>{most_ot['Event']}</b>", PALETTE["teal"]),
                ("Fewest overtakes", f"{int(fewest_ot['Overtakes'])}",
                 f"\U0001F4CD <b>{fewest_ot['Event']}</b>", PALETTE["blue"]),
                ("Most retirements", f"{int(most_dnf['Retirements'])}",
                 f"\U0001F4CD <b>{most_dnf['Event']}</b>", PALETTE["red"]),
            ]
            recovery_df = stats_df.dropna(subset=["RecoveryPlaces"])
            if not recovery_df.empty:
                best_recovery = recovery_df.loc[recovery_df["RecoveryPlaces"].idxmax()]
                season_cards.append((
                    "Best recovery drive",
                    f"+{int(best_recovery['RecoveryPlaces'])} <small>places</small>",
                    f"<b>{best_recovery['RecoveryDriver']}</b> · {best_recovery['Event']}",
                    PALETTE["amber"],
                ))
            stat_cards(season_cards)
            st.write("")

            # New here: the season as a grid. The points-progression lines on
            # the Championship tab answer "who is ahead"; they can't answer
            # "who had a bad run in the middle of the year", because a running
            # total only ever goes up. Points *per race*, one cell per driver
            # per round, makes a dry spell a row of empty cells and a purple
            # patch impossible to miss.
            heat_round = int(event_row["RoundNumber"])
            try:
                heat_progress, _ = load_standings_progression(year, heat_round)
            except Exception:
                heat_progress = pd.DataFrame()
            if not heat_progress.empty:
                running = heat_progress.pivot(index="Driver", columns="Round", values="Points")
                # Column 1 is already a per-race score; every later column is
                # the running total, so the difference between neighbours is
                # what that round was worth.
                per_race = running.diff(axis=1)
                per_race[running.columns[0]] = running[running.columns[0]]
                per_race = per_race.loc[running.iloc[:, -1].sort_values(ascending=False).index]
                # Drivers who haven't scored all season are dropped: their rows
                # are empty by definition and added four blank lines to the
                # bottom of the grid that said nothing the standings table
                # doesn't already say.
                per_race = per_race[per_race.sum(axis=1) > 0]
                rounds = list(running.columns)
                labels = [
                    [f"{v:.0f}" if pd.notna(v) and v > 0 else "" for v in row]
                    for row in per_race.to_numpy()
                ]
                with chart_panel(
                    "Points per race",
                    "Every driver's score round by round · darker means nothing scored",
                    accent=PALETTE["amber"],
                ):
                    fig_heat = base_figure("", "", "Round", hovermode="closest")
                    fig_heat.update_layout(
                        height=max(CHART_HEIGHT, 26 * len(per_race) + 110),
                        margin=dict(l=8, r=20, t=16, b=40),
                    )
                    fig_heat.add_trace(
                        go.Heatmap(
                            z=per_race.to_numpy(),
                            x=[f"R{r}" for r in rounds],
                            y=list(per_race.index),
                            # Sequential, dark to warm: the scale runs from the
                            # panel's own background (a scoreless race should
                            # read as absence, not as a color) up through the
                            # app's blue to its amber for a win.
                            colorscale=[
                                [0.0, "rgba(255,255,255,0.03)"],
                                [0.25, "#1d2b4d"],
                                [0.6, "#3f6fd8"],
                                [1.0, PALETTE["amber"]],
                            ],
                            xgap=3, ygap=3,
                            text=labels, texttemplate="%{text}",
                            textfont=dict(size=9.5, color="rgba(255,255,255,0.85)"),
                            hovertemplate="%{y} · round %{x}: %{z:.0f} pts<extra></extra>",
                            colorbar=dict(
                                title=dict(text="pts", font=dict(size=10)),
                                thickness=10, len=0.5, outlinewidth=0,
                                tickfont=dict(size=10),
                            ),
                        )
                    )
                    fig_heat.update_yaxes(autorange="reversed", showgrid=False)
                    fig_heat.update_xaxes(showgrid=False, showspikes=False, side="top")
                    plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)

                st.write("")

            method_note(
                "Overtakes here use the same net-of-pit-stops, net-of-incidents rule as the "
                "Overtakes chart on the Race pace tab -- an approximation, not an official count.",
                "About the overtake numbers below",
            )
            # Vertical bars with 20+ rotated race names along the bottom were
            # unreadable at anything but full-screen -- horizontal, like the
            # per-driver overtakes chart, so every name reads flat and the
            # chart just grows downward instead of squeezing sideways. Kept in
            # season order (top to bottom = round 1 to the latest) rather than
            # sorted by count, since the point is the season's shape over time,
            # not a leaderboard -- the color gradient carries the ranking.
            ordered_races = stats_df.sort_values("Round", ascending=False)
            with chart_panel(
                "Overtakes by race",
                "In calendar order, latest at the top · color carries the ranking",
                accent=PALETTE["blue"],
            ):
                fig_overtakes_season = base_figure("", "", "Overtakes")
                fig_overtakes_season.update_layout(showlegend=False)
                fig_overtakes_season.add_trace(
                    go.Bar(
                        # "Australian" on its own reads as an adjective looking
                        # for a noun; the full "Grand Prix" is what makes the
                        # y axis three times wider than the bars.
                        x=ordered_races["Overtakes"], y=ordered_races["Event"].str.replace(" Grand Prix", " GP"),
                        orientation="h",
                        marker=dict(
                            color=ordered_races["Overtakes"], colorscale=bar_scale(PALETTE["blue"]),
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=ordered_races["Overtakes"].astype(int),
                        textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} overtakes<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_overtakes_season, ordered_races["Overtakes"].to_numpy())
                style_bars(fig_overtakes_season)
                plotly_chart(fig_overtakes_season, width="stretch", config=PLOTLY_CONFIG)

            st.write("")

            ranked_driver_overtakes = sorted(driver_overtakes.items(), key=lambda kv: kv[1], reverse=True)
            ranked_driver_overtakes = [(d, c) for d, c in ranked_driver_overtakes if c > 0]
            if not ranked_driver_overtakes:
                empty_state("No on-track position gains detected across this season's races yet")
            else:
                with chart_panel(
                    "Overtakes by driver",
                    f"Across all {len(stats_df)} races of the season",
                    accent=PALETTE["teal"],
                ):
                    fig_driver_overtakes = base_figure("", "", "Position gains")
                    fig_driver_overtakes.update_layout(showlegend=False)
                    fig_driver_overtakes.add_trace(
                        go.Bar(
                            x=[c for _, c in ranked_driver_overtakes],
                            y=[d for d, _ in ranked_driver_overtakes],
                            orientation="h",
                            marker=dict(
                                color=[safe_driver_color(d, session) for d, _ in ranked_driver_overtakes],
                                line=dict(color="rgba(255,255,255,0.10)", width=1),
                            ),
                            text=[c for _, c in ranked_driver_overtakes],
                            textposition="outside", cliponaxis=False,
                            hovertemplate="%{y}: %{x} position gains<extra></extra>",
                        )
                    )
                    fig_driver_overtakes.update_yaxes(autorange="reversed")
                    size_horizontal_bars(fig_driver_overtakes, [c for _, c in ranked_driver_overtakes])
                    style_bars(fig_driver_overtakes)
                    plotly_chart(fig_driver_overtakes, width="stretch", config=PLOTLY_CONFIG)

# ------------------------------------------------------------- teammates --

if section == SECTION_SEASON and showing(tab_teammates):
    with tab_teammates:
        section_head(
            "Teammate battles",
            f"{year} season so far",
            "The one fair fight in Formula 1: same car, same team. Who has been ahead in qualifying "
            "and in the races, and by how much -- then pick a pairing to see it race by race.",
            accent=PALETTE["violet"],
        )
        battles = teammate_battles(year, tuple(schedule["EventName"].tolist()))

        short_team = {"Red Bull Racing": "Red Bull", "Haas F1 Team": "Haas"}
        pairs = []
        if not battles.empty:
            for (team, a, b), g in battles.groupby(["Team", "A", "B"]):
                quali, race = g[g["Kind"] == "Qualifying"], g[g["Kind"] == "Race"]
                if len(quali) + len(race) < 3:
                    continue  # a one-off stand-in pairing isn't a season's battle
                pairs.append({
                    "Team": team, "A": a, "B": b,
                    "QA": int(quali["AAhead"].sum()), "QB": int((~quali["AAhead"]).sum()),
                    "RA": int(race["AAhead"].sum()), "RB": int((~race["AAhead"]).sum()),
                    "PtsA": race["PtsA"].sum(), "PtsB": race["PtsB"].sum(),
                    "Gap": quali["Gap"].median(),
                })
        pairs = pd.DataFrame(pairs)

        if pairs.empty:
            empty_state("No completed sessions to compare teammates in yet")
        else:
            pairs["TeamPts"] = pairs["PtsA"] + pairs["PtsB"]
            pairs = pairs.sort_values(["TeamPts", "Team"], ascending=[False, True]).reset_index(drop=True)

            def split(left, right, color):
                total = max(left + right, 1)
                return (
                    f'<div class="tm-split"><b>{left}</b><div class="tm-bar">'
                    f'<i style="width:{100 * left / total:.1f}%;background:{color}"></i>'
                    f'<i style="width:{100 * right / total:.1f}%;background:rgba(238,241,245,0.85)"></i>'
                    f"</div><b>{right}</b></div>"
                )

            body = []
            for r in pairs.itertuples():
                color = team_color(r.Team)
                if pd.notna(r.Gap):
                    gap = f"{r.A if r.Gap > 0 else r.B} +{abs(r.Gap):.3f} s"
                else:
                    gap = "—"
                body.append(
                    "<tr>"
                    f'<td><div class="tm-team"><span class="tm-dot" style="background:{color}"></span>'
                    f"{short_team.get(r.Team, r.Team)}</div></td>"
                    f'<td class="tm-pair" style="color:{color}">{r.A}<span>vs</span>'
                    f'<em style="color:#eef1f5;font-style:normal">{r.B}</em></td>'
                    f"<td>{split(r.QA, r.QB, color)}</td>"
                    f"<td>{split(r.RA, r.RB, color)}</td>"
                    f"<td>{split(int(r.PtsA), int(r.PtsB), color)}</td>"
                    f'<td class="tm-gap">{gap}</td>'
                    "</tr>"
                )
            with chart_panel(
                "Head to head, every pairing",
                "Left count the first driver (team color), right the second (white) · teams by points",
                accent=PALETTE["violet"],
            ):
                st.markdown(
                    '<div class="tm-scroll"><table class="tm-table">'
                    '<colgroup><col class="c-team"><col class="c-pair"><col class="c-split">'
                    '<col class="c-split"><col class="c-split"><col class="c-gap"></colgroup>'
                    "<thead><tr><th>Team</th><th>Pair</th><th>Qualifying</th><th>Race</th>"
                    "<th>Points</th><th>Quali gap</th></tr></thead>"
                    f"<tbody>{''.join(body)}</tbody></table></div>",
                    unsafe_allow_html=True,
                )
            method_note(
                "Qualifying and Race count the sessions each driver finished ahead of the other; a "
                "race counts even when one of them retired, since the classification puts a "
                "retirement behind a finisher. Points are race points. The qualifying gap is the "
                "median, over the season, of the difference in the last part of qualifying both "
                "reached (Q3, else Q2, else Q1), credited to the quicker driver. Pairings that "
                "lasted under three sessions -- a one-off stand-in -- are left out.",
                "How the battles are counted",
            )
            st.write("")

            # ---- one pairing, race by race
            labels = {f"{short_team.get(r.Team, r.Team)} · {r.A} / {r.B}": r for r in pairs.itertuples()}
            pick_key = f"pick_pair_{session_slug()}"
            st.pills(
                "Pairing", list(labels), selection_mode="single", default=list(labels)[0],
                required=True, key=pick_key,
            )
            pill_colors(pick_key, [team_color(r.Team) for r in labels.values()])
            pair = labels.get(st.session_state.get(pick_key)) or next(iter(labels.values()))
            color = team_color(pair.Team)
            g = battles[(battles["Team"] == pair.Team) & (battles["A"] == pair.A) & (battles["B"] == pair.B)]
            quali = g[g["Kind"] == "Qualifying"].sort_values("Round")
            race = g[g["Kind"] == "Race"].sort_values("Round")

            def card(driver, card_color, side):
                pos_q = quali[f"Pos{side}"]
                fin = race[race[f"Fin{side}"]][f"Pos{side}"]
                return (
                    f'<div class="driver-card" style="--card-color:{card_color}">'
                    f'<div class="name">{driver}</div>'
                    f'<div class="laptime">{int(race[f"Pts{side}"].sum())} pts</div>'
                    f'<div class="sub">Qualifying ahead {getattr(pair, "Q" + side)} · race ahead '
                    f'{getattr(pair, "R" + side)}<br>Average grid slot '
                    f'{pos_q.mean():.1f} · average finish {fin.mean():.1f} · '
                    f'{int((~race[f"Fin{side}"]).sum())} DNF</div></div>'
                )

            col_a, col_b = st.columns(2)
            col_a.markdown(card(pair.A, color, "A") if len(race) else "", unsafe_allow_html=True)
            col_b.markdown(card(pair.B, "#eef1f5", "B") if len(race) else "", unsafe_allow_html=True)
            st.write("")

            short_event = lambda e: e.replace(" Grand Prix", "")
            col_q, col_r = st.columns(2)
            with col_q:
                with chart_panel(
                    "Qualifying gap, race by race",
                    f"Up: {pair.A} quicker · down: {pair.B} quicker", accent=color,
                ):
                    qg = quali.dropna(subset=["Gap"])
                    fig = base_figure("", "Seconds", "", hovermode="closest")
                    fig.update_layout(height=360, showlegend=False)
                    fig.add_trace(go.Bar(
                        x=[short_event(e) for e in qg["Event"]], y=qg["Gap"],
                        marker=dict(color=[color if v > 0 else "rgba(238,241,245,0.85)" for v in qg["Gap"]]),
                        # Value on hover only: printed on the bars it turned
                        # sideways and unreadable on every thin one.
                        customdata=[f"{pair.A if v > 0 else pair.B} +{abs(v):.3f}" for v in qg["Gap"]],
                        hovertemplate="%{x}: %{customdata} s<extra></extra>",
                    ))
                    fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)", line_width=1)
                    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
                    style_bars(fig, radius=3)
                    plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            with col_r:
                with chart_panel(
                    "Finishing positions, race by race",
                    "Hollow marker: retired", accent=color,
                ):
                    fig = base_figure("", "Position", "", hovermode="x unified")
                    fig.update_layout(height=360, showlegend=True,
                                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)))
                    worst = int(max(race["PosA"].max(), race["PosB"].max())) if len(race) else 20
                    fig.update_yaxes(range=[worst + 0.8, 0.2], tickvals=[1, 5, 10, 15, 20])
                    for side, driver, c in (("A", pair.A, color), ("B", pair.B, "#eef1f5")):
                        fig.add_trace(go.Scatter(
                            x=[short_event(e) for e in race["Event"]], y=race[f"Pos{side}"], name=driver,
                            mode="lines+markers", line=dict(color=c, width=2.2),
                            marker=dict(size=9, color=[c if f else "rgba(0,0,0,0)" for f in race[f"Fin{side}"]],
                                        line=dict(color=c, width=2)),
                            hovertemplate=f"{driver}: P%{{y}}<extra></extra>",
                        ))
                    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
                    plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

# ------------------------------------------------------------ pit stops --

if section == SECTION_SEASON and showing(tab_pits):
    with tab_pits:
        section_head(
            "Pit stops",
            f"{year} season · DHL stationary times",
            "The fastest stops of the year, the fastest at every Grand Prix, and the fastest DHL has "
            "ever timed.",
            accent=PALETTE["amber"],
        )
        season_dhl = dhl_season(year) if year >= DHL_FIRST_SEASON else None
        if not season_dhl:
            empty_state(f"DHL has no pit stop times for {year} (its records start in {DHL_FIRST_SEASON})")
        else:
            pit_record_cards()
            st.write("")
            today = datetime.date.today().isoformat()
            run = [e for e in season_dhl["events"] if e["day"] <= today]
            winners = []
            for e in run:
                _, stops = dhl_event_stops(year, e["day"])
                if stops:
                    winners.append((e, stops[0]))
            season_best = min((w["duration"] for _, w in winners), default=None)

            # The two top tens side by side -- the same list twice, one for the
            # season and one for all time, so they read as a pair -- then the
            # every-Grand-Prix list at full width, where its circuit names fit.
            col_season, col_ever = st.columns(2)
            with col_season:
                with chart_panel(f"Fastest of {year}", "The ten quickest stops of the season", accent=PALETTE["amber"]):
                    pit_list([{"duration": r["duration"], "name": f"{r['firstName'][0]}. {r['lastName']}",
                               "team": r["team"], "extra": r["abbreviation"]}
                              for r in sorted(season_dhl["fastest"], key=lambda x: x["duration"])[:10]])
            with col_ever:
                with chart_panel("Fastest ever", f"Since {DHL_FIRST_SEASON}, all seasons pooled", accent=PALETTE["amber"]):
                    pit_list([{"duration": r["duration"], "name": f"{r['firstName'][0]}. {r['lastName']}",
                               "team": r["team"], "extra": f"{r['abbreviation']} {r['year']}"}
                              for r in dhl_all_time_fastest()[:10]])
            st.write("")
            with chart_panel("Fastest stop at every Grand Prix", "In calendar order · the season's quickest highlighted",
                             accent=PALETTE["amber"]):
                pit_list(
                    [{"duration": w["duration"], "name": f"{w['firstName'][0]}. {w['lastName']}",
                      "team": w["team"], "extra": e["short_title"], "rank": e["abbr"]}
                     for e, w in winners],
                    best_duration=season_best,
                )
            st.write("")
            if season_dhl["standings"]:
                with chart_panel("DHL Fastest Pit Stop Award", "Team standings · points for the ten quickest stops of every race",
                                 accent=PALETTE["amber"]):
                    standings = season_dhl["standings"]
                    fig = base_figure("", "", "Award points", hovermode="closest")
                    fig.update_layout(showlegend=False)
                    teams = [r["team"] for r in standings][::-1]
                    points = [r["points"] for r in standings][::-1]
                    fig.add_trace(go.Bar(
                        x=points, y=teams, orientation="h",
                        marker=dict(color=[dhl_team_color(t) for t in teams]),
                        text=points, textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} points<extra></extra>",
                    ))
                    size_horizontal_bars(fig, points)
                    style_bars(fig)
                    plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            st.markdown(DHL_SOURCE, unsafe_allow_html=True)

# Fixed colours for the teams that defined an era. Keyed on Ergast's
# constructor names, which FastF1's colour lookup doesn't know for past
# teams (Team Lotus, Brabham) -- it would paint them all grey. Williams is
# drawn white, after its long white-and-blue years, because its blue sat too
# close to Red Bull's. Anything outside this list goes into "Other".
ERA_COLORS = {
    "Ferrari": "#e8002d",
    "McLaren": "#ff8000",
    "Mercedes": "#27f4d2",
    "Red Bull": "#3671c6",
    "Williams": "#e6ebf2",
    "Lotus": "#e0b83b",
    "Brabham": "#9b6bff",
}
ERA_OTHER = "#4a5163"

# Ergast files one team under several names, one per engine deal --
# "Lotus-Climax", "Lotus-Ford", "Team Lotus" -- which split Lotus's 79 wins
# three ways and sent the sixties into "Other". Folded back into one team
# here. Lotus F1 (2012-15) was a different team, Renault renamed, and stays
# apart.
CONSTRUCTOR_GROUPS = {
    "Team Lotus": "Lotus", "Lotus-Climax": "Lotus", "Lotus-Ford": "Lotus", "Lotus-BRM": "Lotus",
    "Brabham-Repco": "Brabham", "Brabham-Climax": "Brabham", "Brabham-Ford": "Brabham",
    "Cooper-Climax": "Cooper", "Cooper-Maserati": "Cooper",
    "McLaren-Ford": "McLaren",
}


def render_eras(chronological, wins_by_team, races_per_season):
    """Every season since 1950 as one column, split by who won its races:
    the eras -- Ferrari at the turn of the century, McLaren in the late
    eighties, Mercedes, Red Bull -- show as blocks of one colour. Shares,
    not counts, since a season has had anywhere from 7 to 24 races."""
    teams = [t for t in wins_by_team.index if t in ERA_COLORS]
    by_season = chronological.groupby(["season", "constructorName"]).size().unstack(fill_value=0)
    seasons = by_season.index.astype(int)
    fig = base_figure("", "Share of the season's races won", "", hovermode="closest")
    fig.update_layout(
        barmode="stack", bargap=0.12, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, traceorder="normal"),
    )
    other = by_season.drop(columns=[t for t in teams if t in by_season.columns]).sum(axis=1)
    series = [(t, by_season[t] if t in by_season else 0 * other, ERA_COLORS[t]) for t in teams]
    series.append(("Other", other, ERA_OTHER))
    for name, counts, color in series:
        share = 100 * counts / races_per_season.reindex(by_season.index)
        # Zeros, not blanks, for a season a team won nothing: plotly stacks a
        # blank by starting the next bar from the axis, so the columns
        # overlapped and stopped short of 100%.
        fig.add_trace(go.Bar(
            x=seasons, y=share, name=name,
            marker=dict(color=color, line=dict(width=0)),
            customdata=np.stack([counts.to_numpy(), races_per_season.reindex(by_season.index).to_numpy()], axis=1),
            hovertemplate=f"%{{x}} · {name}: %{{customdata[0]}} of %{{customdata[1]}} races (%{{y:.0f}}%)<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", dtick=25)
    fig.update_xaxes(dtick=10, showgrid=False)
    with chart_panel(
        "The eras",
        "Every season since 1950, split by the team that won its races",
        accent=PALETTE["red"],
    ):
        plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


RECORD_RACE_COLORS = [
    PALETTE["teal"], PALETTE["red"], PALETTE["blue"], PALETTE["amber"],
    PALETTE["violet"], PALETTE["pink"], "#e6ebf2", "#8fd3ff",
]


def render_record_race(chronological, wins_by_driver, top=8):
    """Career wins piling up over time for the most successful drivers: a
    ranking says who has the most, this shows how and when they got there
    -- and the moment one line overtakes another."""
    leaders = list(wins_by_driver.head(top).index)
    first_win = chronological[chronological["Driver"].isin(leaders)]["raceDate"].min()
    start = pd.to_datetime(first_win) - pd.DateOffset(years=2)
    end = pd.Timestamp.today().normalize()
    fig = base_figure("", "Career wins", "", hovermode="closest")
    height = 480
    fig.update_layout(height=height, showlegend=False, margin=dict(t=16, b=40, l=8, r=150))
    finals = {d: int(wins_by_driver[d]) for d in leaders}
    top_value = max(finals.values()) * 1.06
    offsets = dict(zip(leaders, spread_labels(
        [finals[d] for d in leaders], plot_height_px=height - 16 - 40, value_span=top_value,
    )))
    for index, driver in enumerate(leaders):
        color = RECORD_RACE_COLORS[index % len(RECORD_RACE_COLORS)]
        own = chronological[chronological["Driver"] == driver].sort_values(["season", "round"])
        dates = pd.to_datetime(own["raceDate"]).tolist()
        counts = list(range(1, len(own) + 1))
        # The line runs flat to today (or to the last win of a finished
        # career), so the labels line up at the right edge like a table.
        fig.add_trace(go.Scatter(
            x=dates + [end], y=counts + [counts[-1]], mode="lines", line=dict(color=color, width=2.2, shape="hv"),
            text=[f"{r} {int(s)}" for r, s in zip(own["raceName"], own["season"])] + ["today"],
            hovertemplate=f"{driver}: win %{{y}} · %{{text}}<extra></extra>",
        ))
        fig.add_annotation(
            x=end, y=finals[driver], text=f"<b>{own['familyName'].iloc[0]}</b>  {finals[driver]}",
            showarrow=False, xanchor="left", xshift=8, yshift=-offsets[driver],
            font=dict(size=11.5, color=PALETTE["ink"]),
        )
    fig.update_yaxes(range=[0, top_value])
    fig.update_xaxes(range=[start, end], showgrid=False)
    with chart_panel(
        "The race to the record",
        f"Career wins over time for the {top} most successful drivers",
        accent=PALETTE["amber"],
    ):
        plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


def render_all_time_tables(chronological, poles, wins_by_driver, poles_by_driver, wins_by_team, streaks):
    """The all-time lists as two tables, one row per driver or team with
    every count side by side -- wins, poles, how often a pole became a win,
    the best season, the longest run -- where they used to be seven
    separate bar charts, each ranking the same names on one number."""
    wins_from_pole = chronological[chronological["grid"] == 1]["Driver"].value_counts()
    season_wins = chronological.groupby(["Driver", "season"]).size()
    longest_by_driver = streaks.groupby("Driver")["races"].max()

    driver_rows = []
    for rank, (driver, wins_count) in enumerate(wins_by_driver.head(20).items(), start=1):
        pole_count = int(poles_by_driver.get(driver, 0))
        converted = int(wins_from_pole.get(driver, 0))
        seasons = season_wins[driver]
        driver_rows.append({
            "pos": str(rank), "driver": driver,
            "wins": str(int(wins_count)), "poles": str(pole_count),
            "conv": f"{100 * converted / pole_count:.0f}%" if pole_count else "—",
            "season": f"{int(seasons.max())} · {int(seasons.idxmax())}",
            "streak": str(int(longest_by_driver.get(driver, 1))),
        })

    team_poles = poles["constructorName"].value_counts()
    team_seasons = chronological.groupby(["constructorName", "season"]).size()
    team_rows = []
    for rank, (team, wins_count) in enumerate(wins_by_team.head(15).items(), start=1):
        seasons = team_seasons[team]
        team_rows.append({
            "pos": str(rank), "team": team, "color": ERA_COLORS.get(team, "transparent"),
            "wins": str(int(wins_count)), "poles": str(int(team_poles.get(team, 0))),
            "span": f"{int(seasons.index.min())}–{int(seasons.index.max())}"
            if seasons.index.min() != seasons.index.max() else f"{int(seasons.index.min())}",
            "season": f"{int(seasons.max())} · {int(seasons.idxmax())}",
        })

    # One above the other, not side by side: at half width the names wrapped
    # onto two lines and the last columns ran out of the panel.
    with st.container():
        with chart_panel("Drivers", "The 20 with the most wins", accent=PALETTE["blue"]):
            render_table(driver_rows, [
                ("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("wins", "Wins", "num"),
                ("poles", "Poles", "num"), ("conv", "Pole → win", "num"),
                ("season", "Best season", "mono"), ("streak", "Streak", "num"),
            ])
        st.write("")
        with chart_panel("Constructors", "The 15 with the most wins", accent=PALETTE["teal"]):
            render_table(team_rows, [
                ("pos", "Pos", "pos"), ("team", "Team", "name"), ("wins", "Wins", "num"),
                ("poles", "Poles", "num"), ("span", "Winning years", "mono"),
                ("season", "Best season", "mono"),
            ])
    method_note(
        "**Pole → win** is the share of a driver's poles they turned into a win. **Best season** is "
        "the most wins in one year and the year it came. **Streak** is the most races won in a row, "
        "counted across season boundaries -- a streak doesn't reset in January. **Winning years** "
        "runs from a team's first win to its latest.",
        "What the columns mean",
    )


# ------------------------------------------------------------ all-time --

if section == SECTION_ALL_TIME:
    with st.container():
        section_head(
            "Records",
            "1950 — today",
            "Every world-championship race since 1950. The first load is slow (the whole history "
            "is fetched once), instant after that.",
            accent=PALETTE["violet"],
        )

        try:
            wins = all_time_finishes(results_position=1)
            poles = all_time_finishes(grid_position=1)
        except Exception as exc:
            st.error(f"Couldn't load the historical results: {exc}")
        else:
            wins["Driver"] = wins.apply(full_name, axis=1)
            wins["constructorName"] = wins["constructorName"].replace(CONSTRUCTOR_GROUPS)
            poles["constructorName"] = poles["constructorName"].replace(CONSTRUCTOR_GROUPS)
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

            stat_cards([
                ("Most wins", f"{int(wins_by_driver.iloc[0])}",
                 f"\U0001F3C6 <b>{wins_by_driver.index[0]}</b>", PALETTE["amber"]),
                ("Most poles", f"{int(poles_by_driver.iloc[0])}",
                 f"\U0001F3C1 <b>{poles_by_driver.index[0]}</b>", PALETTE["violet"]),
                ("Most wins, constructor", f"{int(wins_by_team.iloc[0])}",
                 f"\U0001F527 <b>{wins_by_team.index[0]}</b>", PALETTE["teal"]),
            ])
            st.write("")
            stat_cards([
                ("Youngest winner", f"{ages.min():.1f} <small>years</small>",
                 f"<b>{youngest['Driver']}</b> · {int(youngest['season'])} {youngest['raceName']}",
                 PALETTE["blue"]),
                ("Oldest winner", f"{ages.max():.1f} <small>years</small>",
                 f"<b>{oldest['Driver']}</b> · {int(oldest['season'])} {oldest['raceName']}",
                 PALETTE["blue"]),
                ("Win from furthest back", f"P{int(comeback['grid'])}",
                 f"<b>{comeback['Driver']}</b> · {int(comeback['season'])} {comeback['raceName']}",
                 PALETTE["pink"]),
            ])
            st.write("")
            # Seasons, streaks and pole conversion come out of the same two
            # downloads as everything above.
            races_per_season = chronological.groupby("season").size()
            season_wins = chronological.groupby(["season", "Driver"]).size()
            (best_season, best_season_driver), best_season_count = season_wins.idxmax(), int(season_wins.max())
            # A new streak starts wherever the winner differs from the previous
            # race's, so a running count of those changes labels each streak.
            # The grouping key has to be renamed: it inherits the name "Driver"
            # from the column it's derived from, which collides with grouping by
            # the column itself.
            streak_id = (chronological["Driver"] != chronological["Driver"].shift()).cumsum().rename("streak")
            streaks = (
                chronological.groupby([streak_id, chronological["Driver"]])
                .size().rename("races").reset_index()
            )
            longest = streaks.loc[streaks["races"].idxmax()]
            longest_seasons = chronological.loc[streak_id == longest["streak"], "season"]
            longest_span = (
                f"{int(longest_seasons.min())}" if longest_seasons.min() == longest_seasons.max()
                else f"{int(longest_seasons.min())}–{int(longest_seasons.max())}"
            )
            stat_cards([
                (
                    "Wins started from pole",
                    f"{100 * pole_wins / len(chronological):.0f}<small>%</small>",
                    f"{pole_wins} of {len(chronological)} races — well under a coin flip",
                    PALETTE["red"],
                ),
                (
                    "Most wins in a season", f"{best_season_count}",
                    f"<b>{best_season_driver}</b> · {int(best_season)}, "
                    f"{100 * best_season_count / races_per_season[best_season]:.0f}% of the races",
                    PALETTE["amber"],
                ),
                (
                    "Longest winning streak", f"{int(longest['races'])} <small>races</small>",
                    f"<b>{longest['Driver']}</b> · {longest_span}",
                    PALETTE["pink"],
                ),
            ])

            # Career points totals are deliberately absent: F1 has rescored
            # itself repeatedly (8 points for a win until 1960, 9, then 10, then
            # 25 from 2010, half points for short races, best-N-results dropped
            # scores for decades), so an all-time points table ranks eras rather
            # than drivers. Counts of wins and poles mean the same thing in
            # every season.
            method_note(
                "Ranked by wins and poles rather than career points: the points system has been "
                "rewritten often enough (8 for a win in the fifties, 25 today, dropped scores for "
                "decades) that an all-time points table would rank eras, not drivers. Poles are "
                "counted as starts from P1, so a grid penalty can move one.",
                "Why there's no all-time points table",
            )

            st.write("")
            render_eras(chronological, wins_by_team, races_per_season)
            st.write("")
            render_record_race(chronological, wins_by_driver)
            st.write("")
            render_all_time_tables(chronological, poles, wins_by_driver, poles_by_driver, wins_by_team, streaks)
