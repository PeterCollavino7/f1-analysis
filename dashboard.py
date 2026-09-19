"""
F1 session dashboard: head-to-head telemetry, and race pace / tyre
degradation, for any past race weekend and any of its actual sessions.

Run with: venv\\Scripts\\streamlit run dashboard.py
"""
import datetime
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

st.set_page_config(page_title="F1 Dashboard", layout="wide", initial_sidebar_state="expanded")
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
    /* The Deploy button is Streamlit Cloud's call to action; this app runs on
       a laptop and has nowhere to deploy to, so it's dead weight sitting over
       the hero. Same reasoning as the status widget (hidden at the top of
       this file) -- and as there,
       only its visibility is touched, so a selector that stops matching costs
       a stray button, nothing more. The hamburger menu next to it stays:
       rerun, settings and "clear cache" are all genuinely useful here. */
    [data-testid="stAppDeployButton"] { display: none; }

    /* ------------------------------------------------------------ sidebar */
    /* Restyled with background, border, spacing and typography only --
       nothing here hides a Streamlit element, because the internal DOM
       isn't ours and a selector that stops matching after an upgrade would
       silently take a control with it. If a rule below misses, the control
       still renders, just unstyled. */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151a26 0%, #0c0f18 100%);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] label {
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.66rem;
        font-weight: 700;
        color: var(--ink-faint);
    }
    /* The section picker reads as a nav rather than a form control: each
       option gets its own row, and the chosen one carries the accent. */
    [data-testid="stSidebar"] [role="radiogroup"] > label {
        text-transform: none;
        letter-spacing: 0.01em;
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink);
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.3rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(255, 255, 255, 0.02);
        transition: background 0.14s ease, border-color 0.14s ease, transform 0.14s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: var(--line-strong);
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(225, 6, 0, 0.22), rgba(225, 6, 0, 0.04));
        border-color: rgba(225, 6, 0, 0.55);
        box-shadow: inset 3px 0 0 var(--f1-red);
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.04);
        border-color: var(--line);
        border-radius: 10px;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.1rem 0 0.9rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--line);
    }
    .sidebar-brand .flag {
        width: 24px; height: 24px; flex: 0 0 auto; border-radius: 5px;
        background-image:
            linear-gradient(45deg, #e8ebef 25%, transparent 25%, transparent 75%, #e8ebef 75%),
            linear-gradient(45deg, #e8ebef 25%, #12151d 25%, #12151d 75%, #e8ebef 75%);
        background-size: 12px 12px;
        background-position: 0 0, 6px 6px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    .sidebar-brand .word {
        font-size: 1.1rem; font-weight: 700; letter-spacing: 0.08em; color: #f2f4f6; line-height: 1.15;
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
    .sidebar-foot {
        margin-top: 1.5rem; padding-top: 0.9rem;
        border-top: 1px solid var(--line);
        font-size: 0.7rem; line-height: 1.7; color: var(--ink-faint);
    }
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
    .hero-track { position: relative; z-index: 1; flex: 0 0 auto; opacity: 0.8; }
    .hero-track svg {
        display: block; height: 112px; width: auto;
        filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.18));
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
    .tm-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .tm-table th {
        text-align: left; font-weight: 600; font-size: 0.72rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--ink-faint); padding: 0 10px 8px;
    }
    .tm-table td { padding: 9px 10px; border-top: 1px solid var(--line); vertical-align: middle; }
    .tm-team { display: flex; align-items: center; gap: 9px; color: var(--ink-dim); white-space: nowrap; }
    .tm-dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
    .tm-pair { font-family: var(--display); font-weight: 700; letter-spacing: 0.04em; white-space: nowrap; }
    .tm-pair span { color: var(--ink-faint); font-weight: 400; margin: 0 5px; }
    .tm-split { display: flex; align-items: center; gap: 8px; min-width: 150px; }
    .tm-split b { min-width: 2.2em; text-align: center; white-space: nowrap; font-family: var(--mono); font-weight: 500; }
    .tm-bar { flex: 1; display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--line); }
    .tm-bar i { display: block; height: 100%; }
    .tm-gap { font-family: var(--mono); font-size: 0.82rem; color: var(--ink-dim); white-space: nowrap; }

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
            driver_rows.append({"Round": rnd, "Driver": name, "Points": r["points"]})
        for _, r in constructor_table.iterrows():
            constructor_rows.append({"Round": rnd, "Constructor": r["constructorName"], "Points": r["points"]})
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
def load_schedule(year):
    with FF1_LOCK:
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
def published_assets(year):
    """{asset name: API url} for the season's release; empty if there's no
    release for that year."""
    release = requests.get(
        f"https://api.github.com/repos/{DATA_REPO}/releases/tags/{year}", headers=github_headers(), timeout=20,
    )
    if release.status_code == 404:
        return {}
    release.raise_for_status()
    assets, page = {}, 1
    while True:
        # The release object's own asset list isn't the place to read a full
        # season from; this endpoint pages through all of them.
        batch = requests.get(
            f"https://api.github.com/repos/{DATA_REPO}/releases/{release.json()['id']}/assets",
            headers=github_headers(), params={"per_page": 100, "page": page}, timeout=20,
        ).json()
        assets.update({a["name"]: a["url"] for a in batch})
        if len(batch) < 100:
            return assets
        page += 1


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
    for _, event in load_schedule(year).iterrows():
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


def render_hero(kicker_pill, kicker_text, title, subtitle, artwork="", chip_items=()):
    """The page header: kicker, title, one line of context, and a row of
    chips for the facts that used to be crammed into that one line
    (circuit, date, session, race count) separated by middle dots."""
    chips_html = (
        '<div class="hero-chips">' + "".join(f"<span>{c}</span>" for c in chip_items) + "</div>"
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

st.sidebar.markdown(
    '<div class="sidebar-brand"><div class="flag"></div>'
    '<div class="word"><em>F1</em> DASHBOARD'
    '<span>TELEMETRY · STANDINGS · RECORDS</span></div></div>',
    unsafe_allow_html=True,
)
section = st.sidebar.radio("Section", [SECTION_WEEKEND, SECTION_SEASON, SECTION_ALL_TIME])

session = None
if section in (SECTION_WEEKEND, SECTION_SEASON):
    st.sidebar.divider()
    year = st.sidebar.selectbox("Year", options=available_years())
    schedule = load_schedule(year)
    published = published_sessions(year) if data_store_token() is not None else None
    if published is not None:
        schedule = schedule[schedule["EventName"].isin(published)]
    event_name = st.sidebar.selectbox(
        "Grand Prix" if section == SECTION_WEEKEND else "Standings after",
        options=schedule["EventName"].tolist(),
    )
    event_row = schedule[schedule["EventName"] == event_name].iloc[0]
    # The season views don't compare laps, but they do color drivers and
    # teams, and fastf1.plotting needs a loaded session to do that -- the
    # weekend's race is the one that's always there.
    session_name = (
        st.sidebar.selectbox(
            "Session",
            options=[n for n in session_names_for(event_row) if published is None or n in published[event_name]],
        )
        if section == SECTION_WEEKEND else "Race"
    )
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
        ],
    )
    # The headline numbers sit above the tabs, not inside one of them: they
    # describe the session itself, and a reader shouldn't have to guess which
    # tab hides "who won".
    stat_cards(weekend_headline_cards(session, session_name))
    st.write("")
    # Results first, then pace, then telemetry: the order a weekend is
    # actually read -- what happened, how the race ran, then the lap-by-lap
    # forensics. (Telemetry used to be first, which opened the page on the
    # deepest view of all.)
    # A qualifying session has no race to pace: its middle tab is "Stats"
    # instead (gap to pole, ideal lap, the season's poles) -- see
    # render_quali_stats().
    tab_classification, tab_pace, tab_telemetry = st.tabs(
        ["Results", "Stats" if "Qualifying" in session_name else "Race pace", "Head-to-head"]
    )
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
    tab_standings, tab_season_stats, tab_teammates = st.tabs(["Championship", "Race by race", "Teammates"])
else:
    render_hero(
        "1950 — today", "World championship",
        "All-time records",
        "Every world-championship race ever run, ranked",
        chip_items=["\U0001F3C6 wins", "\U0001F3C1 poles", "\U0001F4C8 streaks", f"\U0001F551 {datetime.date.today().year - 1949} seasons"],
    )

st.sidebar.markdown(
    '<div class="sidebar-foot">'
    '<b>Data</b> · FastF1 (official timing &amp; telemetry) + Ergast<br>'
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


def driver_picker(name, label, help_text, eligible, on_pick=None, limit=MAX_DRIVERS, default=()):
    """A two-driver choice: one pill per driver, teammates side by side, each
    marked with its team color. Used by the Head-to-head tab and by the race
    pace comparison; `name` keeps their selections apart, `eligible` is who
    can be picked there. Replaced a multiselect dropdown,
    which took an open-scroll-pick-reopen round per driver, offered a "Select
    all" that made no sense with a limit of two, and simply refused a third
    pick. Here every driver is in view, one click toggles, and a third pick
    replaces the older of the two.

    Starts empty (a choice, not whichever two come first), and the widget key
    carries the session, so switching race or session starts over rather
    than carrying over drivers the new session may not even have.
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
    # given both a default and a state-set value warns on screen.
    if key not in st.session_state:
        initial = [d for d in default if d in order][-limit:]
        st.session_state[key] = initial
        st.session_state[history_key] = initial
    st.pills(
        label, options=order, selection_mode="multi", key=key,
        on_change=keep_newest_two, help=help_text,
    )
    selected = [d for d in st.session_state.get(history_key, []) if d in order]

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
        session.laps["Driver"].unique(),
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
                st.plotly_chart(fig_sectors, width="stretch", config=PLOTLY_CONFIG)

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
        st.plotly_chart(fig_telemetry, width="stretch", config=PLOTLY_CONFIG)


if section == SECTION_WEEKEND:
    with tab_telemetry:
        render_telemetry_tab()

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

            st.plotly_chart(fig_position, width="stretch", config=PLOTLY_CONFIG)


def render_quali_stats():
    """Gap to pole and ideal lap, the two numbers a qualifying session is
    read by. Deleted laps (track limits) are left out of both: a lap that
    didn't count for the grid shouldn't count here either."""
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
            st.plotly_chart(fig_gap, width="stretch", config=PLOTLY_CONFIG)

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
            st.plotly_chart(fig_ideal, width="stretch", config=PLOTLY_CONFIG)
    method_note(
        "**Ideal lap** is the sum of a driver's three best sector times, wherever in the "
        "session each was set. The bar is how much slower their best actual lap was than that "
        "-- a driver who strung their best sectors together on one lap scores zero. Drivers "
        "stay in order of their best lap, as in the gap chart beside it, so the two read row by row."
    )


def render_pit_stops():
    """Time spent in the pit lane at every stop, by team: entry line to exit
    line, which is what a stop actually costs in the race. FastF1 has no
    stationary time, so a slow wheel change and a slow pit lane read alike;
    the note says so."""
    # Race-like sessions only: in practice a trip down the lane is a run
    # ending in the garage, not a stop.
    if session.laps["Position"].isna().all():
        return
    laps = session.laps.sort_values(["Driver", "LapNumber"])
    team_of = session.results.set_index("Abbreviation")["TeamName"].to_dict()
    stops = []
    for driver, d in laps.groupby("Driver"):
        d = d.set_index("LapNumber")
        for lap_number, row in d.iterrows():
            if pd.isna(row["PitInTime"]) or (lap_number + 1) not in d.index:
                continue
            out = d.loc[lap_number + 1, "PitOutTime"]
            if pd.isna(out):
                continue
            seconds = (out - row["PitInTime"]).total_seconds()
            # Under 10 s isn't a stop through the lane; over 60 s is a red
            # flag or a repair in the garage, not a pit stop.
            if 10 < seconds < 60:
                stops.append({"Team": team_of.get(driver, "?"), "Driver": driver,
                              "Lap": int(lap_number), "Seconds": seconds})
    if not stops:
        return
    stops = pd.DataFrame(stops)
    medians = stops.groupby("Team")["Seconds"].median().sort_values()
    teams = list(reversed(medians.index))  # quickest at the top

    def team_color(name):
        try:
            return fastf1.plotting.get_team_color(name, session)
        except Exception:
            return "#999999"

    with chart_panel(
        "Pit lane time by team",
        "Every stop, entry to exit · the tick is each team's median, quickest at the top",
        accent=PALETTE["amber"],
    ):
        # Dots and a median tick, not bars: the times all sit between about
        # 20 and 45 s, so the axis can't start at zero without squashing the
        # differences into a sliver -- and a bar on an axis that doesn't start
        # at zero draws lengths that mean nothing.
        fig = base_figure("", "", "Seconds in the pit lane", hovermode="closest")
        fig.update_layout(height=max(260, 34 * len(teams) + 90), showlegend=False, margin=dict(r=70))
        fig.update_yaxes(categoryorder="array", categoryarray=teams, tickfont=dict(size=11))
        fig.add_trace(go.Scatter(
            x=[medians[t] for t in teams], y=teams, mode="markers",
            marker=dict(symbol="line-ns", size=24, line=dict(width=3, color=[team_color(t) for t in teams])),
            hovertemplate="%{y}: median %{x:.1f} s<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stops["Seconds"], y=stops["Team"], mode="markers",
            marker=dict(size=9, color=[hex_to_rgba(team_color(t), 0.55) for t in stops["Team"]],
                        line=dict(color=[team_color(t) for t in stops["Team"]], width=1.5)),
            text=[f"{r.Driver} · lap {r.Lap}: {r.Seconds:.1f} s" for r in stops.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))
        right = stops["Seconds"].max() + 2
        fig.update_xaxes(range=[stops["Seconds"].min() - 2, right])
        # The median as a figure, in a column of its own past the last stop.
        for t in teams:
            fig.add_annotation(
                x=right, y=t, xanchor="left", xshift=10, showarrow=False, text=f"{medians[t]:.1f} s",
                font=dict(size=11, color="rgba(226,232,240,0.8)"),
            )
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    method_note(
        "Time from the pit entry line to the exit line, from the official timing, for each stop "
        "that took between 10 and 60 seconds (shorter isn't a stop through the lane; longer is a "
        "red flag or a repair). It includes the drive down the speed-limited lane, so it varies "
        "from track to track, and it can't tell a slow wheel change from a slow lane: FastF1 has "
        "no stationary time. A drive-through or stop-and-go penalty looks like a stop too.",
        "What counts as pit lane time",
    )
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

    def team_color(name):
        try:
            return fastf1.plotting.get_team_color(name, session)
        except Exception:
            return "#999999"

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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    method_note(
        "One lap per team, the quickest either driver set in this session. **Top speed** is that "
        "lap's highest reading. **Corner speed** is the average of its minimum speeds, one per "
        "corner, found in the telemetry as a speed dip of at least 25 km/h. A team up and to the "
        "left runs less wing (quick on the straights, slower in the corners); down and to the right "
        "runs more. The dotted lines are the field's medians. It's one lap each, so a tow on the "
        "straight or a scruffy corner moves a point: read it as a tendency, not a measurement.",
        "How the two speeds are taken",
    )


def render_pace_tab():
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

            st.plotly_chart(fig_strategy, width="stretch", config=PLOTLY_CONFIG)
            # Compound colors as chips instead of a Plotly legend: the legend
            # entry for a stacked bar chart is one swatch per trace, and there
            # is one trace per stint -- roughly sixty of them.
            chips([
                (COMPOUND_COLORS.get(c, "#999999"), c.title())
                for c in sorted(used_compounds, key=lambda c: list(COMPOUND_COLORS).index(c) if c in COMPOUND_COLORS else 9)
            ])

    st.write("")

    # Head-to-head race pace: two drivers' lap times side by side, on the tyre
    # each was actually running. The degradation fits further down answer "how
    # fast does this compound wear for the whole field"; they can't answer "was
    # he quicker than the car he was racing, and when" -- which is the question
    # a race is argued over, and the one the tyre-age chart kept being asked to
    # do and couldn't (its x axis is tyre life, so two drivers on different
    # strategies get laid on top of each other out of sequence).
    render_pit_stops()

    h2h_laps = session.laps.dropna(subset=["LapTime", "LapNumber"])
    if not h2h_laps.empty and h2h_laps["Driver"].nunique() >= 2:
        # Opens on the winner against the runner-up, so the chart is there
        # from the start rather than behind a choice; the pills live in a
        # popover inside the panel and close themselves on each pick (a new
        # driver replaces the older of the two). The title needs the pair
        # before the picker is drawn, so it's read from the order the picker
        # keeps in session state, falling back to that same default pair.
        h2h_eligible = h2h_laps["Driver"].unique()
        default_pair = order_by_classification(session, h2h_eligible)[:2]
        picked = [
            d for d in st.session_state.get(picker_key("pace") + "_order", default_pair) if d in h2h_eligible
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
                    h2h_eligible, on_pick=close_pace_popover, default=default_pair,
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
                st.plotly_chart(fig_h2h, width="stretch", config=PLOTLY_CONFIG)
                chips(
                    [
                        (COMPOUND_COLORS.get(c, "#999999"), str(c).title())
                        for c in compounds_seen
                    ]
                    + ([("rgba(255,179,64,0.5)", "Safety car / VSC")] if neutralised else [])
                )

        st.write("")

    overtake_laps = session.laps.dropna(subset=["Position", "LapNumber"])
    if overtake_laps.empty:
        empty_state("No lap-by-lap position data available to count overtakes in this session")
    else:
        method_note(
            "On-track position gains, lap over lap, with a gain credited only for what's left "
            "after excluding four sources of *free* positions: the driver's own pit in/out lap "
            "or a lap where their own time got deleted for track limits (race control's own call "
            "that the lap wasn't clean); however many rivals ahead of them pitted (in or out) "
            "that same lap; and however many rivals ahead of them lost 3+ places themselves that "
            "lap without pitting -- a swing that big in one lap is a spin, contact or mechanical "
            "issue, not a queue of cars each individually passing them. A rival losing 1-2 places "
            "still counts as fair game, since that's within range of a normal on-track pass -- so "
            "this can still include a position gained off a smaller, undetectable mistake, or a "
            "steward's call this data has no way to see. **Treat this as an approximation, not an "
            "official count.**",
            "How an overtake is counted here",
        )
        overtake_counts = count_overtakes(session)

        ranked_overtakes = sorted(overtake_counts.items(), key=lambda kv: kv[1], reverse=True)
        ranked_overtakes = [(d, c) for d, c in ranked_overtakes if c > 0]
        if not ranked_overtakes:
            empty_state("No on-track position gains detected in this session")
        else:
            overtake_styles = build_driver_styles([d for d, _ in ranked_overtakes], session)
            with chart_panel(
                "Overtakes by driver",
                "On-track position gains, net of pit stops and incidents",
                accent=PALETTE["teal"],
            ):
                fig_overtakes = base_figure("", "", "Position gains")
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
                        hovertemplate="%{y}: %{x} position gains<extra></extra>",
                    )
                )
                fig_overtakes.update_yaxes(autorange="reversed")
                size_horizontal_bars(fig_overtakes, [c for _, c in ranked_overtakes])
                style_bars(fig_overtakes)
                st.plotly_chart(fig_overtakes, width="stretch", config=PLOTLY_CONFIG)

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
            st.plotly_chart(fig_spread, width="stretch", config=PLOTLY_CONFIG)

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
        # the rerun it triggered starts.
        pace_drivers = [
            d for d in st.session_state.get(picker_key("laps") + "_order", []) if d in pace_driver_options
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
            st.plotly_chart(fig_pace, width="stretch", config=PLOTLY_CONFIG)

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
            render_quali_stats()
            st.write("")
            render_car_characteristics()
            st.write("")
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
                        st.plotly_chart(fig_poles_driver, width="stretch", config=PLOTLY_CONFIG)

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
            section_head(
                "Fastest laps",
                session_name,
                f"{session_name} has no official classification -- this ranks every driver by "
                "their best lap of the session instead.",
                accent=PALETTE["blue"],
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
            with chart_panel("Session ranking", "By best lap", accent=PALETTE["blue"]):
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
                rows.append(row)

            columns = [("pos", "Pos", "pos"), ("driver", "Driver", "name"), ("team", "Team", "team")]
            if has_quali_times:
                columns += [("q1", "Q1", "mono"), ("q2", "Q2", "mono"), ("q3", "Q3", "mono")]
            else:
                columns += [
                    ("grid", "Grid", "num"), ("delta", "+/−", "delta"), ("gap", "Gap", "mono"),
                    ("pts", "Pts", "num"), ("status", "Status", "status"),
                ]
            panel_title = "Qualifying classification" if has_quali_times else "Race classification"
            with chart_panel(
                panel_title,
                f"{event_name} · {session_name}",
                accent=PALETTE["red"],
            ):
                render_table(rows, columns)

            if not has_quali_times:
                st.write("")
                render_position_chart()

# ------------------------------------------------------------- standings --

if section == SECTION_SEASON:
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

            def team_color(name):
                try:
                    return fastf1.plotting.get_team_color(name, session)
                except Exception:
                    return "#999999"

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
                    st.plotly_chart(fig_driver_progress, width="stretch", config=PLOTLY_CONFIG)
                st.write("")

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
            with chart_panel("Drivers' standings", f"after round {round_number}"):
                render_table(driver_rows, [
                    ("pos", "Pos", "pos"), ("name", "Driver", "name"), ("team", "Team", "team"),
                    ("points", "Points", "points"), ("wins", "Wins", "num"),
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
                    st.plotly_chart(fig_constructor_progress, width="stretch", config=PLOTLY_CONFIG)
                st.write("")

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
            with chart_panel("Constructors' standings", f"after round {round_number}", accent=PALETTE["blue"]):
                render_table(constructor_rows, [
                    ("pos", "Pos", "pos"), ("name", "Constructor", "name"),
                    ("points", "Points", "points"), ("wins", "Wins", "num"),
                ])

            st.write("")

            total_rounds = total_rounds_in_season(year)
            remaining_rounds = total_rounds - round_number
            section_head(
                "Title fight",
                f"{remaining_rounds} race{'s' if remaining_rounds != 1 else ''} left",
                "How the championship looks from here, simulated from what each contender has "
                "actually been scoring.",
                accent=PALETTE["amber"],
            )
            method_note(
                "A simplified Monte Carlo estimate limited to the current top 3: each remaining race "
                "is resampled from that driver's (or team's) own points scored per race so far, "
                "independently of who else is on track -- so a driver's simulated score doesn't take "
                "points away from a rival the way a real race would -- and anyone outside the top 3 "
                "is ignored. **Treat this as a feel for how close the fight is, not a forecast.**",
                "How the odds are simulated",
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
                        marker=dict(
                            colors=[c for _, _, c in ranked],
                            line=dict(color="#0d1017", width=3),
                        ),
                        # The favourite's slice is pulled a hair out of the ring
                        # so the chart has a subject, not just three shares.
                        pull=[0.035] + [0] * (len(ranked) - 1),
                        hole=0.62, sort=False,
                        # No slice labels at all. Plotly stacks the outside
                        # ones in the order it draws them, so the two
                        # no-hope contenders -- whose slices are a sliver or
                        # nothing -- piled their names on top of each other
                        # above the ring, next to a leader line pointing at a
                        # slice too thin to see. The three names and shares
                        # are listed as chips under the chart instead, where
                        # they can't collide, and the favourite's own share
                        # is in the hole.
                        textinfo="none",
                        hovertemplate="%{label}: %{percent}<extra></extra>",
                    )
                )
                fig.update_layout(
                    **DARK_LAYOUT,
                    height=270, showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
                )
                # Two annotations, not one with a <span> in it: a mixed-size
                # <br> block is laid out on the base font's line height, so the
                # 26px number rode up into the name above it.
                fig.add_annotation(
                    text=f"<b>{ranked[0][0]}</b>", showarrow=False,
                    font=dict(size=13, color=ranked[0][2]), x=0.5, y=0.60,
                )
                fig.add_annotation(
                    text=f"{format_odds(ranked[0][1] * 100)}", showarrow=False,
                    font=dict(size=27, color=ranked[0][2]), x=0.5, y=0.44,
                )
                return fig

            col_drivers_odds, col_constructors_odds = st.columns(2)

            with col_drivers_odds:
                st.markdown(
                    '<div class="panel-head" style="--accent:' + PALETTE["red"] + '">'
                    '<div class="title">Drivers\' title odds</div></div>',
                    unsafe_allow_html=True,
                )
                top3_drivers = driver_standings.sort_values("position").head(3)
                names = [
                    r["driverCode"] if pd.notna(r["driverCode"]) else f"{r['givenName']} {r['familyName']}"
                    for _, r in top3_drivers.iterrows()
                ]
                histories = [race_history(driver_progress, "Driver", n) for n in names]
                odds = simulate_top3_odds(names, top3_drivers["points"].tolist(), histories, remaining_rounds)
                colors = [driver_color(n) for n in names]
                st.plotly_chart(
                    odds_chart("", names, [odds[n] for n in names], colors),
                    width="stretch", config=PLOTLY_CONFIG,
                )
                chips([
                    (color, f"{name} {format_odds(odds[name] * 100)}")
                    for name, color in sorted(
                        zip(names, colors), key=lambda pair: odds[pair[0]], reverse=True,
                    )
                ])

            with col_constructors_odds:
                st.markdown(
                    '<div class="panel-head" style="--accent:' + PALETTE["blue"] + '">'
                    '<div class="title">Constructors\' title odds</div></div>',
                    unsafe_allow_html=True,
                )
                top3_constructors = constructor_standings.sort_values("position").head(3)
                names_c = top3_constructors["constructorName"].tolist()
                histories_c = [race_history(constructor_progress, "Constructor", n) for n in names_c]
                odds_c = simulate_top3_odds(names_c, top3_constructors["points"].tolist(), histories_c, remaining_rounds)
                colors_c = [team_color(n) for n in names_c]
                st.plotly_chart(
                    odds_chart("", names_c, [odds_c[n] for n in names_c], colors_c),
                    width="stretch", config=PLOTLY_CONFIG,
                )
                chips([
                    (color, f"{name.replace(' F1 Team', '')} {format_odds(odds_c[name] * 100)}")
                    for name, color in sorted(
                        zip(names_c, colors_c), key=lambda pair: odds_c[pair[0]], reverse=True,
                    )
                ])

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

if section == SECTION_SEASON:
    with tab_season_stats:
        section_head(
            "The season so far",
            f"{len(schedule)} races",
            f"Computed across every completed race of the {year} season. The first load is slow "
            "(each race's full session is fetched once), instant after that.",
            accent=PALETTE["teal"],
        )
        stats_df, driver_overtakes = season_stats(year, tuple(schedule["EventName"].tolist()))

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
                    st.plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)

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
                st.plotly_chart(fig_overtakes_season, width="stretch", config=PLOTLY_CONFIG)

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
                    st.plotly_chart(fig_driver_overtakes, width="stretch", config=PLOTLY_CONFIG)

# ------------------------------------------------------------- teammates --

if section == SECTION_SEASON:
    with tab_teammates:
        section_head(
            "Teammate battles",
            f"{year} season so far",
            "The one fair fight in Formula 1: same car, same team. Who has been ahead in qualifying "
            "and in the races, and by how much -- then pick a pairing to see it race by race.",
            accent=PALETTE["violet"],
        )
        battles = teammate_battles(year, tuple(schedule["EventName"].tolist()))

        def team_color(name):
            try:
                return fastf1.plotting.get_team_color(name, session)
            except Exception:
                return "#999999"

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
                    '<table class="tm-table"><thead><tr><th>Team</th><th>Pair</th>'
                    "<th>Qualifying</th><th>Race</th><th>Points</th><th>Median quali gap</th></tr></thead>"
                    f"<tbody>{''.join(body)}</tbody></table>",
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
                    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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
                    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

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
            # per_row=3 so this single card keeps the width of one card in the
            # rows above it, instead of stretching across the whole page.
            stat_cards([(
                "Wins started from pole",
                f"{100 * pole_wins / len(chronological):.0f}<small>%</small>",
                f"{pole_wins} of {len(chronological)} races — pole is worth well under a coin flip",
                PALETTE["red"],
            )], per_row=3)

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

            with chart_panel("Most race wins", "All-time top 15", accent=PALETTE["blue"]):
                top_wins = wins_by_driver.head(15).sort_values()
                fig_wins = base_figure("", "", "Wins")
                fig_wins.update_layout(showlegend=False)
                fig_wins.add_trace(
                    go.Bar(
                        x=top_wins.to_numpy(), y=top_wins.index, orientation="h",
                        marker=dict(
                            color=top_wins.to_numpy(), colorscale=bar_scale(PALETTE["blue"]),
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=top_wins.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} wins<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_wins, top_wins.to_numpy())
                style_bars(fig_wins)
                st.plotly_chart(fig_wins, width="stretch", config=PLOTLY_CONFIG)

            st.write("")

            with chart_panel("Most pole positions", "All-time top 15", accent=PALETTE["violet"]):
                top_poles = poles_by_driver.head(15).sort_values()
                fig_poles_all = base_figure("", "", "Poles")
                fig_poles_all.update_layout(showlegend=False)
                fig_poles_all.add_trace(
                    go.Bar(
                        x=top_poles.to_numpy(), y=top_poles.index, orientation="h",
                        marker=dict(
                            color=top_poles.to_numpy(), colorscale=bar_scale(PALETTE["violet"]),
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=top_poles.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} poles<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_poles_all, top_poles.to_numpy())
                style_bars(fig_poles_all)
                st.plotly_chart(fig_poles_all, width="stretch", config=PLOTLY_CONFIG)

            st.write("")

            # New here: what a pole was actually worth to each of them. The two
            # charts above rank wins and poles separately, so the reader can
            # see that a driver has many of both without ever learning how
            # often the one turned into the other -- which is the interesting
            # part, and the per-driver version of the "wins started from pole"
            # card at the top of the page.
            wins_from_pole = chronological[chronological["grid"] == 1]["Driver"].value_counts()
            conversion = [
                (driver, int(poles), int(wins_from_pole.get(driver, 0)))
                for driver, poles in poles_by_driver.head(12).items()
            ]
            conversion.sort(key=lambda row: row[2] / row[1] if row[1] else 0)
            if conversion:
                with chart_panel(
                    "Pole-to-win conversion",
                    "Of the 12 biggest pole sitters · filled part is poles converted into wins",
                    accent=PALETTE["violet"],
                ):
                    fig_conv = base_figure("", "", "Poles", hovermode="closest")
                    fig_conv.update_layout(showlegend=False, barmode="overlay")
                    names_conv = [row[0] for row in conversion]
                    poles_conv = [row[1] for row in conversion]
                    won_conv = [row[2] for row in conversion]
                    rates = [100 * w / p if p else 0 for p, w in zip(poles_conv, won_conv)]
                    # The pale bar is the poles, the solid one the wins from
                    # them, drawn over it -- so the gap between the two *is*
                    # the poles that got away, with no second axis or
                    # percentage chart needed to see it.
                    fig_conv.add_trace(
                        go.Bar(
                            x=poles_conv, y=names_conv, orientation="h",
                            marker=dict(color="rgba(255,255,255,0.07)",
                                        line=dict(color="rgba(255,255,255,0.12)", width=1)),
                            text=[f"{r:.0f}%" for r in rates],
                            textposition="outside", cliponaxis=False,
                            hoverinfo="skip",
                        )
                    )
                    fig_conv.add_trace(
                        go.Bar(
                            x=won_conv, y=names_conv, orientation="h",
                            marker=dict(color=PALETTE["violet"]),
                            customdata=[[p, r] for p, r in zip(poles_conv, rates)],
                            hovertemplate=(
                                "%{y}: %{x} wins from %{customdata[0]} poles"
                                " (%{customdata[1]:.0f}%)<extra></extra>"
                            ),
                        )
                    )
                    size_horizontal_bars(fig_conv, poles_conv, pad_frac=0.22)
                    style_bars(fig_conv)
                    st.plotly_chart(fig_conv, width="stretch", config=PLOTLY_CONFIG)

            st.write("")

            with chart_panel("Most wins by constructor", "All-time top 15", accent=PALETTE["teal"]):
                top_team_wins = wins_by_team.head(15).sort_values()
                fig_team_wins = base_figure("", "", "Wins")
                fig_team_wins.update_layout(showlegend=False)
                fig_team_wins.add_trace(
                    go.Bar(
                        x=top_team_wins.to_numpy(), y=top_team_wins.index, orientation="h",
                        marker=dict(
                            color=top_team_wins.to_numpy(), colorscale=bar_scale(PALETTE["teal"]),
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=top_team_wins.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} wins<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_team_wins, top_team_wins.to_numpy())
                style_bars(fig_team_wins)
                st.plotly_chart(fig_team_wins, width="stretch", config=PLOTLY_CONFIG)

            st.write("")
            races_per_season = chronological.groupby("season").size()
            season_wins = chronological.groupby(["season", "Driver"]).size().sort_values(ascending=False)
            top_season_wins = season_wins.head(12).sort_values()
            season_labels = [f"{driver} · {int(season)}" for season, driver in top_season_wins.index]
            season_shares = [
                100 * count / races_per_season[season]
                for (season, _), count in top_season_wins.items()
            ]
            fig_season_wins = base_figure("", "", "Wins")
            fig_season_wins.update_layout(showlegend=False)
            fig_season_wins.add_trace(
                go.Bar(
                    x=top_season_wins.to_numpy(), y=season_labels, orientation="h",
                    marker=dict(
                        color=top_season_wins.to_numpy(), colorscale=bar_scale(PALETTE["amber"]),
                        line=dict(color="rgba(255,255,255,0.12)", width=1),
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
            style_bars(fig_season_wins)
            with chart_panel(
                "Most wins in a single season",
                "Top 12 · with the share of that season's races, since the calendar has grown from 7 to 24",
                accent=PALETTE["amber"],
            ):
                st.plotly_chart(fig_season_wins, width="stretch", config=PLOTLY_CONFIG)

            st.write("")
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
            fig_streaks = base_figure("", "", "Races in a row")
            fig_streaks.update_layout(showlegend=False)
            fig_streaks.add_trace(
                go.Bar(
                    x=top_streaks["races"].to_numpy(), y=streak_labels, orientation="h",
                    marker=dict(
                        color=top_streaks["races"].to_numpy(), colorscale=bar_scale(PALETTE["pink"]),
                        line=dict(color="rgba(255,255,255,0.12)", width=1),
                    ),
                    text=top_streaks["races"].to_numpy(), textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x} in a row<extra></extra>",
                )
            )
            size_horizontal_bars(fig_streaks, top_streaks["races"].to_numpy())
            style_bars(fig_streaks)
            with chart_panel(
                "Longest winning streaks",
                "Consecutive races won, counted across season boundaries — a streak doesn't reset in January",
                accent=PALETTE["pink"],
            ):
                st.plotly_chart(fig_streaks, width="stretch", config=PLOTLY_CONFIG)

            st.write("")

            with chart_panel("Most poles by constructor", "All-time top 12", accent=PALETTE["violet"]):
                poles_by_team = poles["constructorName"].value_counts().head(12).sort_values()
                fig_team_poles = base_figure("", "", "Poles")
                fig_team_poles.update_layout(showlegend=False)
                fig_team_poles.add_trace(
                    go.Bar(
                        x=poles_by_team.to_numpy(), y=poles_by_team.index, orientation="h",
                        marker=dict(
                            color=poles_by_team.to_numpy(), colorscale=bar_scale(PALETTE["violet"]),
                            line=dict(color="rgba(255,255,255,0.12)", width=1),
                        ),
                        text=poles_by_team.to_numpy(), textposition="outside", cliponaxis=False,
                        hovertemplate="%{y}: %{x} poles<extra></extra>",
                    )
                )
                size_horizontal_bars(fig_team_poles, poles_by_team.to_numpy())
                style_bars(fig_team_poles)
                st.plotly_chart(fig_team_poles, width="stretch", config=PLOTLY_CONFIG)
