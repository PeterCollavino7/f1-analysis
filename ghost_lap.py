"""The ghost lap: two drivers' fastest laps raced against each other on the
circuit map, with each car's speed, gear and pedals and the live gap between
them -- the Head-to-head tab's lap trace, played back instead of plotted.

It lives in its own file because it is mostly a page of JavaScript: a canvas
animation can't be drawn by Streamlit, so dashboard.py builds the data and
this renders it as a self-contained HTML document for
streamlit.components.v1.html (an iframe -- its script can't reach, or break,
the rest of the app). No libraries: the whole thing is a few hundred numbers
per channel and a requestAnimationFrame loop.
"""
import json

import numpy as np

SAMPLES = 700  # points per channel -- smooth at any playback speed, ~60 KB of JSON


def ghost_lap_payload(distance, track_x, track_y, drivers, sector_distances=(), speed_unit="km/h", speed_factor=1.0):
    """Everything the page needs, downsampled onto one shared distance grid.

    distance: metres, the grid every channel is already resampled onto.
    track_x/track_y: the reference lap's position on that grid.
    drivers: [{"code", "color", "lap_time", "elapsed", "speed", "throttle",
    "brake", "gear"}], channels on the same grid.
    sector_distances: where sectors 1 and 2 end, in metres.
    """
    picks = np.linspace(0, len(distance) - 1, min(SAMPLES, len(distance))).astype(int)
    x = np.asarray(track_x, dtype=float)[picks]
    # Screen y grows downward; the track would come out mirrored.
    y = -np.asarray(track_y, dtype=float)[picks]
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-6)
    x = (x - x.min()) / span
    y = (y - y.min()) / span

    def channel(values, digits=1):
        return [round(float(v), digits) for v in np.asarray(values, dtype=float)[picks]]

    total = float(distance[-1]) or 1.0
    payload = {
        "x": [round(float(v), 4) for v in x],
        "y": [round(float(v), 4) for v in y],
        "sectors": [round(float(d) / total, 4) for d in sector_distances],
        "unit": speed_unit,
        "drivers": [],
    }
    for d in drivers:
        # Elapsed time must only ever grow along the lap, or the playback's
        # time-to-position lookup (a binary search) could jump backwards.
        elapsed = np.maximum.accumulate(np.asarray(d["elapsed"], dtype=float))
        payload["drivers"].append({
            "code": d["code"],
            "color": d["color"],
            "lap": round(float(d["lap_time"]), 3),
            "t": [round(float(v), 3) for v in elapsed[picks]],
            "speed": channel(np.asarray(d["speed"], dtype=float) * speed_factor, 0),
            "throttle": channel(d["throttle"], 0),
            "brake": channel(d["brake"], 0),
            "gear": channel(d["gear"], 0),
        })
    return payload


def ghost_lap_html(payload, height):
    return (
        PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__HEIGHT__", str(int(height)))
    )


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #eef1f5; --dim: rgba(226,232,240,0.62); --faint: rgba(226,232,240,0.40);
    --line: rgba(255,255,255,0.08); --red: #e10600;
    --display: 'Titillium Web', 'Segoe UI', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: transparent; color: var(--ink); font-family: var(--display); overflow: hidden; }
  .wrap { display: flex; flex-direction: column; height: __HEIGHT__px; padding: 4px 6px 6px; gap: 10px; }
  .hud { display: grid; grid-template-columns: 1fr auto 1fr; gap: 10px; align-items: stretch; }
  .car {
    position: relative; overflow: hidden; border-radius: 12px; padding: 9px 13px;
    background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
    border: 1px solid var(--line); display: grid;
    grid-template-columns: auto 1fr auto; grid-template-rows: auto auto; column-gap: 12px; align-items: center;
  }
  .car::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--c); }
  .car.right { grid-template-columns: auto 1fr auto; }
  .code { font-weight: 900; font-size: 1.15rem; letter-spacing: 0.05em; color: var(--c); grid-row: span 2; min-width: 3.2rem; }
  .speed { font: 700 1.55rem/1 var(--mono); letter-spacing: -0.02em; text-align: right; }
  .speed small { font: 600 0.62rem var(--display); color: var(--faint); margin-left: 3px; letter-spacing: 0.08em; }
  .gear {
    grid-row: span 2; width: 2.3rem; height: 2.3rem; border-radius: 9px; display: grid; place-items: center;
    border: 1.5px solid var(--line); font: 700 1.1rem var(--mono); color: var(--ink);
  }
  .pedals { display: flex; gap: 6px; align-items: center; justify-content: flex-end; }
  .pedals b { font: 600 0.58rem var(--display); letter-spacing: 0.12em; color: var(--faint); }
  .bar { width: 64px; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.07); overflow: hidden; }
  .bar i { display: block; height: 100%; width: 0; border-radius: 3px; }
  .thr i { background: #2ee86e; }
  .brk i { background: #ff4d4d; }
  .gap {
    display: flex; flex-direction: column; justify-content: center; align-items: center; min-width: 9.5rem;
    border-radius: 12px; border: 1px solid var(--line); background: rgba(0,0,0,0.25); padding: 6px 12px;
  }
  .gap .who { font: 700 0.62rem var(--display); letter-spacing: 0.14em; text-transform: uppercase; color: var(--faint); }
  .gap .val { font: 700 1.35rem/1.2 var(--mono); }
  .gap .clock { font: 500 0.7rem var(--mono); color: var(--faint); }
  .stage { position: relative; flex: 1; min-height: 0; }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  .controls { display: flex; align-items: center; gap: 10px; }
  button {
    font: 700 0.78rem var(--display); letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink);
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.14); border-radius: 999px;
    padding: 6px 14px; cursor: pointer; transition: background 0.15s ease, border-color 0.15s ease;
  }
  button:hover { background: rgba(255,255,255,0.1); }
  button.play { background: var(--red); border-color: var(--red); min-width: 5.6rem; }
  button.play:hover { background: #ff2a1f; }
  .rates { display: flex; gap: 4px; }
  .rates button { padding: 5px 9px; font-family: var(--mono); letter-spacing: 0; text-transform: none; }
  .rates button.on { border-color: var(--ink); background: rgba(255,255,255,0.14); }
  .scrub { position: relative; flex: 1; height: 28px; display: flex; align-items: center; }
  input[type=range] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 4px; margin: 0; border-radius: 2px; cursor: pointer;
    background: linear-gradient(90deg, var(--red) var(--p, 0%), rgba(255,255,255,0.12) var(--p, 0%));
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: #fff; border: 3px solid var(--red); box-shadow: 0 0 10px rgba(225,6,0,0.6);
  }
  input[type=range]::-moz-range-thumb {
    width: 10px; height: 10px; border-radius: 50%; background: #fff; border: 3px solid var(--red);
  }
  .ticks { position: absolute; left: 0; right: 0; top: 0; bottom: 0; pointer-events: none; }
  .ticks span { position: absolute; top: -2px; transform: translateX(-50%); font: 600 0.55rem var(--display); color: var(--faint); letter-spacing: 0.1em; }
  @media (max-width: 640px) {
    .hud { grid-template-columns: 1fr 1fr; }
    .gap { grid-column: span 2; order: -1; flex-direction: row; gap: 10px; min-width: 0; }
    .gear, .pedals { display: none; }
    .car { grid-template-columns: auto 1fr; padding: 7px 10px; }
    .speed { font-size: 1.2rem; }
    .rates button:nth-child(1) { display: none; }
  }
</style></head>
<body>
<div class="wrap">
  <div class="hud">
    <div class="car" id="car0"><div class="code"></div><div class="speed"></div><div class="gear"></div>
      <div class="pedals"><b>THR</b><div class="bar thr"><i></i></div><b>BRK</b><div class="bar brk"><i></i></div></div></div>
    <div class="gap"><div class="who">Gap</div><div class="val">—</div><div class="clock">0.000</div></div>
    <div class="car right" id="car1"><div class="code"></div><div class="speed"></div><div class="gear"></div>
      <div class="pedals"><b>THR</b><div class="bar thr"><i></i></div><b>BRK</b><div class="bar brk"><i></i></div></div></div>
  </div>
  <div class="stage"><canvas></canvas></div>
  <div class="controls">
    <button class="play">Play</button>
    <div class="scrub"><input type="range" min="0" max="1000" value="0" step="1"><div class="ticks"></div></div>
    <div class="rates"><button data-r="0.5">0.5×</button><button data-r="1">1×</button><button data-r="2" class="on">2×</button><button data-r="4">4×</button></div>
  </div>
</div>
<script>
const D = __DATA__;
const N = D.x.length;
const cars = D.drivers;
const end = Math.max(...cars.map(c => c.t[N - 1]));
const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let t = 0, rate = 2, playing = false, last = null, started = false;

const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
const stage = document.querySelector(".stage");
const range = document.querySelector("input[type=range]");
const playBtn = document.querySelector("button.play");

// Where a car is at time s: the fractional grid index, by binary search on
// its elapsed-time channel.
function indexAt(car, s) {
  const tt = car.t;
  if (s <= tt[0]) return 0;
  if (s >= tt[N - 1]) return N - 1;
  let lo = 0, hi = N - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (tt[mid] <= s) lo = mid; else hi = mid; }
  return lo + (s - tt[lo]) / Math.max(tt[hi] - tt[lo], 1e-9);
}
const lerp = (arr, f) => { const i = Math.floor(f), j = Math.min(i + 1, N - 1); return arr[i] + (arr[j] - arr[i]) * (f - i); };
// The time a car reached grid point f -- for the gap at the leader's spot.
const timeAt = (car, f) => lerp(car.t, f);

let box = { s: 1, ox: 0, oy: 0 };
function layout() {
  const r = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, r.width * dpr); canvas.height = Math.max(1, r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const maxX = Math.max(...D.x), maxY = Math.max(...D.y), pad = 22;
  const s = Math.min((r.width - 2 * pad) / maxX, (r.height - 2 * pad) / maxY);
  box = { s, ox: (r.width - maxX * s) / 2, oy: (r.height - maxY * s) / 2 };
  draw();
}
const P = f => [box.ox + lerp(D.x, f) * box.s, box.oy + lerp(D.y, f) * box.s];

function trackPath() {
  ctx.beginPath();
  for (let i = 0; i < N; i++) { const x = box.ox + D.x[i] * box.s, y = box.oy + D.y[i] * box.s; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
}
function mark(f, label) {
  const [x, y] = P(f), [x2, y2] = P(Math.min(f + 1, N - 1));
  const a = Math.atan2(y2 - y, x2 - x) + Math.PI / 2;
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x - Math.cos(a) * 11, y - Math.sin(a) * 11); ctx.lineTo(x + Math.cos(a) * 11, y + Math.sin(a) * 11); ctx.stroke();
  ctx.fillStyle = "rgba(226,232,240,0.5)"; ctx.font = "600 10px 'Titillium Web', sans-serif";
  ctx.fillText(label, x + Math.cos(a) * 15, y + Math.sin(a) * 15 + 3);
}

function draw() {
  const r = stage.getBoundingClientRect();
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.lineJoin = ctx.lineCap = "round";
  trackPath(); ctx.strokeStyle = "#232838"; ctx.lineWidth = 15; ctx.stroke();
  trackPath(); ctx.strokeStyle = "rgba(255,255,255,0.07)"; ctx.lineWidth = 1.5; ctx.stroke();
  mark(0, "START");
  // Sector boundaries, labelled with the sector that starts there.
  D.sectors.forEach((s, i) => mark(s * (N - 1), "S" + (i + 2)));

  const pos = cars.map(c => indexAt(c, t));
  // The one further round the lap is drawn last, on top.
  const order = pos[0] >= pos[1] ? [1, 0] : [0, 1];
  order.forEach(k => {
    const c = cars[k], f = pos[k];
    // A trail over the last ~4% of the lap, fading out behind the car.
    const tail = Math.max(0, f - N * 0.04);
    for (let g = tail; g < f; g += 1) {
      const [x1, y1] = P(g), [x2, y2] = P(Math.min(g + 1, f));
      ctx.strokeStyle = c.color; ctx.globalAlpha = 0.85 * (g - tail) / Math.max(f - tail, 1);
      ctx.lineWidth = 5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    // Side by side rather than on top of each other: each car sits a few
    // pixels off the racing line, one either side.
    const [x, y] = P(f), [xa, ya] = P(Math.min(f + 1, N - 1));
    const a = Math.atan2(ya - y, xa - x) + Math.PI / 2, off = k ? -4 : 4;
    const cx = x + Math.cos(a) * off, cy = y + Math.sin(a) * off;
    ctx.shadowColor = c.color; ctx.shadowBlur = 16;
    ctx.fillStyle = c.color; ctx.beginPath(); ctx.arc(cx, cy, 7.5, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0; ctx.lineWidth = 2; ctx.strokeStyle = "#0b0e15"; ctx.stroke();
    // The name on a small plate, off the track on the car's own side of it,
    // so the two labels never sit on each other or on the tarmac.
    const side = k ? -1 : 1, lx = cx + Math.cos(a) * 24 * side, ly = cy + Math.sin(a) * 24 * side;
    ctx.font = "900 11px 'Titillium Web', sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const w = ctx.measureText(c.code).width + 10;
    ctx.fillStyle = "rgba(11,14,21,0.85)"; ctx.strokeStyle = c.color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(lx - w / 2, ly - 8, w, 16, 4); ctx.fill(); ctx.stroke();
    ctx.fillStyle = c.color; ctx.fillText(c.code, lx, ly + 0.5);
    ctx.textAlign = "start"; ctx.textBaseline = "alphabetic";
  });
  hud(pos);
}

function hud(pos) {
  cars.forEach((c, k) => {
    const el = document.getElementById("car" + k), f = pos[k];
    el.style.setProperty("--c", c.color);
    el.querySelector(".code").textContent = c.code;
    el.querySelector(".speed").innerHTML = Math.round(lerp(c.speed, f)) + "<small>" + D.unit.toUpperCase() + "</small>";
    el.querySelector(".gear").textContent = Math.round(c.gear[Math.round(f)]) || "N";
    el.querySelector(".thr i").style.width = Math.max(0, Math.min(100, lerp(c.throttle, f))) + "%";
    el.querySelector(".brk i").style.width = (c.brake[Math.round(f)] > 0 ? 100 : 0) + "%";
  });
  const lead = pos[0] >= pos[1] ? 0 : 1, back = 1 - lead;
  const finished = cars.map((c, k) => t >= c.t[N - 1]);
  let gap;
  if (finished[0] && finished[1]) {
    const w = cars[0].lap <= cars[1].lap ? 0 : 1;
    gap = [w, Math.abs(cars[0].lap - cars[1].lap), "faster by"];
  } else {
    // Gap = how long the car behind takes to reach where the leader is now.
    const f = pos[lead];
    gap = [lead, Math.max(0, timeAt(cars[back], f) - timeAt(cars[lead], f)), "ahead by"];
  }
  const g = document.querySelector(".gap");
  g.querySelector(".who").innerHTML = "<span style='color:" + cars[gap[0]].color + "'>" + cars[gap[0]].code + "</span> " + gap[2];
  g.querySelector(".val").textContent = gap[1].toFixed(3) + " s";
  const shown = Math.min(t, end), m = Math.floor(shown / 60), s = shown - m * 60;
  g.querySelector(".clock").textContent = (m ? m + ":" + s.toFixed(3).padStart(6, "0") : s.toFixed(3));
  range.value = Math.round(1000 * t / end);
  range.style.setProperty("--p", (100 * t / end).toFixed(2) + "%");
}

function frame(now) {
  if (!playing) { last = null; return; }
  if (last !== null) t += (now - last) / 1000 * rate;
  last = now;
  if (t >= end) { t = end; setPlaying(false); playBtn.textContent = "Replay"; }
  draw();
  if (playing) requestAnimationFrame(frame);
}
function setPlaying(on) {
  playing = on; playBtn.textContent = on ? "Pause" : "Play";
  if (on) { if (t >= end) t = 0; requestAnimationFrame(frame); }
}
playBtn.onclick = () => { started = true; setPlaying(!playing); };
range.oninput = () => { t = end * range.value / 1000; draw(); };
document.querySelectorAll(".rates button").forEach(b => b.onclick = () => {
  rate = parseFloat(b.dataset.r);
  document.querySelectorAll(".rates button").forEach(o => o.classList.toggle("on", o === b));
});
const ticks = document.querySelector(".ticks");
D.sectors.forEach((s, i) => {
  const at = cars[0].t[Math.round(s * (N - 1))] / end;
  ticks.insertAdjacentHTML("beforeend", "<span style='left:" + (at * 100).toFixed(1) + "%'>S" + (i + 2) + "</span>");
});

new ResizeObserver(layout).observe(stage);
layout();
// Starts on its own the first time it scrolls into view -- unless the
// reader has asked for less motion, in which case it waits for Play.
if (!reduced) {
  new IntersectionObserver((entries, obs) => {
    if (entries.some(e => e.isIntersecting) && !started) { started = true; setPlaying(true); obs.disconnect(); }
  }, { threshold: 0.5 }).observe(stage);
}
</script>
</body></html>
"""
