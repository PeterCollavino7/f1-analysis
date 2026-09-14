# F1 Analysis

Personal F1 data analysis project, built on [FastF1](https://docs.fastf1.dev/) (official timing
and telemetry data).

## Setup

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
mkdir cache
```

## Scripts

- `tyre_degradation.py` — lap time vs. tyre age per compound for a given race. Fits
  `LapTime ~ TyreLife + LapNumber` per compound (multiple linear regression) instead of a plain
  `LapTime ~ TyreLife` fit, since FastF1 doesn't expose fuel mass directly and `LapNumber` is
  used as its proxy (fuel burns off roughly linearly with laps completed) — without this, a long
  stint (typically Hard) shows lap times trending down for a reason that has nothing to do with
  the tyre. The `TyreLife` coefficient is the fuel-corrected degradation slope; on the first race
  tested it correctly orders Hard (~flat) < Medium < Soft (fastest degrading), matching real
  tyre physics. Also saves `tyre_degradation_by_driver.png`, the same idea broken down per
  driver per compound — who manages each tyre best. That breakdown fits only `TyreLife` per
  driver (not `TyreLife + LapNumber` again): a driver who ran a compound in a single stint has
  `TyreLife` and `LapNumber` climbing together 1-for-1, so the two-variable fit is collinear and
  the coefficients blow up (this produced nonsense ±20-30 s/lap bars at first). The fix is to
  take the fuel slope from the pooled, well-conditioned fit and subtract it out of each driver's
  lap times before fitting their tyre slope alone.
- `dashboard.py` — Streamlit app for head-to-head telemetry. Pick a year, a Grand Prix, and one
  of its actual sessions (the dropdown is built from that weekend's real `Session1..5` names in
  the FastF1 schedule, so Sprint Qualifying/Sprint show up only on sprint weekends, and a normal
  weekend shows Practice 1-3/Qualifying/Race), then up to 3 drivers via a multiselect. Charts:
  speed, throttle, brake, and a delta-time chart (gap to the fastest of the selected laps,
  computed by interpolating the other laps' elapsed time onto the reference lap's distance grid
  — done directly with `numpy.interp` rather than `fastf1.utils.delta_time`, which the library's
  own docs flag as deprecated and not very accurate). All four charts share one fixed height and
  sit in a 2x2 grid. Data is fetched from the F1 API on first request for a given
  year/event/session and cached locally after that (`cache/`) — there's no bulk pre-download of
  every past race, new races just show up in the dropdown once they've happened and get fetched
  the first time someone picks them. Run with `venv\Scripts\streamlit run dashboard.py`.
