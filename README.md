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
  tyre physics.
