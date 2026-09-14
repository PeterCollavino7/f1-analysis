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
