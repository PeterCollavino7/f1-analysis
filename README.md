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

- `tyre_degradation.py` — lap time vs. tyre age per compound for a given race, with a linear
  degradation trend line per compound. First pipeline test; note the trend is not yet corrected
  for fuel load, so long stints (typically Hard) show a lap-time slope dominated by the car
  getting lighter, not by the tyre itself degrading.
