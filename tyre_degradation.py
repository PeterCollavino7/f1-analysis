"""
First pipeline test: tyre degradation in the most recently completed race.

Loads race lap data via FastF1, filters to green-flag racing laps (no pit
in/out laps, no laps under yellow/VSC/SC), and plots lap time vs. tyre age
per driver, split by compound, to see the degradation slope.
"""
import fastf1
import numpy as np
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache("cache")

YEAR = 2026
EVENT = "Spanish Grand Prix"

session = fastf1.get_session(YEAR, EVENT, "R")
session.load()

laps = session.laps.pick_quicklaps()
laps = laps[laps["TrackStatus"] == "1"]  # green flag only

compound_colors = {
    "SOFT": "tab:red",
    "MEDIUM": "tab:orange",
    "HARD": "tab:gray",
    "INTERMEDIATE": "tab:green",
    "WET": "tab:blue",
}

fig, ax = plt.subplots(figsize=(10, 6))

# One line per driver+stint (not per driver+compound): a driver can run the
# same compound in two separate stints, and joining those with a single line
# draws a fake diagonal jumping from high tyre age back down to zero.
for (driver, stint), stint_laps in laps.groupby(["Driver", "Stint"]):
    if len(stint_laps) < 3:
        continue
    compound = stint_laps["Compound"].iloc[0]
    color = compound_colors.get(compound, "black")
    stint_laps = stint_laps.sort_values("TyreLife")
    ax.plot(
        stint_laps["TyreLife"],
        stint_laps["LapTime"].dt.total_seconds(),
        marker="o",
        markersize=3,
        alpha=0.25,
        color=color,
        linewidth=1,
    )

# Per-compound degradation trend, controlled for fuel load. Raw lap time vs.
# tyre age (fit in the first version of this script) conflates two effects:
# the tyre wearing in, and the car getting lighter as the race goes on. Since
# FastF1 doesn't expose fuel mass directly, LapNumber is used as its proxy
# (fuel burns off roughly linearly with laps completed) and a multiple linear
# regression LapTime ~ TyreLife + LapNumber separates the two: the TyreLife
# coefficient is then the degradation slope with fuel effect held constant,
# instead of the fuel effect leaking into it.
print(f"{'Compound':<8} {'tyre s/lap':>12} {'fuel+track s/lap':>18}")
for compound, compound_laps in laps.groupby("Compound"):
    if len(compound_laps) < 10:
        continue
    tyre_life = compound_laps["TyreLife"].to_numpy(dtype=float)
    lap_number = compound_laps["LapNumber"].to_numpy(dtype=float)
    lap_time = compound_laps["LapTime"].dt.total_seconds().to_numpy()

    design = np.column_stack([tyre_life, lap_number, np.ones_like(tyre_life)])
    (tyre_coef, lap_coef, intercept), *_ = np.linalg.lstsq(design, lap_time, rcond=None)
    print(f"{compound:<8} {tyre_coef:>+12.3f} {lap_coef:>+18.3f}")

    # Trend line at the compound's mean lap number, so it isolates the tyre
    # effect instead of also drifting with race progression.
    mean_lap_number = lap_number.mean()
    x_fit = np.linspace(tyre_life.min(), tyre_life.max(), 2)
    y_fit = tyre_coef * x_fit + lap_coef * mean_lap_number + intercept
    color = compound_colors.get(compound, "black")
    ax.plot(
        x_fit,
        y_fit,
        color=color,
        linewidth=3,
        solid_capstyle="round",
        label=f"{compound} ({tyre_coef:+.2f} s/lap, fuel-corrected)",
    )

ax.legend(title="Compound (degradation trend)")
ax.set_xlabel("Tyre life (laps)")
ax.set_ylabel("Lap time (s)")
ax.set_title(f"{YEAR} {EVENT} — lap time vs. tyre age, all drivers")
fig.tight_layout()
fig.savefig("tyre_degradation.png", dpi=150)
print("Saved tyre_degradation.png")
