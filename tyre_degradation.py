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

# Per-compound linear degradation trend, fit across all stints of that
# compound, to show the overall slope through the per-lap scatter.
for compound, compound_laps in laps.groupby("Compound"):
    if len(compound_laps) < 10:
        continue
    x = compound_laps["TyreLife"].to_numpy()
    y = compound_laps["LapTime"].dt.total_seconds().to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 2)
    color = compound_colors.get(compound, "black")
    ax.plot(
        x_fit,
        slope * x_fit + intercept,
        color=color,
        linewidth=3,
        solid_capstyle="round",
        label=f"{compound} ({slope:+.2f} s/lap)",
    )

ax.legend(title="Compound (degradation trend)")
ax.set_xlabel("Tyre life (laps)")
ax.set_ylabel("Lap time (s)")
ax.set_title(f"{YEAR} {EVENT} — lap time vs. tyre age, all drivers")
fig.tight_layout()
fig.savefig("tyre_degradation.png", dpi=150)
print("Saved tyre_degradation.png")
