"""
Publish FastF1 session caches to the private f1-data repo, for the hosted
dashboard to read.

The official timing feed (livetiming.formula1.com) answers 403 to datacenter
IPs -- Streamlit Community Cloud and GitHub's own runners both get refused --
so the hosted app can't download a session itself. A home connection is
served normally, so this script runs here instead: it loads every finished
session through FastF1 (which fills the local cache/ folder), zips that
session's cache folder and uploads it as release assets of
PeterCollavino7/f1-data, one release per season. dashboard.py downloads and
unpacks them into its own cache/ folder, and FastF1 then opens the session
without touching the feed at all.

Each session becomes two assets, so the season views -- which read every
race of the year but never its telemetry -- don't have to download ~20 MB of
car and position data per race:
    <event>__<session>.base.zip        timing, laps, results, messages, weather
    <event>__<session>.telemetry.zip   car_data + position_data

Idempotent: anything already uploaded is skipped, so it's safe to run every
15 minutes (the scheduled task does) and to rerun after a failure. A session
is published as soon as it is complete, not on a fixed delay: the feed's
SessionStatus stream ends with "Ends" a few minutes after the flag, and a
race or sprint also waits for its official classification on Jolpica, which
the hosted app reads itself and caches. Until then the session's local cache
is thrown away and the next run tries again -- an asset uploaded
half-complete would never be replaced. Six hours after the start a session
is published with whatever there is, as it always used to be.

Run with:  venv\\Scripts\\python publish_data.py [YEAR ...]   (default: this year)
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import fastf1

DATA_REPO = "PeterCollavino7/f1-data"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
LOG_FILE = os.path.join(HERE, "publish_data.log")
TELEMETRY_FILES = {"car_data.ff1pkl", "position_data.ff1pkl"}
# No point loading a session before it can possibly be over: its scheduled
# length from the start. Loading one still running is harmless (it fails the
# readiness check and is retried), just wasted downloads.
MIN_LENGTH = {
    "Race": datetime.timedelta(minutes=100),
    "Sprint": datetime.timedelta(minutes=35),
    "Sprint Qualifying": datetime.timedelta(minutes=45),
    "Sprint Shootout": datetime.timedelta(minutes=45),
}
DEFAULT_LENGTH = datetime.timedelta(minutes=60)
# Past this, publish whatever the feed has -- a session that never sends
# "Ends", or a classification Jolpica hasn't posted, must not hold it forever.
SETTLE_TIME = datetime.timedelta(hours=6)
RACE_LIKE = {"Race", "Sprint"}
# gh isn't on PATH for the scheduled task (or Git Bash); this is where it lives.
GH = os.environ.get("GH_EXE", r"C:\Users\peter\tools\bin\gh.exe")
if not os.path.exists(GH):
    GH = "gh"


def log(message):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    if sys.stdout is not None:  # None under pythonw.exe, which the scheduled task uses
        print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gh(*args):
    # CREATE_NO_WINDOW: run from pythonw.exe by the scheduled task, each gh
    # call would otherwise flash a console window on screen.
    return subprocess.run(
        [GH, *args], capture_output=True, text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def published_assets(year):
    """Asset names already in the season's release, creating the release
    the first time a season is published."""
    result = gh("release", "view", str(year), "--repo", DATA_REPO, "--json", "assets")
    if result.returncode != 0:
        created = gh(
            "release", "create", str(year), "--repo", DATA_REPO, "--title", str(year),
            "--notes", f"FastF1 session caches for the {year} season, written by publish_data.py.",
        )
        if created.returncode != 0:
            raise RuntimeError(f"couldn't create release {year}: {created.stderr.strip()}")
        return set()
    return {asset["name"] for asset in json.loads(result.stdout)["assets"]}


def session_key(session):
    # api_path is "/static/<year>/<event folder>/<session folder>/", which is
    # also where FastF1 keeps the session under its cache directory.
    _, _, event_dir, session_dir = session.api_path.strip("/").split("/")
    return f"{event_dir}__{session_dir}"


def session_cache_folder(session):
    return os.path.join(CACHE_DIR, *session.api_path.strip("/").split("/")[1:])


def zip_files(folder, names, target):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in names:
            zf.write(os.path.join(folder, name), name)


def finished_sessions(year):
    """(round, session name, settled) for every session that can be over by
    now; settled once SETTLE_TIME has passed since its start."""
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name, start = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
            if isinstance(name, str) and name and start is not None and not isinstance(start, float):
                if start + MIN_LENGTH.get(name, DEFAULT_LENGTH) <= now:
                    yield int(event["RoundNumber"]), name, start + SETTLE_TIME <= now


def not_ready(session):
    """Why the session isn't complete yet, or None when it is."""
    status = session.session_status
    if status is None or not status["Status"].isin(["Ends", "Finalised"]).any():
        return "timing feed still open"
    if session.name in RACE_LIKE:
        classified = session.results["ClassifiedPosition"].fillna("").astype(str).str.strip()
        if (classified == "").all():
            return "no official classification on Jolpica yet"
    return None


def forget(session, round_number):
    """Drop everything cached for a session that wasn't ready -- its pickles
    and the HTTP responses behind them, timing feed and Jolpica alike --
    so the next run downloads it afresh instead of reusing what was cached
    half-way (FastF1 keeps HTTP responses for 12 hours)."""
    shutil.rmtree(session_cache_folder(session), ignore_errors=True)
    http = fastf1.req.Cache._requests_session_cached
    if http is None:
        return
    jolpica = f"/{session.event.year}/{round_number}/"
    stale = [u for u in http.cache.urls() if session.api_path in u or ("jolpi" in u and jolpica in u)]
    http.cache.delete(urls=stale)


def publish_year(year):
    existing = published_assets(year)
    published = skipped = failed = waiting = 0
    for round_number, session_name, settled in finished_sessions(year):
        session = fastf1.get_session(year, round_number, session_name)
        key = session_key(session)
        if f"{key}.base.zip" in existing:
            skipped += 1
            continue
        try:
            session.load()
            session.laps  # raises if the timing data didn't come through
        except Exception as exc:
            if settled:
                log(f"  {key}: not loaded, will retry next run ({exc!r:.200})")
                failed += 1
            else:
                waiting += 1  # most likely not on the feed yet; quiet retry
            forget(session, round_number)
            continue
        reason = not_ready(session)
        if reason and not settled:
            log(f"  {key}: waiting ({reason})")
            waiting += 1
            forget(session, round_number)
            continue
        folder = session_cache_folder(session)
        files = os.listdir(folder)
        base = [f for f in files if f.endswith(".ff1pkl") and f not in TELEMETRY_FILES]
        telemetry = [f for f in files if f in TELEMETRY_FILES]
        tmp = tempfile.mkdtemp()
        try:
            assets = [os.path.join(tmp, f"{key}.base.zip")]
            zip_files(folder, base, assets[0])
            if telemetry:
                # Telemetry first in the upload, base last: the dashboard
                # (and this script's skip check) treat the base asset as
                # "this session is published", so it must never exist
                # without its telemetry beside it.
                assets.insert(0, os.path.join(tmp, f"{key}.telemetry.zip"))
                zip_files(folder, telemetry, assets[0])
            for asset in assets:
                result = gh("release", "upload", str(year), asset, "--repo", DATA_REPO, "--clobber")
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip())
            size = sum(os.path.getsize(a) for a in assets) / 1e6
            log(f"  {key}: published ({size:.1f} MB){f', settled without: {reason}' if reason else ''}")
            published += 1
        except Exception as exc:
            log(f"  {key}: upload failed, will retry next run ({exc!r:.200})")
            failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        del session
    # Run every 15 minutes, a line per run would bury the log in "nothing to
    # do"; only runs that did or tried something are written down.
    if published or failed or waiting:
        log(f"{year}: {published} published, {skipped} already there, {waiting} waiting, {failed} failed")
    return failed


def main():
    years = [int(y) for y in sys.argv[1:]] or [datetime.date.today().year]
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("WARNING")
    if sys.stdout is not None:  # the log only records runs that did something
        print(f"publishing {', '.join(map(str, years))}", flush=True)
    failed = 0
    for year in years:
        try:
            failed += publish_year(year)
        except Exception as exc:
            log(f"{year}: aborted ({exc!r:.300})")
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
