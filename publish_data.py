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

Idempotent: anything already uploaded is skipped, so it's safe to run daily
(the scheduled task does) and to rerun after a failure. A session is only
published 6 hours after it ended -- the feed fills in over the first hour or
so, and an asset uploaded half-complete would never be replaced.

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
SETTLE_TIME = datetime.timedelta(hours=6)
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
    """(round, session name) for every session that ended long enough ago."""
    cutoff = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - SETTLE_TIME
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name, start = event.get(f"Session{i}"), event.get(f"Session{i}DateUtc")
            if isinstance(name, str) and name and start is not None and not isinstance(start, float):
                if start <= cutoff:
                    yield int(event["RoundNumber"]), name


def publish_year(year):
    existing = published_assets(year)
    published = skipped = failed = 0
    for round_number, session_name in finished_sessions(year):
        session = fastf1.get_session(year, round_number, session_name)
        key = session_key(session)
        if f"{key}.base.zip" in existing:
            skipped += 1
            continue
        try:
            session.load()
            session.laps  # raises if the timing data didn't come through
        except Exception as exc:
            log(f"  {key}: not loaded, will retry next run ({exc!r:.200})")
            failed += 1
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
            log(f"  {key}: published ({size:.1f} MB)")
            published += 1
        except Exception as exc:
            log(f"  {key}: upload failed, will retry next run ({exc!r:.200})")
            failed += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        del session
    log(f"{year}: {published} published, {skipped} already there, {failed} failed")
    return failed


def main():
    years = [int(y) for y in sys.argv[1:]] or [datetime.date.today().year]
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("WARNING")
    log(f"publishing {', '.join(map(str, years))}")
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
