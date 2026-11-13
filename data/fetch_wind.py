"""Fetch fire-season hourly 10 m wind for the Khao Yai map window: data/wind_fire_season.npz.

One Open-Meteo historical-archive request per year (1 February to 30 April, 2012 to 2024), for the
centre of the 40 x 40 grid of 500 m cells (data/khao_yai_500m.npz). The archive is reanalysis on a
coarse grid, so this is one wind series for the whole map, not a per-cell field.

Raw JSON responses are cached in ../data_cache/open_meteo/ (outside the repo) and reused on re-run.
Only the npz is committed. Standard library only (urllib, as in fetch_khao_yai.py).
"""

import datetime
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("QP_DATA_CACHE", HERE.parent.parent / "data_cache"))
OPEN_METEO_CACHE = CACHE / "open_meteo"
OUT_NPZ = HERE / "wind_fire_season.npz"
UA = "q-pithak-data-fetch"          # generic client name, no personal details
ACCESS_DATE = "13 November 2026"
YEARS = list(range(2012, 2025))

# Corners of the 40 x 40 grid (data/khao_yai_meta.json, region_bounds_deg). Centre = midpoint.
GRID_NORTH, GRID_SOUTH = 14.479641, 14.298767
GRID_WEST, GRID_EAST = 101.108157, 101.293586
CENTRE_LAT = round((GRID_NORTH + GRID_SOUTH) / 2, 6)
CENTRE_LON = round((GRID_WEST + GRID_EAST) / 2, 6)

API = "https://archive-api.open-meteo.com/v1/archive"
EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def request_url(year):
    params = {
        "latitude": CENTRE_LAT,
        "longitude": CENTRE_LON,
        "start_date": f"{year}-02-01",
        "end_date": f"{year}-04-30",
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    return API + "?" + urllib.parse.urlencode(params)


def fetch_year(year, state):
    """Return the parsed JSON for one year, from cache if present, else over HTTP."""
    OPEN_METEO_CACHE.mkdir(parents=True, exist_ok=True)
    raw = OPEN_METEO_CACHE / f"archive_{year}_{CENTRE_LAT}_{CENTRE_LON}.json"
    if raw.exists() and raw.stat().st_size > 1000:
        print(f"  {year} cached   {raw.name}")
        return json.loads(raw.read_text())
    if state["fetched"] > 0:
        time.sleep(1.0)                 # pause between uncached requests
    url = request_url(year)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
    except (urllib.error.URLError, OSError) as ex:
        die(f"{year}: request failed ({ex}); url: {url}")
    state["fetched"] += 1
    data = json.loads(body)
    if data.get("error"):
        die(f"{year}: API error: {data.get('reason', data)}")
    tmp = raw.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(raw)
    print(f"  {year} fetched  {raw.name} ({len(body) / 1e3:.0f} kB)")
    return data


def to_hours(iso_times):
    """'YYYY-MM-DDTHH:MM' in UTC -> int64 hours since 1970-01-01T00:00Z."""
    out = np.empty(len(iso_times), dtype=np.int64)
    for i, s in enumerate(iso_times):
        t = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=datetime.timezone.utc)
        secs = (t - EPOCH).total_seconds()
        out[i] = int(round(secs / 3600.0))
    return out


def main():
    print(f"Centre of the 40 x 40 grid: lat {CENTRE_LAT}, lon {CENTRE_LON}")
    print(f"cache: {OPEN_METEO_CACHE}")
    state = {"fetched": 0}
    times, speeds, dirs = [], [], []
    used_lat = used_lon = None
    dropped = 0
    total = 0
    for y in YEARS:
        d = fetch_year(y, state)
        if used_lat is None:
            used_lat, used_lon = float(d["latitude"]), float(d["longitude"])
        elif (float(d["latitude"]), float(d["longitude"])) != (used_lat, used_lon):
            die(f"{y}: grid point differs from earlier years")
        h = d["hourly"]
        t, s, dr = h["time"], h["wind_speed_10m"], h["wind_direction_10m"]
        if not (len(t) == len(s) == len(dr)):
            die(f"{y}: hourly arrays have different lengths")
        total += len(t)
        for ti, si, di in zip(t, s, dr):
            if si is None or di is None or (isinstance(si, float) and math.isnan(si)) \
                    or (isinstance(di, float) and math.isnan(di)):
                dropped += 1
                continue
            times.append(ti)
            speeds.append(float(si))
            dirs.append(float(di) % 360.0)

    time_utc = to_hours(times)
    if len(time_utc) > 1 and not np.all(np.diff(time_utc) > 0):
        die("time is not strictly increasing")
    speed = np.asarray(speeds, dtype=np.float32)
    direction = np.asarray(dirs, dtype=np.float32)
    # float32 rounding can push 359.99999 to 360.0; fold it back into [0, 360).
    direction[direction >= 360.0] = 0.0

    np.savez_compressed(
        OUT_NPZ,
        time_utc=time_utc,
        speed_ms=speed,
        direction_deg=direction,
        lat=np.float64(used_lat),
        lon=np.float64(used_lon),
    )
    size = OUT_NPZ.stat().st_size
    print(f"Hours in responses: {total}; dropped (missing value): {dropped}; kept: {len(time_utc)}")
    print(f"Grid point used by API: lat {used_lat}, lon {used_lon}")
    print(f"Wrote {OUT_NPZ} ({size / 1e3:.1f} kB)")
    if size >= 500_000:
        die("npz is 500 kB or larger")


if __name__ == "__main__":
    main()
