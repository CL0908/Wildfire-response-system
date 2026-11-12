#!/usr/bin/env python
"""Export the FIRMS fire detections inside the dashboard's map window for date replay.

Reads `bounds` and the 20 x 20 grid shape from dashboards/public/results.json, reads the
yearly VIIRS S-NPP Thailand CSVs from the raw cache (QP_DATA_CACHE, default
../data_cache/firms next to the repo), keeps detections strictly inside the bounds and
writes dashboards/public/fires.json.

Row filter (same as load_fires in data/fetch_khao_yai.py): confidence in {n, h} and
type == 0 when the year's file has a usable type column; otherwise all n/h rows.

Usage: python data/export_fire_history.py
"""
import csv
import json
import math
import os
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent / "dashboards" / "public"
CACHE = Path(os.environ.get("QP_DATA_CACHE", HERE.parent.parent / "data_cache")) / "firms"
YEARS = range(2012, 2026)
SOURCE = "NASA FIRMS VIIRS 375 m active fire detections"
NOTE = "Each entry is a satellite detection of a hot spot, located to about 375 m. Not a fire perimeter."


def cell_of(lat, lon, bounds, rows, cols):
    """(row, col) of the grid cell containing a point strictly inside bounds, else None.
    Row 0 is the north edge, column 0 the west edge."""
    n, s, w, e = bounds["north"], bounds["south"], bounds["west"], bounds["east"]
    if not (s < lat < n and w < lon < e):
        return None
    row = min(int((n - lat) / (n - s) * rows), rows - 1)
    col = min(int((lon - w) / (e - w) * cols), cols - 1)
    return row, col


def keep_row(confidence, type_value, type_usable):
    if confidence.strip().lower() not in ("n", "h"):
        return False
    return (type_value.strip() == "0") if type_usable else True


def read_year(path, bounds):
    """Returns the list of kept raw rows (dicts) for one CSV, inside bounds."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        if fields[:2] != ["latitude", "longitude"]:
            raise SystemExit(f"unexpected header in {path.name}: {fields[:4]}")
        rows = []
        for r in rd:
            lat, lon = float(r["latitude"]), float(r["longitude"])
            if not (bounds["south"] < lat < bounds["north"] and bounds["west"] < lon < bounds["east"]):
                continue
            rows.append(r)
    type_usable = "type" in fields and any(
        r["type"].strip() != "" for r in rows if r["confidence"].strip().lower() in ("n", "h"))
    return [r for r in rows if keep_row(r["confidence"], r.get("type", "") or "", type_usable)]


def main():
    res = json.loads((PUBLIC / "results.json").read_text())
    bounds = res["bounds"]
    rows_n, cols_n = res["grid"]["rows"], res["grid"]["cols"]

    seen = set()
    records = []
    for y in YEARS:
        path = CACHE / f"viirs-snpp_{y}_Thailand.csv"
        if not path.exists():
            raise SystemExit(f"missing raw file {path}; run data/fetch_khao_yai.py first")
        for r in read_year(path, bounds):
            key = (r["latitude"], r["longitude"], r["acq_date"], r["acq_time"])
            if key in seen:
                continue
            seen.add(key)
            lat, lon = float(r["latitude"]), float(r["longitude"])
            rc = cell_of(lat, lon, bounds, rows_n, cols_n)
            if rc is None:
                continue
            records.append({
                "date": r["acq_date"], "time": int(r["acq_time"]), "lon": round(lon, 5),
                "lat": round(lat, 5), "frp": round(float(r["frp"]), 1),
                "row": rc[0], "col": rc[1], "night": 1 if r["daynight"].strip() == "N" else 0,
            })

    records.sort(key=lambda d: (d["date"], d["time"], d["lon"], d["lat"]))
    detections = [[d["date"], d["lon"], d["lat"], d["frp"], d["row"], d["col"], d["night"]]
                  for d in records]
    out = {
        "source": SOURCE,
        "note": NOTE,
        "bounds": bounds,
        "grid": {"rows": rows_n, "cols": cols_n},
        "first_date": detections[0][0] if detections else None,
        "last_date": detections[-1][0] if detections else None,
        "count": len(detections),
        "fields": ["date", "lon", "lat", "frp", "row", "col", "night"],
        "detections": detections,
    }
    body = json.dumps({k: v for k, v in out.items() if k != "detections"}, indent=1)
    head = body[:body.rstrip().rfind("}")].rstrip()
    det = ",\n".join("  " + json.dumps(d, separators=(",", ":")) for d in detections)
    text = f"{head},\n  \"detections\": [\n{det}\n  ]\n}}\n"
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "fires.json").write_text(text)
    print(f"detections {len(detections)}  first {out['first_date']}  last {out['last_date']}  "
          f"fires.json {len(text.encode())} bytes")


if __name__ == "__main__":
    main()
