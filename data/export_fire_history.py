#!/usr/bin/env python
"""Export the FIRMS fire detections inside the dashboard's map window for date replay.

Reads `bounds` and the 20 x 20 grid shape from dashboards/public/results.json, reads the
yearly VIIRS S-NPP Thailand CSVs from the raw cache (QP_DATA_CACHE, default
../data_cache/firms next to the repo), keeps detections strictly inside the bounds and
writes dashboards/public/fires.json.

With --park, the same export runs on the whole-park grid: bounds from
data/park_500m_meta.json (`bounds_deg`), grid shape from its `grid_shape`, output
dashboards/public/fires_park.json. Each detection gets one extra field, `inside`, which is
1 when its cell has inside_park > 0.5 in data/park_500m.npz, else 0.

Row filter (same as load_fires in data/fetch_khao_yai.py): confidence in {n, h} and
type == 0 when the year's file has a usable type column; otherwise all n/h rows.

Usage: python data/export_fire_history.py [--park]
"""
import argparse
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
PARK_SOURCE = SOURCE + ", binned to the whole-park grid"
PARK_NOTE = (NOTE + " Grid: the whole of Khao Yai National Park as 131 x 187 cells of 500 m, "
             "the park plus a 5 km margin on every side. inside is 1 when more than half of the "
             "cell's sample points lie inside the park polygon, else 0.")
PARK_META = HERE / "park_500m_meta.json"
PARK_NPZ = HERE / "park_500m.npz"
FIELDS = ["date", "lon", "lat", "frp", "row", "col", "night"]


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


def collect_records(bounds, rows_n, cols_n):
    """Kept, de-duplicated detections across all years, each with its grid row and col."""
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
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--park", action="store_true",
                        help="export the whole-park grid to dashboards/public/fires_park.json")
    args = parser.parse_args()

    if args.park:
        meta = json.loads(PARK_META.read_text())
        bounds = meta["bounds_deg"]
        rows_n, cols_n = meta["grid_shape"]
        import numpy as np
        inside = np.load(PARK_NPZ)["inside_park"] > 0.5
        fields = FIELDS + ["inside"]
        source, note = PARK_SOURCE, PARK_NOTE
        out_name = "fires_park.json"
    else:
        res = json.loads((PUBLIC / "results.json").read_text())
        bounds = res["bounds"]
        rows_n, cols_n = res["grid"]["rows"], res["grid"]["cols"]
        inside = None
        fields = FIELDS
        source, note = SOURCE, NOTE
        out_name = "fires.json"

    records = collect_records(bounds, rows_n, cols_n)
    detections = []
    for d in records:
        row = [d["date"], d["lon"], d["lat"], d["frp"], d["row"], d["col"], d["night"]]
        if inside is not None:
            row.append(int(inside[d["row"], d["col"]]))
        detections.append(row)
    out = {
        "source": source,
        "note": note,
        "bounds": bounds,
        "grid": {"rows": rows_n, "cols": cols_n},
        "first_date": detections[0][0] if detections else None,
        "last_date": detections[-1][0] if detections else None,
        "count": len(detections),
        "fields": fields,
        "detections": detections,
    }
    body = json.dumps({k: v for k, v in out.items() if k != "detections"}, indent=1)
    head = body[:body.rstrip().rfind("}")].rstrip()
    det = ",\n".join("  " + json.dumps(d, separators=(",", ":")) for d in detections)
    text = f"{head},\n  \"detections\": [\n{det}\n  ]\n}}\n"
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / out_name).write_text(text)
    print(f"detections {len(detections)}  first {out['first_date']}  last {out['last_date']}  "
          f"{out_name} {len(text.encode())} bytes")


if __name__ == "__main__":
    main()
