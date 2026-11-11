#!/usr/bin/env python
"""One-off fetch of real geodata for Khao Yai National Park.

Produces, next to this file:
    khao_yai_500m.npz   per-cell arrays for a 40 x 40 cell (20 km x 20 km) region
    khao_yai_meta.json  bounds, sources, filters, statistics
    preview.png         overview figure

Raw downloads (FIRMS CSVs, park boundary) go to a cache OUTSIDE the repo
(default: <repo>/../data_cache, override with QP_DATA_CACHE). The DEM and land
cover rasters are read as windows over HTTP range requests and never stored.

Only needed to regenerate data/. The main pipeline reads only the npz.
Usage: python data/fetch_khao_yai.py
"""
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("QP_DATA_CACHE", HERE.parent.parent / "data_cache"))
UA = "q-pithak-data-fetch"
ACCESS_DATE = "12 November 2026"

# ---- sources ---------------------------------------------------------------
DEM_URL = ("https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
           "Copernicus_DSM_COG_10_N14_00_E101_00_DEM/Copernicus_DSM_COG_10_N14_00_E101_00_DEM.tif")
LC_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
          "ESA_WorldCover_10m_2021_v200_N12E099_Map.tif")
FIRMS_URL = ("https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{y}/"
             "viirs-snpp_{y}_Thailand.csv")
OSM_URL = "https://polygons.openstreetmap.fr/get_geojson.py?id=2971358&params=0"
YEARS = list(range(2012, 2026))

# ---- geometry --------------------------------------------------------------
PARK_W, PARK_E, PARK_S, PARK_N = 101.1267, 101.8960, 14.0773, 14.5746
CELL_M = 500.0
MARGIN_M = 2000.0
REGION = 40          # cells per side of the study region
SUB = 20             # cells per side of the default window
THRESHOLDS = [60, 50, 40, 30, 20, 10]   # percent of cells with inside_park > 0.5
INSIDE_CELL = 0.5    # a cell counts as "inside" when more than this fraction is inside

LAT0 = (PARK_S + PARK_N) / 2.0
DLAT = CELL_M / 110574.0
DLON = CELL_M / (111320.0 * math.cos(math.radians(LAT0)))
M_LAT_DEG = 110574.0
M_LON_DEG = 111320.0 * math.cos(math.radians(LAT0))

LC_CLASSES = [("lc_tree", 10), ("lc_shrub", 20), ("lc_grass", 30), ("lc_crop", 40),
              ("lc_built", 50), ("lc_bare", 60), ("lc_water", 80), ("lc_wetland", 90)]

bytes_downloaded = {"http": 0}


# ---- helpers ---------------------------------------------------------------
def die(msg):
    sys.exit("ERROR: " + msg)


def download(url, dest, min_bytes=1000):
    """Download url to dest unless present. Returns True if downloaded."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  cached   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
            n = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                n += len(chunk)
    except (urllib.error.URLError, OSError) as e:
        part.unlink(missing_ok=True)
        die(f"could not download {url}: {e}")
    if n < min_bytes:
        part.unlink(missing_ok=True)
        die(f"download of {url} returned only {n} bytes")
    part.replace(dest)
    bytes_downloaded["http"] += n
    print(f"  download {dest.name} ({n / 1e6:.1f} MB)")
    return True


def open_raster(url, name):
    """Open a COG via /vsicurl range requests; fall back to a cached full download."""
    import rasterio
    try:
        return rasterio.open("/vsicurl/" + url)
    except Exception as e:  # noqa: BLE001
        print(f"  windowed read of {name} failed ({e}); downloading whole file to cache")
        dest = CACHE / name
        download(url, dest, min_bytes=1_000_000)
        return rasterio.open(dest)


def read_window(src, west, east, south, north):
    """Read the pixels covering the lon/lat box (outward-rounded). Returns array and the
    1D pixel-centre longitudes / latitudes."""
    from rasterio.windows import from_bounds, Window
    w = from_bounds(west, south, east, north, src.transform)
    r0 = max(int(math.floor(w.row_off)), 0)
    c0 = max(int(math.floor(w.col_off)), 0)
    r1 = min(int(math.ceil(w.row_off + w.height)), src.height)
    c1 = min(int(math.ceil(w.col_off + w.width)), src.width)
    arr = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    t = src.transform
    lons = t.c + (np.arange(c0, c1) + 0.5) * t.a
    lats = t.f + (np.arange(r0, r1) + 0.5) * t.e
    return arr, lons, lats


def write_npz(path, arrays):
    """Deterministic npz (fixed zip timestamps) so reruns are byte-identical."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, a in arrays.items():
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            with z.open(zi, "w", force_zip64=False) as f:
                np.lib.format.write_array(f, np.ascontiguousarray(a), allow_pickle=False)


# ---- park boundary ---------------------------------------------------------
def load_boundary():
    dest = CACHE / "osm_relation_2971358.geojson"
    download(OSM_URL, dest, min_bytes=5000)
    gj = json.loads(dest.read_text())
    polys = []

    def walk(g):
        t = g["type"]
        if t == "FeatureCollection":
            for f in g["features"]:
                walk(f)
        elif t == "Feature":
            walk(g["geometry"])
        elif t == "GeometryCollection":
            for x in g["geometries"]:
                walk(x)
        elif t == "Polygon":
            polys.append([np.asarray(r, float)[:, :2] for r in g["coordinates"]])
        elif t == "MultiPolygon":
            for p in g["coordinates"]:
                polys.append([np.asarray(r, float)[:, :2] for r in p])

    walk(gj)
    if not polys:
        die("park boundary GeoJSON contains no polygons")
    return polys


def points_in_polygons(lon, lat, polys):
    """Even-odd point-in-polygon over all rings of all polygons (outer rings and holes
    toggle the result, which is correct for non-overlapping MultiPolygons with holes)."""
    from matplotlib.path import Path as MPath
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    inside = np.zeros(len(pts), bool)
    for poly in polys:
        for ring in poly:
            x0, y0 = ring.min(0)
            x1, y1 = ring.max(0)
            cand = np.flatnonzero((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                                  (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
            if cand.size:
                hit = MPath(ring).contains_points(pts[cand])
                inside[cand[hit]] ^= True
    return inside.reshape(np.shape(lon))


# ---- fire detections -------------------------------------------------------
def load_fires(grid):
    """Returns dict year -> (lat, lon) arrays of filtered detections inside the grid
    extent, plus per-year stats."""
    print("FIRMS VIIRS S-NPP yearly Thailand CSVs")
    fires, stats = {}, {}
    for i, y in enumerate(YEARS):
        dest = CACHE / "firms" / f"viirs-snpp_{y}_Thailand.csv"
        if download(FIRMS_URL.format(y=y), dest, min_bytes=100_000):
            time.sleep(1.0)
        with open(dest, newline="") as f:
            rd = csv.reader(f)
            hdr = next(rd)
            if hdr[:2] != ["latitude", "longitude"]:
                die(f"unexpected header in {dest.name}: {hdr[:4]}")
            ix = {k: hdr.index(k) for k in ("latitude", "longitude", "confidence")}
            it = hdr.index("type") if "type" in hdr else None
            rows_total = rows_box = 0
            cand = []   # (lat, lon, type_str) for n/h rows inside grid extent
            for row in rd:
                rows_total += 1
                lat = float(row[ix["latitude"]])
                lon = float(row[ix["longitude"]])
                if not (grid["south"] <= lat <= grid["north"] and grid["west"] <= lon <= grid["east"]):
                    continue
                rows_box += 1
                if row[ix["confidence"]].strip().lower() not in ("n", "h"):
                    continue
                cand.append((lat, lon, row[it].strip() if it is not None else ""))
        type_usable = it is not None and any(c[2] != "" for c in cand)
        if type_usable:
            kept = [c for c in cand if c[2] == "0"]
        else:
            kept = cand
        fires[y] = (np.array([c[0] for c in kept], float), np.array([c[1] for c in kept], float))
        stats[y] = {"rows_in_file": rows_total, "rows_in_grid_extent": rows_box,
                    "rows_confidence_n_or_h": len(cand), "rows_kept": len(kept),
                    "type_filter_applied": bool(type_usable)}
    return fires, stats


# ---- grid ------------------------------------------------------------------
def make_grid():
    mlat = MARGIN_M / M_LAT_DEG
    mlon = MARGIN_M / M_LON_DEG
    north, west = PARK_N + mlat, PARK_W - mlon
    nrows = int(math.ceil((north - (PARK_S - mlat)) / DLAT))
    ncols = int(math.ceil(((PARK_E + mlon) - west) / DLON))
    return {"north": north, "west": west, "nrows": nrows, "ncols": ncols,
            "south": north - nrows * DLAT, "east": west + ncols * DLON}


def cell_index(lat, lon, grid):
    r = np.floor((grid["north"] - lat) / DLAT).astype(int)
    c = np.floor((lon - grid["west"]) / DLON).astype(int)
    ok = (r >= 0) & (r < grid["nrows"]) & (c >= 0) & (c < grid["ncols"])
    return r, c, ok


def inside_fraction(grid, polys, n=5):
    """Fraction of each cell inside the park, from an n x n grid of points per cell."""
    off = (np.arange(n) + 0.5) / n
    rr = (np.arange(grid["nrows"])[:, None] + off[None, :]).ravel()   # sub-row positions
    cc = (np.arange(grid["ncols"])[:, None] + off[None, :]).ravel()
    lat = grid["north"] - rr * DLAT
    lon = grid["west"] + cc * DLON
    LON, LAT = np.meshgrid(lon, lat)
    ins = points_in_polygons(LON, LAT, polys)
    ins = ins.reshape(grid["nrows"], n, grid["ncols"], n)
    return ins.mean(axis=(1, 3)).astype(np.float32)


def best_window(inside, fire, size):
    """Window of size x size cells with the most detections among windows where the share
    of cells with inside > INSIDE_CELL is >= threshold; relax threshold in 10-point steps."""
    def box(a):
        s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
        s[1:, 1:] = a.cumsum(0).cumsum(1)
        return s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]
    n_in = box((inside > INSIDE_CELL).astype(np.int64))
    tot = box(fire.astype(np.int64))
    for pct in THRESHOLDS:
        ok = n_in * 100 >= pct * size * size
        if ok.any():
            r, c = np.unravel_index(np.argmax(np.where(ok, tot, -1)), tot.shape)  # first max, row-major
            return {"row0": int(r), "col0": int(c), "threshold_pct": pct,
                    "share_cells_inside": float(n_in[r, c] / size ** 2),
                    "fire_total": int(tot[r, c]), "windows_meeting_threshold": int(ok.sum())}
    die("no window satisfies even the relaxed park-coverage thresholds")


# ---- rasters binned to cells -----------------------------------------------
def bin_indices(lons, lats, reg):
    r = np.floor((reg["north"] - lats) / DLAT).astype(int)
    c = np.floor((lons - reg["west"]) / DLON).astype(int)
    r = np.where((r >= 0) & (r < REGION), r, -1)
    c = np.where((c >= 0) & (c < REGION), c, -1)
    flat = np.where((r[:, None] >= 0) & (c[None, :] >= 0), r[:, None] * REGION + c[None, :], -1)
    return flat


def bin_dem(arr, lons, lats, reg):
    flat = bin_indices(lons, lats, reg)
    m = flat >= 0
    idx = flat[m]
    v = arr[m].astype(np.float64)
    cnt = np.bincount(idx, minlength=REGION ** 2)
    if (cnt == 0).any():
        die("some cells have no DEM pixels")
    mean = np.bincount(idx, weights=v, minlength=REGION ** 2) / cnt
    mx = np.full(REGION ** 2, -np.inf)
    np.maximum.at(mx, idx, v)
    return (mean.reshape(REGION, REGION).astype(np.float32),
            mx.reshape(REGION, REGION).astype(np.float32), cnt.reshape(REGION, REGION))


def bin_landcover(arr, lons, lats, reg):
    flat = bin_indices(lons, lats, reg)
    lut = np.full(256, len(LC_CLASSES), np.int64)     # last slot = other
    for k, (_, code) in enumerate(LC_CLASSES):
        lut[code] = k
    m = flat >= 0
    key = flat[m] * (len(LC_CLASSES) + 1) + lut[arr[m]]
    counts = np.bincount(key, minlength=REGION ** 2 * (len(LC_CLASSES) + 1))
    counts = counts.reshape(REGION, REGION, len(LC_CLASSES) + 1)
    tot = counts.sum(-1)
    frac = counts / tot[..., None]
    out = {name: frac[..., k].astype(np.float32) for k, (name, _) in enumerate(LC_CLASSES)}
    out["lc_other"] = frac[..., -1].astype(np.float32)
    nodata_share = float((arr[m] == 0).mean())
    return out, tot, nodata_share


def stats(a):
    a = np.asarray(a, np.float64)
    return {"min": round(float(a.min()), 6), "mean": round(float(a.mean()), 6),
            "max": round(float(a.max()), 6)}


# ---- main ------------------------------------------------------------------
def main():
    t_start = time.time()
    import rasterio
    CACHE.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    print(f"Grid: {grid['nrows']} x {grid['ncols']} cells of {DLAT:.6f} deg lat x {DLON:.6f} deg lon; "
          f"cache: {CACHE}")

    print("Park boundary (OSM relation 2971358)")
    polys = load_boundary()
    print(f"  {len(polys)} polygon(s), {sum(len(r) for p in polys for r in p)} vertices")

    fires, fire_stats = load_fires(grid)

    # whole-park coarse grid
    inside = inside_fraction(grid, polys)
    by_year = np.zeros((len(YEARS), grid["nrows"], grid["ncols"]), np.int32)
    for k, y in enumerate(YEARS):
        r, c, ok = cell_index(fires[y][0], fires[y][1], grid)
        np.add.at(by_year[k], (r[ok], c[ok]), 1)
    coarse_fire = by_year.sum(0)
    print(f"Coarse grid: {int((inside > INSIDE_CELL).sum())} cells inside park, "
          f"{int(coarse_fire.sum())} detections in grid extent")

    # choose region and default window
    sel = best_window(inside, coarse_fire, REGION)
    R0, C0 = sel["row0"], sel["col0"]
    reg = {"north": grid["north"] - R0 * DLAT, "west": grid["west"] + C0 * DLON}
    reg["south"] = reg["north"] - REGION * DLAT
    reg["east"] = reg["west"] + REGION * DLON
    inside_r = inside[R0:R0 + REGION, C0:C0 + REGION].copy()
    fire_r_by_year = by_year[:, R0:R0 + REGION, C0:C0 + REGION]
    fire_count = fire_r_by_year.sum(0).astype(np.int32)
    fire_years = (fire_r_by_year > 0).sum(0).astype(np.int16)
    sub = best_window(inside_r, fire_count, SUB)
    print(f"Region: grid offset (row {R0}, col {C0}), threshold {sel['threshold_pct']}%, "
          f"{sel['fire_total']} detections; default window offset ({sub['row0']}, {sub['col0']}), "
          f"threshold {sub['threshold_pct']}%, {sub['fire_total']} detections")

    # rasters, windowed
    import rasterio
    pad = 2 * DLAT
    box = (reg["west"] - pad, reg["east"] + pad, reg["south"] - pad, reg["north"] + pad)
    env = rasterio.Env(GDAL_HTTP_USERAGENT=UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env:
        print("Copernicus DEM GLO-30 (windowed read)")
        try:
            with open_raster(DEM_URL, "Copernicus_DEM_N14_E101.tif") as src:
                dem_arr, dem_lon, dem_lat = read_window(src, *box)
                dem_mean, dem_max, dem_cnt = bin_dem(dem_arr, dem_lon, dem_lat, reg)
                checks = dem_checks(src, dem_arr, dem_lon, dem_lat, dem_mean, reg)
        except rasterio.errors.RasterioIOError as e:
            die(f"cannot read DEM: {e}")
        print(f"  {dem_arr.shape} pixels read")
        print("ESA WorldCover 2021 v200 (windowed read)")
        try:
            with open_raster(LC_URL, "ESA_WorldCover_N12E099.tif") as src:
                lc_arr, lc_lon, lc_lat = read_window(src, *box)
        except rasterio.errors.RasterioIOError as e:
            die(f"cannot read WorldCover: {e}")
        print(f"  {lc_arr.shape} pixels read")
    lc, lc_cnt, lc_nodata = bin_landcover(lc_arr, lc_lon, lc_lat, reg)

    lat = reg["north"] - (np.arange(REGION) + 0.5) * DLAT
    lon = reg["west"] + (np.arange(REGION) + 0.5) * DLON
    LON, LAT = np.meshgrid(lon, lat)

    arrays = {"dem_m": dem_mean, "dem_max_m": dem_max, "fire_count": fire_count,
              "fire_years": fire_years}
    arrays.update(lc)
    arrays["inside_park"] = inside_r
    arrays["lat"] = LAT
    arrays["lon"] = LON
    write_npz(HERE / "khao_yai_500m.npz", arrays)

    # sanity numbers
    lc_sum = sum(lc[k].astype(np.float64) for k in lc)
    shares = {k: float(v.mean()) for k, v in lc.items()}
    in_park_cells = (inside_r > INSIDE_CELL)
    per_year_box = {}
    per_year_region = {}
    for k, y in enumerate(YEARS):
        la, lo = fires[y]
        per_year_box[y] = int(((la >= PARK_S) & (la <= PARK_N) & (lo >= PARK_W) & (lo <= PARK_E)).sum())
        per_year_region[y] = int(fire_r_by_year[k].sum())
    check = {
        "dem_min_max_m": [float(dem_mean.min()), float(dem_max.max())],
        "dem_min_pixels_per_cell": int(dem_cnt.min()),
        "orientation": {
            "lat_decreases_with_row": bool((np.diff(LAT, axis=0) < 0).all()),
            "lon_increases_with_col": bool((np.diff(LON, axis=1) > 0).all()),
            **checks["orientation"]},
        "khao_rom_check": checks["khao_rom"],
        "lc_fraction_sum_min_max": [float(lc_sum.min()), float(lc_sum.max())],
        "lc_pixels_per_cell_min_max": [int(lc_cnt.min()), int(lc_cnt.max())],
        "lc_nodata_pixel_share": round(lc_nodata, 6),
        "lc_region_shares": {k: round(v, 4) for k, v in shares.items()},
        "fire_total_region": int(fire_count.sum()),
        "fire_cells_zero": int((fire_count == 0).sum()),
        "fire_per_year_park_bbox": per_year_box,
        "inside_cells_gt_0.5_share": float(in_park_cells.mean()),
        "inside_mean_fraction": float(inside_r.mean()),
    }

    reg_bounds = {"north": reg["north"], "south": reg["south"], "west": reg["west"], "east": reg["east"]}
    sw = {"row0": sub["row0"], "col0": sub["col0"], "n_rows": SUB, "n_cols": SUB,
          "north": reg["north"] - sub["row0"] * DLAT,
          "south": reg["north"] - (sub["row0"] + SUB) * DLAT,
          "west": reg["west"] + sub["col0"] * DLON,
          "east": reg["west"] + (sub["col0"] + SUB) * DLON}
    meta = {
        "description": "500 m cell aggregates of real geodata for a 20 km x 20 km region of Khao Yai "
                       "National Park, Thailand. Only derived aggregates are stored.",
        "access_date": ACCESS_DATE,
        "region_bounds_deg": {k: round(v, 6) for k, v in reg_bounds.items()},
        "cell_size_m": CELL_M,
        "cell_size_deg": {"lat": DLAT, "lon": DLON, "lat0_for_lon_scale": LAT0,
                          "metres_per_deg_lat": M_LAT_DEG, "metres_per_deg_lon": M_LON_DEG},
        "grid_shape": [REGION, REGION],
        "orientation": "row 0 is the north edge, column 0 is the west edge; lat decreases with "
                       "row, lon increases with column; lat/lon are cell centres",
        "default_window": {**{k: (round(v, 6) if isinstance(v, float) else v) for k, v in sw.items()},
                           "offset_in_region": [sub["row0"], sub["col0"]]},
        "selection": {
            "rule": "window of 40 x 40 cells with the highest total filtered fire-detection count "
                    "(2012 to 2025) among windows where the share of cells with inside_park > 0.5 is "
                    "at least the threshold; threshold relaxed in 10-point steps if no window "
                    "qualifies; ties go to the first window in row-major order",
            "search_grid": {"rows": grid["nrows"], "cols": grid["ncols"],
                            "north": round(grid["north"], 6), "west": round(grid["west"], 6),
                            "margin_m": MARGIN_M, "region_offset_row_col": [R0, C0]},
            "region_threshold_pct_used": sel["threshold_pct"],
            "region_share_cells_inside": sel["share_cells_inside"],
            "region_windows_meeting_threshold": sel["windows_meeting_threshold"],
            "region_fire_total": sel["fire_total"],
            "default_window_rule": "same rule for 20 x 20 cells inside the region",
            "default_window_threshold_pct_used": sub["threshold_pct"],
            "default_window_share_cells_inside": sub["share_cells_inside"],
            "default_window_fire_total": sub["fire_total"],
        },
        "sources": {
            "elevation": {
                "name": "Copernicus DEM GLO-30 (DSM, COG, 1 arc-second)", "url": DEM_URL,
                "licence": "Copernicus DEM licence (free of charge, incl. commercial use, with attribution)",
                "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
                               "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the "
                               "European Union and ESA; all rights reserved",
                "processing": "window read over HTTP; mean and max of the pixels whose centre is in each cell"},
            "fire": {
                "name": "NASA FIRMS VIIRS S-NPP 375 m, yearly Thailand CSVs, 2012 to 2025",
                "url": FIRMS_URL, "licence": "NASA open data policy; free to use with acknowledgement",
                "attribution": "We acknowledge the use of data and/or imagery from NASA's Fire Information "
                               "for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), "
                               "part of NASA's Earth Science Data and Information System (ESDIS).",
                "filter": "confidence in {n, h} and type == 0 (presumed vegetation fire); if a year's file "
                          "lacks a usable type column all n/h rows are kept (see per_year)",
                "per_year_file_stats": {str(y): fire_stats[y] for y in YEARS},
                "per_year_detections_in_region": {str(y): per_year_region[y] for y in YEARS},
                "per_year_detections_in_park_bbox": {str(y): per_year_box[y] for y in YEARS}},
            "landcover": {
                "name": "ESA WorldCover 10 m 2021 v200", "url": LC_URL, "licence": "CC BY 4.0",
                "attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data "
                               "(2021) processed by ESA WorldCover consortium",
                "citation": "Zanaga et al. 2022, doi 10.5281/zenodo.7254221",
                "processing": "window read over HTTP; fraction of pixels per class in each cell; classes "
                              "10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 80 water, "
                              "90 wetland, anything else (incl. 0 no data) in lc_other"},
            "park_boundary": {
                "name": "OpenStreetMap relation 2971358", "url": OSM_URL, "licence": "ODbL 1.0",
                "attribution": "© OpenStreetMap contributors",
                "licence_url": "https://www.openstreetmap.org/copyright",
                "processing": "inside_park = share of a 5 x 5 grid of points per cell inside the polygon "
                              "(even-odd over all rings, so holes are excluded)"}},
        "array_stats": {k: stats(v) for k, v in arrays.items()},
        "sanity_checks": check,
    }
    (HERE / "khao_yai_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    make_preview(arrays, inside, coarse_fire, polys, grid, reg, R0, C0, sub)
    print("Sanity checks:")
    print(json.dumps(check, indent=2))
    print(f"Downloaded this run: {bytes_downloaded['http'] / 1e6:.1f} MB of cached files "
          "(DEM/WorldCover windows are range-read each run, a few MB, not byte-counted)")
    print(f"Wrote {HERE / 'khao_yai_500m.npz'} ({(HERE / 'khao_yai_500m.npz').stat().st_size} bytes), "
          f"meta, preview in {time.time() - t_start:.0f} s")


def dem_checks(src, arr, lons, lats, mean, reg):
    """Orientation and Khao Rom checks, reading raw pixels straight from the source."""
    rng = np.random.default_rng(0)
    cells = [(0, 0), (0, REGION - 1), (REGION - 1, 0), (REGION - 1, REGION - 1), (20, 20)]
    cells += [tuple(int(v) for v in rng.integers(0, REGION, 2)) for _ in range(5)]
    spot = []
    for r, c in cells:
        clat = reg["north"] - (r + 0.5) * DLAT
        clon = reg["west"] + (c + 0.5) * DLON
        px = float(next(src.sample([(clon, clat)]))[0])
        spot.append({"row": r, "col": c, "raw_pixel_m": round(px, 1),
                     "cell_mean_m": round(float(mean[r, c]), 1)})
    # raw pixel at every cell centre, independent indexing via the dataset's own transform
    LAT = reg["north"] - (np.arange(REGION) + 0.5) * DLAT
    LON = reg["west"] + (np.arange(REGION) + 0.5) * DLON
    rows, cols = np.meshgrid(*[np.arange(REGION)] * 2, indexing="ij")
    pr, pc = rasterio_rowcol(src, np.tile(LON, REGION), np.repeat(LAT, REGION))
    t = src.transform
    r_off = int(round((lats[0] - t.f) / t.e - 0.5))
    c_off = int(round((lons[0] - t.c) / t.a - 0.5))
    raw = arr[pr - r_off, pc - c_off].reshape(REGION, REGION).astype(np.float64)
    corr = float(np.corrcoef(raw.ravel(), mean.ravel())[0, 1])
    corr_flip = float(np.corrcoef(raw[::-1].ravel(), mean.ravel())[0, 1])
    # Khao Rom
    from rasterio.windows import Window
    kl, kn = 101.40, 14.37
    a, lo, la = read_window(src, kl - 0.04, kl + 0.04, kn - 0.04, kn + 0.04)
    dx = (lo[None, :] - kl) * M_LON_DEG
    dy = (la[:, None] - kn) * M_LAT_DEG
    d = np.hypot(dx, dy)
    out = {}
    for rad in (1000, 3000):
        m = d <= rad
        i = np.argmax(np.where(m, a, -np.inf))
        ri, ci = np.unravel_index(i, a.shape)
        out[f"max_within_{rad}m"] = {"dem_m": round(float(a[ri, ci]), 1),
                                     "lat": round(float(la[ri]), 5), "lon": round(float(lo[ci]), 5)}
    nearest = np.unravel_index(np.argmin(d), d.shape)
    out["value_at_point_m"] = round(float(a[nearest]), 1)
    return {"orientation": {
        "spot_checks": spot,
        "max_abs_diff_spot_m": round(max(abs(s["raw_pixel_m"] - s["cell_mean_m"]) for s in spot), 1),
        "corr_raw_centre_pixel_vs_cell_mean": round(corr, 5),
        "corr_if_array_flipped_north_south": round(corr_flip, 5)}, "khao_rom": out}


def rasterio_rowcol(src, lons, lats):
    cols, rows = ~src.transform * (np.asarray(lons), np.asarray(lats))
    return np.floor(rows).astype(int), np.floor(cols).astype(int)


# ---- preview ---------------------------------------------------------------
def make_preview(arrays, inside, coarse_fire, polys, grid, reg, R0, C0, sub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    km = CELL_M / 1000.0
    ext_r = [0, REGION * km, REGION * km, 0]

    def ring_km(ring, origin_n, origin_w):
        return ((ring[:, 0] - origin_w) / DLON * km, (origin_n - ring[:, 1]) / DLAT * km)

    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    fire_cmap = plt.get_cmap("YlOrRd").copy()
    fire_cmap.set_bad("#e6e6e6")
    panels = [
        ("Elevation (mean per cell)", arrays["dem_m"], "viridis", None, "m above sea level"),
        ("Fire detections 2012-2025 (grey = none)", np.ma.masked_equal(arrays["fire_count"], 0),
         fire_cmap, LogNorm(vmin=1, vmax=max(2, arrays["fire_count"].max())), "detections per cell (log scale)"),
        ("Tree cover", arrays["lc_tree"], "Greens", None, "fraction of cell area"),
        ("Grass + shrub cover", arrays["lc_grass"] + arrays["lc_shrub"], "YlOrBr", None,
         "fraction of cell area"),
        ("Inside park", arrays["inside_park"], "Blues", None, "fraction of cell area"),
    ]
    for ax, (title, data, cmap, norm, label) in zip(axs.ravel()[:5], panels):
        kw = {"norm": norm} if norm is not None else {}
        if norm is None and "fraction" in label:
            kw.update(vmin=0, vmax=1)
        im = ax.imshow(data, extent=ext_r, cmap=cmap, interpolation="nearest", **kw)
        for p in polys:
            for ring in p:
                x, y = ring_km(ring, reg["north"], reg["west"])
                ax.plot(x, y, color="k", lw=0.8)
        ax.set_xlim(0, REGION * km)
        ax.set_ylim(REGION * km, 0)
        ax.set_title(title)
        ax.set_xlabel("east from region west edge (km)")
        ax.set_ylabel("south from region north edge (km)")
        fig.colorbar(im, ax=ax, label=label, shrink=0.85)
    ax = axs[1, 2]
    cf = np.ma.masked_equal(coarse_fire, 0)
    ext = [0, grid["ncols"] * km, grid["nrows"] * km, 0]
    im = ax.imshow(cf, extent=ext, cmap=fire_cmap, norm=LogNorm(vmin=1, vmax=max(2, coarse_fire.max())),
                   interpolation="nearest")
    for p in polys:
        for ring in p:
            x, y = ring_km(ring, grid["north"], grid["west"])
            ax.plot(x, y, color="k", lw=0.8)
    ax.add_patch(Rectangle((C0 * km, R0 * km), REGION * km, REGION * km, fill=False, ec="#0072B2", lw=2,
                           label="40 x 40 region"))
    ax.add_patch(Rectangle(((C0 + sub["col0"]) * km, (R0 + sub["row0"]) * km), SUB * km, SUB * km,
                           fill=False, ec="#009E73", lw=2, ls="--", label="20 x 20 default window"))
    ax.set_xlim(0, grid["ncols"] * km)
    ax.set_ylim(grid["nrows"] * km, 0)
    ax.set_title("Whole park: fire detections 2012-2025")
    ax.set_xlabel("east from grid west edge (km)")
    ax.set_ylabel("south from grid north edge (km)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax, label="detections per cell (log scale)", shrink=0.85)
    fig.savefig(HERE / "preview.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
