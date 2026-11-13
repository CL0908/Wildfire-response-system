"""Build the wider hazard-training grid: data/hazard_training_500m.npz + hazard_training_meta.json.

A 200 x 200 grid of 500 m cells (about 100 km x 100 km) centred on, and lattice-aligned with,
the 40 x 40 Khao Yai extract (data/khao_yai_500m.npz). Per cell: elevation, land cover,
park membership and distance to the park edge, FIRMS detections per year (2012 to 2025),
distance to OpenStreetMap roads and settlements. Nothing is fitted here.

Reuses the readers, filters and definitions of data/fetch_khao_yai.py (imported, not copied).
Only the cell-binning is re-written, because fetch_khao_yai.bin_* hard-code a 40-cell side.
Raw downloads stay in ../data_cache/ (FIRMS CSVs, OSM responses); nothing raw is committed.

Run from the repo root:  ../.venv/bin/python data/fetch_hazard_training.py
"""
import datetime
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_khao_yai as fk  # noqa: E402

HERE = fk.HERE
CACHE = fk.CACHE
OSM_CACHE = CACHE / "osm"
UA = fk.UA   # generic client name, no personal details
OUT_NPZ = HERE / "hazard_training_500m.npz"
OUT_META = HERE / "hazard_training_meta.json"
EXISTING_NPZ = HERE / "khao_yai_500m.npz"

N = 200                      # cells per side of the new grid
OLD_N = fk.REGION            # 40
OFFSET = (N - OLD_N) // 2    # 80 cells of 500 m on each side of the existing region
OSM_MARGIN_M = 10_000.0
DENSIFY_M = 50.0             # spacing of boundary / road sample points (error < 25 m before refinement)

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# one request per feature class
OSM_CLASSES = {
    "roads_major": ('way["highway"~"^(motorway|trunk|primary|secondary|tertiary)$"]', "out geom qt;"),
    "roads_minor": ('way["highway"~"^(unclassified|residential)$"]', "out geom qt;"),
    "roads_track": ('way["highway"="track"]', "out geom qt;"),
    "places": ('node["place"~"^(city|town|village|hamlet)$"]', "out qt;"),
}
DEM_TILE = ("https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
            "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def english_date(d):
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


# ---- distance helpers (planar metres) -----------------------------------------
def point_segment_distance(p, a, b):
    """Exact distance from points p (M,2) to segments a-b (M,2 each), planar."""
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    L2 = (ab ** 2).sum(-1)
    t = np.where(L2 > 0, ((p - a) * ab).sum(-1) / np.where(L2 > 0, L2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.sqrt(((p - proj) ** 2).sum(-1))


def build_segments(lines):
    """lines: list of (k,2) arrays of planar vertices -> segment endpoint arrays A, B."""
    A, B = [], []
    for ln in lines:
        ln = np.asarray(ln, float)
        if len(ln) >= 2:
            A.append(ln[:-1])
            B.append(ln[1:])
    if not A:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.concatenate(A), np.concatenate(B)


def densify(A, B, spacing):
    """Sample points on segments A-B at midpoints of sub-pieces no longer than `spacing`, so
    every point of a segment is within spacing/2 of a sample. Returns (points, segment index)."""
    length = np.sqrt(((B - A) ** 2).sum(-1))
    n = np.maximum(np.ceil(length / spacing).astype(np.int64), 1)
    seg = np.repeat(np.arange(len(A)), n)
    start = np.cumsum(n) - n
    k = np.arange(n.sum()) - np.repeat(start, n)
    t = (k + 0.5) / n[seg]
    return A[seg] + t[:, None] * (B[seg] - A[seg]), seg


def distance_to_lines(query, A, B, spacing=DENSIFY_M, k=8):
    """Distance in metres from query points (M,2) to the nearest segment of A-B.
    KD-tree over densified samples finds candidate segments; the answer is the exact
    point-to-segment distance minimised over the k nearest samples' segments (k = 8)."""
    from scipy.spatial import cKDTree
    pts, seg = densify(A, B, spacing)
    tree = cKDTree(pts)
    k = min(k, len(pts))
    _, idx = tree.query(query, k=k)
    idx = idx.reshape(len(query), -1)
    best = np.full(len(query), np.inf)
    for j in range(idx.shape[1]):
        s = seg[idx[:, j]]
        best = np.minimum(best, point_segment_distance(query, A[s], B[s]))
    return best


def distance_to_points(query, pts):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(query)
    return d


# ---- OSM via Overpass -----------------------------------------------------------
def overpass_fetch(name, selector, out_stmt, bbox):
    """Fetch one feature class, cached as raw JSON under data_cache/osm. Returns (json, info)."""
    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    raw = OSM_CACHE / f"overpass_{name}.json"
    info_f = OSM_CACHE / f"overpass_{name}.info.json"
    if raw.exists() and raw.stat().st_size > 100 and info_f.exists():
        print(f"  cached   {raw.name} ({raw.stat().st_size / 1e6:.1f} MB)")
        return json.loads(raw.read_text()), json.loads(info_f.read_text())
    s, w, n, e = bbox
    q = (f"[out:json][timeout:300];{selector}({s:.6f},{w:.6f},{n:.6f},{e:.6f});{out_stmt}")
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for mirror in OVERPASS_MIRRORS:
        for attempt, wait in enumerate((0, 20, 60)):
            if wait:
                time.sleep(wait)
            try:
                req = urllib.request.Request(mirror, data=data, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=420) as r:
                    body = r.read()
                js = json.loads(body)
                if "elements" not in js:
                    raise ValueError("no 'elements' in response")
                if js.get("remark"):
                    raise ValueError("overpass remark: " + js["remark"][:120])
                raw.write_bytes(body)
                info = {"mirror": mirror, "query": q,
                        "downloaded": english_date(datetime.date.today()),
                        "osm_base_timestamp": js.get("osm3s", {}).get("timestamp_osm_base"),
                        "bytes": len(body), "elements": len(js["elements"])}
                info_f.write_text(json.dumps(info, indent=1))
                print(f"  download {name} from {mirror}: {len(js['elements'])} elements, "
                      f"{len(body) / 1e6:.1f} MB")
                time.sleep(5)    # be polite between classes
                return js, info
            except (urllib.error.URLError, OSError, ValueError) as ex:
                last = f"{mirror}: {ex}"
                print(f"  attempt {attempt + 1} failed on {mirror}: {ex}")
    return None, {"error": f"all mirrors failed for {name}; last: {last}"}


# ---- rasters binned onto the 200-cell grid ---------------------------------------
class Lattice:
    def __init__(self, north, west):
        self.north, self.west = north, west
        self.south = north - N * fk.DLAT
        self.east = west + N * fk.DLON

    def rc(self, lat, lon):
        r = np.floor((self.north - lat) / fk.DLAT).astype(np.int64)
        c = np.floor((lon - self.west) / fk.DLON).astype(np.int64)
        return r, c

    def flat_index(self, lons, lats):
        """(len(lats), len(lons)) flat cell index, -1 outside the grid."""
        r, _ = self.rc(lats, lons[:1].repeat(len(lats)))
        _, c = self.rc(lats[:1].repeat(len(lons)), lons)
        r = np.where((r >= 0) & (r < N), r, -1)
        c = np.where((c >= 0) & (c < N), c, -1)
        return np.where((r[:, None] >= 0) & (c[None, :] >= 0), r[:, None] * N + c[None, :], -1)


def bin_dem_into(acc, arr, lons, lats, lat_):
    flat = lat_.flat_index(lons, lats)
    m = flat >= 0
    idx = flat[m]
    acc["sum"] += np.bincount(idx, weights=arr[m].astype(np.float64), minlength=N * N)
    acc["cnt"] += np.bincount(idx, minlength=N * N)


def bin_lc_into(counts, arr, lons, lats, lat_):
    ncls = len(fk.LC_CLASSES) + 1
    lut = np.full(256, ncls - 1, np.int64)
    for k, (_, code) in enumerate(fk.LC_CLASSES):
        lut[code] = k
    flat = lat_.flat_index(lons, lats)
    m = flat >= 0
    key = flat[m] * ncls + lut[arr[m]]
    counts += np.bincount(key, minlength=N * N * ncls).reshape(N * N, ncls)


def read_dem(lat_):
    import rasterio
    acc = {"sum": np.zeros(N * N), "cnt": np.zeros(N * N, np.int64)}
    tiles = []
    for ilat in range(math.floor(lat_.south), math.floor(lat_.north) + 1):
        for ilon in range(math.floor(lat_.west), math.floor(lat_.east) + 1):
            url = DEM_TILE.format(ns="N" if ilat >= 0 else "S", lat=abs(ilat),
                                  ew="E" if ilon >= 0 else "W", lon=abs(ilon))
            name = url.rsplit("/", 1)[1]
            with fk.open_raster(url, name) as src:
                arr, lons, lats = fk.read_window(src, lat_.west, lat_.east, lat_.south, lat_.north)
            print(f"  DEM {name}: {arr.shape} pixels")
            bin_dem_into(acc, arr, lons, lats, lat_)
            tiles.append(url)
    if (acc["cnt"] == 0).any():
        fk.die("some cells have no DEM pixels")
    return (acc["sum"] / acc["cnt"]).reshape(N, N).astype(np.float32), int(acc["cnt"].min()), tiles


def read_landcover(lat_, strip=1000):
    import rasterio
    from rasterio.windows import from_bounds, Window
    counts = np.zeros((N * N, len(fk.LC_CLASSES) + 1), np.int64)
    with fk.open_raster(fk.LC_URL, "ESA_WorldCover_N12E099.tif") as src:
        w = from_bounds(lat_.west, lat_.south, lat_.east, lat_.north, src.transform)
        r0, r1 = max(int(math.floor(w.row_off)), 0), min(int(math.ceil(w.row_off + w.height)), src.height)
        c0, c1 = max(int(math.floor(w.col_off)), 0), min(int(math.ceil(w.col_off + w.width)), src.width)
        t = src.transform
        lons = t.c + (np.arange(c0, c1) + 0.5) * t.a
        total = 0
        for rs in range(r0, r1, strip):
            re = min(rs + strip, r1)
            arr = src.read(1, window=Window(c0, rs, c1 - c0, re - rs))
            lats = t.f + (np.arange(rs, re) + 0.5) * t.e
            bin_lc_into(counts, arr, lons, lats, lat_)
            total += arr.size
        print(f"  WorldCover: {r1 - r0} x {c1 - c0} pixels read at full 10 m resolution")
    tot = counts.sum(1)
    frac = counts / tot[:, None]
    out = {name: frac[:, k].reshape(N, N).astype(np.float32) for k, (name, _) in enumerate(fk.LC_CLASSES)}
    out["lc_other"] = frac[:, -1].reshape(N, N).astype(np.float32)
    return out, int(tot.min()), int(tot.max())


# ---- helpers for the report ------------------------------------------------------
def stats(a):
    a = np.asarray(a, np.float64)
    return {"min": round(float(a.min()), 6), "mean": round(float(a.mean()), 6),
            "max": round(float(a.max()), 6)}


def project(lon, lat, lon0, lat0):
    """Local equirectangular metres, with the same metres-per-degree as the cell lattice."""
    return np.column_stack([(np.asarray(lon) - lon0) * fk.M_LON_DEG,
                            (np.asarray(lat) - lat0) * fk.M_LAT_DEG])


def way_lines(js, lon0, lat0):
    lines = []
    for el in js["elements"]:
        g = el.get("geometry")
        if el.get("type") == "way" and g and len(g) >= 2:
            lines.append(project([p["lon"] for p in g], [p["lat"] for p in g], lon0, lat0))
    return lines


# ---- main ------------------------------------------------------------------------
def main():
    t0 = time.time()
    import rasterio
    old = np.load(EXISTING_NPZ)
    old_north = float(old["lat"][0, 0] + fk.DLAT / 2)
    old_west = float(old["lon"][0, 0] - fk.DLON / 2)
    lat_ = Lattice(old_north + OFFSET * fk.DLAT, old_west - OFFSET * fk.DLON)
    grid = {"north": lat_.north, "south": lat_.south, "west": lat_.west, "east": lat_.east,
            "nrows": N, "ncols": N}
    lat = lat_.north - (np.arange(N) + 0.5) * fk.DLAT
    lon = lat_.west + (np.arange(N) + 0.5) * fk.DLON
    LON, LAT = np.meshgrid(lon, lat)
    print(f"Grid {N} x {N}: lat {lat_.south:.6f}..{lat_.north:.6f}, lon {lat_.west:.6f}..{lat_.east:.6f}; "
          f"existing region at offset ({OFFSET}, {OFFSET})")
    lon0, lat0 = lat_.west, lat_.north      # projection origin (any fixed point works)

    arrays = {"lat": LAT, "lon": LON}

    # --- elevation, land cover
    env = rasterio.Env(GDAL_HTTP_USERAGENT=UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env:
        print("Copernicus DEM GLO-30")
        dem, dem_min_px, dem_tiles = read_dem(lat_)
        print("ESA WorldCover 2021 v200")
        lc, lc_px_min, lc_px_max = read_landcover(lat_)
    arrays["dem_m"] = dem
    arrays.update(lc)

    # --- park
    print("Park boundary (OSM relation 2971358)")
    polys = fk.load_boundary()
    inside = fk.inside_fraction(grid, polys)
    arrays["inside_park"] = inside
    rings = [r for p in polys for r in p]
    A, B = build_segments([project(r[:, 0], r[:, 1], lon0, lat0) for r in rings])
    q = project(LON.ravel(), LAT.ravel(), lon0, lat0)
    arrays["dist_park_edge_m"] = distance_to_lines(q, A, B).reshape(N, N).astype(np.float32)

    # --- fires
    fires, fire_stats = fk.load_fires(grid)
    years = np.array(fk.YEARS, np.int16)
    fire_by_year = np.zeros((len(years), N, N), np.int16)
    for k, y in enumerate(fk.YEARS):
        r, c, ok = fk.cell_index(fires[y][0], fires[y][1], grid)
        np.add.at(fire_by_year[k], (r[ok], c[ok]), 1)
    arrays["fire_by_year"] = fire_by_year
    arrays["years"] = years

    # --- OSM
    bm = OSM_MARGIN_M / fk.M_LAT_DEG
    bn = OSM_MARGIN_M / fk.M_LON_DEG
    bbox = (lat_.south - bm, lat_.west - bn, lat_.north + bm, lat_.east + bn)
    print(f"OpenStreetMap via Overpass, bbox S{bbox[0]:.4f} W{bbox[1]:.4f} N{bbox[2]:.4f} E{bbox[3]:.4f}")
    osm_info, osm_js, failed = {}, {}, []
    for name, (sel, out) in OSM_CLASSES.items():
        js, info = overpass_fetch(name, sel, out, bbox)
        osm_info[name] = info
        if js is None:
            failed.append(name)
        else:
            osm_js[name] = js
    if "roads_major" in osm_js:
        lines_major = way_lines(osm_js["roads_major"], lon0, lat0)
        Am, Bm = build_segments(lines_major)
        arrays["dist_major_road_m"] = distance_to_lines(q, Am, Bm).reshape(N, N).astype(np.float32)
        osm_info["roads_major"]["ways_used"] = len(lines_major)
    if all(c in osm_js for c in ("roads_major", "roads_minor", "roads_track")):
        lines_all = []
        for c in ("roads_major", "roads_minor", "roads_track"):
            ln = way_lines(osm_js[c], lon0, lat0)
            osm_info[c]["ways_used"] = len(ln)
            lines_all += ln
        Aa, Ba = build_segments(lines_all)
        d_all = distance_to_lines(q, Aa, Ba).reshape(N, N)
        # the major roads are a subset of all roads, so the all-roads distance cannot exceed it
        arrays["dist_road_m"] = np.minimum(d_all, arrays["dist_major_road_m"]).astype(np.float32)
        osm_info["roads_all_segments"] = int(len(Aa))
    if "places" in osm_js:
        nodes = [e for e in osm_js["places"]["elements"] if e.get("type") == "node"]
        pts = project([e["lon"] for e in nodes], [e["lat"] for e in nodes], lon0, lat0)
        arrays["dist_settlement_m"] = distance_to_points(q, pts).reshape(N, N).astype(np.float32)
        kinds = {}
        for e in nodes:
            k = e.get("tags", {}).get("place", "?")
            kinds[k] = kinds.get(k, 0) + 1
        osm_info["places"]["nodes_used"] = len(nodes)
        osm_info["places"]["by_type"] = kinds

    # --- checks against the existing extract
    o = OFFSET
    sl = (slice(o, o + OLD_N), slice(o, o + OLD_N))
    sub = {}
    for k in ["dem_m"] + [n for n, _ in fk.LC_CLASSES] + ["lc_other", "inside_park"]:
        d = np.abs(arrays[k][sl].astype(np.float64) - old[k].astype(np.float64))
        sub[k] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
    lat_diff = float(np.abs(LAT[sl] - old["lat"]).max())
    lon_diff = float(np.abs(LON[sl] - old["lon"]).max())
    fire_sub = fire_by_year[(slice(None),) + sl].astype(np.int64)
    fire_equal = bool(np.array_equal(fire_sub.sum(0), old["fire_count"]))
    years_equal = bool(np.array_equal((fire_sub > 0).sum(0), old["fire_years"]))
    expect = [30, 29, 82, 24, 223, 13, 4, 26, 416, 5, 3, 152, 7, 3]
    per_year_sub = [int(x) for x in fire_sub.sum((1, 2))]
    ins = arrays["inside_park"] > 0.5
    checks = {
        "sub_block_offset_row_col": [o, o],
        "sub_block_lat_max_abs_diff_deg": lat_diff,
        "sub_block_lon_max_abs_diff_deg": lon_diff,
        "sub_block_value_agreement": sub,
        "fire_sub_block_sum_equals_existing_fire_count": fire_equal,
        "fire_years_sub_block_equals_existing_fire_years": years_equal,
        "fire_sub_block_per_year": dict(zip(fk.YEARS, per_year_sub)),
        "fire_sub_block_per_year_expected": dict(zip(fk.YEARS, expect)),
        "fire_sub_block_per_year_matches": per_year_sub == expect,
        "fire_total_grid_per_year": dict(zip(fk.YEARS, [int(x) for x in fire_by_year.sum((1, 2))])),
        "fire_total_inside_park_cells_per_year": dict(
            zip(fk.YEARS, [int(x) for x in fire_by_year[:, ins].sum(1)])),
        "fire_max_per_cell_year": int(fire_by_year.max()),
        "lc_fraction_sum_min_max": [float(sum(lc[k].astype(np.float64) for k in lc).min()),
                                    float(sum(lc[k].astype(np.float64) for k in lc).max())],
        "dem_min_pixels_per_cell": dem_min_px,
        "lc_pixels_per_cell_min_max": [lc_px_min, lc_px_max],
        "cells_inside_park_gt_0.5": int(ins.sum()),
    }
    for k, v in arrays.items():
        if k.startswith("dist_"):
            checks[f"{k}_finite_nonneg"] = bool(np.isfinite(v).all() and (v >= 0).all())

    fk.write_npz(OUT_NPZ, arrays)

    meta = {
        "description": ("500 m cell grid, 200 x 200 cells (about 100 km x 100 km), centred on and "
                        "lattice-aligned with the 40 x 40 Khao Yai extract, for fitting a wildfire "
                        "hazard model. Only derived aggregates are stored."),
        "build_date": english_date(datetime.date.today()),
        "grid_shape": [N, N],
        "orientation": "row 0 is the north edge, column 0 is the west edge; lat/lon are cell centres (2-D arrays)",
        "cell_size_m": fk.CELL_M,
        "cell_size_deg": {"lat": fk.DLAT, "lon": fk.DLON, "metres_per_deg_lat": fk.M_LAT_DEG,
                          "metres_per_deg_lon": fk.M_LON_DEG,
                          "note": "same lattice as khao_yai_meta.json"},
        "bounds_deg": {"north": lat_.north, "south": lat_.south, "west": lat_.west, "east": lat_.east},
        "existing_region_offset_row_col": [o, o],
        "existing_region_shape": [OLD_N, OLD_N],
        "arrays": {
            "lat, lon": "cell-centre coordinates, float64, shape (200, 200)",
            "dem_m": "mean Copernicus DEM GLO-30 elevation (m) of pixels whose centre is in the cell",
            "lc_tree ... lc_wetland, lc_other": "share of WorldCover 10 m pixels per class (full resolution)",
            "inside_park": "share of a 5 x 5 point grid per cell inside the park polygon (even-odd)",
            "dist_park_edge_m": "metres from cell centre to nearest point of any park boundary ring; positive on both sides, inside_park says which",
            "fire_by_year": "int16 (14, 200, 200) FIRMS VIIRS S-NPP detections per cell and year, confidence n/h, type 0",
            "years": "2012 to 2025",
            "dist_road_m": "metres to nearest OSM way tagged highway = motorway, trunk, primary, secondary, tertiary, unclassified, residential or track (link ramps excluded)",
            "dist_major_road_m": "same, motorway to tertiary only",
            "dist_settlement_m": "metres to nearest OSM node tagged place = city, town, village or hamlet",
        },
        "distance_method": (f"local equirectangular metres (same metres per degree as the cell lattice); "
                            f"segments sampled every <= {DENSIFY_M:.0f} m into a KD-tree, then the exact "
                            f"point-to-segment distance over the 8 nearest samples' segments; error well under 1 m against brute force on a sample of cells. "
                            f"Scale error of the single-latitude projection is about 0.2 percent at the north and south edges."),
        "osm_bbox_deg_south_west_north_east": list(bbox),
        "osm_margin_m": OSM_MARGIN_M,
        "osm_note": "Distances to features further than the 10 km margin beyond the grid are not seen; this only affects the outer rim.",
        "osm": osm_info,
        "osm_failed_classes": failed,
        "sources": {
            "elevation": {"name": "Copernicus DEM GLO-30 (DSM, COG, 1 arc-second)", "tiles": dem_tiles,
                          "licence": "Copernicus DEM licence (free of charge, incl. commercial use, with attribution)",
                          "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
                          "download_date": english_date(datetime.date.today()),
                          "resolution": "full 1 arc-second (about 30 m)"},
            "landcover": {"name": "ESA WorldCover 10 m 2021 v200", "url": fk.LC_URL, "licence": "CC BY 4.0",
                          "attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
                          "download_date": english_date(datetime.date.today()),
                          "resolution": "full 10 m"},
            "fire": {"name": "NASA FIRMS VIIRS S-NPP 375 m, yearly Thailand CSVs, 2012 to 2025",
                     "url": fk.FIRMS_URL, "licence": "NASA open data policy; free to use with acknowledgement",
                     "filter": "confidence in {n, h} and type == 0 (same load_fires as fetch_khao_yai.py)",
                     "per_year_file_stats_in_grid_extent": fire_stats,
                     "files_downloaded": "12 November 2026 (cached in data_cache/firms)"},
            "park_boundary": {"name": "OpenStreetMap relation 2971358", "url": fk.OSM_URL,
                              "licence": "ODbL 1.0", "attribution": "© OpenStreetMap contributors",
                              "download_date": "12 November 2026 (cached in data_cache)"},
            "osm_roads_places": {"name": "OpenStreetMap highways and place nodes via the Overpass API",
                                 "endpoints_tried": OVERPASS_MIRRORS,
                                 "licence": "ODbL 1.0", "attribution": "© OpenStreetMap contributors",
                                 "licence_url": "https://www.openstreetmap.org/copyright"},
        },
        "array_stats": {k: stats(v) for k, v in arrays.items() if k not in ("years",)},
        "checks": checks,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Wrote {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1e6:.2f} MB) and {OUT_META.name}; "
          f"{time.time() - t0:.0f} s")
    if failed:
        print("WARNING: OSM classes failed:", failed)


if __name__ == "__main__":
    main()
