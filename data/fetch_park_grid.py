"""Build the whole-park grid: data/park_500m.npz + park_500m_meta.json.

Covers the ENTIRE Khao Yai National Park polygon plus a margin of 10 cells (5 km) on every
side, on the same 500 m lattice as data/hazard_training_500m.npz (200 x 200) and
data/khao_yai_500m.npz (40 x 40). The 200 x 200 grid stops at longitude 101.664, so about a
quarter of the park lies outside it; this grid has no such cut-off.

Per cell: elevation (mean and max), land cover, park membership, distance to the park edge,
FIRMS detections per year (2012 to 2025). Not included: road and settlement distances (the
cached OpenStreetMap extract stops short of the park's east edge; refitting the hazard model
is out of scope here). Nothing is fitted.

Reuses data/fetch_khao_yai.py (boundary reader, point-in-polygon, FIRMS loader/filters, cell
index, npz writer, lattice constants) and data/fetch_hazard_training.py (distance helpers,
projection) by import. Re-written here because the originals hard-code a 40 or 200 cell side
or a single tile: the DEM/land-cover binning, the tile reading and the extent selection.
Raw downloads stay in ../data_cache/ (FIRMS CSVs, park boundary, binned raster cache).

Run from the repo root:  ../.venv/bin/python data/fetch_park_grid.py
"""
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_khao_yai as fk  # noqa: E402
import fetch_hazard_training as fh  # noqa: E402

HERE = fk.HERE
CACHE = fk.CACHE
OUT_NPZ = HERE / "park_500m.npz"
OUT_META = HERE / "park_500m_meta.json"
OLD40_NPZ = HERE / "khao_yai_500m.npz"
OLD200_N = fh.N                  # 200
OLD200_OFFSET = fh.OFFSET        # the 40 block sits at (80, 80) in the 200 grid
OLD40_N = fk.REGION              # 40
MARGIN_CELLS = 10                # 5 km on every side of the park polygon's bounding box
WC_TILE = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
           "ESA_WorldCover_10m_2021_v200_{ns}{lat:02d}{ew}{lon:03d}_Map.tif")


# ---- lattice and extent -------------------------------------------------------------
def lattice_200():
    """North and west edges of the 200 x 200 grid, rebuilt exactly as fetch_hazard_training does."""
    old = np.load(OLD40_NPZ)
    old_north = float(old["lat"][0, 0] + fk.DLAT / 2)
    old_west = float(old["lon"][0, 0] - fk.DLON / 2)
    return old_north + OLD200_OFFSET * fk.DLAT, old_west - OLD200_OFFSET * fk.DLON


def make_grid(polys, north200, west200):
    """Grid dict (as in fetch_khao_yai) on the 200-grid lattice: park bbox + MARGIN_CELLS."""
    pts = np.concatenate([r for p in polys for r in p])
    lon_min, lat_min = pts.min(0)
    lon_max, lat_max = pts.max(0)
    r_first = int(math.floor((north200 - lat_max) / fk.DLAT))
    r_last = int(math.floor((north200 - lat_min) / fk.DLAT))
    c_first = int(math.floor((lon_min - west200) / fk.DLON))
    c_last = int(math.floor((lon_max - west200) / fk.DLON))
    row0, col0 = r_first - MARGIN_CELLS, c_first - MARGIN_CELLS     # offsets in the 200 lattice
    nrows = r_last + MARGIN_CELLS + 1 - row0
    ncols = c_last + MARGIN_CELLS + 1 - col0
    north = north200 - row0 * fk.DLAT
    west = west200 + col0 * fk.DLON
    return {"north": north, "west": west, "nrows": nrows, "ncols": ncols,
            "south": north - nrows * fk.DLAT, "east": west + ncols * fk.DLON,
            "row0_in_200": row0, "col0_in_200": col0,
            "park_bbox_deg": {"west": float(lon_min), "east": float(lon_max),
                              "south": float(lat_min), "north": float(lat_max)}}


def polygon_area_km2(polys):
    """Park area from the polygon itself. Each polygon: first ring outer, the rest holes.
    Two metrics: 'lattice' (planar, metres per degree of the cell lattice, matching the grid's
    0.25 km2 cells) and 'local' (longitude scaled by cos(latitude) vertex by vertex)."""
    def shoelace(x, y):
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    out = {}
    for name, scale in (("lattice", lambda lat: fk.M_LON_DEG),
                        ("local", lambda lat: 111320.0 * np.cos(np.radians(lat)))):
        total = 0.0
        for poly in polys:
            for i, ring in enumerate(poly):
                x = (ring[:, 0] - ring[0, 0]) * scale(ring[:, 1])
                y = (ring[:, 1] - ring[0, 1]) * fk.M_LAT_DEG
                a = shoelace(x, y)
                total += a if i == 0 else -a
        out[name] = total / 1e6
    return out


# ---- rasters binned onto this grid ----------------------------------------------------
class Binner:
    def __init__(self, grid):
        self.g = grid
        self.nr, self.nc = grid["nrows"], grid["ncols"]

    def flat_index(self, lons, lats):
        r = np.floor((self.g["north"] - lats) / fk.DLAT).astype(np.int64)
        c = np.floor((lons - self.g["west"]) / fk.DLON).astype(np.int64)
        r = np.where((r >= 0) & (r < self.nr), r, -1)
        c = np.where((c >= 0) & (c < self.nc), c, -1)
        return np.where((r[:, None] >= 0) & (c[None, :] >= 0), r[:, None] * self.nc + c[None, :], -1)


def dem_tile_urls(grid):
    urls = []
    for ilat in range(math.floor(grid["south"]), math.floor(grid["north"]) + 1):
        for ilon in range(math.floor(grid["west"]), math.floor(grid["east"]) + 1):
            urls.append(fh.DEM_TILE.format(ns="N" if ilat >= 0 else "S", lat=abs(ilat),
                                           ew="E" if ilon >= 0 else "W", lon=abs(ilon)))
    return urls


def wc_tile_urls(grid):
    urls = []
    for ilat in range(3 * math.floor(grid["south"] / 3), math.floor(grid["north"]) + 1, 3):
        for ilon in range(3 * math.floor(grid["west"] / 3), math.floor(grid["east"]) + 1, 3):
            urls.append(WC_TILE.format(ns="N" if ilat >= 0 else "S", lat=abs(ilat),
                                       ew="E" if ilon >= 0 else "W", lon=abs(ilon)))
    return urls


def tile_bounds(src):
    b = src.bounds
    return {"west": b.left, "south": b.bottom, "east": b.right, "north": b.top}


def covers(b, grid):
    return (b["west"] <= grid["west"] and b["east"] >= grid["east"]
            and b["south"] <= grid["south"] and b["north"] >= grid["north"])


def read_dem(grid):
    nr, nc = grid["nrows"], grid["ncols"]
    n = nr * nc
    ssum, cnt = np.zeros(n), np.zeros(n, np.int64)
    mx = np.full(n, -np.inf)
    binner = Binner(grid)
    tiles = []
    for url in dem_tile_urls(grid):
        name = url.rsplit("/", 1)[1]
        with fk.open_raster(url, name) as src:
            tb = tile_bounds(src)
            arr, lons, lats = fk.read_window(src, grid["west"], grid["east"], grid["south"], grid["north"])
        print(f"  DEM {name}: {arr.shape} pixels; tile bounds {tb}")
        flat = binner.flat_index(lons, lats)
        m = flat >= 0
        idx, v = flat[m], arr[m].astype(np.float64)
        ssum += np.bincount(idx, weights=v, minlength=n)
        cnt += np.bincount(idx, minlength=n)
        np.maximum.at(mx, idx, v)
        tiles.append({"url": url, "bounds_deg": tb, "pixels_read": int(arr.size),
                      "pixels_in_grid": int(m.sum())})
    if (cnt == 0).any():
        fk.die("some cells have no DEM pixels")
    return ((ssum / cnt).reshape(nr, nc).astype(np.float32), mx.reshape(nr, nc).astype(np.float32),
            int(cnt.min()), tiles)


def read_landcover(grid, strip=1000):
    from rasterio.windows import from_bounds, Window
    nr, nc = grid["nrows"], grid["ncols"]
    ncls = len(fk.LC_CLASSES) + 1
    counts = np.zeros((nr * nc, ncls), np.int64)
    lut = np.full(256, ncls - 1, np.int64)
    for k, (_, code) in enumerate(fk.LC_CLASSES):
        lut[code] = k
    binner = Binner(grid)
    tiles = []
    for url in wc_tile_urls(grid):
        name = url.rsplit("/", 1)[1]
        with fk.open_raster(url, name) as src:
            tb = tile_bounds(src)
            w = from_bounds(grid["west"], grid["south"], grid["east"], grid["north"], src.transform)
            r0, r1 = max(int(math.floor(w.row_off)), 0), min(int(math.ceil(w.row_off + w.height)), src.height)
            c0, c1 = max(int(math.floor(w.col_off)), 0), min(int(math.ceil(w.col_off + w.width)), src.width)
            t = src.transform
            lons = t.c + (np.arange(c0, c1) + 0.5) * t.a
            used = 0
            for rs in range(r0, r1, strip):
                re = min(rs + strip, r1)
                arr = src.read(1, window=Window(c0, rs, c1 - c0, re - rs))
                lats = t.f + (np.arange(rs, re) + 0.5) * t.e
                flat = binner.flat_index(lons, lats)
                m = flat >= 0
                key = flat[m] * ncls + lut[arr[m]]
                counts += np.bincount(key, minlength=nr * nc * ncls).reshape(nr * nc, ncls)
                used += int(m.sum())
        print(f"  WorldCover {name}: {r1 - r0} x {c1 - c0} pixels read at 10 m; tile bounds {tb}")
        tiles.append({"url": url, "bounds_deg": tb, "pixels_read": int((r1 - r0) * (c1 - c0)),
                      "pixels_in_grid": used})
    tot = counts.sum(1)
    if (tot == 0).any():
        fk.die("some cells have no WorldCover pixels")
    frac = counts / tot[:, None]
    out = {name: frac[:, k].reshape(nr, nc).astype(np.float32) for k, (name, _) in enumerate(fk.LC_CLASSES)}
    out["lc_other"] = frac[:, -1].reshape(nr, nc).astype(np.float32)
    return out, int(tot.min()), int(tot.max()), tiles


def english_date(d):
    return fh.english_date(d)


# ---- main ------------------------------------------------------------------------------
def main():
    t0 = time.time()
    import rasterio
    CACHE.mkdir(parents=True, exist_ok=True)
    north200, west200 = lattice_200()
    print("Park boundary (OSM relation 2971358)")
    polys = fk.load_boundary()
    grid = make_grid(polys, north200, west200)
    nr, nc = grid["nrows"], grid["ncols"]
    row0, col0 = grid["row0_in_200"], grid["col0_in_200"]
    b40 = (OLD200_OFFSET - row0, OLD200_OFFSET - col0)      # 40 block position inside this grid
    if not (0 <= b40[0] <= nr - OLD40_N and 0 <= b40[1] <= nc - OLD40_N):
        fk.die("the 40 x 40 block is not inside the new grid")
    print(f"Grid {nr} x {nc}: offset in 200 grid (row {row0}, col {col0}); 40 block at {b40}; "
          f"lat {grid['south']:.6f}..{grid['north']:.6f}, lon {grid['west']:.6f}..{grid['east']:.6f}")

    lat = grid["north"] - (np.arange(nr) + 0.5) * fk.DLAT
    lon = grid["west"] + (np.arange(nc) + 0.5) * fk.DLON
    LON, LAT = np.meshgrid(lon, lat)
    arrays = {"lat": LAT, "lon": LON}

    # --- elevation, land cover
    env = rasterio.Env(GDAL_HTTP_USERAGENT=fk.UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env:
        print("Copernicus DEM GLO-30")
        dem, dem_max, dem_min_px, dem_tiles = read_dem(grid)
        print("ESA WorldCover 2021 v200")
        lc, lc_min, lc_max, lc_tiles = read_landcover(grid)
    for tiles in (dem_tiles, lc_tiles):
        if len(tiles) == 1 and not covers(tiles[0]["bounds_deg"], grid):
            fk.die(f"single raster tile does not cover the grid: {tiles[0]['url']}")
    arrays["dem_m"] = dem
    arrays["dem_max_m"] = dem_max
    arrays.update(lc)

    # --- park
    inside = fk.inside_fraction(grid, polys)
    arrays["inside_park"] = inside
    lon0, lat0 = west200, north200      # same projection origin as the 200 grid, so distances agree there
    rings = [r for p in polys for r in p]
    A, B = fh.build_segments([fh.project(r[:, 0], r[:, 1], lon0, lat0) for r in rings])
    q = fh.project(LON.ravel(), LAT.ravel(), lon0, lat0)
    arrays["dist_park_edge_m"] = fh.distance_to_lines(q, A, B).reshape(nr, nc).astype(np.float32)
    area = polygon_area_km2(polys)

    # --- fires
    fires, fire_stats = fk.load_fires(grid)
    years = np.array(fk.YEARS, np.int16)
    fire_by_year = np.zeros((len(years), nr, nc), np.int16)
    for k, y in enumerate(fk.YEARS):
        r, c, ok = fk.cell_index(fires[y][0], fires[y][1], grid)
        np.add.at(fire_by_year[k], (r[ok], c[ok]), 1)
    arrays["fire_by_year"] = fire_by_year
    arrays["years"] = years
    arrays["fire_count"] = fire_by_year.sum(0).astype(np.int32)
    arrays["fire_years"] = (fire_by_year > 0).sum(0).astype(np.int16)

    # --- numbers for the meta file
    ins = inside > fk.INSIDE_CELL
    east_cols = np.arange(nc) + col0 >= OLD200_N         # columns beyond the 200 grid's east edge
    per_year_in_park = [int(x) for x in fire_by_year[:, ins].sum(1)]
    lc_sum = sum(lc[k].astype(np.float64) for k in lc)
    lc_in_park = {k: round(float(v[ins].mean()), 4) for k, v in lc.items()}
    ring = np.zeros((nr, nc), bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    overlap_r = (max(0, -row0), min(nr, OLD200_N - row0))
    overlap_c = (max(0, -col0), min(nc, OLD200_N - col0))
    checks = {
        "cells_inside_park_gt_0.5": int(ins.sum()),
        "area_from_grid_km2": round(float(inside.sum()) * 0.25, 2),
        "area_from_grid_cells_gt_0.5_km2": round(float(ins.sum()) * 0.25, 2),
        "outer_ring_max_inside_park": float(inside[ring].max()),
        "lc_fraction_sum_min_max": [float(lc_sum.min()), float(lc_sum.max())],
        "lc_pixels_per_cell_min_max": [lc_min, lc_max],
        "dem_min_pixels_per_cell": dem_min_px,
        "fire_total_grid": int(fire_by_year.sum()),
        "fire_total_inside_park_cells": int(fire_by_year[:, ins].sum()),
        "fire_total_inside_park_cells_east_of_200_grid": int(fire_by_year[:, :, east_cols][:, ins[:, east_cols]].sum()),
        "fire_total_east_of_200_grid_all_cells": int(fire_by_year[:, :, east_cols].sum()),
        "fire_per_year_grid": dict(zip(fk.YEARS, [int(x) for x in fire_by_year.sum((1, 2))])),
        "fire_per_year_inside_park_cells": dict(zip(fk.YEARS, per_year_in_park)),
        "lc_shares_mean_over_inside_park_cells": lc_in_park,
        "dem_range_m_inside_park_cells": [float(dem[ins].min()), float(dem_max[ins].max())],
    }
    fk.write_npz(OUT_NPZ, arrays)

    bounds = {"north": grid["north"], "south": grid["south"], "west": grid["west"], "east": grid["east"]}
    meta = {
        "description": ("500 m cell grid covering the whole of Khao Yai National Park plus a margin of "
                        f"{MARGIN_CELLS} cells (5 km) on every side, on the same lattice as "
                        "hazard_training_500m.npz (200 x 200) and khao_yai_500m.npz (40 x 40). "
                        "Only derived aggregates are stored. No road or settlement distances."),
        "build_date": english_date(datetime.date.today()),
        "grid_shape": [nr, nc],
        "orientation": "row 0 is the north edge, column 0 is the west edge; lat/lon are cell centres (2-D arrays)",
        "cell_size_m": fk.CELL_M,
        "cell_size_deg": {"lat": fk.DLAT, "lon": fk.DLON, "metres_per_deg_lat": fk.M_LAT_DEG,
                          "metres_per_deg_lon": fk.M_LON_DEG,
                          "note": "same lattice as khao_yai_meta.json and hazard_training_meta.json"},
        "bounds_deg": bounds,
        "offset_in_hazard_training_grid_row_col": [row0, col0],
        "offset_note": ("cell (r, c) here is cell (r + row0, c + col0) of the 200 x 200 grid; columns "
                        ">= 200 - col0 lie beyond that grid's east edge"),
        "overlap_with_hazard_training_grid": {
            "rows_here": list(overlap_r), "cols_here": list(overlap_c),
            "shape": [overlap_r[1] - overlap_r[0], overlap_c[1] - overlap_c[0]]},
        "khao_yai_block_offset_row_col": list(b40),
        "khao_yai_block_shape": [OLD40_N, OLD40_N],
        "extent_rule": (f"bounding box of the OSM park polygon, rounded outward to whole cells of the "
                        f"200-grid lattice, plus {MARGIN_CELLS} cells on every side"),
        "park_polygon_bbox_deg": grid["park_bbox_deg"],
        "park_polygon_area_km2": {k: round(v, 2) for k, v in area.items()},
        "park_polygon_area_note": ("'lattice': planar shoelace with the metres per degree of the cell "
                                   "lattice (what a 0.25 km2 cell assumes); 'local': longitude scaled by "
                                   "cos(latitude) at each vertex. Outer ring minus holes."),
        "arrays": {
            "lat, lon": f"cell-centre coordinates, float64, shape ({nr}, {nc})",
            "dem_m": "mean Copernicus DEM GLO-30 elevation (m) of pixels whose centre is in the cell, float32",
            "dem_max_m": "maximum of the same pixels, float32",
            "lc_tree ... lc_wetland, lc_other": "share of WorldCover 10 m pixels per class (full resolution), float32",
            "inside_park": "share of a 5 x 5 point grid per cell inside the park polygon (even-odd), float32",
            "dist_park_edge_m": "metres from cell centre to nearest point of any park boundary ring; positive on both sides, inside_park says which; float32",
            "fire_by_year": f"int16 (14, {nr}, {nc}) FIRMS VIIRS S-NPP detections per cell and year, confidence n/h, type 0",
            "years": "int16, 2012 to 2025",
            "fire_count": "int32, sum of fire_by_year over years (same definition as khao_yai_500m.npz)",
            "fire_years": "int16, number of years with at least one detection (same as khao_yai_500m.npz)",
        },
        "not_included": ("dist_road_m, dist_major_road_m, dist_settlement_m. They are only used for "
                         "refitting the hazard model, the cached OpenStreetMap extract stops short of the "
                         "park's east edge, and re-fetching it was out of scope."),
        "sources": {
            "elevation": {"name": "Copernicus DEM GLO-30 (DSM, COG, 1 arc-second)", "tiles": dem_tiles,
                          "licence": "Copernicus DEM licence (free of charge, incl. commercial use, with attribution)",
                          "attribution": "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved",
                          "download_date": english_date(datetime.date.today()),
                          "resolution": "full 1 arc-second (about 30 m)"},
            "landcover": {"name": "ESA WorldCover 10 m 2021 v200", "tiles": lc_tiles, "licence": "CC BY 4.0",
                          "attribution": "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
                          "download_date": english_date(datetime.date.today()),
                          "resolution": "full 10 m"},
            "fire": {"name": "NASA FIRMS VIIRS S-NPP 375 m, yearly Thailand CSVs, 2012 to 2025",
                     "url": fk.FIRMS_URL, "licence": "NASA open data policy; free to use with acknowledgement",
                     "filter": "confidence in {n, h} and type == 0 (same fk.load_fires as the other extracts)",
                     "per_year_file_stats_in_grid_extent": fire_stats,
                     "files": "cached in data_cache/firms"},
            "park_boundary": {"name": "OpenStreetMap relation 2971358", "url": fk.OSM_URL,
                              "licence": "ODbL 1.0", "attribution": "© OpenStreetMap contributors"},
        },
        "array_stats": {k: fh.stats(v) for k, v in arrays.items() if k not in ("years",)},
        "checks": checks,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    OUT_META.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"Wrote {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1e6:.2f} MB) and {OUT_META.name}; "
          f"{time.time() - t0:.0f} s")
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("fire_per_year")}, indent=1))
    print("polygon area km2:", area)


if __name__ == "__main__":
    main()
