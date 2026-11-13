#!/usr/bin/env python
"""Export the whole-park data for the dashboard's park view into dashboards/public/.

Writes, next to the existing window files (which are not touched):
    park.json            500 m layers, placements with their covered cells, coverage curve, labels
    park_terrain.bin     Copernicus DEM GLO-30 over the whole park grid extent at about 150 m,
    park_terrain.json    same binary layout and json fields as terrain.bin / terrain.json
    park_landcover.bin   ESA WorldCover 2021 on the same 150 m grid, same layout as landcover.bin
    park_landcover.json  and landcover.json

Placements and metrics are copied from park/results.json (never recomputed); covered cells come
from the cached visibility matrix of park/instance.py. The terrain and land cover are read as
windows over HTTP from the same public COGs the other exporters use (the raw rasters are not kept
in ../data_cache); needs rasterio and network access, no key.

Usage: python data/export_park_dashboard.py [--skip-rasters]
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (HERE, REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import export_landcover_mesh as elm  # noqa: E402  (class codes, names, tree class)
import fetch_khao_yai as fky  # noqa: E402

PUBLIC = REPO / "dashboards" / "public"
RESULTS = REPO / "park" / "results.json"
PARK_META = REPO / "data" / "park_500m_meta.json"
OLD_META = REPO / "data" / "khao_yai_meta.json"
TARGET_PIXEL_M = 150.0
MAX_DM = 65535
RULES = ("coverage", "fire_history", "timed_greedy")
STRIP = 1000          # 10 m land-cover rows per read

LABELS = {
    "real": {
        "elevation": "Copernicus DEM GLO-30 (30 m), averaged to 500 m cells",
        "land_cover": "ESA WorldCover 2021 (10 m), as shares per 500 m cell",
        "fire_detections": "NASA FIRMS VIIRS S-NPP 375 m detections, 2012 to 2025, counted per 500 m cell",
        "park_boundary": "OpenStreetMap relation 2971358 (ODbL), as the share of each cell inside the park",
    },
    "modelled": {
        "priority": "hand-weighted priority surface (fire history, fuel, dryness, spread, park value), not measured",
        "visibility": "line-of-sight from a 30 m mast with a 4 km range over the DEM, no vegetation or weather",
    },
    "simulated": {
        "detection_times": "simulated fires on a simplified, uncalibrated spread model, not real detection times",
    },
}


# ---- park.json --------------------------------------------------------------
def covered_cells(sees, site_cells, sites_rc, cols):
    """Sorted flat indices of every cell seen by at least one of the sites ([row, col] pairs)."""
    flat = np.array([r * cols + c for r, c in sites_rc], dtype=np.int64)
    idx = np.searchsorted(site_cells, flat)
    if np.any(idx >= len(site_cells)) or np.any(site_cells[np.minimum(idx, len(site_cells) - 1)] != flat):
        raise ValueError("a placement site is not a candidate cell of the instance")
    mask = np.zeros(sees.shape[1], dtype=bool)
    for i in idx:
        mask[sees.indices[sees.indptr[i]:sees.indptr[i + 1]]] = True
    return np.flatnonzero(mask).tolist()


def build_park_json(inst, results, grid_meta, window_corner, window_size):
    rows, cols = inst.shape
    b = grid_meta["bounds_deg"]
    placements = {}
    for k in results["k_values"]:
        entry = results["headline"][str(k)]
        placements[str(k)] = {}
        for rule in RULES:
            if rule not in entry:
                continue
            p = entry[rule]
            placements[str(k)][rule] = {
                "sites": p["sites"],
                "covered": covered_cells(inst.sees, inst.site_cells, p["sites"], cols),
                "metrics": {key: v for key, v in p.items() if key != "sites"},
            }
    return {
        "meta": {
            "rows": rows, "cols": cols, "cell_m": inst.cell_m,
            "bounds": {"north": round(b["north"], 6), "south": round(b["south"], 6),
                       "west": round(b["west"], 6), "east": round(b["east"], 6)},
            "window": {"row0": int(window_corner[0]), "col0": int(window_corner[1]),
                       "n_rows": int(window_size[0]), "n_cols": int(window_size[1])},
            "mast_height_m": inst.params["mast_height_m"], "range_m": inst.params["max_range_m"],
            "k_values": results["k_values"],
            "generated": date.today().isoformat(),
            "generated_from": ("park/results.json (placements, metrics, coverage curve), "
                               "park/instance.py (layers, covered cells from the cached visibility), "
                               "data/park_500m.npz"),
            "note": results["note"],
            "labels": LABELS,
        },
        "layers": {
            "inside_park": np.round(inst.inside_park.ravel(), 2).tolist(),
            "priority": np.round(inst.priority.ravel(), 3).tolist(),
            "fire_count": np.asarray(inst.fire_count).ravel().astype(int).tolist(),
            "elevation_m": np.rint(inst.elevation.ravel()).astype(int).tolist(),
        },
        "placements": placements,
        "coverage_curve": results["coverage_curve"],
    }


# ---- 150 m grid, terrain and land cover -----------------------------------------
def grid_150m(bounds, target_m=TARGET_PIXEL_M):
    """rows, cols and bounds of a pixel grid over `bounds` with pixels of about `target_m`."""
    width_m = (bounds["east"] - bounds["west"]) * fky.M_LON_DEG
    height_m = (bounds["north"] - bounds["south"]) * fky.M_LAT_DEG
    return int(round(height_m / target_m)), int(round(width_m / target_m)), dict(bounds)


def _pixel_index(lats, lons, b, rows, cols):
    ri = np.floor((b["north"] - lats) / (b["north"] - b["south"]) * rows).astype(np.int64)
    ci = np.floor((lons - b["west"]) / (b["east"] - b["west"]) * cols).astype(np.int64)
    return ri, ci, (ri >= 0) & (ri < rows), (ci >= 0) & (ci < cols)


def resample_terrain(arr, lons, lats, bounds, rows, cols):
    """Mean of the DEM pixels (by centre) per output pixel, as uint16 decimetres (rows, cols)."""
    ri, ci, rv, cv = _pixel_index(lats, lons, bounds, rows, cols)
    sub = arr[np.ix_(rv, cv)].astype(np.float64)
    flat = (ri[rv][:, None] * cols + ci[cv][None, :]).ravel()
    n = rows * cols
    cnt = np.bincount(flat, minlength=n)
    if cnt.min() == 0:
        raise ValueError("output pixel without any DEM pixel")
    mean = np.bincount(flat, weights=sub.ravel(), minlength=n) / cnt
    dm = np.rint(mean * 10.0)
    if dm.min() < 0 or dm.max() > MAX_DM:
        raise ValueError(f"elevation outside 0..6553.5 m: {dm.min() / 10} .. {dm.max() / 10}")
    return dm.reshape(rows, cols).astype("<u2")


def terrain_meta(dm, bounds, source, pixel_m):
    rows, cols = dm.shape
    return {
        "rows": rows, "cols": cols, "unit": "decimetres", "dtype": "uint16-le",
        "min_m": float(dm.min() / 10), "max_m": float(dm.max() / 10),
        "bounds": dict(bounds),
        "width_m": round((bounds["east"] - bounds["west"]) * fky.M_LON_DEG, 1),
        "height_m": round((bounds["north"] - bounds["south"]) * fky.M_LAT_DEG, 1),
        "source": source,
        "pixel_m": round(pixel_m, 1),
        "resampling": "mean of the 1 arc-second pixels whose centre falls in the output pixel",
        "generated": date.today().isoformat(),
    }


N_SLOTS = len(elm.CODES) + 1      # last slot counts nodata (0)


def accumulate_landcover(counts, arr, lons, lats, bounds, rows, cols):
    """Add the class counts of a strip of 10 m pixels to counts (rows*cols, N_SLOTS)."""
    ri, ci, rv, cv = _pixel_index(lats, lons, bounds, rows, cols)
    sub = arr[np.ix_(rv, cv)]
    lut = np.full(256, -1, np.int64)
    for k, code in enumerate(elm.CODES):
        lut[code] = k
    lut[0] = len(elm.CODES)
    slot = lut[sub.ravel()]
    if np.any(slot < 0):
        raise ValueError(f"unexpected WorldCover codes: {sorted(set(np.unique(sub)) - set(elm.CODES) - {0})}")
    flat = (ri[rv][:, None] * cols + ci[cv][None, :]).ravel()
    counts += np.bincount(flat * N_SLOTS + slot, minlength=rows * cols * N_SLOTS).reshape(rows * cols, N_SLOTS)


def finish_landcover(counts, rows, cols):
    """(class uint8, tree_fraction uint8) flat arrays: majority class (ties to the lowest code)."""
    total = counts.sum(1)
    if total.min() == 0:
        raise ValueError("output pixel without any WorldCover pixel")
    mode = counts.argmax(1)
    if np.any(mode == len(elm.CODES)):
        raise ValueError("output pixel whose majority is nodata")
    cls = np.array(elm.CODES, np.uint8)[mode]
    tree = np.rint(counts[:, elm.CODES.index(elm.TREE)] / total * 255).astype(np.uint8)
    return cls, tree


def landcover_meta(cls, rows, cols, source, pixel_m):
    present = {}
    for code in elm.CODES:
        share = float((cls == code).mean())
        if share > 0:
            present[str(code)] = {"name": elm.NAMES[code], "share": round(share, 5)}
    return {
        "rows": rows, "cols": cols,
        "layout": "class[rows*cols] then tree_fraction[rows*cols], uint8, row-major, row 0 north",
        "classes": present,
        "source": source,
        "pixel_m": round(pixel_m, 1),
        "generated": date.today().isoformat(),
    }


# ---- main -------------------------------------------------------------------
def write_json(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")


def export_park_json(out_dir=PUBLIC):
    from park.instance import load_park_instance, window_cells_in_park
    inst = load_park_instance()
    results = json.loads(RESULTS.read_text())
    grid_meta = json.loads(PARK_META.read_text())
    win = json.loads(OLD_META.read_text())["default_window"]
    doc = build_park_json(inst, results, grid_meta, window_cells_in_park(),
                          (win["n_rows"], win["n_cols"]))
    write_json(out_dir / "park.json", doc)
    return out_dir / "park.json"


def export_rasters(out_dir=PUBLIC):
    import rasterio
    b = json.loads(PARK_META.read_text())["bounds_deg"]
    rows, cols, bounds = grid_150m(b)
    pixel_m = ((b["east"] - b["west"]) * fky.M_LON_DEG / cols + (b["north"] - b["south"]) * fky.M_LAT_DEG / rows) / 2
    env = rasterio.Env(GDAL_HTTP_USERAGENT=fky.UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env:
        with fky.open_raster(fky.DEM_URL, "Copernicus_DEM_N14_E101.tif") as src:
            arr, lons, lats = fky.read_window(src, b["west"], b["east"], b["south"], b["north"])
            nodata, t = src.nodata, src.transform
        if not (t.a > 0 and t.e < 0):
            fky.die("unexpected DEM orientation (expected north-up)")
        if (nodata is not None and np.any(arr == nodata)) or not np.all(np.isfinite(arr)):
            fky.die("DEM window contains nodata or non-finite pixels")
        dm = resample_terrain(arr, lons, lats, bounds, rows, cols)
        write_json(out_dir / "park_terrain.json", terrain_meta(dm, bounds, "Copernicus DEM GLO-30", pixel_m))
        (out_dir / "park_terrain.bin").write_bytes(dm.tobytes())
        print(f"terrain {rows} x {cols} = {rows * cols} vertices, {dm.min() / 10} .. {dm.max() / 10} m")
        del arr

        from rasterio.windows import Window, from_bounds
        counts = np.zeros((rows * cols, N_SLOTS), np.int64)
        with fky.open_raster(fky.LC_URL, "ESA_WorldCover_N12E099.tif") as src:
            t = src.transform
            if not (t.a > 0 and t.e < 0):
                fky.die("unexpected WorldCover orientation (expected north-up)")
            w = from_bounds(b["west"], b["south"], b["east"], b["north"], src.transform)
            r0, r1 = max(int(np.floor(w.row_off)), 0), min(int(np.ceil(w.row_off + w.height)), src.height)
            c0, c1 = max(int(np.floor(w.col_off)), 0), min(int(np.ceil(w.col_off + w.width)), src.width)
            lons = t.c + (np.arange(c0, c1) + 0.5) * t.a
            for rs in range(r0, r1, STRIP):
                re_ = min(rs + STRIP, r1)
                strip = src.read(1, window=Window(c0, rs, c1 - c0, re_ - rs))
                lats = t.f + (np.arange(rs, re_) + 0.5) * t.e
                accumulate_landcover(counts, strip, lons, lats, bounds, rows, cols)
        nodata_share = counts[:, -1].sum() / counts.sum()
        if nodata_share > elm.MAX_NODATA:
            fky.die(f"WorldCover nodata share {nodata_share:.4%} exceeds {elm.MAX_NODATA:.2%}")
        cls, tree = finish_landcover(counts, rows, cols)
        write_json(out_dir / "park_landcover.json",
                   landcover_meta(cls, rows, cols, "ESA WorldCover 2021 v200", pixel_m))
        (out_dir / "park_landcover.bin").write_bytes(cls.tobytes() + tree.tobytes())
        print(f"landcover {rows} x {cols}, 10 m pixels per output pixel {counts.sum(1).min()}.."
              f"{counts.sum(1).max()}, nodata {nodata_share:.4%}, mean tree fraction {tree.mean() / 255:.3f}")


def main():
    PUBLIC.mkdir(parents=True, exist_ok=True)
    p = export_park_json()
    print(f"{p.name} {p.stat().st_size} bytes")
    if "--skip-rasters" not in sys.argv:
        export_rasters()


if __name__ == "__main__":
    main()
