#!/usr/bin/env python
"""Export the native-resolution Copernicus DEM GLO-30 window used by the dashboard's 3D view.

Reads `bounds` from dashboards/public/results.json, reads the DEM pixels covering those
bounds (no resampling, pixels rounded outward) and writes next to it:
    terrain.bin    uint16 little-endian, row-major, row 0 = north, column 0 = west,
                   elevation in decimetres above sea level
    terrain.json   shape, units, elevation range and the pixel block's outer edges

Needs rasterio and network access (public bucket, no key).
Usage: python data/export_terrain_mesh.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetch_khao_yai as fky  # noqa: E402

PUBLIC = HERE.parent / "dashboards" / "public"
MAX_DM = 65535


def main():
    b = json.loads((PUBLIC / "results.json").read_text())["bounds"]
    import rasterio
    env = rasterio.Env(GDAL_HTTP_USERAGENT=fky.UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env, fky.open_raster(fky.DEM_URL, "Copernicus_DEM_N14_E101.tif") as src:
        arr, lons, lats = fky.read_window(src, b["west"], b["east"], b["south"], b["north"])
        nodata = src.nodata
        t = src.transform
    if not (t.a > 0 and t.e < 0):
        fky.die("unexpected raster orientation (expected north-up)")
    if nodata is not None and np.any(arr == nodata):
        fky.die("DEM window contains nodata pixels")
    if not np.all(np.isfinite(arr)):
        fky.die("DEM window contains non-finite values")
    dm = np.rint(arr.astype(np.float64) * 10.0)
    if dm.min() < 0 or dm.max() > MAX_DM:
        fky.die(f"elevation outside 0..6553.5 m: {dm.min() / 10} .. {dm.max() / 10}")

    rows, cols = dm.shape
    west = float(lons[0] - t.a / 2)
    east = float(lons[-1] + t.a / 2)
    north = float(lats[0] - t.e / 2)
    south = float(lats[-1] + t.e / 2)
    meta = {
        "rows": rows, "cols": cols, "unit": "decimetres", "dtype": "uint16-le",
        "min_m": float(dm.min() / 10), "max_m": float(dm.max() / 10),
        "bounds": {"north": north, "south": south, "west": west, "east": east},
        "width_m": round((east - west) * fky.M_LON_DEG, 1),
        "height_m": round((north - south) * fky.M_LAT_DEG, 1),
        "source": "Copernicus DEM GLO-30",
        "generated": date.today().isoformat(),
    }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "terrain.bin").write_bytes(dm.astype("<u2").tobytes())
    (PUBLIC / "terrain.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    size = (PUBLIC / "terrain.bin").stat().st_size
    print(f"rows {rows}  cols {cols}  min {meta['min_m']} m  max {meta['max_m']} m  "
          f"terrain.bin {size} bytes")


if __name__ == "__main__":
    main()
