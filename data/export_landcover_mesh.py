#!/usr/bin/env python
"""Export ESA WorldCover 2021 land cover on the same pixel grid as the 3D terrain file.

Reads rows, cols and the pixel block's outer edges from dashboards/public/terrain.json, reads the
10 m WorldCover pixels covering that block and assigns each one, by its centre, to the terrain
pixel it falls in (a terrain pixel is 1/3600 degree, so it holds 3 or 4 WorldCover pixels per
axis). Writes next to terrain.bin:
    landcover.bin    uint8, row-major, row 0 = north, column 0 = west:
                     class[rows*cols] (majority class of the 10 m pixels, ties go to the lowest
                     code) followed by tree_fraction[rows*cols] (share of class 10, 0..255)
    landcover.json   layout, classes present with their share of the window, source, date

Needs rasterio and network access (public bucket, no key).
Usage: python data/export_landcover_mesh.py
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
CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
NAMES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
         60: "bare / sparse vegetation", 70: "snow and ice", 80: "permanent water body",
         90: "herbaceous wetland", 95: "mangroves", 100: "moss and lichen"}
TREE = 10
MAX_NODATA = 0.001   # share of 10 m pixels with value 0 tolerated in the window


def main():
    terrain = json.loads((PUBLIC / "terrain.json").read_text())
    rows, cols, b = terrain["rows"], terrain["cols"], terrain["bounds"]
    import rasterio
    env = rasterio.Env(GDAL_HTTP_USERAGENT=fky.UA, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                       CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with env, fky.open_raster(fky.LC_URL, "ESA_WorldCover_N12E099.tif") as src:
        arr, lons, lats = fky.read_window(src, b["west"], b["east"], b["south"], b["north"])
        t = src.transform
    if not (t.a > 0 and t.e < 0):
        fky.die("unexpected raster orientation (expected north-up)")

    # Terrain pixel of every 10 m pixel, by its centre; centres outside the block are dropped.
    ri = np.floor((b["north"] - lats) / (b["north"] - b["south"]) * rows).astype(np.int64)
    ci = np.floor((lons - b["west"]) / (b["east"] - b["west"]) * cols).astype(np.int64)
    rv = (ri >= 0) & (ri < rows)
    cv = (ci >= 0) & (ci < cols)
    sub = arr[np.ix_(rv, cv)]
    if not (rv.sum() >= rows and cv.sum() >= cols):
        fky.die("WorldCover window does not cover the terrain block")
    rr, cc = np.meshgrid(ri[rv], ci[cv], indexing="ij")
    flat = (rr * cols + cc).ravel()

    nodata_share = float((sub == 0).mean())
    if nodata_share > MAX_NODATA:
        fky.die(f"WorldCover nodata share {nodata_share:.4%} exceeds {MAX_NODATA:.2%}")
    lut = np.full(256, -1, np.int64)
    for k, code in enumerate(CODES):
        lut[code] = k
    lut[0] = len(CODES)                        # nodata slot
    slot = lut[sub.ravel()]
    if np.any(slot < 0):
        fky.die(f"unexpected WorldCover codes: {sorted(set(np.unique(sub)) - set(CODES) - {0})}")
    n_slots = len(CODES) + 1
    counts = np.bincount(flat * n_slots + slot, minlength=rows * cols * n_slots)
    counts = counts.reshape(rows * cols, n_slots)
    total = counts.sum(1)
    if total.min() < 4:
        fky.die(f"terrain pixel with only {total.min()} WorldCover pixels (shape mismatch)")
    mode = counts.argmax(1)
    if np.any(mode == len(CODES)):
        fky.die("terrain pixel whose majority is nodata")
    cls = np.array(CODES, np.uint8)[mode]
    tree = np.rint(counts[:, CODES.index(TREE)] / total * 255).astype(np.uint8)

    present = {}
    for code in CODES:
        share = float((cls == code).mean())
        if share > 0:
            present[str(code)] = {"name": NAMES[code], "share": round(share, 5)}
    meta = {
        "rows": rows, "cols": cols,
        "layout": "class[rows*cols] then tree_fraction[rows*cols], uint8, row-major, row 0 north",
        "classes": present,
        "source": "ESA WorldCover 2021 v200",
        "generated": date.today().isoformat(),
    }
    (PUBLIC / "landcover.bin").write_bytes(cls.tobytes() + tree.tobytes())
    (PUBLIC / "landcover.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    size = (PUBLIC / "landcover.bin").stat().st_size
    print(f"rows {rows}  cols {cols}  10 m pixels per terrain pixel {total.min()}..{total.max()}  "
          f"nodata {nodata_share:.4%}  landcover.bin {size} bytes")
    for code, v in present.items():
        print(f"  {code:>3} {v['name']:<26} {v['share']:.2%}")
    print(f"  mean tree fraction {tree.mean() / 255:.3f}")


if __name__ == "__main__":
    main()
