"""
Step E: Visualize priority heatmap, candidate sites, chosen sensors, and viewsheds.

Produces:
  - static matplotlib figure (placement_map.png)
  - interactive folium HTML map (placement_map.html) with real Khao Yai coordinates
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers the 3-D projection


# Khao Yai approximate bounding box
KY_LAT_MIN, KY_LAT_MAX = 14.25, 14.55
KY_LON_MIN, KY_LON_MAX = 101.35, 101.55
KY_CENTER = (14.44, 101.43)


def cell_to_latlon(r, c, grid_shape):
    """Map grid (r,c) → approximate (lat, lon) within Khao Yai bounds."""
    rows, cols = grid_shape
    lat = KY_LAT_MAX - (r / rows) * (KY_LAT_MAX - KY_LAT_MIN)
    lon = KY_LON_MIN + (c / cols) * (KY_LON_MAX - KY_LON_MIN)
    return lat, lon


def plot_placement_static(
    surf: dict,
    sites: list[tuple[int, int]],
    chosen: list[int],
    viewsheds: dict[int, set[int]],
    out_path: str = "placement_map.png",
):
    grid_shape = surf["grid_shape"]
    rows, cols = grid_shape
    priority = surf["priority"]

    from matplotlib.lines import Line2D

    dem_m = surf.get("dem_m")   # may be absent in old surf dicts
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── left: priority heatmap + DEM contours + candidate sites ───────────────
    ax = axes[0]
    im = ax.imshow(priority, cmap="YlOrRd", origin="upper", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Priority (hazard × value)", fraction=0.046)

    if dem_m is not None:
        ax.contour(dem_m, levels=8, colors="k", linewidths=0.4, alpha=0.35, origin="upper")

    for i, (r, c) in enumerate(sites):
        color = "#1a6dd4" if i in chosen else "#888888"
        size = 120 if i in chosen else 40
        ax.scatter(c, r, c=color, s=size, zorder=5,
                   marker="^" if i in chosen else "o")

    ax.set_title("Priority + Elevation Contours + Placement", fontsize=10)
    ax.set_xlabel("Column (×500 m)"); ax.set_ylabel("Row (×500 m)")
    legend = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#1a6dd4",
               markersize=10, label="Chosen sensor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=6, label="Candidate site"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    # ── right: DEM hillshade + viewshed footprints ─────────────────────────────
    ax = axes[1]
    if dem_m is not None:
        ax.imshow(dem_m, cmap="terrain", origin="upper", alpha=0.55)
    else:
        ax.imshow(priority, cmap="Greys", origin="upper", alpha=0.4, vmin=0, vmax=1)

    cmap_vs = plt.colormaps["tab10"]
    for ci, site_idx in enumerate(chosen):
        vs_mask = np.zeros((rows, cols), dtype=float)
        for cell in viewsheds.get(site_idx, set()):
            r2, c2 = divmod(cell, cols)
            vs_mask[r2, c2] = 1.0
        color = cmap_vs(ci % 10)
        colored = np.zeros((rows, cols, 4))
        colored[vs_mask == 1] = (*color[:3], 0.45)
        ax.imshow(colored, origin="upper")

        r, c = sites[site_idx]
        ax.scatter(c, r, c=[color], s=160, marker="^", zorder=6,
                   edgecolors="k", linewidths=0.8)
        ax.text(c + 0.4, r - 0.5, f"S{ci+1}", color="k", fontsize=8, fontweight="bold")

    if dem_m is not None:
        ax.contour(dem_m, levels=8, colors="k", linewidths=0.4, alpha=0.4, origin="upper")

    vs_sizes = [len(viewsheds.get(s, set())) for s in chosen]
    ax.set_title(
        f"Viewshed footprints — {len(chosen)} sensors  "
        f"(med {int(np.median(vs_sizes))} cells/sensor)",
        fontsize=10,
    )
    ax.set_xlabel("Column (×500 m)"); ax.set_ylabel("Row (×500 m)")

    plt.suptitle("QPreFire — Khao Yai Sensor Placement (500 m cells, 4 km range)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    plt.close()


def plot_folium_map(
    surf: dict,
    sites: list[tuple[int, int]],
    chosen: list[int],
    viewsheds: dict[int, set[int]],
    out_path: str = "placement_map.html",
):
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError:
        print("folium not installed — skipping HTML map")
        return

    grid_shape = surf["grid_shape"]
    rows, cols = grid_shape
    priority = surf["priority"]

    m = folium.Map(location=KY_CENTER, zoom_start=11,
                   tiles="CartoDB positron")

    # Priority heatmap
    heat_data = []
    for r in range(rows):
        for c in range(cols):
            if priority[r, c] > 0.1:
                lat, lon = cell_to_latlon(r, c, grid_shape)
                heat_data.append([lat, lon, float(priority[r, c])])
    HeatMap(heat_data, radius=18, blur=15, max_zoom=13,
            gradient={"0.3": "yellow", "0.6": "orange", "1.0": "red"}).add_to(m)

    # Candidate sites
    for i, (r, c) in enumerate(sites):
        lat, lon = cell_to_latlon(r, c, grid_shape)
        if i in chosen:
            folium.Marker(
                [lat, lon],
                icon=folium.Icon(color="blue", icon="tower", prefix="fa"),
                tooltip=f"Sensor S{chosen.index(i)+1} (site {i})",
            ).add_to(m)
            # viewshed polygon (convex hull of visible cells)
            vs_latlons = []
            for cell in viewsheds.get(i, set()):
                vr, vc = divmod(cell, cols)
                vs_latlons.append(cell_to_latlon(vr, vc, grid_shape))
            if vs_latlons:
                folium.PolyLine(vs_latlons[:50], color="blue", weight=0.3, opacity=0.3).add_to(m)
        else:
            folium.CircleMarker(
                [lat, lon], radius=3, color="gray", fill=True, fill_opacity=0.5,
                tooltip=f"Candidate {i}",
            ).add_to(m)

    # Khao Yai boundary (approximate rectangle)
    folium.Rectangle(
        bounds=[[KY_LAT_MIN, KY_LON_MIN], [KY_LAT_MAX, KY_LON_MAX]],
        color="green", weight=2, fill=False, tooltip="Khao Yai NP (approx)"
    ).add_to(m)

    m.save(out_path)
    print(f"Saved {out_path}")


def plot_terrain_slope(surf: dict, out_path: str = "terrain_slope.png"):
    """
    Two-panel figure:
      Left  — 3-D surface of the DEM (elevation in metres)
      Right — slope map in degrees  [slope = atan(|∇DEM|) using 500 m cells]

    The slope map shows WHY viewsheds are irregular: ridgelines are high-slope
    lines that block line of sight.
    """
    dem_m     = surf["dem_m"]
    slope_deg = surf["slope_deg"]
    rows, cols = dem_m.shape
    cell_size  = 500.0   # metres

    fig = plt.figure(figsize=(14, 5))

    # ── Left: 3-D terrain surface ──────────────────────────────────────────────
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    x_km = np.arange(cols) * cell_size / 1000   # east  (km)
    y_km = np.arange(rows) * cell_size / 1000   # north (km)
    xx, yy = np.meshgrid(x_km, y_km)
    ax3d.plot_surface(
        xx, yy, dem_m,
        cmap="terrain", rstride=1, cstride=1,
        linewidth=0, antialiased=True, alpha=0.92,
    )
    ax3d.set_xlabel("East (km)", fontsize=8, labelpad=4)
    ax3d.set_ylabel("North (km)", fontsize=8, labelpad=4)
    ax3d.set_zlabel("Elevation (m)", fontsize=8, labelpad=4)
    ax3d.set_title("DEM — elevation", fontsize=10)
    ax3d.view_init(elev=30, azim=-120)
    ax3d.tick_params(labelsize=7)

    # ── Right: slope map ──────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(slope_deg, cmap="hot_r", origin="upper",
                    vmin=0, vmax=max(slope_deg.max(), 30))
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, label="Slope (°)")
    cb.ax.tick_params(labelsize=8)

    # Overlay elevation contours so ridgelines pop
    ax2.contour(dem_m, levels=10, colors="steelblue", linewidths=0.6,
                alpha=0.6, origin="upper")

    ax2.set_title("Slope map (° from horizontal) with elevation contours", fontsize=10)
    ax2.set_xlabel("Column (×500 m)", fontsize=9)
    ax2.set_ylabel("Row (×500 m)", fontsize=9)

    # Annotate steep zones
    steep = slope_deg > (np.percentile(slope_deg, 85))
    ax2.contourf(steep.astype(float), levels=[0.5, 1.5],
                 colors=["none", "#ff000033"])
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="#ff000055", label="Top-15% slope (ridgelines)")],
               fontsize=8, loc="lower right")

    fig.suptitle("Khao Yai — terrain & slope  (cell size 500 m)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from priority_surface import build_priority_surface
    from viewshed import generate_candidate_sites, compute_viewsheds, MAX_RANGE_CELLS
    from solvers.greedy import greedy_coverage

    surf = build_priority_surface()
    sites = generate_candidate_sites(
        (20, 20), n_candidates=16,
        priority=surf["priority"], dem_norm=surf["dem"],
    )
    vs = compute_viewsheds(surf["dem_m"], sites)
    weights = {r * 20 + c: float(surf["priority"][r, c]) for r in range(20) for c in range(20)}
    chosen, _ = greedy_coverage(vs, weights, n_sensors=4)

    out = os.path.dirname(os.path.abspath(__file__)) + "/.."
    plot_placement_static(surf, sites, chosen, vs, out_path=f"{out}/placement_map.png")
    plot_terrain_slope(surf, out_path=f"{out}/terrain_slope.png")
    plot_folium_map(surf, sites, chosen, vs, out_path=f"{out}/placement_map.html")
