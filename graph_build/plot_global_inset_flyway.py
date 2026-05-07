#!/usr/bin/env python3
"""
Flyway / node visualization (Natural Earth 110m + optional layouts).

1) world_overview_robinson_lon90w.png  
   Global Robinson projection (central meridian 90°W, Americas centered); boxes for B1/B2.

2) detail_B1_*.png / detail_B2_*.png  
   Standalone WGS84 zoom panels for layout software.

3) global_inset_flyway.png  
   Americas WGS84 context + two zoom panels + rectangle/link overlays.

Default Natural Earth path: `<workspace>/110m_cultural/` (often sibling to the FlywayBench repo).

Robinson projection is software-defined (PROJ: `+proj=robin +lon_0=-90`), not a separate download.
Prefer **Cartopy** (`pip install cartopy`) over raw GeoPandas `to_crs` for smoother world maps.

Optional smoother land: Natural Earth Physical `ne_110m_land` next to `110m_cultural` (see `--land-shp`).

Examples:
  python3 plot_global_inset_flyway.py
  python3 plot_global_inset_flyway.py --countries /path/to/ne_110m_admin_0_countries.shp
  python3 plot_global_inset_flyway.py --skip-combined
  python3 plot_global_inset_flyway.py --skip-world --skip-details
  python3 plot_global_inset_flyway.py --land-shp /path/to/ne_110m_land.shp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch, Rectangle
from shapely.geometry import box

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE.parent.parent  # FlywayBench root’s parent (sibling data dirs)
FIG = BASE / "figures"
DEFAULT_NODES = BASE / "node_features_imputed_4554.csv"
DEFAULT_Y = BASE / "Y_binary_4554x155.csv"
DEFAULT_FLYWAY = DATA_ROOT / "birdlife_americas_flyway" / "americas_flyway_subregions.geojson"
DEFAULT_NE_DIR = DATA_ROOT / "110m_cultural"

# Robinson, central meridian 90°W (Americas-centered)
ROBIN_LON0 = "-90"
ROBIN_CRS = f"+proj=robin +lon_0={ROBIN_LON0} +datum=WGS84 +units=m +no_defs"

ZOOM_BOXES: list[tuple[str, str, tuple[float, float], tuple[float, float]]] = [
    # Two zoom windows ~24° lon × 15° lat
    ("B1", "B1 Great Lakes hotspot", (-96.0, -72.0), (40.0, 55.0)),
    ("B2", "B2 Brazilian Amazon sparse area", (-74.0, -50.0), (-12.0, 3.0)),
]

# High-contrast zoom-box colors (avoid inferno/viridis hues)
BOX_EDGE = "#00E5FF"
BOX_LABEL_BG = "#005F73"
LINK_COLOR = "#00ACC1"


def load_flyway_wgs84(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(3857)
    return g.to_crs(4326)


def resolve_countries_shp(countries_arg: Path | None) -> Path | None:
    if countries_arg is None:
        d = DEFAULT_NE_DIR
        if not d.is_dir():
            return None
        cand = list(d.glob("ne_110m_admin_0_countries.shp"))
        if not cand:
            return None
        return cand[0]
    p = countries_arg
    if p.is_dir():
        cand = list(p.glob("ne_110m_admin_0_countries.shp"))
        if not cand:
            raise SystemExit(f"No ne_110m_admin_0_countries.shp under {p}")
        p = cand[0]
    if not p.exists():
        raise SystemExit(f"Countries shapefile not found: {p}")
    return p


def load_world_wgs84(shp: Path) -> gpd.GeoDataFrame:
    w = gpd.read_file(shp)
    if w.crs is None:
        w = w.set_crs(4326)
    else:
        w = w.to_crs(4326)
    return w


def load_color_series(
    *,
    nodes_csv: Path,
    y_csv: Path,
    color: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series, str]:
    """
    Returns (lon, lat, c, color_label).
    If color != 'richness_count': take column from node_features.
    If color == 'richness_count': row-sum species columns in Y_binary (excluding cell_lat/cell_lon).
    """
    df_nodes = pd.read_csv(nodes_csv)
    lon = df_nodes["cell_lon"].to_numpy(float) + 0.5
    lat = df_nodes["cell_lat"].to_numpy(float) + 0.5
    if color != "richness_count":
        c = pd.to_numeric(df_nodes[color], errors="coerce")
        return lon, lat, c, color

    if not y_csv.exists():
        raise SystemExit(f"Y matrix not found for richness mode: {y_csv}")
    y = pd.read_csv(y_csv)
    id_cols = {"cell_lat", "cell_lon"}
    sp_cols = [c for c in y.columns if c not in id_cols]
    if not sp_cols:
        raise SystemExit("Y matrix has no species columns.")
    # Align row order with nodes; merge on coordinates if needed.
    if (
        len(y) != len(df_nodes)
        or not np.array_equal(y["cell_lat"].to_numpy(), df_nodes["cell_lat"].to_numpy())
        or not np.array_equal(y["cell_lon"].to_numpy(), df_nodes["cell_lon"].to_numpy())
    ):
        y = df_nodes[["cell_lat", "cell_lon"]].merge(y, on=["cell_lat", "cell_lon"], how="left", validate="one_to_one")
        y[sp_cols] = y[sp_cols].fillna(0)
    c = y[sp_cols].sum(axis=1).astype(float)
    return lon, lat, c, "species_richness_count"


def add_zoom_frame_and_links(
    fig: plt.Figure,
    ax_main: plt.Axes,
    ax_detail: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    edgecolor: str = BOX_EDGE,
    linecolor: str = LINK_COLOR,
    lw_rect: float = 2.2,
    lw_line: float = 1.2,
) -> None:
    w = x1 - x0
    h = y1 - y0
    rect = Rectangle(
        (x0, y0),
        w,
        h,
        fill=False,
        edgecolor=edgecolor,
        linewidth=lw_rect,
        linestyle="-",
        zorder=8,
        transform=ax_main.transData,
    )
    ax_main.add_patch(rect)
    corners_main = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    corners_detail = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for (xa, ya), (xb, yb) in zip(corners_main, corners_detail):
        con = ConnectionPatch(
            xyA=(xa, ya),
            xyB=(xb, yb),
            coordsA="data",
            coordsB="axes fraction",
            axesA=ax_main,
            axesB=ax_detail,
            color=linecolor,
            linewidth=lw_line,
            linestyle="-",
            alpha=0.85,
            zorder=6,
            clip_on=False,
        )
        fig.add_artist(con)


def plot_detail_standalone(
    out: Path,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    world: gpd.GeoDataFrame,
    fly: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    c: pd.Series,
    color_col: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    world.plot(ax=ax, color="#e8eaed", edgecolor="#9aa7b3", linewidth=0.3, zorder=0)
    fly.boundary.plot(ax=ax, color="#c1121f", linewidth=0.9, zorder=2, alpha=0.95)
    vmax = float(np.nanpercentile(c.to_numpy(dtype=float), 98))
    if vmax <= 0:
        vmax = float(np.nanmax(c.to_numpy(dtype=float)))
    sc = ax.scatter(
        lon,
        lat,
        c=c,
        cmap="inferno" if "richness" in color_col else "viridis",
        s=32,
        alpha=0.9,
        edgecolors="none",
        zorder=3,
        vmin=0,
        vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude °E")
    ax.set_ylabel("Latitude °N")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_facecolor("#ffffff")
    fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02, label=color_col)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _plot_world_robinson_geopandas_fallback(
    out: Path,
    world: gpd.GeoDataFrame,
    fly: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    c: pd.Series,
    color_col: str,
) -> None:
    """Fallback without Cartopy: GeoPandas to_crs(Robinson); may show seams/artifacts."""
    world_r = world.to_crs(ROBIN_CRS)
    fly_r = fly.to_crs(ROBIN_CRS)
    pts = gpd.GeoDataFrame({color_col: c}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
    pts_r = pts.to_crs(ROBIN_CRS)

    fig, ax = plt.subplots(figsize=(12.5, 7.0), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    world_r.plot(ax=ax, color="#e2e6ea", edgecolor="#8b95a5", linewidth=0.2, zorder=0)
    fly_r.boundary.plot(ax=ax, color="#c1121f", linewidth=0.9, zorder=2, alpha=0.95)
    vmax = float(np.nanpercentile(c.to_numpy(dtype=float), 98))
    if vmax <= 0:
        vmax = float(np.nanmax(c.to_numpy(dtype=float)))
    sc = ax.scatter(
        pts_r.geometry.x,
        pts_r.geometry.y,
        c=pts_r[color_col],
        cmap="inferno" if "richness" in color_col else "viridis",
        s=6,
        alpha=0.75,
        edgecolors="none",
        zorder=3,
        rasterized=True,
        vmin=0,
        vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
    )

    for tag, _title, (x0, x1), (y0, y1) in ZOOM_BOXES:
        b = gpd.GeoDataFrame(geometry=[box(x0, y0, x1, y1)], crs=4326).to_crs(ROBIN_CRS)
        b.boundary.plot(ax=ax, color=BOX_EDGE, linewidth=2.0, linestyle="-", zorder=9)
        cx, cy = b.geometry.iloc[0].centroid.x, b.geometry.iloc[0].centroid.y
        ax.annotate(
            tag,
            (cx, cy),
            fontsize=9,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=BOX_LABEL_BG, edgecolor="none"),
            zorder=10,
        )

    wx0, wy0, wx1, wy1 = world_r.total_bounds
    padx = (wx1 - wx0) * 0.02
    pady = (wy1 - wy0) * 0.04
    ax.set_xlim(wx0 - padx, wx1 + padx)
    ax.set_ylim(wy0 - pady, wy1 + pady)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(
        f"Global context — Robinson projection (lon₀ = {ROBIN_LON0}°), study boxes (GeoPandas fallback)",
        fontsize=12,
        pad=12,
    )
    cax = fig.add_axes([0.22, 0.06, 0.56, 0.028])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label=color_col)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_world_robinson_with_boxes(
    out: Path,
    countries_shp: Path,
    land_shp: Path | None,
    world: gpd.GeoDataFrame,
    fly: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    c: pd.Series,
    color_col: str,
) -> None:
    """Prefer Cartopy + local NE shapefiles; drop Antarctica polygon to reduce polar artifacts."""
    try:
        import cartopy.crs as ccrs
        import cartopy.io.shapereader as shpreader
    except ImportError:
        _plot_world_robinson_geopandas_fallback(out, world, fly, lon, lat, c, color_col)
        return

    fig = plt.figure(figsize=(12.5, 7.0), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-90.0))
    ax.set_facecolor("#ffffff")

    if land_shp is not None and land_shp.exists():
        lr = shpreader.Reader(str(land_shp))
        ax.add_geometries(
            lr.geometries(),
            crs=ccrs.PlateCarree(),
            facecolor="#e2e6ea",
            edgecolor="none",
            linewidth=0,
            zorder=0,
        )

    reader = shpreader.Reader(str(countries_shp))
    geoms = []
    for rec in reader.records():
        a = rec.attributes
        if a.get("ADM0_A3") == "ATA" or a.get("ADMIN") == "Antarctica":
            continue
        g = rec.geometry
        if g is None or g.is_empty:
            continue
        geoms.append(g)

    ax.add_geometries(
        geoms,
        crs=ccrs.PlateCarree(),
        facecolor="#e2e6ea",
        edgecolor="#8b95a5",
        linewidth=0.2,
        zorder=1,
    )

    for geom in fly.geometry:
        if geom is None or geom.is_empty:
            continue
        ax.add_geometries(
            [geom],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#c1121f",
            linewidth=1.05,
            zorder=3,
        )

    vmax = float(np.nanpercentile(c.to_numpy(dtype=float), 98))
    if vmax <= 0:
        vmax = float(np.nanmax(c.to_numpy(dtype=float)))
    sc = ax.scatter(
        lon,
        lat,
        c=c,
        cmap="inferno" if "richness" in color_col else "viridis",
        s=6,
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=4,
        edgecolors="none",
        rasterized=True,
        vmin=0,
        vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
    )

    pc = ccrs.PlateCarree()
    for tag, _title, (x0, x1), (y0, y1) in ZOOM_BOXES:
        b = box(x0, y0, x1, y1)
        ax.add_geometries(
            [b],
            crs=pc,
            facecolor="none",
            edgecolor=BOX_EDGE,
            linewidth=2.2,
            linestyle="-",
            zorder=10,
        )
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        ax.text(
            cx,
            cy,
            tag,
            transform=pc,
            fontsize=9,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox=dict(boxstyle="square,pad=0.25", facecolor=BOX_LABEL_BG, edgecolor="none"),
            zorder=11,
        )

    ax.set_global()
    ax.set_title(
        "Global context — Robinson (lon₀ = 90°W), Natural Earth 110m — Cartopy",
        fontsize=12,
        pad=12,
    )
    try:
        ax.spines["geo"].set_linewidth(0.2)
        ax.spines["geo"].set_edgecolor("#8b95a5")
    except Exception:
        pass

    cax = fig.add_axes([0.22, 0.06, 0.56, 0.028])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label=color_col)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_combined_americas(
    out: Path,
    world: gpd.GeoDataFrame,
    fly: gpd.GeoDataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    c: pd.Series,
    color_col: str,
) -> None:
    """Americas WGS84 map + right-column zooms + connector lines (legacy layout; NE land underneath)."""
    fig = plt.figure(figsize=(14.5, 7.8), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.35, 0.95],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.32,
        left=0.06,
        right=0.97,
        top=0.90,
        bottom=0.10,
    )
    ax_main = fig.add_subplot(gs[:, 0])
    ax_b1 = fig.add_subplot(gs[0, 1])
    ax_b2 = fig.add_subplot(gs[1, 1])

    world.plot(ax=ax_main, color="#e2e6ea", edgecolor="#9aa7b3", linewidth=0.25, zorder=0)
    fly.boundary.plot(ax=ax_main, color="#c1121f", linewidth=1.0, zorder=2, alpha=0.9)
    vmax = float(np.nanpercentile(c.to_numpy(dtype=float), 98))
    if vmax <= 0:
        vmax = float(np.nanmax(c.to_numpy(dtype=float)))
    cmap_name = "inferno" if "richness" in color_col else "viridis"
    sc = ax_main.scatter(
        lon,
        lat,
        c=c,
        cmap=cmap_name,
        s=11,
        alpha=0.86,
        edgecolors="none",
        zorder=3,
        rasterized=True,
        vmin=0,
        vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
    )

    minx, miny, maxx, maxy = fly.total_bounds
    pad = 4.0
    ax_main.set_xlim(minx - pad, maxx + pad)
    ax_main.set_ylim(miny - pad, maxy + pad)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("Longitude °E")
    ax_main.set_ylabel("Latitude °N")
    ax_main.set_title(
        "A  Americas context (WGS84) — flyway + 1° nodes",
        loc="left",
        fontsize=12,
    )
    ax_main.grid(True, alpha=0.2, linestyle=":")
    ax_main.set_facecolor("#ffffff")

    cax = fig.add_axes([0.08, 0.03, 0.48, 0.022])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label=color_col)
    cax.tick_params(labelsize=8)

    ax_list = [ax_b1, ax_b2]
    for ax_ins, (_tag, title, (x0, x1), (y0, y1)) in zip(ax_list, ZOOM_BOXES):
        world.plot(ax=ax_ins, color="#eceff2", edgecolor="#a8b0ba", linewidth=0.2, zorder=0)
        fly.boundary.plot(ax=ax_ins, color="#c1121f", linewidth=0.75, zorder=2, alpha=0.95)
        ax_ins.scatter(
            lon,
            lat,
            c=c,
            cmap=cmap_name,
            s=26,
            alpha=0.92,
            edgecolors="none",
            zorder=3,
            vmin=0,
            vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
        )
        ax_ins.set_xlim(x0, x1)
        ax_ins.set_ylim(y0, y1)
        ax_ins.set_aspect("equal")
        ax_ins.tick_params(labelsize=8)
        ax_ins.set_title(title, fontsize=10, pad=4)
        for spine in ax_ins.spines.values():
            spine.set_edgecolor(BOX_EDGE)
            spine.set_linewidth(2.2)
        ax_ins.set_facecolor("#ffffff")
        ax_ins.grid(True, alpha=0.2, linestyle=":")
        add_zoom_frame_and_links(fig, ax_main, ax_ins, x0, x1, y0, y1)

    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    ap.add_argument(
        "--y-matrix",
        type=Path,
        default=DEFAULT_Y,
        help="Y_binary_*.csv (used when --color richness_count)",
    )
    ap.add_argument("--flyway", type=Path, default=DEFAULT_FLYWAY)
    ap.add_argument(
        "--countries",
        type=Path,
        default=None,
        help=".shp path or directory containing ne_110m_admin_0_countries.shp (default: DATA_ROOT/110m_cultural)",
    )
    ap.add_argument(
        "--land-shp",
        type=Path,
        default=None,
        help="Optional ne_110m_land.shp (Physical 110m) for smoother land fill under Cartopy",
    )
    ap.add_argument("--color", default="water_frac")
    ap.add_argument("--out-suffix", default="", help="Suffix for output filenames (e.g. _richness)")
    ap.add_argument("--skip-world", action="store_true", help="Skip Robinson world overview")
    ap.add_argument("--skip-details", action="store_true", help="Skip standalone B1/B2 PNGs")
    ap.add_argument("--skip-combined", action="store_true", help="Skip global_inset_flyway composite")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)

    shp = resolve_countries_shp(args.countries)
    if shp is None:
        raise SystemExit(
            "Natural Earth ne_110m_admin_0_countries.shp not found.\n"
            f"Unzip 110m_cultural under: {DEFAULT_NE_DIR}\n"
            "Or pass: --countries /path/to/ne_110m_admin_0_countries.shp"
        )

    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    lon, lat, c, color_label = load_color_series(nodes_csv=args.nodes, y_csv=args.y_matrix, color=args.color)

    fly = load_flyway_wgs84(args.flyway)
    world = load_world_wgs84(shp)

    land_shp = args.land_shp
    if land_shp is not None and not land_shp.exists():
        raise SystemExit(f"--land-shp not found: {land_shp}")

    if not args.skip_world:
        out_w = FIG / f"world_overview_robinson_lon90w{args.out_suffix}.png"
        plot_world_robinson_with_boxes(out_w, shp, land_shp, world, fly, lon, lat, c, color_label)
        print("Wrote:", out_w.resolve())

    if not args.skip_details:
        for tag, title, (x0, x1), (y0, y1) in ZOOM_BOXES:
            slug = tag.lower().replace(" ", "_")
            outp = FIG / f"detail_{slug}_wgs84{args.out_suffix}.png"
            plot_detail_standalone(outp, title, (x0, x1), (y0, y1), world, fly, lon, lat, c, color_label)
            print("Wrote:", outp.resolve())

    if not args.skip_combined:
        out_c = FIG / f"global_inset_flyway{args.out_suffix}.png"
        plot_combined_americas(out_c, world, fly, lon, lat, c, color_label)
        print("Wrote:", out_c.resolve())


if __name__ == "__main__":
    main()
