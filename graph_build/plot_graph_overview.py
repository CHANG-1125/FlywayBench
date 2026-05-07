#!/usr/bin/env python3
"""
High-resolution graph-data overview figures under graph_build/figures/.

Basemap styles (aligned with plot_global_inset_flyway Americas panel):
  - Default --basemap ne: Natural Earth ne_110m_admin_0_countries + flyway outline in red;
    axis #f4f6f8, land #e2e6ea / borders #9aa7b3.
  - --basemap flyway_only: white background + flyway outline only.
  - --basemap none: no vector basemap (scatter + light gray background).

Requires pandas, matplotlib, numpy; ne / flyway_only need geopandas.

Examples:
  python3 plot_graph_overview.py
  python3 plot_graph_overview.py --basemap flyway_only
  python3 plot_graph_overview.py --flyway-focus
      # Nodes inside flyway union only; three large maps; filenames flyway_*.png;
      # default shifts lon >150° by −360° (Americas-centered) + tight extent from points.
  python3 plot_graph_overview.py --flyway-focus --no-amers-lon-shift
      # Keep raw WGS84 lon and flyway total_bounds (wide map).

Flyway GeoJSON + geopandas required for --isolate-points / --flyway-focus.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE.parent.parent
FIG = BASE / "figures"
X_PATH = BASE / "node_features_imputed_4554.csv"
Y_PATH = BASE / "Y_binary_4554x155.csv"
DEFAULT_NE_SHP = DATA_ROOT / "110m_cultural" / "ne_110m_admin_0_countries.shp"
DEFAULT_FLYWAY = DATA_ROOT / "birdlife_americas_flyway" / "americas_flyway_subregions.geojson"

# Colors aligned with plot_global_inset flyway Americas WGS84 panel
LAND_FACE = "#e2e6ea"
LAND_EDGE = "#9aa7b3"
LAND_LW = 0.25
AXIS_FACE_NE = "#ffffff"
AXIS_FACE_NONE = "#ffffff"
FLY_EDGE_DETAIL = "#c1121f"
FLY_EDGE_MINIMAL = "#1a1a1a"


def _centers(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (df["cell_lon"].to_numpy(dtype=float) + 0.5), (df["cell_lat"].to_numpy(dtype=float) + 0.5)


def load_flyway(path: Path) -> gpd.GeoDataFrame | None:
    if not path.exists():
        return None
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(3857)
    return g.to_crs(4326)


def load_world(shp: Path) -> gpd.GeoDataFrame | None:
    if not shp.exists():
        return None
    w = gpd.read_file(shp)
    if w.crs is None:
        w = w.set_crs(4326)
    else:
        w = w.to_crs(4326)
    return w


def filter_to_flyway_polygon(
    df: pd.DataFrame,
    Y: pd.DataFrame,
    fly: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows whose 1° grid centers fall inside the flyway union (aligned with Y)."""
    if fly is None or fly.empty:
        return df, Y
    union = unary_union(fly.geometry)
    lon = df["cell_lon"].to_numpy(dtype=float) + 0.5
    lat = df["cell_lat"].to_numpy(dtype=float) + 0.5
    pts = gpd.GeoSeries.from_xy(lon, lat, crs="EPSG:4326")
    mask = pts.within(union).to_numpy()
    n = int(mask.sum())
    if n == 0:
        raise SystemExit("No grid nodes inside flyway union: check flyway GeoJSON vs node coordinates.")
    return df.loc[mask].reset_index(drop=True), Y.loc[mask].reset_index(drop=True)


def set_extent_from_flyway(ax, fly: gpd.GeoDataFrame | None, pad_deg: float = 4.0) -> None:
    if fly is None or fly.empty:
        return
    minx, miny, maxx, maxy = fly.total_bounds
    ax.set_xlim(minx - pad_deg, maxx + pad_deg)
    ax.set_ylim(miny - pad_deg, maxy + pad_deg)


# Pacific flyway spans -180/180 in WGS84; shift longitudes for a single Americas-centered map
AMERS_LON_SHIFT_THRESHOLD = 150.0


def lon_amers_display(lon: np.ndarray, threshold: float = AMERS_LON_SHIFT_THRESHOLD) -> np.ndarray:
    x = np.asarray(lon, dtype=float)
    return np.where(x > threshold, x - 360.0, x)


def shift_lon_gdf(g: gpd.GeoDataFrame | None, threshold: float = AMERS_LON_SHIFT_THRESHOLD) -> gpd.GeoDataFrame | None:
    if g is None or g.empty:
        return g

    def _fn(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        xa = np.where(xa > threshold, xa - 360.0, xa)
        return xa, ya

    g2 = g.copy()
    g2["geometry"] = g2.geometry.map(lambda geom: shp_transform(_fn, geom) if geom is not None else None)
    return g2


def set_extent_from_points(ax, lon: np.ndarray, lat: np.ndarray, pad_deg: float = 1.0) -> None:
    ax.set_xlim(float(np.nanmin(lon)) - pad_deg, float(np.nanmax(lon)) + pad_deg)
    ax.set_ylim(float(np.nanmin(lat)) - pad_deg, float(np.nanmax(lat)) + pad_deg)


def draw_basemap(
    ax,
    world: gpd.GeoDataFrame | None,
    fly: gpd.GeoDataFrame | None,
    basemap: str,
) -> None:
    if basemap == "flyway_only":
        ax.set_facecolor("#ffffff")
        if fly is not None:
            fly.boundary.plot(
                ax=ax,
                color=FLY_EDGE_MINIMAL,
                linewidth=0.85,
                zorder=1,
            )
        return
    if basemap == "ne" and world is not None:
        ax.set_facecolor(AXIS_FACE_NE)
        world.plot(
            ax=ax,
            color=LAND_FACE,
            edgecolor=LAND_EDGE,
            linewidth=LAND_LW,
            zorder=0,
        )
        if fly is not None:
            fly.boundary.plot(
                ax=ax,
                color=FLY_EDGE_DETAIL,
                linewidth=0.95,
                zorder=1,
                alpha=0.92,
            )
        return
    ax.set_facecolor(AXIS_FACE_NONE)


def plot_maps_2x2(
    df: pd.DataFrame,
    out: Path,
    *,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    world: gpd.GeoDataFrame | None,
    fly: gpd.GeoDataFrame | None,
    basemap: str,
    figsize: tuple[float, float] = (11.5, 9.0),
    scatter_s: float = 14.0,
    save_dpi: int = 200,
    extent_mode: str = "flyway",
) -> None:
    if lon is None or lat is None:
        lon, lat = _centers(df)
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#ffffff")

    panels = [
        ("water_frac", "Water Fraction (water_frac)", "viridis", None),
        ("NDVI_mean", "Mean NDVI", "YlGn", None),
        ("LST_day_mean", "Daytime LST (°C)", "coolwarm", None),
        ("DEM", "Elevation DEM (m)", "terrain", None),
    ]

    for ax, (col, title, cmap, _) in zip(axes.flat, panels):
        draw_basemap(ax, world, fly, basemap)
        v = df[col].to_numpy()
        sc = ax.scatter(
            lon,
            lat,
            c=v,
            cmap=cmap,
            s=scatter_s,
            alpha=0.88,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18 if basemap == "ne" else 0.25, linestyle=":")
        if extent_mode == "points":
            set_extent_from_points(ax, lon, lat, pad_deg=1.2)
        else:
            set_extent_from_flyway(ax, fly)
    fig.suptitle("Americas Flyway 1° Grid Nodes — Multi-source Environmental Features (Imputed)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    specs = [
        ("NDVI_mean", "NDVI", "#2d6a4f"),
        ("LST_day_mean", "LST (°C)", "#bc4749"),
        ("DEM", "DEM (m)", "#6c584c"),
        ("water_frac", "water_frac", "#1d3557"),
    ]
    for ax, (col, label, color) in zip(axes.flat, specs):
        x = df[col].to_numpy()
        ax.hist(x, bins=48, color=color, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{label} distribution")
        ax.set_ylabel("Node count")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Node feature distributions (n = 4554)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_species_richness(
    Y: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    out: Path,
    *,
    world: gpd.GeoDataFrame | None,
    fly: gpd.GeoDataFrame | None,
    basemap: str,
    figsize: tuple[float, float] = (10.5, 7.2),
    scatter_s: float = 18.0,
    save_dpi: int = 200,
    extent_mode: str = "flyway",
) -> None:
    sp_cols = [c for c in Y.columns if c not in ("cell_lat", "cell_lon")]
    rich = Y[sp_cols].sum(axis=1).to_numpy()
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#ffffff")
    draw_basemap(ax, world, fly, basemap)
    sc = ax.scatter(
        lon,
        lat,
        c=rich,
        cmap="inferno",
        s=scatter_s,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
        zorder=3,
        norm=mcolors.Normalize(vmin=0, vmax=np.percentile(rich, 98)),
    )
    plt.colorbar(sc, ax=ax, label="Species count (y=1)")
    ax.set_title("Target species richness per cell (2000–2025)", fontsize=13)
    ax.set_xlabel("Longitude °E")
    ax.set_ylabel("Latitude °N")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18 if basemap == "ne" else 0.25, linestyle=":")
    if extent_mode == "points":
        set_extent_from_points(ax, lon, lat, pad_deg=1.2)
    else:
        set_extent_from_flyway(ax, fly)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out: Path) -> None:
    cols = ["NDVI_mean", "LST_day_mean", "DEM", "water_frac"]
    C = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    labels = ["NDVI", "LST", "DEM", "water"]
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Pearson correlation (node features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_missing_map(
    df: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    out: Path,
    *,
    world: gpd.GeoDataFrame | None,
    fly: gpd.GeoDataFrame | None,
    basemap: str,
    figsize: tuple[float, float] = (10.5, 7.2),
    scatter_s: float = 20.0,
    save_dpi: int = 200,
    extent_mode: str = "flyway",
) -> None:
    m = df["miss_NDVI"] + df["miss_LST"] + df["miss_DEM"]
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("#ffffff")
    draw_basemap(ax, world, fly, basemap)
    cmap = plt.colormaps["Reds"].resampled(4)
    sc = ax.scatter(
        lon,
        lat,
        c=m.to_numpy(),
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        s=scatter_s,
        alpha=0.92,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["0", "1", "2", "3"])
    cbar.set_label("Missing dims (NDVI/LST/DEM)")
    ax.set_title("Missing-indicator sum before imputation", fontsize=13)
    ax.set_xlabel("Longitude °E")
    ax.set_ylabel("Latitude °N")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.18 if basemap == "ne" else 0.25, linestyle=":")
    if extent_mode == "points":
        set_extent_from_points(ax, lon, lat, pad_deg=1.2)
    else:
        set_extent_from_flyway(ax, fly)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=save_dpi)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--basemap",
        choices=("ne", "flyway_only", "none"),
        default="ne",
        help="ne=Natural Earth + flyway outline; flyway_only=white + flyway outline; none=no vectors",
    )
    ap.add_argument(
        "--ne-shp",
        type=Path,
        default=DEFAULT_NE_SHP,
        help="Path to ne_110m_admin_0_countries.shp",
    )
    ap.add_argument("--flyway", type=Path, default=DEFAULT_FLYWAY, help="americas_flyway_subregions.geojson")
    ap.add_argument(
        "--isolate-points",
        action="store_true",
        help="Keep only nodes whose grid centers lie in the flyway polygon union",
    )
    ap.add_argument("--only-maps", action="store_true", help="Skip histograms and correlation matrix")
    ap.add_argument(
        "--large",
        action="store_true",
        help="Larger map canvas and higher DPI (flyway-focused posters)",
    )
    ap.add_argument(
        "--flyway-focus",
        action="store_true",
        help="Same as --isolate-points --only-maps --large; writes flyway_*.png",
    )
    ap.add_argument(
        "--no-amers-lon-shift",
        action="store_true",
        help="Disable Americas lon shift / tight extent under --flyway-focus (global extent)",
    )
    args = ap.parse_args()

    if args.flyway_focus:
        args.isolate_points = True
        args.only_maps = True
        args.large = True

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#ffffff",
        }
    )

    if not X_PATH.exists() or not Y_PATH.exists():
        raise SystemExit("Run prepare_node_features_and_Y.py first to produce CSV inputs.")

    fly = load_flyway(args.flyway)
    if fly is None:
        raise SystemExit(f"Flyway GeoJSON not found: {args.flyway}")
    world = load_world(args.ne_shp) if args.basemap == "ne" else None
    if args.basemap == "ne" and world is None:
        print("WARN: Natural Earth shapefile missing; using basemap=none. Path:", args.ne_shp)
        args.basemap = "none"

    df = pd.read_csv(X_PATH)
    Y = pd.read_csv(Y_PATH)
    if args.isolate_points:
        df, Y = filter_to_flyway_polygon(df, Y, fly)
    lon, lat = _centers(df)

    if args.flyway_focus and not args.no_amers_lon_shift:
        lon_plot = lon_amers_display(lon)
        world_disp = shift_lon_gdf(world)
        fly_disp = shift_lon_gdf(fly)
        extent_mode = "flyway"
    else:
        lon_plot = lon
        world_disp = world
        fly_disp = fly
        extent_mode = "flyway"

    basemap_plot = args.basemap
    if args.flyway_focus and basemap_plot == "ne":
        # Flyway-focused maps skip NE land fill to reduce seam artifacts after lon wrapping
        basemap_plot = "flyway_only"

    prefix = "flyway_" if args.flyway_focus else ""
    if args.large:
        fs_2x2 = (14.0, 11.0)
        s_2x2 = 18.0
        dpi_out = 240
        fs_1 = (13.0, 9.0)
        s_sp = 26.0
        s_miss = 28.0
    else:
        fs_2x2 = (11.5, 9.0)
        s_2x2 = 14.0
        dpi_out = 200
        fs_1 = (10.5, 7.2)
        s_sp = 18.0
        s_miss = 20.0

    plot_maps_2x2(
        df,
        FIG / f"{prefix}overview_maps_2x2.png",
        lon=lon_plot,
        lat=lat,
        world=world_disp,
        fly=fly_disp,
        basemap=basemap_plot,
        figsize=fs_2x2,
        scatter_s=s_2x2,
        save_dpi=dpi_out,
        extent_mode=extent_mode,
    )
    if not args.only_maps:
        plot_distributions(df, FIG / "feature_histograms.png")
    plot_species_richness(
        Y,
        lon_plot,
        lat,
        FIG / f"{prefix}species_richness_map.png",
        world=world_disp,
        fly=fly_disp,
        basemap=basemap_plot,
        figsize=fs_1,
        scatter_s=s_sp,
        save_dpi=dpi_out,
        extent_mode=extent_mode,
    )
    if not args.only_maps:
        plot_correlation(df, FIG / "feature_correlation.png")
    plot_missing_map(
        df,
        lon_plot,
        lat,
        FIG / f"{prefix}missing_dims_map.png",
        world=world_disp,
        fly=fly_disp,
        basemap=basemap_plot,
        figsize=fs_1,
        scatter_s=s_miss,
        save_dpi=dpi_out,
        extent_mode=extent_mode,
    )

    print(
        "Saved figures to:",
        FIG.resolve(),
        "| basemap =",
        basemap_plot,
        "| isolate =",
        args.isolate_points,
        "| only_maps =",
        args.only_maps,
        "| large =",
        args.large,
        "| amers_lon_shift =",
        args.flyway_focus and not args.no_amers_lon_shift,
    )
    for p in sorted(FIG.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
