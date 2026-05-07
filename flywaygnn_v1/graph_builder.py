from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import torch


def build_spatial_edges(cell_lat: np.ndarray, cell_lon: np.ndarray) -> set[tuple[int, int]]:
    """Moore neighborhood (8-neighbor) undirected edges."""
    idx_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(cell_lat, cell_lon))}
    edges: set[tuple[int, int]] = set()
    for i, (a, b) in enumerate(zip(cell_lat, cell_lon)):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if da == 0 and db == 0:
                    continue
                j = idx_of.get((int(a + da), int(b + db)))
                if j is None or i == j:
                    continue
                u, v = (i, j) if i < j else (j, i)
                edges.add((u, v))
    return edges


def load_flyway_wgs84(path: Path) -> gpd.GeoDataFrame:
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs(3857)
    return g.to_crs(4326)


def assign_corridor_id(
    fly: gpd.GeoDataFrame,
    lon_center: np.ndarray,
    lat_center: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Assign each 1° grid center to a BirdLife flyway subregion polygon (point-in-polygon).
    If polygons overlap, first feature row in the GeoJSON wins; unassigned nodes get -1.
    """
    n = len(lon_center)
    ids = np.full(n, -1, dtype=np.int16)
    names = [str(nm) for nm in fly["name"].tolist()]
    pts = gpd.GeoSeries.from_xy(lon_center, lat_center, crs="EPSG:4326")
    for i, (_, row) in enumerate(fly.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        m = pts.within(geom).to_numpy() & (ids < 0)
        ids[m] = np.int16(i)
    return ids, names


def build_flyway_corridor_edges(
    y: np.ndarray,
    fly: gpd.GeoDataFrame,
    lon_center: np.ndarray,
    lat_center: np.ndarray,
    *,
    breeding_lat_min: float,
    winter_lat_max: float,
    max_nodes_per_side: int,
    train_row_mask: np.ndarray | None = None,
) -> set[tuple[int, int]]:
    """
    Species-aware long-range flyway edges E_f:
    1) Y_{i,s}=Y_{j,s}=1
    2) breeding-side φ_i >= φ_B, wintering-side φ_j <= φ_W (grid-center latitude)
    3) corridor(v_i)=corridor(v_j) from BirdLife americas_flyway_subregions point-in-polygon
    For each (species, corridor), keep top-K nodes per side by species richness, then connect cross pairs.

    train_row_mask: if provided, only train-supervised rows contribute labels for co-occurrence (extrapolation
    rows treated as unlabeled) to avoid label leakage into E_f during extrapolation evaluation.
    """
    _, n_species = y.shape
    if train_row_mask is not None:
        tr = train_row_mask.astype(np.float32)
        y_eff = y * tr[:, np.newaxis]
    else:
        y_eff = y
    richness = y_eff.sum(axis=1)
    corridor_id, _names = assign_corridor_id(fly, lon_center, lat_center)
    edges: set[tuple[int, int]] = set()

    for s in range(n_species):
        pos = np.where(y_eff[:, s] > 0.5)[0]
        if len(pos) < 2:
            continue
        north = pos[lat_center[pos] >= breeding_lat_min]
        south = pos[lat_center[pos] <= winter_lat_max]
        if len(north) == 0 or len(south) == 0:
            continue

        n_corridors = len(fly)
        for c in range(n_corridors):
            n_idx = [int(i) for i in north if corridor_id[i] == c]
            s_idx = [int(i) for i in south if corridor_id[i] == c]
            if len(n_idx) == 0 or len(s_idx) == 0:
                continue
            n_idx = sorted(n_idx, key=lambda i: (-richness[i], i))[:max_nodes_per_side]
            s_idx = sorted(s_idx, key=lambda i: (-richness[i], i))[:max_nodes_per_side]

            for i in n_idx:
                for j in s_idx:
                    if i == j:
                        continue
                    u, v = (i, j) if i < j else (j, i)
                    edges.add((u, v))
    return edges


def edge_set_to_undirected_index(edge_set: set[tuple[int, int]], n_nodes: int) -> torch.Tensor:
    if not edge_set:
        return torch.empty((2, 0), dtype=torch.long)
    src = []
    dst = []
    for u, v in sorted(edge_set):
        if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes:
            continue
        src.extend([u, v])
        dst.extend([v, u])
    return torch.tensor([src, dst], dtype=torch.long)
