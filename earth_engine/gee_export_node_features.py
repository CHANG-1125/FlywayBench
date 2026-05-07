#!/usr/bin/env python3
"""
Google Earth Engine: extract multi-source environmental features per 1° grid cell for 2000–2025 (no habitat masking).

Spatial reducer settings (match exported tables):
  - NDVI_mean / LST_day_mean / DEM_mean: reduceRegion(mean) on cell polygon, scale=500 m
  - water_frac: JRC water area / cell area; water sum uses reduceRegion(sum), scale=30 m

Sources:
  - NDVI: MODIS MOD13A2; LST: MOD11A2; DEM: Copernicus GLO30 (COPERNICUS/DEM/GLO30, DEM band)
  - water_frac: JRC GSW max_extent (continuous 0–1)

Before running:
  pip install earthengine-api pandas
  earthengine authenticate

Optional:
  export EARTHENGINE_PROJECT=ee-your-project-id
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import ee
import pandas as pd

BASE = Path(__file__).resolve().parent
GRID_CSV = BASE.parent / "grid_cells_active_baseline.csv"
START = "2000-01-01"
END = "2025-12-31"
# Match paper/history: NDVI/LST/DEM means at 500 m; water body sum at 30 m for coastline detail
STACK_SCALE_M = 500
WATER_SCALE_M = 30


def cell_polygon(row) -> ee.Geometry:
    lat0 = float(row.cell_lat)
    lon0 = float(row.cell_lon)
    return ee.Geometry.Rectangle([lon0, lat0, lon0 + 1.0, lat0 + 1.0], geodesic=False)


def main() -> None:
    proj = os.environ.get("EARTHENGINE_PROJECT")
    try:
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()
    except Exception as e:
        print("ee.Initialize() failed:", e, file=sys.stderr)
        print("Run: earthengine authenticate", file=sys.stderr)
        print("Or: export EARTHENGINE_PROJECT=ee-your-project-id", file=sys.stderr)
        sys.exit(1)

    if not GRID_CSV.exists():
        print("Missing:", GRID_CSV, file=sys.stderr)
        sys.exit(1)

    grids = pd.read_csv(GRID_CSV)
    features = []
    for _, r in grids.iterrows():
        geom = cell_polygon(r)
        features.append(ee.Feature(geom, {"cell_lat": int(r.cell_lat), "cell_lon": int(r.cell_lon)}))

    fc = ee.FeatureCollection(features)
    region_union = fc.geometry()

    mod13 = ee.ImageCollection("MODIS/061/MOD13A2").filterDate(START, END).filterBounds(region_union)

    def mask_mod13(img):
        qa = img.select("DetailedQA")
        ndvi = img.select("NDVI")
        return ndvi.updateMask(qa.bitwiseAnd(3).eq(0)).copyProperties(img, ["system:time_start"])

    mod13_clean = mod13.map(mask_mod13)
    ndvi_mean = mod13_clean.mean().multiply(0.0001).rename("NDVI_mean")

    mod11 = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterDate(START, END)
        .filterBounds(region_union)
        .select("LST_Day_1km")
    )

    def scale_lst(img):
        return (
            img.select("LST_Day_1km")
            .multiply(0.02)
            .subtract(273.15)
            .rename("LST_C")
        )

    lst_mean = mod11.map(scale_lst).mean().rename("LST_day_mean")

    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("DEM_mean")

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("max_extent")
    pixel_area = ee.Image.pixelArea()

    def add_water_frac(feat: ee.Feature) -> ee.Feature:
        geom = feat.geometry()
        water_area_img = jrc.gt(0).multiply(pixel_area).rename("water_area")
        summed = water_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=WATER_SCALE_M,
            maxPixels=1e13,
        )
        water_area = summed.get("water_area")
        total_area = geom.area(maxError=1)
        wf = ee.Number(water_area).divide(total_area)
        return feat.set({"water_frac": wf})

    stack = ndvi_mean.addBands(lst_mean).addBands(dem)

    def sample_stack(feat: ee.Feature) -> ee.Feature:
        geom = feat.geometry()
        stat = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=STACK_SCALE_M,
            maxPixels=1e13,
        )
        return feat.set(stat)

    fc2 = fc.map(sample_stack).map(add_water_frac)

    task = ee.batch.Export.table.toDrive(
        collection=fc2,
        description="flyway_grid_node_features_2000_2025",
        folder="ee_export_flyway",
        fileNamePrefix="grid_node_features_2000_2025",
        fileFormat="CSV",
    )
    task.start()
    print("Started Export.table.toDrive:", task.id)
    print("stack_reduceRegion_scale_m=", STACK_SCALE_M, "water_reduceRegion_scale_m=", WATER_SCALE_M)
    print("Check Earth Engine Tasks panel to download CSV when complete.")


if __name__ == "__main__":
    main()
