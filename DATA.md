# Data provenance and licensing

This document summarizes **which external data FlywayBench uses**, **what this repository does not redistribute**, and **how to obtain or regenerate inputs** for local runs.

FlywayBench omits large raw tables and intermediate CSVs from version control by default (see `.gitignore`). Regenerate them using the scripts in this repository.

**Disclaimer:** This is **not legal advice**. Confirm current license and terms-of-use text on each provider’s website before redistributing mirrors or derivatives.

---

## Summary

| Layer | Source | Role in FlywayBench | Redistribution notes |
|-------|--------|---------------------|----------------------|
| eBird observations | [eBird](https://ebird.org) (Cornell Lab of Ornithology) | Checklist-level records; retrieval stratified by taxonomy/year | Do **not** commit bulk raw exports unless your agreement allows it. Prefer **documented acquisition steps** and links to **official** eBird access routes. |
| Americas flyway geometry | BirdLife International (flyway boundaries) | Spatial mask and corridor topology (`americas_flyway_system.geojson` / subregions) | Use **official downloads**; keep **attribution** required by BirdLife; avoid re-hosting if terms disallow. |
| Americas Flyway System species list | BirdLife (Data Zone–style listings) | Taxonomic intersection with eBird for the 155-species modeling set | Cite BirdLife; sharing **names + methodology** is usually sufficient—confirm license on your copy before publishing full listing files. |
| WPE taxonomy | Wetlands International (*Waterbird Population Estimates*) | Taxonomic framing (e.g. threat status intersections) | **Cite WPE**; do not redistribute commercial/table extracts without permission. |
| Remote sensing covariates | [Google Earth Engine](https://earthengine.google.com/) (MODIS NDVI/LST, Copernicus DEM GLO30, JRC Global Surface Water) | Grid summaries: `NDVI_mean`, `LST_day_mean`, `DEM`, `water_frac` | Do **not** republish underlying imagery stacks as your own. Provide **`gee_export_node_features.py`** and **dataset citations** so users regenerate exports under their own GEE account. |
| Natural Earth | [Natural Earth](https://www.naturalearthdata.com/) | Optional basemap vectors (e.g. 110 m cultural) | Use per Natural Earth terms (attribution required). |

---

## Temporal coverage

Reference experiments and bundled outputs use data constrained to **2000-01-01 — 2025-12-31** (inclusive end dates where applicable). That window matches how observation strata, grid activity, and Earth Engine summaries were aligned for the published benchmark snapshot.

**Reproducers and extenders** are welcome to **refresh or lengthen** the interval—for example, pulling eBird through a later year or regenerating GEE exports with more recent imagery—provided you keep acquisition steps documented and respect each provider’s terms. Expect **row counts, train/val/test splits, and metrics** to differ from the frozen reference run whenever you change the temporal window or upstream catalogue versions.

---

## What this repository includes

- **Source code** for preprocessing, GEE export, graph construction, training, and evaluation.
- **Documentation** for column definitions, time range (2000-01-01 — 2025-12-31), \(1^\circ\) grid definition, extrapolation threshold \(\tau=-12^\circ\), and benchmark hyperparameters.
- **Small reference outputs** where permitted (e.g. bundled benchmark JSON/tables under `flywaygnn_v1/runs/paper_experiments/`).

---

## What this repository excludes by default

- Full **eBird observation dumps** at workflow scale (tens of millions of rows).
- Bulk **GEE raster stacks** or large exported CSVs **unless** you have verified redistribution rights and publish them with explicit license metadata (e.g. on Zenodo after a legal/compliance check).

Users should build intermediates by following **[`README.md`](README.md)**.

---

## Building inputs from sources

Order matters:

1. **eBird** — Acquire observations following the project protocol (scientific name × year strata; reference window **2000-01-01 — 2025-12-31**). You may use **newer end dates** or refreshed dumps for your own runs; document the window you used. Normalize columns for downstream scripts (`LATITUDE`, `LONGITUDE`, `OBSERVATION DATE`, `SCIENTIFIC NAME`, `OBSERVATION COUNT`, etc.).
2. **Taxonomic filter** — Intersect with the Americas Flyway System species universe and the 155 taxa used in the graph.
3. **Spatial clip** — Point-in-polygon against BirdLife Americas flyway polygons (WGS84). Scale after clip is on the order of **14M** records for the Americas corridor workflow (see `ebird_overlap_155_scientific_americas_clipped/README_americas_clip.md` in a full workspace checkout).
4. **Debiasing & gridding** — `run_312_313.py`: debias, assign \(1^\circ\) grid IDs, static presence \(y_{i,s}\), active grid list.
5. **GEE node features** — `earth_engine/gee_export_node_features.py`, aligned with `grid_cells_active_baseline.csv`.
6. **Graph tables** — `graph_build/prepare_node_features_and_Y.py` → `node_features_imputed_4554.csv`, `Y_binary_4554x155.csv`.
7. **Models** — `flywaygnn_v1/train.py` or `run_paper_experiments.py`.

Filenames and CLI defaults are documented in **[`README.md`](README.md)** and **[`earth_engine/README_GEE.md`](earth_engine/README_GEE.md)**.

---

## Geometry paths expected by code

| File | Consumers |
|------|-----------|
| `birdlife_americas_flyway/americas_flyway_subregions.geojson` | FlywayGNN corridor edges (`--flyway`; default relative layout in [`README.md`](README.md)) |
| `birdlife_americas_flyway/americas_flyway_system.geojson` | Documented for continental clip workflows |

Place these next to the repository root or override paths via CLI flags.

---

## Example attribution text (publications)

You may adapt the following in papers or reports (verify against final publisher and data policies):

> Observation records come from eBird under applicable terms of use; we do not redistribute raw eBird extracts in this repository. Flyway boundaries follow BirdLife International materials with citation. Environmental covariates were computed in Google Earth Engine from publicly catalogued collections; we provide scripts to reproduce exported summaries on our \(1^\circ\) grid.

---

## Contact

Questions about **this codebase**: open a **GitHub Issue**. Questions about **third-party data terms**: contact the respective data providers.
