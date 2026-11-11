# Data sources for data/khao_yai_500m.npz

Only derived 500 m cell aggregates are stored in this repository, not the source data. All sources were accessed on 12 November 2026 without accounts or API keys. `fetch_khao_yai.py` regenerates the extract; the main pipeline reads only the npz.

## Elevation: Copernicus DEM GLO-30

Tile N14_00_E101_00 (1 arc-second, about 30 m), read as a window over HTTP. Stored: mean and maximum pixel elevation per 500 m cell (`dem_m`, `dem_max_m`).
Source: https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/Copernicus_DSM_COG_10_N14_00_E101_00_DEM/Copernicus_DSM_COG_10_N14_00_E101_00_DEM.tif
Licence: Copernicus DEM licence (free, including commercial use, with attribution).
Attribution: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved

## Fire detections: NASA FIRMS

VIIRS S-NPP 375 m active fire detections, yearly Thailand CSVs for 2012 to 2025. Kept: confidence `n` or `h` and `type` == 0 (presumed vegetation fire). Stored: detection count per cell (`fire_count`) and number of distinct years with a detection (`fire_years`).
Source: https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/YEAR/viirs-snpp_YEAR_Thailand.csv
Licence: NASA open data policy; free to use with acknowledgement.
Attribution: We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Earth Science Data and Information System (ESDIS).

## Land cover: ESA WorldCover 2021 v200

10 m land-cover map, tile N12E099, read as a window over HTTP. Stored: share of pixels per class in each cell (tree, shrub, grass, crop, built, bare, water, wetland, other).
Source: https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N12E099_Map.tif
Licence: CC BY 4.0.
Attribution: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium
Citation: Zanaga et al. 2022, doi 10.5281/zenodo.7254221

## Park boundary: OpenStreetMap

Khao Yai National Park, relation 2971358, obtained as GeoJSON from polygons.openstreetmap.fr. Stored: share of each cell inside the polygon (`inside_park`, from a 5 x 5 point grid per cell).
Source: https://polygons.openstreetmap.fr/get_geojson.py?id=2971358&params=0
Licence: ODbL 1.0, https://www.openstreetmap.org/copyright
Attribution: © OpenStreetMap contributors
