"""Data-source managers for WW-DHSVM.

This module is a lightly edited fork of Watershed Workflow's
``sources`` package.  The manager architecture is unchanged -- that is
the part of Watershed Workflow most worth inheriting -- and consists of
three layers:

``Manager``
    Cache infrastructure common to everything: canonical folder layout,
    bounding-box snapping, filename generation, and *superset detection*
    so that a second request contained within a first one reuses the
    cached file instead of re-downloading.

``ManagerShapes`` / ``ManagerDataset``
    Vector and raster specializations, handling CRS normalization,
    buffering, clipping, and standard column naming.

Concrete managers
    One per data product.

Changes from Watershed Workflow
-------------------------------
* :class:`ManagerHRRR` is **new** -- NOAA's 3 km hourly HRRR, retrieved
  with Herbie, added because DHSVM is frequently applied to recent
  high-resolution event studies that a reanalysis cannot serve.
* The MODIS AppEEARS and HF-Hydrodata managers are retained but are no
  longer part of the default set: DHSVM's vegetation table is
  climatological, so a transient LAI product is optional rather than
  required as it is for ATS.
* The default meteorology is **AORC**, as in Watershed Workflow, with
  Daymet and HRRR as alternates.
* The default land cover is NLCD, the default soil source SSURGO, and
  the default DEM 3DEP at 30 m rather than 60 m -- DHSVM grids are
  typically 30--150 m, finer than the ATS meshes Watershed Workflow
  defaults were chosen for.
"""

import logging
from typing import Dict, Any

from .manager_shapefile import ManagerShapefile
from .manager_raster import ManagerRaster

from .manager_wbd import ManagerWBD
from .manager_nhd import ManagerNHD
from .manager_3dep import Manager3DEP
from .manager_nrcs import ManagerNRCS
from .manager_soilgrids import ManagerSoilGrids
from .manager_soilgrids_2017 import ManagerSoilGrids2017
from .manager_polaris import ManagerPOLARIS
from .manager_pelletier_dtb import ManagerPelletierDTB
from .manager_nlcd import ManagerNLCD
from .manager_glhymps import ManagerGLHYMPS

# Meteorology
from .manager_daymet import ManagerDaymet
from .manager_aorc import ManagerAORC
from .manager_hrrr import ManagerHRRR

# Optional / advanced
from .manager_modis_appeears import ManagerMODISAppEEARS


# ---------------------------------------------------------------------------
# Watershed boundaries
# ---------------------------------------------------------------------------
huc_sources: Dict[str, Any] = {
    'WBD': ManagerWBD('WBD'),
    'WaterData WBD': ManagerWBD('WaterData'),
}
default_huc_source = 'WBD'

# ---------------------------------------------------------------------------
# Hydrography -- used to burn the DEM and to validate the derived network,
# never to define DHSVM's channel topology directly.
# ---------------------------------------------------------------------------
hydrography_sources: Dict[str, Any] = {
    'NHDPlus MR v2.1': ManagerNHD('NHDPlus MR v2.1'),
    'NHD MR': ManagerNHD('NHD MR'),
    'NHDPlus HR': ManagerNHD('NHDPlus HR'),
}
default_hydrography_source = 'NHDPlus MR v2.1'

# ---------------------------------------------------------------------------
# Elevation -- the most important DHSVM input.
# ---------------------------------------------------------------------------
dem_sources: Dict[str, Any] = {
    '3DEP 60m': Manager3DEP(60),
    '3DEP 30m': Manager3DEP(30),
    '3DEP 10m': Manager3DEP(10),
}
default_dem_source = '3DEP 30m'

# ---------------------------------------------------------------------------
# Soil texture and depth to bedrock.
# ---------------------------------------------------------------------------
soil_sources: Dict[str, Any] = {
    'NRCS SSURGO': ManagerNRCS(),
    'SoilGrids': ManagerSoilGrids(),
    'SoilGrids2017': ManagerSoilGrids2017(),
    'POLARIS': ManagerPOLARIS(),
}
default_soil_source = 'NRCS SSURGO'

depth_to_bedrock_sources: Dict[str, Any] = {
    'Pelletier DTB': ManagerPelletierDTB(),
    'SoilGrids2017': ManagerSoilGrids2017(),
}
default_depth_to_bedrock = 'Pelletier DTB'

# ---------------------------------------------------------------------------
# Land cover.
# ---------------------------------------------------------------------------
land_cover_sources: Dict[str, Any] = {
    'NLCD (L48)': ManagerNLCD(location='L48'),
    'NLCD (AK)': ManagerNLCD(location='AK'),
    'MODIS': ManagerMODISAppEEARS(),
}
default_land_cover = 'NLCD (L48)'

# ---------------------------------------------------------------------------
# Meteorology.
# ---------------------------------------------------------------------------
met_sources: Dict[str, Any] = {
    'AORC': ManagerAORC(),
    'DayMet': ManagerDaymet(),
    'HRRR': ManagerHRRR(),
}
default_met = 'AORC'

#: Which met sources can drive DHSVM at a sub-daily timestep.  Daymet is
#: daily-only and must be disaggregated, which DHSVM does not do
#: internally -- see :mod:`ww_dhsvm.meteorology`.
met_is_subdaily = {'AORC': True, 'DayMet': False, 'HRRR': True}


def getDefaultSources() -> Dict[str, Any]:
    """A complete default source set for a CONUS DHSVM build.

    Returns
    -------
    dict
        Keyed by ``'HUC'``, ``'hydrography'``, ``'DEM'``, ``'soil'``,
        ``'depth to bedrock'``, ``'land cover'`` and ``'meteorology'``.
    """
    return {
        'HUC': huc_sources[default_huc_source],
        'hydrography': hydrography_sources[default_hydrography_source],
        'DEM': dem_sources[default_dem_source],
        'soil': soil_sources[default_soil_source],
        'depth to bedrock': depth_to_bedrock_sources[default_depth_to_bedrock],
        'land cover': land_cover_sources[default_land_cover],
        'meteorology': met_sources[default_met],
    }


def logSources(sources: Dict[str, Any]) -> None:
    """Log the source set, for provenance at the top of a notebook run."""
    logging.info('Data sources')
    logging.info('-' * 60)
    for stype, s in sources.items():
        if s is None:
            logging.info(f'  {stype:<18s}: None')
        else:
            logging.info(f'  {stype:<18s}: {s.name}')
            logging.info(f'  {"":<18s}  via {s.source}')
