"""WW-DHSVM: Watershed Workflow for the Distributed Hydrology Soil Vegetation Model.

WW-DHSVM is a fork of `Watershed Workflow
<https://github.com/environmental-modeling-workflows/watershed-workflow>`_
(v2.1.0, BSD licensed) retargeted from ATS to DHSVM.  It keeps Watershed
Workflow's central idea -- that a hyper-resolution hydrologic model setup
should be reproducible from a *watershed identifier* plus a set of
declarative data-source managers -- and replaces everything downstream of
the watershed boundary with DHSVM's requirements.

What was kept from Watershed Workflow
-------------------------------------
* the **source-manager** architecture (:mod:`ww_dhsvm.sources`): a small
  set of abstract base classes that normalize CRS handling, buffering,
  bounding-box snapping, and an on-disk cache with spatial and temporal
  superset detection, so that re-running a notebook re-downloads nothing;
* the data-source implementations themselves -- WBD, NHD, 3DEP, NLCD,
  SSURGO, SoilGrids, POLARIS, AORC and Daymet;
* CRS, warping, colormap and plotting infrastructure.

What was removed
----------------
Everything specific to ATS's unstructured discretization: 2D/3D mesh
generation, Delaunay triangulation, stream-aligned river meshing,
mesh extrusion, ExodusII and VTK writers, labeled-set regions, the
river-tree data structure, and the ATS XML input-spec writers.  DHSVM
runs on a single regular grid, so none of it has an analog.

What was added
--------------
* :mod:`ww_dhsvm.grid`         -- the regular DHSVM model grid
* :mod:`ww_dhsvm.binary`       -- DHSVM's native binary map format
* :mod:`ww_dhsvm.terrain`      -- pit filling, D8 routing, slope/aspect
* :mod:`ww_dhsvm.streams`      -- channel network and the three stream files
* :mod:`ww_dhsvm.soils`        -- soil texture classes and DHSVM soil parameters
* :mod:`ww_dhsvm.vegetation`   -- NLCD to DHSVM vegetation classes and parameters
* :mod:`ww_dhsvm.shading`      -- topographic shading and sky-view factor maps
* :mod:`ww_dhsvm.states`       -- initial model-state files
* :mod:`ww_dhsvm.meteorology`  -- AORC / Daymet / **HRRR** to DHSVM forcing
* :mod:`ww_dhsvm.config_writer`-- the DHSVM configuration file
* :mod:`ww_dhsvm.diagnostics`  -- input QA and consistency checks
* :mod:`ww_dhsvm.output`       -- readers for DHSVM's output files
* :mod:`ww_dhsvm.workflow`     -- the high-level orchestration entry points

Typical use
-----------
The frontend is a Jupyter notebook that supplies a HUC code or a shapefile
and a handful of parameters, and calls into this package::

    import ww_dhsvm
    import ww_dhsvm.sources

    sources = ww_dhsvm.sources.getDefaultSources()
    watershed = ww_dhsvm.getWatershed('01080205', sources)
    grid = ww_dhsvm.grid.ModelGrid.fromShape(watershed, cellsize=150)
    ...

See ``notebooks/`` for the fully worked Connecticut River Basin
example.
"""

__version__ = '1.0.0'
__all__ = [
    'getWatershed',
    'findHUC',
    'grid',
    'terrain',
    'streams',
    'soils',
    'vegetation',
    'shading',
    'states',
    'meteorology',
    'binary',
    'config_writer',
    'diagnostics',
    'output',
    'plot',
    'workflow',
]

from typing import Any, Optional, List
import logging

import numpy as np
import shapely.geometry
import geopandas as gpd

import ww_dhsvm.config
import ww_dhsvm.crs
import ww_dhsvm.warp
import ww_dhsvm.sources.standard_names as names


# -----------------------------------------------------------------------------
# Watershed boundary acquisition -- the entry point of every workflow
# -----------------------------------------------------------------------------
def getWatershed(huc: str,
                 sources: Optional[dict] = None,
                 crs: Optional[Any] = None,
                 source: Optional[Any] = None) -> gpd.GeoDataFrame:
    """Download a watershed boundary by HUC code.

    This is the primary entry point: a DHSVM setup begins with a domain,
    and in the United States a domain is most conveniently named by its
    USGS Hydrologic Unit Code.

    Parameters
    ----------
    huc : str
        A HUC code of any even level from 2 to 12, e.g. ``'0108'`` for
        the Connecticut River Basin.  Leading zeros matter and must be
        preserved, so pass a string, never an int.
    sources : dict, optional
        Source dictionary from :func:`ww_dhsvm.sources.getDefaultSources`.
        Only ``sources['HUC']`` is used.
    crs : CRS, optional
        CRS to return the boundary in.  Defaults to the source's native
        CRS (NAD83 geographic for WBD).
    source : manager, optional
        A HUC manager, overriding ``sources['HUC']``.

    Returns
    -------
    gpd.GeoDataFrame
        One row per hydrologic unit, with standard ``ID``/``name``/``area``
        columns.
    """
    if source is None:
        if sources is None:
            import ww_dhsvm.sources
            sources = ww_dhsvm.sources.getDefaultSources()
        source = sources['HUC']

    huc = str(huc).strip()
    if len(huc) % 2 or not (2 <= len(huc) <= 12):
        raise ValueError(f'"{huc}" is not a valid HUC code: HUC codes have an '
                         f'even number of digits between 2 and 12.')

    logging.info(f'Downloading HUC{len(huc)} {huc} from {source.name}')
    df = source.getShapesByID([huc], out_crs=crs)
    if len(df) == 0:
        raise ValueError(f'No hydrologic unit found for HUC {huc}.')

    total_area = df.to_crs(df.estimate_utm_crs()).area.sum() / 1e6
    logging.info(f'  found {len(df)} unit(s), total area {total_area:.1f} km^2')
    return df


def getWatershedFromShapefile(filename: str,
                              crs: Optional[Any] = None,
                              index: Optional[int] = None) -> gpd.GeoDataFrame:
    """Read a watershed boundary from a user-supplied shapefile.

    The alternative entry point for domains that are not a whole
    hydrologic unit -- a gaged catchment, a research forest, a utility's
    service area.

    Parameters
    ----------
    filename : str
        Path to any vector format GeoPandas can read (shapefile,
        GeoPackage, GeoJSON, ...).
    crs : CRS, optional
        CRS to return the boundary in.  Defaults to the file's own.
    index : int, optional
        Select a single feature by positional index.  By default all
        features are returned and later dissolved into one domain.

    Returns
    -------
    gpd.GeoDataFrame
    """
    import ww_dhsvm.sources
    logging.info(f'Reading watershed boundary from {filename}')
    mgr = ww_dhsvm.sources.ManagerShapefile(filename)
    df = mgr.getShapes(out_crs=crs)
    if index is not None:
        df = df.iloc[[index]]
    area = df.to_crs(df.estimate_utm_crs()).area.sum() / 1e6
    logging.info(f'  {len(df)} feature(s), total area {area:.1f} km^2')
    return df


def findHUC(source: Any,
            shape: shapely.geometry.base.BaseGeometry,
            shape_crs: Any,
            hint: Optional[str] = None,
            level: Optional[int] = None) -> str:
    """Find the smallest HUC that fully contains a shape.

    Useful when a user supplies a shapefile but a HUC-keyed dataset (for
    example NHDPlus HR, which is distributed by HUC4) still needs to know
    which unit to fetch.

    Parameters
    ----------
    source : manager
        A HUC source manager, e.g. ``sources['HUC']``.
    shape : shapely geometry
        The shape to locate.
    shape_crs : CRS
        CRS of ``shape``.
    hint : str, optional
        A HUC known to contain the shape, to narrow the search.  Without
        one the search starts from the containing HUC2.
    level : int, optional
        Stop at this HUC level rather than descending as far as possible.

    Returns
    -------
    str
        The HUC code.
    """
    def _contains(huc_code):
        df = source.getShapesByID([huc_code])
        geom = ww_dhsvm.warp.shply(shape, shape_crs, df.crs)
        return df.union_all().buffer(0).contains(geom.buffer(0))

    if hint is None:
        latlon = ww_dhsvm.crs.from_epsg(4326)
        pt = ww_dhsvm.warp.shply(shape.centroid, shape_crs, latlon)
        source.setLevel(2)
        df2 = source.getShapesByGeometry(pt, latlon)
        if len(df2) == 0:
            raise ValueError('Could not locate a containing HUC2 for this shape.')
        hint = str(df2.iloc[0][names.ID])

    current = hint
    max_level = level if level is not None else 12
    while len(current) < max_level:
        next_level = len(current) + 2
        source.setLevel(next_level)
        children = source.getShapesByGeometry(shape, shape_crs)
        found = None
        for _, row in children.iterrows():
            code = str(row[names.ID])
            if code.startswith(current) and _contains(code):
                found = code
                break
        if found is None:
            break
        current = found
        logging.info(f'  descended to HUC{len(current)} {current}')
    return current
