"""The DHSVM model grid.

This module plays the role that ``mesh.py`` plays in Watershed Workflow.
Where ATS is discretized on an unstructured, stream-aligned, mixed-
polyhedral mesh, DHSVM is discretized on a single **regular, north-up,
square-celled grid in a projected CRS**.  Every DHSVM input -- elevation,
mask, soil class, soil depth, vegetation class, the stream map, even the
meteorological interpolation weights -- is defined on this one grid, so
establishing it correctly is the first and most consequential step of a
DHSVM setup.

The grid is fully described by six numbers plus a CRS:

* ``ncols``, ``nrows``  -- shape
* ``cellsize``          -- spacing, metres, identical in x and y
* ``xllcorner``, ``yllcorner`` -- south-west **corner** (not cell centre)
* ``crs``               -- a projected CRS, conventionally UTM

DHSVM's ``[AREA]`` section instead names the *north-west* corner, as
``Extreme West`` (= ``xllcorner``) and ``Extreme North``
(= ``yllcorner + nrows * cellsize``).  This was verified against
``InitConstants.c``, which reads them into ``Map->Xorig`` / ``Map->Yorig``,
and against ``InitMetSources.c``, which converts a station's UTM position
to a grid index with::

    row = round(((Yorig - 0.5*DY) - North) / DY)
    col = round((East - (Xorig + 0.5*DX)) / DX)

i.e. cell centres sit at ``Xorig + (col+0.5)*DX`` and
``Yorig - (row+0.5)*DY``.  Row 0 is therefore the **northernmost** row,
which is also the order in which the binary maps are written.  That
convention matches a standard GDAL/rasterio north-up affine transform, so
``ModelGrid`` interoperates directly with rasterio and rioxarray.
"""

from typing import Optional, Tuple, Any
import math
import logging

import numpy as np
import shapely.geometry
import geopandas as gpd
import rasterio.transform
import rasterio.features
import pyproj

import ww_dhsvm.crs
from ww_dhsvm.crs import CRS
import ww_dhsvm.warp


class ModelGrid:
    """A regular, north-up, square-celled grid in a projected CRS.

    Attributes
    ----------
    nrows, ncols : int
        Grid shape.  Row 0 is the northernmost row.
    cellsize : float
        Grid spacing in projected units (metres).
    xllcorner, yllcorner : float
        South-west corner of the grid, in ``crs``.
    crs : CRS
        Projected coordinate reference system.
    mask : np.ndarray or None
        ``uint8`` array shaped ``(nrows, ncols)``; non-zero means inside
        the basin.  ``None`` until :meth:`setMaskFromShape` is called.
    """

    def __init__(self,
                 nrows: int,
                 ncols: int,
                 cellsize: float,
                 xllcorner: float,
                 yllcorner: float,
                 crs: CRS,
                 mask: Optional[np.ndarray] = None):
        self.nrows = int(nrows)
        self.ncols = int(ncols)
        self.cellsize = float(cellsize)
        self.xllcorner = float(xllcorner)
        self.yllcorner = float(yllcorner)
        self.crs = crs
        self.mask = mask

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def fromShape(cls,
                  shape: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
                  cellsize: float,
                  shape_crs: Optional[CRS] = None,
                  crs: Optional[CRS] = None,
                  buffer_cells: int = 3,
                  align: bool = True) -> 'ModelGrid':
        """Build a grid that comfortably contains a watershed polygon.

        Parameters
        ----------
        shape : shapely geometry or GeoDataFrame
            The watershed boundary.
        cellsize : float
            Desired grid spacing in metres.  DHSVM is typically run at
            30--150 m; below ~10 m the 3-layer soil column and the
            saturated-subsurface-flow parameterization stop being
            physically defensible.
        shape_crs : CRS, optional
            CRS of ``shape``.  Required when ``shape`` is a bare shapely
            geometry.
        crs : CRS, optional
            Target projected CRS.  When ``None``, the UTM zone containing
            the shape's centroid is chosen automatically.
        buffer_cells : int, optional
            Extra cells of padding added on every side.  A few cells of
            margin keep the basin boundary away from the array edge,
            where DHSVM's D4/D8 routing stencils would otherwise be
            truncated.  Default 3.
        align : bool, optional
            Snap the origin to a whole multiple of ``cellsize``.  This
            makes grids reproducible and makes two runs at the same
            resolution share cell edges.  Default ``True``.

        Returns
        -------
        ModelGrid
        """
        if isinstance(shape, gpd.GeoDataFrame):
            if shape_crs is not None:
                raise ValueError('shape_crs should not be given with a GeoDataFrame')
            shape_crs = shape.crs
            geom = shape.union_all()
        else:
            if shape_crs is None:
                raise ValueError('shape_crs is required when shape is a shapely geometry')
            geom = shape

        if crs is None:
            crs = cls.guessUTM(geom, shape_crs)
            logging.info(f'  auto-selected projected CRS: {ww_dhsvm.crs.toString(crs)}')

        geom_p = ww_dhsvm.warp.shply(geom, shape_crs, crs)
        xmin, ymin, xmax, ymax = geom_p.bounds

        pad = buffer_cells * cellsize
        xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad

        if align:
            xmin = math.floor(xmin / cellsize) * cellsize
            ymin = math.floor(ymin / cellsize) * cellsize
            xmax = math.ceil(xmax / cellsize) * cellsize
            ymax = math.ceil(ymax / cellsize) * cellsize

        ncols = int(round((xmax - xmin) / cellsize))
        nrows = int(round((ymax - ymin) / cellsize))

        grid = cls(nrows, ncols, cellsize, xmin, ymin, crs)
        logging.info(f'  model grid: {nrows} rows x {ncols} cols @ {cellsize:g} m '
                     f'= {grid.area_km2:.1f} km^2 bounding box')
        return grid

    @staticmethod
    def guessUTM(shape: shapely.geometry.base.BaseGeometry, shape_crs: CRS) -> CRS:
        """Pick the UTM zone containing a shape's centroid.

        DHSVM's ``[AREA]`` section supports ``Coordinate System = UTM``,
        and its shading/solar routines assume a projected, equal-scale
        grid.  UTM is the conventional and safest choice.
        """
        latlon = ww_dhsvm.crs.from_epsg(4326)
        centroid = ww_dhsvm.warp.shply(shape.centroid, shape_crs, latlon)
        lon, lat = centroid.x, centroid.y
        zone = int(math.floor((lon + 180.0) / 6.0) % 60) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        return ww_dhsvm.crs.from_epsg(epsg)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """``(nrows, ncols)``."""
        return (self.nrows, self.ncols)

    @property
    def transform(self):
        """The rasterio north-up affine transform for this grid."""
        return rasterio.transform.from_origin(
            self.xllcorner, self.yurcorner, self.cellsize, self.cellsize)

    @property
    def xurcorner(self) -> float:
        """Easting of the east edge."""
        return self.xllcorner + self.ncols * self.cellsize

    @property
    def yurcorner(self) -> float:
        """Northing of the north edge -- DHSVM's ``Extreme North``."""
        return self.yllcorner + self.nrows * self.cellsize

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` in ``crs``."""
        return (self.xllcorner, self.yllcorner, self.xurcorner, self.yurcorner)

    @property
    def area_km2(self) -> float:
        """Area of the full grid bounding box, km^2."""
        return self.nrows * self.ncols * self.cellsize ** 2 / 1e6

    @property
    def cell_area(self) -> float:
        """Area of one cell, m^2."""
        return self.cellsize ** 2

    @property
    def x(self) -> np.ndarray:
        """Easting of each column's cell centre, west to east."""
        return self.xllcorner + (np.arange(self.ncols) + 0.5) * self.cellsize

    @property
    def y(self) -> np.ndarray:
        """Northing of each row's cell centre, **north to south**."""
        return self.yurcorner - (np.arange(self.nrows) + 0.5) * self.cellsize

    def meshgrid(self) -> Tuple[np.ndarray, np.ndarray]:
        """2D arrays of cell-centre eastings and northings."""
        return np.meshgrid(self.x, self.y)

    def rowcol(self, easting: float, northing: float) -> Tuple[int, int]:
        """Convert projected coordinates to ``(row, col)``.

        Uses exactly the rounding DHSVM uses in ``InitMetSources.c``, so
        a station this method places in cell ``(r, c)`` is the same cell
        DHSVM will place it in.
        """
        row = int(round(((self.yurcorner - 0.5 * self.cellsize) - northing) / self.cellsize))
        col = int(round((easting - (self.xllcorner + 0.5 * self.cellsize)) / self.cellsize))
        return row, col

    def xy(self, row: int, col: int) -> Tuple[float, float]:
        """Convert ``(row, col)`` to the cell centre's projected coordinates."""
        return (self.xllcorner + (col + 0.5) * self.cellsize,
                self.yurcorner - (row + 0.5) * self.cellsize)

    def polygon(self) -> shapely.geometry.Polygon:
        """The grid's bounding box as a polygon in ``crs``."""
        return shapely.geometry.box(*self.bounds)

    def centerLatLon(self) -> Tuple[float, float]:
        """Latitude and longitude of the grid centre, degrees.

        DHSVM needs these for its solar-geometry routines
        (``Center Latitude`` / ``Center Longitude`` in ``[AREA]``).
        """
        cx = 0.5 * (self.xllcorner + self.xurcorner)
        cy = 0.5 * (self.yllcorner + self.yurcorner)
        pt = ww_dhsvm.warp.shply(shapely.geometry.Point(cx, cy),
                                 self.crs, ww_dhsvm.crs.from_epsg(4326))
        return pt.y, pt.x

    def timeZoneMeridian(self) -> float:
        """Standard-meridian longitude for the grid centre.

        DHSVM uses ``Time Zone Meridian`` together with ``Center
        Longitude`` to convert local standard time to solar time.  The
        meridian is the centre of the 15-degree-wide time zone, so
        forcing timestamps must be in that zone's *standard* time (never
        daylight-saving, never UTC).  ``ww_dhsvm`` writes forcing in UTC
        and therefore returns 0.0 here -- see
        :mod:`ww_dhsvm.meteorology`.
        """
        _, lon = self.centerLatLon()
        return round(lon / 15.0) * 15.0

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def setMaskFromShape(self,
                         shape: shapely.geometry.base.BaseGeometry | gpd.GeoDataFrame,
                         shape_crs: Optional[CRS] = None,
                         all_touched: bool = False) -> np.ndarray:
        """Rasterize a watershed polygon into the basin mask.

        DHSVM's ``Outside Basin Value`` is set to 0 by ``ww_dhsvm``, so
        the mask is 1 inside the basin and 0 outside.

        Parameters
        ----------
        shape : shapely geometry or GeoDataFrame
            The watershed boundary.
        shape_crs : CRS, optional
            CRS of ``shape``; required for a bare shapely geometry.
        all_touched : bool, optional
            Include every cell the polygon touches rather than only those
            whose centre it contains.  Default ``False`` (centre rule),
            which keeps the masked area closest to the true basin area.

        Returns
        -------
        np.ndarray
            The ``uint8`` mask, also stored as ``self.mask``.
        """
        if isinstance(shape, gpd.GeoDataFrame):
            if shape_crs is not None:
                raise ValueError('shape_crs should not be given with a GeoDataFrame')
            shape_crs = shape.crs
            geom = shape.union_all()
        else:
            if shape_crs is None:
                raise ValueError('shape_crs is required when shape is a shapely geometry')
            geom = shape

        geom_p = ww_dhsvm.warp.shply(geom, shape_crs, self.crs)
        mask = rasterio.features.rasterize(
            [(geom_p, 1)],
            out_shape=self.shape,
            transform=self.transform,
            fill=0,
            all_touched=all_touched,
            dtype='uint8')

        self.mask = mask
        n = int(mask.sum())
        logging.info(f'  basin mask: {n} of {self.nrows*self.ncols} cells active '
                     f'({100.0*n/(self.nrows*self.ncols):.1f}%), '
                     f'{n*self.cell_area/1e6:.1f} km^2')
        return mask

    @property
    def n_active(self) -> int:
        """Number of in-basin cells."""
        if self.mask is None:
            return self.nrows * self.ncols
        return int(np.count_nonzero(self.mask))

    @property
    def basin_area_km2(self) -> float:
        """Area of the masked basin, km^2."""
        return self.n_active * self.cell_area / 1e6

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------

    def toDataArray(self, array: np.ndarray, name: str = 'data'):
        """Wrap a 2D array on this grid as a georeferenced xr.DataArray."""
        import xarray as xr
        da = xr.DataArray(array, dims=('y', 'x'),
                          coords={'y': self.y, 'x': self.x}, name=name)
        da = da.rio.write_crs(self.crs)
        da = da.rio.write_transform(self.transform)
        return da

    def rasterProfile(self, dtype: str = 'float32', nodata: Any = None) -> dict:
        """A rasterio profile dict for writing GeoTIFFs on this grid."""
        return dict(driver='GTiff', height=self.nrows, width=self.ncols, count=1,
                    dtype=dtype, crs=self.crs, transform=self.transform,
                    nodata=nodata, compress='deflate', tiled=True)

    def __repr__(self) -> str:
        return (f'ModelGrid({self.nrows}x{self.ncols} @ {self.cellsize:g} m, '
                f'll=({self.xllcorner:.1f}, {self.yllcorner:.1f}), '
                f'crs={ww_dhsvm.crs.toString(self.crs)})')

    def summary(self) -> str:
        """A human-readable multi-line description, for notebook logging."""
        lat, lon = self.centerLatLon()
        lines = [
            'DHSVM model grid',
            '-' * 60,
            f'  shape            : {self.nrows} rows x {self.ncols} cols '
            f'= {self.nrows*self.ncols:,} cells',
            f'  cell size        : {self.cellsize:g} m  '
            f'({self.cell_area/1e4:.2f} ha per cell)',
            f'  CRS              : {ww_dhsvm.crs.toString(self.crs)}',
            f'  Extreme West     : {self.xllcorner:.6f}',
            f'  Extreme North    : {self.yurcorner:.6f}',
            f'  bounding box     : {self.area_km2:.1f} km^2',
            f'  centre lat/lon   : {lat:.6f}, {lon:.6f}',
        ]
        if self.mask is not None:
            lines.append(f'  active cells     : {self.n_active:,} '
                         f'({100.0*self.n_active/(self.nrows*self.ncols):.1f}%)')
            lines.append(f'  basin area       : {self.basin_area_km2:.2f} km^2')
        return '\n'.join(lines)
