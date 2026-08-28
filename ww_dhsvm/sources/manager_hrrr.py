"""Manager for NOAA's High-Resolution Rapid Refresh (HRRR), via Herbie.

HRRR is NOAA's operational 3 km convection-allowing analysis and forecast
system over CONUS.  For DHSVM it is attractive for three reasons: it is
**3 km** rather than AORC's 1 km-but-reanalysis or Daymet's 1 km daily;
it is **hourly**; and it is **near-real-time**, so a DHSVM setup can be
run for the current water year rather than only for the period a
reanalysis has been extended through.

Its limitations matter too, and this docstring is the right place for
them: HRRR begins **2014-07-30**, its early years (through the v2
upgrade in Aug 2016 and v3 in Jul 2018) are not homogeneous, and it is a
*model analysis*, not an observational product -- its precipitation in
particular is a short-range forecast field and carries convective
placement error.  For multi-decade DHSVM calibration, AORC remains the
better choice; HRRR is for recent, high-resolution, event-scale work.

Which fields, and from which forecast hour
------------------------------------------
DHSVM needs seven quantities.  All are present in HRRR's ``sfc``
(``wrfsfc``) product, but **not all at forecast hour 0**: accumulated
precipitation does not exist in the analysis, and the radiation fluxes
are period-averages.  This manager therefore builds each valid hour
``H`` from the run initialized at ``H-1`` evaluated at ``fxx=1``:

=========================  =========================================
DHSVM variable             HRRR field (``fxx=1``)
=========================  =========================================
air temperature            ``TMP:2 m above ground``
wind speed                 ``UGRD``/``VGRD:10 m above ground``
relative humidity          ``RH:2 m above ground``
incoming shortwave         ``DSWRF:surface``
incoming longwave          ``DLWRF:surface``
precipitation              ``APCP:surface`` (0--1 hour accumulation)
=========================  =========================================

This gives a seamless hourly series in which precipitation is a true
1-hour accumulation ending at the timestamp and the radiation fluxes are
means over the preceding hour -- exactly DHSVM's convention for a 1-hour
timestep.

Projection
----------
HRRR is on a Lambert Conformal Conic grid, which Herbie returns with 2D
``latitude``/``longitude`` coordinate arrays rather than 1D axes.  This
manager subsets in native grid space using a lat/lon bounding-box mask
and keeps the 2D coordinates, since :mod:`ww_dhsvm.meteorology` samples
at station points rather than requiring a regular axis.
"""

from typing import Optional, List, Tuple, Any
import os
import gc
import logging
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import cftime
import shapely.geometry

import ww_dhsvm.crs
from ww_dhsvm.crs import CRS

from . import manager_dataset


class ManagerHRRR(manager_dataset.ManagerDataset):
    """NOAA HRRR hourly surface analysis/forecast, retrieved with Herbie.

    Parameters
    ----------
    product : str, optional
        HRRR product.  ``'sfc'`` (default) is the 2D surface file and
        holds every field DHSVM needs.
    fxx : int, optional
        Forecast hour used for each valid time.  Default 1, which is
        required for accumulated precipitation -- see the module
        docstring.  Do not set 0 unless you have arranged precipitation
        another way.
    priority : list of str, optional
        Herbie source priority, e.g. ``['aws', 'nomads', 'google']``.
    """

    class Request(manager_dataset.ManagerDataset.Request):
        """HRRR request carrying the resolved cache filename."""
        def __init__(self, request, filename: str = ''):
            super().copyFromExisting(request)
            self.filename = filename

    #: DHSVM-relevant HRRR fields, as Herbie regex search strings.
    SEARCH = {
        'TMP': r':TMP:2 m above ground:',
        'RH': r':RH:2 m above ground:',
        'UGRD': r':UGRD:10 m above ground:',
        'VGRD': r':VGRD:10 m above ground:',
        'DSWRF': r':DSWRF:surface:',
        'DLWRF': r':DLWRF:surface:',
        'APCP': r':APCP:surface:',
    }

    VALID_VARIABLES = list(SEARCH.keys())
    DEFAULT_VARIABLES = list(SEARCH.keys())

    #: HRRR's first operational cycle.
    HRRR_START = (2014, 7, 30)

    def __init__(self,
                 product: str = 'sfc',
                 fxx: int = 1,
                 priority: Optional[List[str]] = None):
        self.product = product
        self.fxx = fxx
        self.priority = priority

        native_start = cftime.datetime(*self.HRRR_START, calendar='standard')
        # No fixed end: HRRR is operational.  Use "now" so validation
        # rejects requests for the future but permits today.
        now = pd.Timestamp.now('UTC').tz_localize(None)
        native_end = cftime.datetime(now.year, now.month, now.day,
                                     calendar='standard')

        super().__init__(
            name=f'HRRR {product} f{fxx:02d}',
            source='NOAA HRRR via Herbie (AWS/NODD)',
            native_resolution=0.03,      # ~3 km expressed in degrees
            native_crs_in=CRS.from_epsg(4326),
            native_crs_out=CRS.from_epsg(4326),
            native_start=native_start,
            native_end=native_end,
            valid_variables=self.VALID_VARIABLES,
            default_variables=self.DEFAULT_VARIABLES,
            cache_category='meteorology',
            cache_extension='nc',
            has_varname=False,
            has_resampling=True,
            short_name='HRRR',
        )
        os.makedirs(self._cacheFolder(), exist_ok=True)

    # ------------------------------------------------------------------

    def _requestDataset(self,
                        request: manager_dataset.ManagerDataset.Request,
                        temporal_resampling: Optional[str] = None,
                        force: bool = False,
                        max_workers: int = 4,
                        ) -> manager_dataset.ManagerDataset.Request:
        """Download (or locate in cache) the HRRR series for this request."""
        assert request.start is not None and request.end is not None

        start_year, end_year = request.start.year, request.end.year
        filename = self._cacheFilename(request.snapped_bounds,
                                       start_year=start_year, end_year=end_year,
                                       temporal_resampling=temporal_resampling)

        if not force and not os.path.exists(filename):
            superset = self._checkCache(request.geometry.bounds,
                                        request.snapped_bounds,
                                        start_year=start_year, end_year=end_year,
                                        temporal_resampling=temporal_resampling)
            if superset is not None and self._cacheCoversPeriod(
                    superset, request.start, request.end):
                logging.info(f'  using superset cache: {superset}')
                hrrr_req = self.Request(request, superset)
                hrrr_req.is_ready = True
                return hrrr_req

        if force or not os.path.exists(filename):
            self._download(request, filename, temporal_resampling, max_workers)
        elif not self._cacheCoversPeriod(filename, request.start, request.end):
            logging.info(f'  existing cache does not span the requested period; '
                         f're-downloading')
            self._download(request, filename, temporal_resampling, max_workers)
        else:
            logging.info(f'  using existing cache: {filename}')

        hrrr_req = self.Request(request, filename)
        hrrr_req.is_ready = True
        return hrrr_req



    @staticmethod
    def _partIsComplete(path, expected_times) -> bool:
        """Is a cached monthly part complete for the hours this request needs?

        Resumability must not silently reuse a month that was written by
        an earlier, shorter request -- so a part is reused only when it
        contains at least 99% of the hours the current request wants from
        that month.
        """
        try:
            with xr.open_dataset(path) as ds:
                have = pd.DatetimeIndex(ds['time'].values)
        except Exception:
            return False
        want = pd.DatetimeIndex(expected_times)
        n_have = len(want.intersection(have))
        return n_have >= 0.99 * len(want)

    @staticmethod
    def _cacheCoversPeriod(path, start, end) -> bool:
        """Does a cached file actually span the requested period?

        The inherited cache-filename scheme encodes only a *year* range,
        so a file built for three days in June 2025 and a request for all
        of 2025 generate the same name.  Sub-annual requests are normal
        for DHSVM, so the file's own time axis is checked rather than
        trusting the filename.
        """
        try:
            with xr.open_dataset(path) as ds:
                if 'time' not in ds.coords or ds.sizes.get('time', 0) == 0:
                    return False
                t0 = pd.Timestamp(ds['time'].values[0])
                t1 = pd.Timestamp(ds['time'].values[-1])
        except Exception:
            return False
        want0 = pd.Timestamp(start.strftime('%Y-%m-%d %H:00'))
        want1 = pd.Timestamp(end.strftime('%Y-%m-%d %H:00'))
        # One hour of slack: the first valid time may lag the request by fxx.
        covers = (t0 <= want0 + pd.Timedelta(hours=1)) and (t1 >= want1 - pd.Timedelta(hours=1))
        if not covers:
            logging.info(f'  cache spans {t0} to {t1}; request needs {want0} to {want1}')
        return covers

    def _download(self, request, filename, temporal_resampling, max_workers):
        """Fetch HRRR day by day and write one consolidated netCDF.

        Memory, not bandwidth, is the binding constraint.  HRRR fields
        arrive on the **full CONUS grid** (1059 x 1799), so one day of
        seven variables is ~1.7 GB before subsetting.  Two earlier designs
        were killed by the OOM reaper: accumulating a month of full-CONUS
        blocks, and then accumulating a month of *subsets* in a list --
        Herbie and cfgrib keep enough alive that even the small subsets
        dragged the large parents along.

        So each day is fetched, clipped, written to its own small netCDF
        and dropped.  The month is then assembled by reopening those files
        lazily.  Peak memory is bounded at roughly one day no matter how
        long the request is, and every daily part doubles as a resume
        point.
        """
        start = pd.Timestamp(request.start.strftime('%Y-%m-%d %H:00'))
        end = pd.Timestamp(request.end.strftime('%Y-%m-%d %H:00'))

        # Each valid hour H comes from the run initialized at H - fxx.
        valid_hours = pd.date_range(start, end, freq='1h')
        logging.info(f'HRRR: {len(valid_hours)} hourly steps from '
                     f'{valid_hours[0]} to {valid_hours[-1]} '
                     f'(runs f{self.fxx:02d}, {len(request.variables)} variables)')

        search = '|'.join(f'({self.SEARCH[k]})' for k in request.variables)
        bounds = tuple(request.snapped_bounds)

        part_dir = os.path.join(self._cacheFolder(), '_parts')
        os.makedirs(part_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]

        monthly_files: list = []
        missing_days: list = []
        t_start = pd.Timestamp.now()

        for month, group in pd.Series(valid_hours, index=valid_hours).groupby(
                pd.Grouper(freq='MS', level=0)):
            if len(group) == 0:
                continue
            part = os.path.join(part_dir, f'{stem}__{month:%Y%m}.nc')
            if os.path.exists(part) and self._partIsComplete(part, group.index):
                logging.info(f'  {month:%Y-%m}: reusing complete part '
                             f'{os.path.basename(part)}')
                monthly_files.append(part)
                continue

            day_dir = os.path.join(part_dir, f'{stem}__{month:%Y%m}_days')
            os.makedirs(day_dir, exist_ok=True)
            day_files = []
            n_days = len(set(group.index.date))
            for k, (day, hours) in enumerate(
                    pd.Series(group.values, index=group.index).groupby(
                        pd.Grouper(freq='D', level=0)), start=1):
                if len(hours) == 0:
                    continue
                dpath = os.path.join(day_dir, f'{day:%Y%m%d}.nc')
                if os.path.exists(dpath):
                    day_files.append(dpath)
                    continue
                runs = pd.DatetimeIndex(hours.index) - pd.Timedelta(hours=self.fxx)
                ds = self._fetchChunk(list(runs), search, request.variables,
                                      bounds, max_workers)
                if ds is None:
                    missing_days.append(day)
                    continue
                ds.to_netcdf(dpath, encoding={v: dict(zlib=True, complevel=4)
                                              for v in ds.data_vars})
                ds.close()
                del ds
                gc.collect()
                day_files.append(dpath)
                if k % 5 == 0 or k == n_days:
                    el = (pd.Timestamp.now() - t_start).total_seconds()
                    logging.info(f'  {month:%Y-%m}: {k}/{n_days} days '
                                 f'({el/60:.1f} min elapsed)')

            if not day_files:
                logging.warning(f'  {month:%Y-%m}: no data retrieved')
                continue

            # Nested concat along time: the file order is ours (sorted by
            # date), so no coordinate inference is needed and a degenerate
            # single-hour day cannot break the merge.
            dsm = xr.open_mfdataset(sorted(day_files), combine='nested',
                                    concat_dim='time').sortby('time').load()
            dsm.to_netcdf(part, encoding={v: dict(zlib=True, complevel=4)
                                          for v in dsm.data_vars})
            dsm.close()
            del dsm
            gc.collect()
            logging.info(f'  {month:%Y-%m}: wrote {os.path.basename(part)} '
                         f'({os.path.getsize(part)/1e6:.1f} MB) from '
                         f'{len(day_files)} daily parts')
            monthly_files.append(part)

        if not monthly_files:
            raise RuntimeError('HRRR: no data could be retrieved for this request.')
        if missing_days:
            logging.warning(f'  {len(missing_days)} day(s) could not be retrieved and '
                            f'will appear as gaps: '
                            f'{[str(pd.Timestamp(d).date()) for d in missing_days[:8]]}'
                            f'{" ..." if len(missing_days) > 8 else ""}')

        out = xr.open_mfdataset(sorted(monthly_files), combine='nested',
                                concat_dim='time').sortby('time').load()
        if temporal_resampling is not None:
            out = out.resample(time=temporal_resampling).mean()

        out = out.rio.write_crs(self.native_crs_out)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        comp = {v: dict(zlib=True, complevel=4) for v in out.data_vars}
        out.to_netcdf(filename, encoding=comp)
        logging.info(f'  wrote HRRR cache: {filename} '
                     f'({os.path.getsize(filename)/1e6:.1f} MB, '
                     f'{out.sizes.get("time", 0)} timesteps)')

    def _fetchChunk(self, run_times, search, variables, bounds, max_workers,
                    max_retries: int = 4):
        """Download one day of HRRR runs and clip to the bounding box.

        NOAA's S3 mirror rate-limits aggressive parallel access, returning
        503s and reset connections.  A dropped day would leave a hole in
        the forcing that DHSVM reads as a missing timestep, so failures
        are retried with exponential backoff and a halved thread count
        before being reported.  ``remove_grib=True`` deletes each GRIB
        after it is read -- a year of CONUS files would otherwise be tens
        of gigabytes of scratch.
        """
        from herbie import FastHerbie

        xmin, ymin, xmax, ymax = bounds
        run_index = pd.DatetimeIndex(run_times)
        threads = max_workers

        last_exc = None
        for attempt in range(max_retries):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    FH = FastHerbie(run_index, model='hrrr', product=self.product,
                                    fxx=[self.fxx], priority=self.priority,
                                    max_threads=threads)
                    ds = FH.xarray(search, max_threads=threads, remove_grib=True)
                if ds is not None:
                    break
                last_exc = RuntimeError('Herbie returned no dataset')
            except Exception as exc:
                last_exc = exc
            wait = 2.0 ** attempt
            threads = max(1, threads // 2)
            logging.warning(f'    retry {attempt+1}/{max_retries} for '
                            f'{run_index[0]:%Y-%m-%d} after {type(last_exc).__name__}; '
                            f'sleeping {wait:.0f}s, threads -> {threads}')
            time.sleep(wait)
        else:
            logging.error(f'    GIVING UP on {run_index[0]:%Y-%m-%d}: {last_exc}')
            return None

        # Herbie returns a list when fields sit on different level types.
        #
        # Memory discipline matters here.  Each day of seven variables is a
        # ~1.7 GB **full-CONUS** block (1059 x 1799 x 24 hours), and Herbie
        # materialises it rather than leaving it lazy, because the GRIB is
        # deleted straight after reading.  Subsetting to the basin cuts that
        # to a few MB -- but only if the full-CONUS originals are then
        # released.  Left to reference counting alone they accumulate across
        # a month's loop and exhaust memory; a year's request was killed by
        # the OOM reaper at ~28 GB before this teardown was added.
        drop = ('heightAboveGround', 'surface', 'step', 'gribfile_projection',
                'atmosphereSingleLayer')
        originals = ds if isinstance(ds, list) else [ds]
        if isinstance(ds, list):
            merged = None
            for d in ds:
                d = d.drop_vars([c for c in drop if c in d.coords or c in d.variables],
                                errors='ignore')
                merged = d if merged is None else xr.merge([merged, d],
                                                           compat='override')
            ds = merged
        else:
            ds = ds.drop_vars([c for c in drop if c in ds.coords or c in ds.variables],
                              errors='ignore')

        if ds is None:
            return None

        ds = self._subset(ds, xmin, ymin, xmax, ymax)
        ds = self._standardizeNames(ds)

        # Herbie indexes by run time; DHSVM cares about valid time.
        if 'valid_time' in ds.coords:
            ds = ds.assign_coords(time=ds['valid_time'])
        elif 'time' in ds.coords and self.fxx:
            ds = ds.assign_coords(
                time=ds['time'] + np.timedelta64(self.fxx, 'h'))
        for d in ('valid_time', 'step'):
            if d in ds.coords:
                ds = ds.drop_vars(d)

        # A request whose last day holds a single hour comes back with `time`
        # squeezed to a scalar coordinate rather than a length-1 dimension.
        # Written out that way it has no time index, and reassembling the days
        # then fails with "the coordinate 'time' has no corresponding index".
        if 'time' in ds.coords and 'time' not in ds.dims:
            ds = ds.expand_dims('time')

        out = ds.load()          # materialise the small basin subset ...
        for d in originals:      # ... then drop the full-CONUS blocks
            try:
                d.close()
            except Exception:
                pass
        del originals, ds
        gc.collect()
        return out

    @staticmethod
    def _subset(ds, xmin, ymin, xmax, ymax):
        """Clip to a lat/lon bounding box on HRRR's 2D curvilinear grid."""
        lon = ds['longitude']
        lon = xr.where(lon > 180.0, lon - 360.0, lon)
        lat = ds['latitude']

        inside = ((lon >= xmin) & (lon <= xmax) &
                  (lat >= ymin) & (lat <= ymax))
        dims = [d for d in inside.dims if d in ('y', 'x')]
        if len(dims) != 2:
            raise RuntimeError(
                f'HRRR: expected 2D y/x coordinates to subset on, got dims '
                f'{inside.dims}.  Returning the un-subset CONUS grid here '
                f'would silently exhaust memory, so this is fatal.')

        yi = np.flatnonzero(inside.any(dim='x').values)
        xi = np.flatnonzero(inside.any(dim='y').values)
        if yi.size == 0 or xi.size == 0:
            raise ValueError(
                f'HRRR: bounding box ({xmin:.3f}, {ymin:.3f}, {xmax:.3f}, '
                f'{ymax:.3f}) does not intersect the HRRR CONUS domain.')

        # One extra ring so downstream interpolation is never extrapolating.
        ds = ds.isel(y=slice(max(yi[0] - 1, 0), yi[-1] + 2),
                     x=slice(max(xi[0] - 1, 0), xi[-1] + 2))
        ds = ds.assign_coords(longitude=xr.where(ds['longitude'] > 180.0,
                                                 ds['longitude'] - 360.0,
                                                 ds['longitude']))
        return ds

    @staticmethod
    def _standardizeNames(ds):
        """Rename cfgrib's short names to this manager's HRRR field names."""
        rename = {'t2m': 'TMP', 'r2': 'RH', 'u10': 'UGRD', 'v10': 'VGRD',
                  'dswrf': 'DSWRF', 'dlwrf': 'DLWRF', 'tp': 'APCP',
                  'sdswrf': 'DSWRF', 'sdlwrf': 'DLWRF', 'unknown': 'APCP',
                  'prate': 'APCP'}
        return ds.rename({k: v for k, v in rename.items() if k in ds.data_vars})

    def _fetchDataset(self, request, chunk_time=None) -> xr.Dataset:
        """Open the cached HRRR netCDF."""
        if chunk_time:
            ds = xr.open_dataset(request.filename, chunks={'time': chunk_time})
        else:
            ds = xr.open_dataset(request.filename)
        keep = [v for v in request.variables if v in ds.data_vars]
        return ds[keep] if keep else ds
