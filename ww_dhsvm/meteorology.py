"""Meteorological forcing for DHSVM.

DHSVM reads forcing as a set of **point time series** ("stations"), one
ASCII file each, and interpolates them onto the model grid at every
timestep using inverse-distance, nearest-neighbour or Cressman weights,
with lapse-rate corrections applied from each station's elevation to
each cell's elevation.

The station file format, verified against ``ReadMetRecord.c``, is one
whitespace-delimited row per timestep::

    MM/DD/YYYY-HH  Tair  Wind  RH  Sin  Lin  Precip

============  ======================================  =================
column        quantity                                units
============  ======================================  =================
1             timestamp                               ``MM/DD/YYYY-HH``
2             air temperature                         **degrees C**
3             wind speed                              m/s
4             relative humidity                       **percent**
5             incoming shortwave (direct + diffuse)   W/m^2
6             incoming longwave                       W/m^2
7             precipitation                           **metres per timestep**
============  ======================================  =================

Two of those are easy to get wrong and are worth stating twice:
temperature is in **Celsius, not Kelvin**, and precipitation is a depth
in **metres accumulated over one model timestep**, not a rate.  A
1-hour timestep with 2.5 mm of rain is written as ``0.002500``.  For
sub-hourly runs the timestamp gains a minutes field
(``MM/DD/YYYY-HH:MM``).

Why station mode rather than gridded mode
-----------------------------------------
DHSVM also offers ``Gridded Met data = TRUE``, which scans a directory
for files named ``prefix_LAT_LON`` and infers station positions from the
filenames.  ``ww_dhsvm`` deliberately does **not** use it.  Reading
``InitMetSources.c`` shows that the gridded path never assigns
``Stat[k].Elev``, so it stays at its ``calloc`` value of zero -- and
``MakeLocalMetData.c`` then lapses temperature from *sea level* to each
cell.  In a basin whose mean elevation is 300 m and with the default
-0.0065 C/m lapse rate, that is a systematic -1.95 C cold bias, which
would show up as far too much snow.  Station mode lets ``ww_dhsvm``
write an explicit ``Elevation`` for every station, sampled from the same
DEM the model runs on, so the lapse correction is a small,
physically-meaningful adjustment rather than a large spurious one.

Time convention
---------------
Forcing is written in **UTC**, and the configuration writer sets
``Time Zone Meridian = 0``.  ``CalcSolar.c`` computes the offset between
clock noon and solar noon as ``4 min/deg * (StandardMeridian -
Longitude)``; with a zero meridian and UTC timestamps this reduces to
the site's true longitude offset, so solar geometry is exact and
daylight-saving time never enters.  This is simpler *and* more accurate
than converting to local standard time.
"""

from typing import Optional, Dict, List, Tuple, Any
import os
import logging
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import cftime

from ww_dhsvm.grid import ModelGrid
import ww_dhsvm.crs
import ww_dhsvm.warp


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

def saturationVaporPressure(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water, Pa (Bolton 1980).

    Accurate to 0.1% over -35 to +35 C, which spans anything DHSVM will
    encounter in a temperate basin.
    """
    t = np.asarray(temp_c, dtype='float64')
    return 611.2 * np.exp(17.67 * t / (t + 243.5))


def specificHumidityToRH(q: np.ndarray, pressure_pa: np.ndarray,
                         temp_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from specific humidity, pressure and temperature.

    Uses ``e = q P / (0.622 + 0.378 q)`` for the actual vapour pressure,
    then divides by the saturation value.  Result is clipped to
    ``[1, 100]``; ``ReadMetRecord.c`` clamps anything outside ``[0, 100]``
    and prints a warning per timestep, which would flood the log.
    """
    q = np.asarray(q, dtype='float64')
    e = q * np.asarray(pressure_pa, dtype='float64') / (0.622 + 0.378 * q)
    rh = 100.0 * e / saturationVaporPressure(temp_c)
    return np.clip(rh, 1.0, 100.0)


def vaporPressureToRH(vp_pa: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from vapour pressure and temperature."""
    rh = 100.0 * np.asarray(vp_pa, dtype='float64') / saturationVaporPressure(temp_c)
    return np.clip(rh, 1.0, 100.0)


def estimateLongwave(temp_c: np.ndarray, rh_pct: np.ndarray,
                     cloud_fraction: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate incoming longwave radiation, W/m^2.

    Only needed for Daymet, which does not distribute a longwave field.
    Uses Prata's (1996) clear-sky emissivity in terms of screen-level
    precipitable water, with the Crawford & Duchon (1999) cloud
    correction::

        eps_clear = 1 - (1 + w) exp(-sqrt(1.2 + 3w))
        w         = 46.5 e / T
        eps       = (1 - c) eps_clear + c
        L         = eps sigma T^4

    This is a well-tested parameterization, but it is still an estimate:
    where longwave matters -- winter snowmelt under cloud -- prefer AORC
    or HRRR, which carry a modelled longwave field.
    """
    t_k = np.asarray(temp_c, dtype='float64') + 273.15
    e_pa = np.asarray(rh_pct, dtype='float64') / 100.0 * saturationVaporPressure(temp_c)
    e_hpa = e_pa / 100.0

    w = 46.5 * e_hpa / t_k
    eps_clear = 1.0 - (1.0 + w) * np.exp(-np.sqrt(1.2 + 3.0 * w))

    if cloud_fraction is None:
        cloud_fraction = np.zeros_like(t_k)
    c = np.clip(np.asarray(cloud_fraction, dtype='float64'), 0.0, 1.0)
    eps = (1.0 - c) * eps_clear + c

    sigma = 5.670374419e-8
    return eps * sigma * t_k ** 4


def cloudFractionFromShortwave(sin_wm2: np.ndarray,
                               potential_wm2: np.ndarray) -> np.ndarray:
    """Cloud fraction from the ratio of actual to potential shortwave.

    ``c = 1 - (S / S_pot)`` clipped to ``[0, 1]``, the standard
    Crawford & Duchon (1999) inversion.  Undefined at night, where it is
    carried forward from the daily daytime mean by the caller.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(potential_wm2 > 10.0,
                         np.asarray(sin_wm2) / potential_wm2, np.nan)
    return np.clip(1.0 - ratio, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Source-specific conversion to DHSVM's seven quantities
# ---------------------------------------------------------------------------

def convertAORCToDHSVM(ds: xr.Dataset) -> xr.Dataset:
    """Convert an AORC dataset to DHSVM's variables and units.

    AORC v1.1 provides exactly what DHSVM needs, so this is a pure unit
    and name conversion with no estimation:

    ============================  =========================  ==========
    AORC                          DHSVM                      conversion
    ============================  =========================  ==========
    ``TMP_2maboveground`` [K]     ``Tair`` [C]               -273.15
    ``UGRD``/``VGRD`` [m/s]       ``Wind`` [m/s]             hypot
    ``SPFH``, ``PRES``, ``TMP``   ``RH`` [%]                 psychrometry
    ``DSWRF_surface`` [W/m2]      ``Sin`` [W/m2]             identity
    ``DLWRF_surface`` [W/m2]      ``Lin`` [W/m2]             identity
    ``APCP_surface`` [kg/m2/h]    ``Precip`` [m/h]           /1000
    ============================  =========================  ==========

    Precipitation is left as **metres per hour** here; the conversion to
    metres per model timestep happens in :func:`writeStationFile`, which
    is the only place that knows the timestep.
    """
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    tair_c = ds['TMP_2maboveground'] - 273.15
    out['Tair'] = tair_c
    out['Wind'] = np.sqrt(ds['UGRD_10maboveground'] ** 2 +
                          ds['VGRD_10maboveground'] ** 2)
    out['RH'] = xr.apply_ufunc(specificHumidityToRH,
                               ds['SPFH_2maboveground'], ds['PRES_surface'], tair_c,
                               dask='parallelized', output_dtypes=['float64'])
    out['Sin'] = ds['DSWRF_surface'].clip(min=0.0)
    out['Lin'] = ds['DLWRF_surface'].clip(min=0.0)
    out['Precip'] = (ds['APCP_surface'] / 1000.0).clip(min=0.0)
    out.attrs['met_source'] = 'AORC v1.1'
    out.attrs['wind_reference_height_m'] = 10.0
    logging.info('  converted AORC -> DHSVM variables (no estimated fields)')
    return out


def convertHRRRToDHSVM(ds: xr.Dataset) -> xr.Dataset:
    """Convert an HRRR dataset to DHSVM's variables and units.

    Like AORC, HRRR carries every field DHSVM needs, so nothing is
    estimated:

    ==========================  =====================  ================
    HRRR                        DHSVM                  conversion
    ==========================  =====================  ================
    ``TMP`` [K]                 ``Tair`` [C]           -273.15
    ``UGRD``/``VGRD`` [m/s]     ``Wind`` [m/s]         hypot
    ``RH`` [%]                  ``RH`` [%]             identity
    ``DSWRF`` [W/m2]            ``Sin`` [W/m2]         identity
    ``DLWRF`` [W/m2]            ``Lin`` [W/m2]         identity
    ``APCP`` [kg/m2 per hour]   ``Precip`` [m/h]       /1000
    ==========================  =====================  ================
    """
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    out['Tair'] = ds['TMP'] - 273.15
    out['Wind'] = np.sqrt(ds['UGRD'] ** 2 + ds['VGRD'] ** 2)
    out['RH'] = ds['RH'].clip(min=1.0, max=100.0)
    out['Sin'] = ds['DSWRF'].clip(min=0.0)
    out['Lin'] = ds['DLWRF'].clip(min=0.0)
    out['Precip'] = (ds['APCP'] / 1000.0).clip(min=0.0)
    out.attrs['met_source'] = 'NOAA HRRR'
    out.attrs['wind_reference_height_m'] = 10.0
    logging.info('  converted HRRR -> DHSVM variables (no estimated fields)')
    return out


def convertDaymetToDHSVM(ds: xr.Dataset,
                         wind_speed: float = 2.0,
                         time_dim: str = 'time') -> xr.Dataset:
    """Convert a Daymet dataset to DHSVM's variables and units.

    Daymet is **daily** and lacks both wind and longwave, so this
    conversion estimates more than the others:

    * temperature is the mean of ``tmin`` and ``tmax``;
    * shortwave is ``srad * dayl / 86400``, converting Daymet's
      daylight-average flux to a 24-hour average;
    * relative humidity comes from Daymet's ``vp`` and the mean
      temperature;
    * **wind is assumed constant** at ``wind_speed`` -- Daymet has no
      wind field at all;
    * **longwave is estimated** with :func:`estimateLongwave`.

    Because it is daily, a DHSVM run driven by Daymet must use a 24-hour
    timestep, which suppresses the diurnal snowmelt and ET cycle.  Use
    AORC or HRRR when sub-daily physics matters; Daymet is here for long
    historical periods and for regions AORC does not cover.
    """
    out = xr.Dataset(coords=ds.coords, attrs=dict(ds.attrs))
    tair_c = (ds['tmin'] + ds['tmax']) / 2.0
    out['Tair'] = tair_c
    out['Wind'] = xr.full_like(tair_c, float(wind_speed))
    out['RH'] = xr.apply_ufunc(vaporPressureToRH, ds['vp'], tair_c,
                               dask='parallelized', output_dtypes=['float64'])
    out['Sin'] = (ds['srad'] * ds['dayl'] / 86400.0).clip(min=0.0)
    out['Lin'] = xr.apply_ufunc(estimateLongwave, tair_c, out['RH'],
                                dask='parallelized', output_dtypes=['float64'])
    out['Precip'] = (ds['prcp'] / 1000.0).clip(min=0.0)   # mm/day -> m/day

    out.attrs['met_source'] = 'Daymet v4'
    out.attrs['wind_reference_height_m'] = 10.0
    out.attrs['estimated_fields'] = 'Wind (constant), Lin (Prata 1996)'
    logging.warning(f'  Daymet has no wind field: assuming a constant '
                    f'{wind_speed:g} m/s everywhere')
    logging.warning('  Daymet has no longwave field: estimating it with '
                    'Prata (1996) clear-sky emissivity')
    return out


CONVERTERS = {
    'AORC': convertAORCToDHSVM,
    'HRRR': convertHRRRToDHSVM,
    'DayMet': convertDaymetToDHSVM,
}


def convertToDHSVM(ds: xr.Dataset, source_name: str, **kwargs) -> xr.Dataset:
    """Dispatch to the right converter by source name."""
    for key, fn in CONVERTERS.items():
        if key.lower() in source_name.lower():
            return fn(ds, **kwargs)
    raise ValueError(f'No DHSVM converter registered for met source '
                     f'"{source_name}".  Known: {list(CONVERTERS)}')


# ---------------------------------------------------------------------------
# Station placement
# ---------------------------------------------------------------------------

def toDatetimeIndex(times) -> pd.DatetimeIndex:
    """Coerce any of xarray's time representations to a pandas DatetimeIndex.

    Watershed Workflow's ``ManagerDataset`` normalises every dataset's time
    axis to **cftime** objects, because ATS runs on non-standard calendars
    (Daymet's 365-day ``noleap``, in particular).  DHSVM has no calendar
    abstraction at all -- it writes a literal ``MM/DD/YYYY-HH`` stamp -- and
    ``pandas.to_datetime`` refuses cftime objects outright.

    This converts cftime to real datetimes field by field.  For a proleptic
    or standard Gregorian calendar that is exact.  For ``noleap`` or
    ``360_day`` data it is a *relabelling*: the dates are reinterpreted on
    the real calendar, which is the only sensible thing to hand DHSVM, and
    is why sub-daily DHSVM runs should be driven by AORC or HRRR rather
    than by a noleap product.
    """
    arr = np.asarray(times)
    if arr.size == 0:
        return pd.DatetimeIndex([])

    first = arr.flat[0]
    if isinstance(first, cftime.datetime):
        cal = getattr(first, 'calendar', 'standard')
        if cal not in ('standard', 'gregorian', 'proleptic_gregorian'):
            logging.warning(f'  forcing calendar is "{cal}"; relabelling onto the '
                            f'real calendar for DHSVM, which has no calendar '
                            f'abstraction')
        return pd.DatetimeIndex([
            datetime.datetime(t.year, t.month, t.day, t.hour, t.minute,
                              int(getattr(t, 'second', 0)))
            for t in arr])
    return pd.DatetimeIndex(pd.to_datetime(arr))


class MetStation:
    """One DHSVM meteorological station.

    Attributes
    ----------
    name : str
        Station name, written into the config file.
    easting, northing : float
        Position in the model grid's projected CRS.
    lat, lon : float
        Position in degrees, used to sample the met dataset.
    elevation : float
        Elevation in metres, sampled from the model DEM.  This is the
        reference elevation for DHSVM's lapse-rate corrections.
    row, col : int
        Grid indices of the containing cell.
    filename : str
        Path to the station's forcing file.
    """

    def __init__(self, name, easting, northing, lat, lon, elevation,
                 row, col, filename=''):
        self.name = name
        self.easting = float(easting)
        self.northing = float(northing)
        self.lat = float(lat)
        self.lon = float(lon)
        self.elevation = float(elevation)
        self.row = int(row)
        self.col = int(col)
        self.filename = filename

    def __repr__(self):
        return (f'MetStation({self.name}, ({self.row},{self.col}), '
                f'{self.elevation:.1f} m)')


def placeStations(grid: ModelGrid,
                  dem: np.ndarray,
                  met: xr.Dataset,
                  lat_name: str = 'latitude',
                  lon_name: str = 'longitude',
                  max_stations: Optional[int] = None,
                  inside_only: bool = True) -> List[MetStation]:
    """Place DHSVM stations at the met dataset's native grid points.

    Using the forcing dataset's own cell centres -- rather than an
    arbitrary array of points -- means no spatial interpolation happens
    before DHSVM sees the data, so the model's own inverse-distance
    scheme is the only interpolation in the chain.

    Parameters
    ----------
    grid : ModelGrid
        The model grid.
    dem : np.ndarray
        Elevation on the model grid, used to give each station its
        reference elevation.
    met : xr.Dataset
        Converted met dataset carrying latitude/longitude coordinates,
        either 1D (AORC, Daymet) or 2D curvilinear (HRRR).
    lat_name, lon_name : str, optional
        Coordinate names in ``met``.
    max_stations : int, optional
        Cap on station count.  DHSVM interpolates from *every* station at
        *every* timestep, so cost grows linearly with station count while
        the added information falls off quickly.  When the cap is
        exceeded, stations are thinned on a regular stride **in the
        source dataset's own grid indices**, which keeps the retained
        set spatially even -- a flat stride through a raster-ordered
        list would leave whole rows unrepresented.
    inside_only : bool, optional
        Keep only stations whose cell is inside the basin mask.  DHSVM
        itself discards outside stations unless ``Outside = TRUE``.

    Returns
    -------
    list of MetStation
    """
    lat = met[lat_name].values
    lon = met[lon_name].values
    if lon.max() > 180.0:
        lon = np.where(lon > 180.0, lon - 360.0, lon)

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        idx = [(i, j) for i in range(lat.size) for j in range(lon.size)]
    elif lat.ndim == 2:
        lat2d, lon2d = lat, lon
        idx = [(i, j) for i in range(lat2d.shape[0]) for j in range(lat2d.shape[1])]
    else:
        raise ValueError(f'Unexpected lat/lon dimensionality: '
                         f'{lat.shape} / {lon.shape}')

    latlon = ww_dhsvm.crs.from_epsg(4326)
    import pyproj
    tf = pyproj.Transformer.from_crs(latlon, grid.crs, always_xy=True)

    mask = grid.mask if grid.mask is not None else np.ones(grid.shape, 'uint8')

    stations = []
    for (i, j) in idx:
        la, lo = float(lat2d[i, j]), float(lon2d[i, j])
        if not (np.isfinite(la) and np.isfinite(lo)):
            continue
        east, north = tf.transform(lo, la)
        r, c = grid.rowcol(east, north)
        if not (0 <= r < grid.nrows and 0 <= c < grid.ncols):
            continue
        if inside_only and mask[r, c] == 0:
            continue
        elev = float(dem[r, c])
        if not np.isfinite(elev):
            continue
        stations.append(MetStation(
            name=f'met_{la:.5f}_{lo:.5f}'.replace('-', 'm'),
            easting=east, northing=north, lat=la, lon=lo,
            elevation=elev, row=r, col=c))
        stations[-1].src_index = (i, j)

    if not stations:
        raise ValueError(
            'No meteorological stations fall inside the basin.  The met '
            'dataset may not cover this domain, or the basin may be smaller '
            'than one met grid cell -- in the latter case, pass '
            'inside_only=False to keep the nearest surrounding cells.')

    if max_stations is not None and len(stations) > max_stations:
        stations = _thinSpatially(stations, max_stations)

    elevs = [s.elevation for s in stations]
    logging.info(f'  placed {len(stations)} met stations; elevation '
                 f'{min(elevs):.0f}-{max(elevs):.0f} m '
                 f'(mean {np.mean(elevs):.0f} m)')
    return stations


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _timestampFormat(timestep_hours: float) -> str:
    """Pick DHSVM's timestamp format for a given timestep."""
    if abs(timestep_hours - round(timestep_hours)) < 1e-9 and timestep_hours >= 1:
        return '%m/%d/%Y-%H'
    return '%m/%d/%Y-%H:%M'


def writeStationFile(filename: str,
                     df: pd.DataFrame,
                     timestep_hours: float) -> str:
    """Write one DHSVM station forcing file.

    Parameters
    ----------
    filename : str
        Destination path.
    df : pd.DataFrame
        Indexed by timestamp, with columns ``Tair`` [C], ``Wind`` [m/s],
        ``RH`` [%], ``Sin`` [W/m^2], ``Lin`` [W/m^2] and ``Precip``
        [**metres per hour**].  Precipitation is multiplied by
        ``timestep_hours`` here, which is the one place that conversion
        belongs.
    timestep_hours : float
        Model timestep, hours.

    Returns
    -------
    str
        ``filename``.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fmt = _timestampFormat(timestep_hours)

    df = repairGaps(df, os.path.basename(filename))

    tair = np.asarray(df['Tair'], dtype='float64')
    wind = np.clip(np.asarray(df['Wind'], dtype='float64'), 0.1, None)
    rh = np.clip(np.asarray(df['RH'], dtype='float64'), 1.0, 100.0)
    sin_ = np.clip(np.asarray(df['Sin'], dtype='float64'), 0.0, 1380.0)
    lin = np.clip(np.asarray(df['Lin'], dtype='float64'), 0.0, 1800.0)
    precip = np.clip(np.asarray(df['Precip'], dtype='float64'), 0.0, None) * timestep_hours

    stamps = [t.strftime(fmt) for t in toDatetimeIndex(df.index)]

    with open(filename, 'w') as fid:
        for k in range(len(df)):
            fid.write(f'{stamps[k]} {tair[k]:.4f} {wind[k]:.4f} {rh[k]:.4f} '
                      f'{sin_[k]:.4f} {lin[k]:.4f} {precip[k]:.6f}\n')
    return filename


#: Longest run of consecutive missing timesteps that :func:`repairGaps`
#: will interpolate through before refusing.
MAX_INTERPOLATED_GAP = 8


def repairGaps(df: pd.DataFrame,
               label: str = '',
               max_gap: int = MAX_INTERPOLATED_GAP) -> pd.DataFrame:
    """Fill short gaps in a station series, and refuse long ones.

    This is not defensive boilerplate -- it fixes a failure mode that is
    both real and silent.  AORC v1.1 has isolated missing hours: over the
    Connecticut River Basin in 2025, **five of 46** forcing stations each
    carried two missing values.  DHSVM interpolates its forcing from
    *every* station at *every* timestep, so one bad cell is enough -- in
    an earlier build over a smaller domain a single such station turned
    the last 17% of a year-long simulation into NaN, with no warning from
    the model, and with the run still exiting 0.

    Short gaps are interpolated linearly in time, which for a 3-hour
    timestep is a defensible reconstruction of temperature, humidity,
    wind and radiation, and a conservative one for precipitation (it
    bridges rather than zeroing a storm).  Leading and trailing gaps take
    the nearest value.  A gap longer than ``max_gap`` timesteps raises,
    because at that point the series should be refetched or a different
    product used rather than invented.

    Parameters
    ----------
    df : pd.DataFrame
        Station series indexed by time.
    label : str, optional
        Name used in log messages, usually the station filename.
    max_gap : int, optional
        Longest run of consecutive missing values to interpolate.

    Returns
    -------
    pd.DataFrame
        A copy with gaps filled.
    """
    n_missing = int(df.isna().sum().sum())
    if n_missing == 0:
        return df

    out = df.copy()
    for col in out.columns:
        na = out[col].isna()
        if not na.any():
            continue
        # Longest consecutive run of NaN in this column.
        runs = (na != na.shift()).cumsum()[na]
        longest = int(runs.value_counts().max()) if len(runs) else 0
        if longest > max_gap:
            raise ValueError(
                f'{label or "station"}: {col} has a gap of {longest} consecutive '
                f'timesteps, longer than max_gap={max_gap}.  Refetch the forcing '
                f'or choose another product rather than interpolating this far.')
        out[col] = out[col].interpolate(method='time', limit_direction='both')

    still = int(out.isna().sum().sum())
    if still:
        raise ValueError(f'{label or "station"}: {still} values remain missing '
                         f'after interpolation.')

    logging.warning(f'  {label or "station"}: filled {n_missing} missing value(s) '
                    f'by time interpolation -- a single NaN reaching DHSVM '
                    f'propagates across the whole basin')
    return out


def extractStationSeries(met: xr.Dataset,
                         station: MetStation,
                         time_dim: str = 'time') -> pd.DataFrame:
    """Pull one station's time series out of the converted met dataset."""
    i, j = station.src_index
    dims = [d for d in met['Tair'].dims if d != time_dim]
    sel = {dims[0]: i, dims[1]: j} if len(dims) == 2 else {dims[0]: i}
    sub = met.isel(sel)
    data = {v: np.asarray(sub[v].values, dtype='float64')
            for v in ('Tair', 'Wind', 'RH', 'Sin', 'Lin', 'Precip')}
    times = toDatetimeIndex(met[time_dim].values)
    return pd.DataFrame(data, index=times)


def writeAllStations(outdir: str,
                     met: xr.Dataset,
                     stations: List[MetStation],
                     timestep_hours: float,
                     time_dim: str = 'time') -> List[MetStation]:
    """Write a forcing file for every station and record its path.

    Returns
    -------
    list of MetStation
        The same stations, with ``filename`` populated.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(f'Writing {len(stations)} DHSVM station forcing files '
                 f'({timestep_hours:g} h timestep) -> {outdir}')

    total_precip = []
    for k, st in enumerate(stations):
        df = extractStationSeries(met, st, time_dim)
        path = os.path.join(outdir, f'{st.name}.txt')
        writeStationFile(path, df, timestep_hours)
        st.filename = path
        total_precip.append(float(df['Precip'].sum() * timestep_hours * 1000.0))
        if (k + 1) % 25 == 0 or k == len(stations) - 1:
            logging.info(f'    {k+1}/{len(stations)} written')

    logging.info(f'  basin-mean total precipitation over the period: '
                 f'{np.mean(total_precip):.0f} mm '
                 f'(range {np.min(total_precip):.0f}-{np.max(total_precip):.0f} mm)')
    return stations


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize(met: xr.Dataset, timestep_hours: float = 1.0) -> pd.DataFrame:
    """Summary statistics of the converted forcing, for QA.

    Returns
    -------
    pd.DataFrame
        One row per variable with min / mean / max and units, plus a
        ``plausible`` flag comparing against physically sensible ranges.
    """
    limits = {
        'Tair': (-50.0, 50.0, 'deg C'),
        'Wind': (0.0, 60.0, 'm/s'),
        'RH': (0.0, 100.0, '%'),
        'Sin': (0.0, 1380.0, 'W/m^2'),
        'Lin': (50.0, 600.0, 'W/m^2'),
        'Precip': (0.0, 0.25, 'm/h'),
    }
    rows = []
    for v, (lo, hi, unit) in limits.items():
        if v not in met:
            continue
        raw = np.asarray(met[v].values, dtype='float64')
        n_missing = int(np.count_nonzero(~np.isfinite(raw)))
        a = raw[np.isfinite(raw)]
        if a.size == 0:
            continue
        rows.append(dict(variable=v, units=unit, min=a.min(), mean=a.mean(),
                         max=a.max(), p01=np.percentile(a, 1),
                         p99=np.percentile(a, 99), n_missing=n_missing,
                         plausible=bool(a.min() >= lo - 1e-6 and a.max() <= hi + 1e-6)))
    df = pd.DataFrame(rows)

    logging.info('Forcing summary')
    logging.info('-' * 78)
    for _, r in df.iterrows():
        flag = 'ok' if r['plausible'] else 'OUT OF RANGE'
        gap = '' if not r['n_missing'] else f"   {r['n_missing']} MISSING"
        logging.info(f"  {r['variable']:<8s} {r['units']:<8s} "
                     f"min {r['min']:10.3f}  mean {r['mean']:10.3f}  "
                     f"max {r['max']:10.3f}   {flag}{gap}")
    total_missing = int(df['n_missing'].sum()) if len(df) else 0
    if total_missing:
        logging.warning(f'  {total_missing} missing value(s) in the source '
                        f'forcing; writeStationFile will interpolate short gaps, '
                        f'but a NaN reaching DHSVM propagates basin-wide')
    return df


def annualWaterBalanceCheck(met: xr.Dataset, timestep_hours: float) -> Dict[str, float]:
    """Report annual precipitation and mean radiation, for a sanity check.

    A basin's annual precipitation is the number a hydrologist knows by
    heart; if the forcing says 400 mm for New England, something is wrong
    with units or with the time axis long before DHSVM runs.
    """
    n_steps = met.sizes['time']
    years = n_steps * timestep_hours / 8766.0
    # Average over space *first*, then accumulate over time.  Summing over
    # time first with ``nansum`` turns a grid cell that is missing at every
    # step -- ocean, or outside the product's domain, which is most coastal
    # basins -- into a legitimate-looking 0 mm total, and the spatial mean
    # then reports basin precipitation about as low as the missing fraction
    # is large.  Averaging first simply omits those cells.
    spatial = [d for d in met['Precip'].dims if d != 'time']
    basin_mean = met['Precip'].mean(dim=spatial)
    total_mm = float(basin_mean.sum().values) * timestep_hours * 1000.0
    annual_mm = total_mm / max(years, 1e-9)

    out = dict(
        n_timesteps=int(n_steps),
        years=float(years),
        total_precip_mm=total_mm,
        annual_precip_mm=annual_mm,
        mean_temp_c=float(np.nanmean(met['Tair'].values)),
        mean_sin_wm2=float(np.nanmean(met['Sin'].values)),
        mean_lin_wm2=float(np.nanmean(met['Lin'].values)),
        mean_wind_ms=float(np.nanmean(met['Wind'].values)),
        mean_rh_pct=float(np.nanmean(met['RH'].values)),
    )
    logging.info(f"  period: {out['years']:.2f} yr over {out['n_timesteps']} steps")
    logging.info(f"  annual precipitation : {annual_mm:8.0f} mm/yr")
    logging.info(f"  mean air temperature : {out['mean_temp_c']:8.2f} deg C")
    logging.info(f"  mean shortwave in    : {out['mean_sin_wm2']:8.1f} W/m^2")
    logging.info(f"  mean longwave in     : {out['mean_lin_wm2']:8.1f} W/m^2")
    logging.info(f"  mean wind speed      : {out['mean_wind_ms']:8.2f} m/s")
    logging.info(f"  mean relative humidity: {out['mean_rh_pct']:7.1f} %")
    return out


def _thinSpatially(stations: List[MetStation], max_stations: int) -> List[MetStation]:
    """Thin a station list while keeping it spatially even.

    Stations arrive in raster order, so a flat stride would sample whole
    rows and skip others.  Instead this strides independently in the
    source dataset's row and column indices, which preserves coverage of
    the basin in both directions.
    """
    ii = np.array([s.src_index[0] for s in stations])
    jj = np.array([s.src_index[1] for s in stations])
    n_i = len(np.unique(ii))
    n_j = len(np.unique(jj))

    # Choose per-axis strides whose product thins by about the right factor.
    factor = np.sqrt(len(stations) / float(max_stations))
    si = max(1, int(round(n_i / max(n_i / factor, 1))))
    sj = max(1, int(round(n_j / max(n_j / factor, 1))))

    for _ in range(64):
        keep = [s for s in stations
                if (s.src_index[0] % si == 0) and (s.src_index[1] % sj == 0)]
        if len(keep) <= max_stations:
            break
        si += 1
        sj += 1
    else:
        keep = stations[:max_stations]

    if not keep:                       # strides overshot; fall back
        keep = stations[::max(1, len(stations) // max_stations)][:max_stations]

    logging.info(f'  thinning {len(stations)} candidate stations to {len(keep)} '
                 f'(source-grid stride {si} x {sj}) -- DHSVM interpolates from '
                 f'every station at every timestep, so this bounds runtime')
    return keep
