"""Readers for DHSVM's output files.

DHSVM writes ASCII time series into its ``Output Directory``.  The files
are simple but idiosyncratic -- three different timestamp formats appear
across them -- so this module normalizes everything to a pandas
``DatetimeIndex`` and documents the units, which the files themselves
only partly declare.

============================  ==========================================
file                          content
============================  ==========================================
``Aggregated.Values``         ~60 basin-average state and flux variables
``Mass.Balance``              per-timestep water balance terms
``Mass.Final.Balance``        the closing balance, as text
``Streamflow.Only``           outflow of every ``SAVE`` segment
``Stream.Flow``               inflow / lateral / outflow / storage change
``saturation_extent.txt``     fraction of the basin saturated
============================  ==========================================

Not every build writes every file.  The serial DHSVM 3.2 binary produces
``saturation_extent.txt``; the Global-Arrays parallel build was observed
not to.  :func:`readAll` therefore reports what it found rather than
failing on what is absent, and the figures degrade gracefully.

Units
-----
Channel discharge in ``Stream.Flow`` and ``Streamflow.Only`` is
**cubic metres per model timestep**, not m^3/s.  :func:`readStreamflow`
converts it to m^3/s when given the timestep, which is almost always
what you want for comparison against a USGS gauge.

Most depth-like quantities in ``Aggregated.Values`` are in **metres**
even where the header says otherwise; ``Mass.Balance`` mixes metres and
millimetres, and its column headers name the units explicitly.
"""

from typing import Optional, Dict, List, Any
import os
import re
import logging

import numpy as np
import pandas as pd


def _parseDates(series: pd.Series) -> pd.DatetimeIndex:
    """Parse any of DHSVM's three timestamp spellings.

    ``MM/DD/YYYY-HH:MM:SS`` (aggregated, mass balance),
    ``MM.DD.YYYY-HH:MM:SS`` (channel files), and the bare
    ``MM/DD/YYYY-HH`` used in the configuration file.
    """
    s = series.astype(str).str.strip().str.replace('.', '/', regex=False)
    for fmt in ('%m/%d/%Y-%H:%M:%S', '%m/%d/%Y-%H:%M', '%m/%d/%Y-%H'):
        try:
            return pd.DatetimeIndex(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    return pd.DatetimeIndex(pd.to_datetime(s, format='mixed'))


def readAggregated(filename: str) -> pd.DataFrame:
    """Read ``Aggregated.Values`` -- basin-average states and fluxes.

    Returns
    -------
    pd.DataFrame
        Indexed by time.  Column names come from the file header, which
        DHSVM writes with irregular spacing; they are cleaned here.
    """
    with open(filename) as fid:
        header = fid.readline()
    cols = [c for c in re.split(r'\s+', header.strip()) if c]

    df = pd.read_csv(filename, sep=r'\s+', skiprows=1, header=None,
                     engine='python')
    # DHSVM occasionally writes a trailing separator, giving one extra
    # all-NaN column; drop any columns the header does not name.
    if df.shape[1] > len(cols):
        df = df.iloc[:, :len(cols)]
    elif df.shape[1] < len(cols):
        cols = cols[:df.shape[1]]
    df.columns = cols

    df.index = _parseDates(df[cols[0]])
    df.index.name = 'time'
    df = df.drop(columns=[cols[0]])
    df = df.apply(pd.to_numeric, errors='coerce')
    logging.info(f'  read {len(df)} timesteps x {df.shape[1]} variables '
                 f'from {os.path.basename(filename)}')
    return df


def readMassBalance(filename: str) -> pd.DataFrame:
    """Read ``Mass.Balance`` -- the per-timestep water budget.

    The ``Error`` column is the closure residual in mm; it should stay
    at machine-noise level (|error| < 1e-3 mm per step).  A growing
    residual means a genuine bug or an unstable timestep.
    """
    df = readAggregated(filename)
    if 'Error' in df.columns:
        err = df['Error'].abs()
        logging.info(f'  mass-balance residual: max {err.max():.3e} mm, '
                     f'mean {err.mean():.3e} mm, cumulative {df["Error"].sum():.3e} mm')
    return df


def readStreamflow(filename: str,
                   timestep_hours: Optional[float] = None) -> pd.DataFrame:
    """Read ``Streamflow.Only`` -- outflow of every ``SAVE`` segment.

    Parameters
    ----------
    filename : str
        Path to ``Streamflow.Only``.
    timestep_hours : float, optional
        When given, values are converted from **m^3 per timestep** to
        **m^3/s**, and the columns keep their segment names.

    Returns
    -------
    pd.DataFrame
        Indexed by time, one column per saved segment.
    """
    with open(filename) as fid:
        header = fid.readline()
    cols = [c for c in re.split(r'\s+', header.strip()) if c]

    # DHSVM writes a date-only first data row, before any routing has
    # happened.  Pandas would infer a single column from it, so the column
    # names from the header are supplied explicitly.
    df = pd.read_csv(filename, sep=r'\s+', skiprows=1, header=None,
                     names=cols, engine='python')

    df.index = _parseDates(df[cols[0]])
    df.index.name = 'time'
    df = df.drop(columns=[cols[0]]).apply(pd.to_numeric, errors='coerce')

    # Drop that blank first row so the series starts at a real timestep.
    df = df.dropna(how='all')

    if timestep_hours is not None:
        df = df / (timestep_hours * 3600.0)
        logging.info(f'  converted streamflow from m^3/timestep to m^3/s '
                     f'({timestep_hours:g} h timestep)')

    logging.info(f'  read {len(df)} timesteps x {df.shape[1]} saved segments')
    return df


def readStreamFlowDetail(filename: str,
                         timestep_hours: Optional[float] = None) -> pd.DataFrame:
    """Read ``Stream.Flow`` -- the per-segment channel budget.

    ``channel_save_outflow_text`` in ``channel.c`` writes **two different
    row layouts** into this one file, and they have different field
    counts -- reading it with a single column spec silently mis-assigns
    every total.

    Per-segment rows (six numeric fields)::

        date  segment_id  inflow  lateral_inflow  outflow  storage_change  "name"

    One totals row per timestep (**seven** numeric fields, id 0)::

        date  0  lateral_inflow  outflow  storage  storage_change  error  "Totals"

    Note the totals row has no ``inflow`` column -- internal transfers
    cancel across the network -- and adds ``storage`` and a closure
    ``error``.  The two layouts are returned as two DataFrames.

    All volumes are m^3 per timestep unless ``timestep_hours`` converts
    them to m^3/s.

    Returns
    -------
    segments : pd.DataFrame
        Per-segment rows.
    totals : pd.DataFrame
        One row per timestep of network-wide totals.
    """
    seg_rows, tot_rows = [], []
    with open(filename) as fid:
        for line in fid:
            m = re.search(r'"([^"]*)"', line)
            name = m.group(1) if m else ''
            parts = line[:m.start()].split() if m else line.split()
            if len(parts) < 6:
                continue
            try:
                if name == 'Totals' and len(parts) >= 7:
                    tot_rows.append((parts[0], float(parts[2]), float(parts[3]),
                                     float(parts[4]), float(parts[5]),
                                     float(parts[6])))
                else:
                    seg_rows.append((parts[0], int(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4]),
                                     float(parts[5]), name))
            except ValueError:
                continue

    segs = pd.DataFrame(seg_rows, columns=['date', 'segment_id', 'inflow',
                                           'lateral_inflow', 'outflow',
                                           'storage_change', 'name'])
    tots = pd.DataFrame(tot_rows, columns=['date', 'lateral_inflow', 'outflow',
                                           'storage', 'storage_change', 'error'])
    for df in (segs, tots):
        if len(df):
            df.index = _parseDates(df['date'])
            df.index.name = 'time'
            df.drop(columns=['date'], inplace=True)

    if timestep_hours is not None:
        f = timestep_hours * 3600.0
        for c in ('inflow', 'lateral_inflow', 'outflow', 'storage_change'):
            if c in segs:
                segs[c] = segs[c] / f
        for c in ('lateral_inflow', 'outflow', 'storage_change', 'error'):
            if c in tots:
                tots[c] = tots[c] / f

    logging.info(f'  read {len(segs):,} segment-timestep records and '
                 f'{len(tots):,} network totals from {os.path.basename(filename)}')
    if len(tots):
        closure = tots['error'].abs().mean()
        logging.info(f'  channel-network closure error: mean |error| '
                     f'{closure:.4g} m^3/s')
    return segs, tots


def readSaturationExtent(filename: str) -> pd.Series:
    """Read ``saturation_extent.txt`` -- saturated fraction of the basin.

    A useful diagnostic of whether the subsurface parameterization is
    behaving: a basin permanently at 0 has lateral conductivity far too
    high, one permanently near 1 far too low.
    """
    df = pd.read_csv(filename, sep=r'\s+', header=None,
                     names=['date', 'saturated_fraction'], engine='python')
    s = pd.Series(df['saturated_fraction'].values, index=_parseDates(df['date']),
                  name='saturated_fraction')
    s.index.name = 'time'
    logging.info(f'  saturated fraction: mean {s.mean():.4f}, max {s.max():.4f}')
    return s


def readFinalMassBalance(filename: str) -> Dict[str, float]:
    """Parse ``Mass.Final.Balance`` into a dictionary of terms, mm."""
    out = {}
    with open(filename) as fid:
        for line in fid:
            m = re.match(r'\s*(.+?)\s*\.{3,}\s*(-?[\d.eE+]+)\s*$', line)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
    if out:
        logging.info(f'  final mass balance: {len(out)} terms, '
                     f'error {out.get("Mass Error (mm)", float("nan")):.4f} mm')
    return out


def readAll(output_dir: str, timestep_hours: Optional[float] = None) -> Dict[str, Any]:
    """Read every DHSVM output file present in a directory.

    Returns
    -------
    dict
        Keyed ``'aggregated'``, ``'mass_balance'``, ``'streamflow'``,
        ``'stream_detail'``, ``'saturation'``, ``'final_balance'``.
        Missing files are simply absent from the dict.
    """
    readers = [
        ('aggregated', 'Aggregated.Values', lambda p: readAggregated(p)),
        ('mass_balance', 'Mass.Balance', lambda p: readMassBalance(p)),
        ('streamflow', 'Streamflow.Only', lambda p: readStreamflow(p, timestep_hours)),
        ('stream_detail', 'Stream.Flow', lambda p: readStreamFlowDetail(p, timestep_hours)),
        ('saturation', 'saturation_extent.txt', lambda p: readSaturationExtent(p)),
        ('final_balance', 'Mass.Final.Balance', lambda p: readFinalMassBalance(p)),
    ]
    out = {}
    logging.info(f'Reading DHSVM output from {output_dir}')
    for key, fname, fn in readers:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            logging.info(f'  {fname}: not present')
            continue
        try:
            res = fn(path)
            if key == 'stream_detail' and isinstance(res, tuple):
                out['stream_detail'], out['stream_totals'] = res
            else:
                out[key] = res
        except Exception as exc:
            logging.warning(f'  {fname}: could not be read ({exc})')
    return out


def waterBalanceSummary(results: Dict[str, Any],
                        basin_area_km2: float,
                        timestep_hours: float) -> Dict[str, float]:
    """Close the basin water balance from DHSVM output.

    Reports precipitation, evapotranspiration, streamflow and storage
    change over the run, all as basin-average depths in mm, plus the
    runoff and ET ratios that make the result immediately interpretable.

    Basin outflow is taken from the **network totals** row of
    ``Stream.Flow`` rather than by summing the ``SAVE``-flagged segments.
    Those two agree only when every outlet happens to be flagged; the
    totals row is what ``channel.c`` itself accumulates over all segments
    with no downstream neighbour, so it is the authoritative number.  The
    channel network's own closure error is reported alongside it, because
    a large one means water is disappearing inside the routing scheme --
    most often because reaches on depression-filled flats have no
    gradient to move it.

    Returns
    -------
    dict
        Depths in mm and dimensionless ratios.
    """
    out: Dict[str, float] = {}
    mb = results.get('mass_balance')

    if mb is not None:
        def _sum(col, scale=1.0):
            return float(mb[col].sum() * scale) if col in mb.columns else float('nan')
        out['precip_mm'] = _sum('Precip(m)', 1000.0)
        out['et_mm'] = _sum('TotalET', 1000.0)
        # ChannelInt is the *hillslope* output term in DHSVM's own balance:
        # water leaving the land surface into the channel network.  It is not
        # the same as discharge at the outlet, which is what leaves the
        # channel network -- the two differ by channel storage.
        out['channel_int_mm'] = _sum('ChannelInt', 1000.0)
        out['road_int_mm'] = _sum('RoadInt', 1000.0)
        # Vapour fluxes are INPUTS in MassBalance.c (they are negative when
        # the pack loses mass), so they are carried with their sign.
        out['snow_vapor_flux_mm'] = _sum('SnowVaporFlux', 1000.0)
        out['canopy_snow_vapor_flux_mm'] = _sum('CanopySnowVaporFlux', 1000.0)
        if 'Error' in mb.columns:
            out['dhsvm_closure_error_mm'] = float(mb['Error'].sum() * 1.0)
        # TotSoilMoist is a per-timestep state (m), so storage change is the
        # difference of its endpoints, not a sum.
        if 'TotSoilMoist' in mb.columns:
            tsm = pd.to_numeric(mb['TotSoilMoist'], errors='coerce').dropna()
            if len(tsm) > 1:
                out['soil_storage_change_mm'] = float(
                    (tsm.iloc[-1] - tsm.iloc[0]) * 1000.0)
        if 'Swq' in mb.columns:
            swq = pd.to_numeric(mb['Swq'], errors='coerce').dropna()
            if len(swq) > 1:
                out['snow_storage_change_mm'] = float(
                    (swq.iloc[-1] - swq.iloc[0]) * 1000.0)

    seconds = timestep_hours * 3600.0
    area_m2 = basin_area_km2 * 1e6

    tots = results.get('stream_totals')
    if tots is not None and len(tots):
        # Already m^3/s if the reader was given a timestep.
        q = tots['outflow']
        out['mean_q_m3s'] = float(q.mean())
        out['peak_q_m3s'] = float(q.max())
        out['streamflow_mm'] = float(q.sum() * seconds / area_m2 * 1000.0)
        out['channel_lateral_in_mm'] = float(
            tots['lateral_inflow'].sum() * seconds / area_m2 * 1000.0)
        out['channel_closure_error_mm'] = float(
            tots['error'].sum() * seconds / area_m2 * 1000.0)
    else:
        sf = results.get('streamflow')
        if sf is not None and len(sf.columns):
            outlet_cols = [c for c in sf.columns if c.upper().startswith('OUTLET')]
            q = sf[outlet_cols or list(sf.columns)].sum(axis=1)
            out['mean_q_m3s'] = float(q.mean())
            out['peak_q_m3s'] = float(q.max())
            out['streamflow_mm'] = float(q.sum() * seconds / area_m2 * 1000.0)

    p = out.get('precip_mm', float('nan'))
    if p and np.isfinite(p) and p > 0:
        if 'streamflow_mm' in out:
            out['runoff_ratio'] = out['streamflow_mm'] / p
        if 'et_mm' in out:
            out['et_ratio'] = out['et_mm'] / p

    logging.info('Water balance over the simulation')
    logging.info('-' * 60)
    for k, v in out.items():
        logging.info(f'  {k:<26s} {v:12.3f}')
    return out
