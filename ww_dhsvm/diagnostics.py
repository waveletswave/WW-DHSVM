"""Input quality assurance for WW-DHSVM.

DHSVM's own input validation is thin and its error messages are numeric
codes emitted after several minutes of initialization.  This module runs
the checks *before* the model does, and reports them in terms a
hydrologist can act on.

Three tiers of check:

**Structural** -- things DHSVM will reject outright: a soil class above
``Number of Soil Types``, a channel segment whose outlet does not exist,
a state file with the wrong number of matrices, soil shallower than the
root zone.  These are hard errors.

**Physical** -- things DHSVM will happily run with but which are almost
certainly wrong: a wilting point above field capacity, a basin whose
annual precipitation is half what the region receives, a drainage
density an order of magnitude off the mapped network.  These are
warnings that deserve a human decision.

**Descriptive** -- the numbers a modeller wants to see anyway: basin
area, elevation range, land-cover composition, channel count.  Not
pass/fail, but the fastest way to notice that the domain is not the one
you meant to build.
"""

from typing import Optional, Dict, List, Any
import logging

import numpy as np
import pandas as pd


class Report:
    """Accumulates checks and renders them as a readable report."""

    def __init__(self, title: str = 'WW-DHSVM input diagnostics'):
        self.title = title
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[tuple] = []
        self.sections: List[str] = []

    def section(self, name: str) -> 'Report':
        self.sections.append(name)
        self.info.append(('__section__', name))
        return self

    def error(self, msg: str) -> 'Report':
        self.errors.append(msg)
        self.info.append(('ERROR', msg))
        return self

    def warn(self, msg: str) -> 'Report':
        self.warnings.append(msg)
        self.info.append(('WARN', msg))
        return self

    def note(self, key: str, value: Any) -> 'Report':
        self.info.append((key, value))
        return self

    def check(self, condition: bool, ok_msg: str, fail_msg: str,
              fatal: bool = True) -> 'Report':
        """Record a pass/fail check."""
        if condition:
            self.info.append(('ok', ok_msg))
        elif fatal:
            self.error(fail_msg)
        else:
            self.warn(fail_msg)
        return self

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def render(self) -> str:
        """Render the whole report as text."""
        lines = ['', '=' * 78, f'  {self.title}', '=' * 78]
        for key, val in self.info:
            if key == '__section__':
                lines.append('')
                lines.append(f'  {val}')
                lines.append('  ' + '-' * 74)
            elif key == 'ERROR':
                lines.append(f'    [ERROR]  {val}')
            elif key == 'WARN':
                lines.append(f'    [warn ]  {val}')
            elif key == 'ok':
                lines.append(f'    [ ok  ]  {val}')
            else:
                lines.append(f'    {key:<34s} {val}')
        lines.append('')
        lines.append('  ' + '-' * 74)
        verdict = ('ALL STRUCTURAL CHECKS PASSED' if self.ok
                   else f'{len(self.errors)} STRUCTURAL ERROR(S) -- DHSVM WILL NOT RUN')
        lines.append(f'  {verdict}')
        if self.warnings:
            lines.append(f'  {len(self.warnings)} warning(s) -- review before trusting results')
        lines.append('=' * 78)
        lines.append('')
        return '\n'.join(lines)

    def log(self) -> 'Report':
        for line in self.render().splitlines():
            logging.info(line)
        return self


def _stats(a, mask=None):
    a = np.asarray(a, dtype='float64')
    if mask is not None:
        a = a[mask != 0]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return a


def checkInputs(grid,
                dem: np.ndarray,
                soil_map: np.ndarray,
                soil_depth: np.ndarray,
                veg_map: np.ndarray,
                soil_blocks: List[Dict[str, Any]],
                veg_blocks: List[Dict[str, Any]],
                network: Optional[Dict[str, Any]] = None,
                met_summary: Optional[Dict[str, float]] = None,
                expected_annual_precip_mm: Optional[float] = None,
                report: Optional[Report] = None) -> Report:
    """Run the full input-QA suite and return a :class:`Report`."""
    r = report or Report()
    mask = grid.mask

    # -- domain ------------------------------------------------------
    r.section('Domain')
    r.note('grid shape', f'{grid.nrows} rows x {grid.ncols} cols')
    r.note('cell size', f'{grid.cellsize:g} m')
    r.note('active cells', f'{grid.n_active:,} '
                           f'({100.0*grid.n_active/(grid.nrows*grid.ncols):.1f}%)')
    r.note('basin area', f'{grid.basin_area_km2:.2f} km^2')
    lat, lon = grid.centerLatLon()
    r.note('centre lat/lon', f'{lat:.5f}, {lon:.5f}')
    r.check(grid.n_active > 100,
            f'{grid.n_active:,} active cells is enough to resolve the basin',
            f'Only {grid.n_active} active cells: the grid is too coarse for this '
            f'basin, or the mask failed.  Reduce cell size.')

    # -- elevation ---------------------------------------------------
    r.section('Elevation')
    d = _stats(dem, mask)
    if d is None:
        r.error('DEM has no valid in-basin values.')
    else:
        r.note('elevation range', f'{d.min():.1f} to {d.max():.1f} m '
                                  f'(relief {d.max()-d.min():.1f} m)')
        r.note('mean elevation', f'{d.mean():.1f} m')
        r.check(np.isfinite(dem[mask != 0]).all() if mask is not None
                else np.isfinite(dem).all(),
                'DEM has no NaN inside the basin',
                'DEM contains NaN inside the basin; DHSVM binary maps have no '
                'nodata convention, so these must be filled.')
        r.check(d.min() > -500 and d.max() < 9000,
                'Elevations are physically plausible',
                f'Elevation range {d.min():.1f}-{d.max():.1f} m is implausible; '
                f'check DEM units and vertical datum.')

    # -- soils -------------------------------------------------------
    r.section('Soils')
    r.note('soil types', f'{len(soil_blocks)}')
    hi = int(np.max(soil_map))
    r.check(hi <= len(soil_blocks),
            f'Highest soil class {hi} is within Number of Soil Types '
            f'({len(soil_blocks)})',
            f'soil.bin holds class {hi} but only {len(soil_blocks)} soil types '
            f'are declared (InitTerrainMaps.c error 32).')
    r.check(int(np.min(soil_map)) >= 1,
            'All soil classes are >= 1',
            'soil.bin contains class 0; DHSVM soil classes are 1-based.')

    sd = _stats(soil_depth, mask)
    if sd is not None:
        r.note('soil depth range', f'{sd.min():.2f} to {sd.max():.2f} m '
                                   f'(mean {sd.mean():.2f} m)')
        r.check(sd.min() > 0,
                'Soil depth is positive everywhere',
                'Soil depth is zero or negative somewhere in the basin.')

        # Root zone must fit inside the soil column (CheckOut.c).
        required = np.zeros(256)
        for b in veg_blocks:
            required[b['id']] = float(np.sum(b['root_zone_depths']))
        need = required[veg_map]
        viol = (soil_depth <= need)
        if mask is not None:
            viol &= (mask != 0)
        n_viol = int(viol.sum())
        r.check(n_viol == 0,
                'Soil depth exceeds the root zone in every cell',
                f'{n_viol:,} cells have soil shallower than their vegetation\'s '
                f'root zone; CheckOut.c treats this as fatal.  Call '
                f'terrain.enforceSoilDepthBelowRootZone.')

    # -- vegetation --------------------------------------------------
    r.section('Vegetation')
    r.note('vegetation types', f'{len(veg_blocks)}')
    hiv = int(np.max(veg_map))
    r.check(hiv <= len(veg_blocks),
            f'Highest vegetation class {hiv} is within Number of Vegetation '
            f'Types ({len(veg_blocks)})',
            f'veg.bin holds class {hiv} but only {len(veg_blocks)} types are '
            f'declared.')
    r.check(int(np.min(veg_map)) >= 1,
            'All vegetation classes are >= 1',
            'veg.bin contains class 0; DHSVM vegetation classes are 1-based.')

    tallest = max(b['height'][0] for b in veg_blocks)
    r.note('tallest canopy', f'{tallest:.1f} m')

    forest_frac = 0.0
    active = veg_map[mask != 0] if mask is not None else veg_map
    for b in veg_blocks:
        if b['overstory']:
            forest_frac += np.count_nonzero(active == b['id']) / max(active.size, 1)
    r.note('forested fraction', f'{100*forest_frac:.1f}%')

    # -- channels ----------------------------------------------------
    if network is not None:
        r.section('Channel network')
        segs = network['segments']
        r.note('segments', f'{len(segs):,}')
        r.note('channel cells', f'{int(network["channel_mask"].sum()):,}')
        r.note('drainage density', f'{network["drainage_density"]:.3f} km/km^2')
        r.note('max Strahler order', f'{int(segs["order"].max())}')
        n_out = int((segs['outlet'] == 0).sum())
        r.note('outlet segments', f'{n_out}')
        r.check((segs['slope'] > 0).all(),
                'All channel slopes are positive',
                'Some channel segments have slope <= 0; channel.c rejects these.')
        r.check((segs['length'] > 0).all(),
                'All channel lengths are positive',
                'Some channel segments have length <= 0; channel.c rejects these.')
        dd = network['drainage_density']
        r.check(0.2 <= dd <= 5.0,
                f'Drainage density {dd:.2f} km/km^2 is in the usual range',
                f'Drainage density {dd:.2f} km/km^2 is outside the 0.2-5 km/km^2 '
                f'range typical of humid basins; revisit the channel-initiation '
                f'threshold.', fatal=False)
        if n_out > 1:
            r.warn(f'{n_out} outlet segments.  Expected for a hydrologic unit that '
                   f'is not a single closed watershed (a main-stem HUC8, say), but '
                   f'a headwater catchment should have exactly one.')

    # -- meteorology -------------------------------------------------
    if met_summary is not None:
        r.section('Meteorological forcing')
        for k in ('n_timesteps', 'years', 'annual_precip_mm', 'mean_temp_c',
                  'mean_sin_wm2', 'mean_lin_wm2', 'mean_wind_ms', 'mean_rh_pct'):
            if k in met_summary:
                r.note(k, f'{met_summary[k]:.3f}' if isinstance(met_summary[k], float)
                          else met_summary[k])
        p = met_summary.get('annual_precip_mm')
        if p is not None:
            r.check(200 <= p <= 6000,
                    f'Annual precipitation {p:.0f} mm/yr is physically plausible',
                    f'Annual precipitation {p:.0f} mm/yr is outside 200-6000 mm/yr; '
                    f'check the precipitation units and the time axis.',
                    fatal=False)
            if expected_annual_precip_mm:
                rel = abs(p - expected_annual_precip_mm) / expected_annual_precip_mm
                r.check(rel < 0.35,
                        f'Annual precipitation is within 35% of the expected '
                        f'{expected_annual_precip_mm:.0f} mm/yr for this region',
                        f'Annual precipitation {p:.0f} mm/yr differs by '
                        f'{100*rel:.0f}% from the expected '
                        f'{expected_annual_precip_mm:.0f} mm/yr.', fatal=False)
        t = met_summary.get('mean_temp_c')
        if t is not None:
            r.check(-25 <= t <= 35,
                    f'Mean air temperature {t:.1f} C is plausible',
                    f'Mean air temperature {t:.1f} C is implausible; check whether '
                    f'the forcing is still in Kelvin.', fatal=False)
        s = met_summary.get('mean_sin_wm2')
        if s is not None:
            r.check(40 <= s <= 350,
                    f'Mean shortwave {s:.0f} W/m^2 is plausible',
                    f'Mean shortwave {s:.0f} W/m^2 is outside 40-350 W/m^2; check '
                    f'whether a daylight-average flux was mistaken for a '
                    f'24-hour average.', fatal=False)

    return r


def compareOutputToExpectation(water_balance: Dict[str, float],
                               basin_area_km2: float,
                               report: Optional[Report] = None) -> Report:
    """Sanity-check a completed run's water balance.

    Checks that runoff and ET ratios are in the range a temperate basin
    can actually produce, and that the two plus storage change account
    for precipitation.
    """
    r = report or Report('WW-DHSVM output diagnostics')
    r.section('Water balance')
    for k, v in water_balance.items():
        r.note(k, f'{v:.3f}')

    rr = water_balance.get('runoff_ratio')
    if rr is not None and np.isfinite(rr):
        r.check(0.0 <= rr <= 1.0,
                f'Runoff ratio {rr:.3f} is within [0, 1]',
                f'Runoff ratio {rr:.3f} is outside [0, 1], which is impossible '
                f'without an external inflow.', fatal=False)
        r.check(0.15 <= rr <= 0.75,
                f'Runoff ratio {rr:.3f} is typical of a humid temperate basin',
                f'Runoff ratio {rr:.3f} is outside the 0.15-0.75 typical of humid '
                f'temperate basins.  For a short run this usually means the basin '
                f'is still filling storage from its cold start rather than that '
                f'anything is wrong.', fatal=False)

    ce = water_balance.get('channel_closure_error_mm')
    li = water_balance.get('channel_lateral_in_mm')
    if ce is not None and li:
        r.check(abs(ce) < 0.15 * abs(li),
                f'Channel routing closes to {ce:.1f} mm against '
                f'{li:.1f} mm of lateral inflow',
                f'The channel network loses {ce:.1f} mm against {li:.1f} mm of '
                f'lateral inflow.  This usually means reaches on '
                f'depression-filled flats have too little gradient to move '
                f'water -- raise min_slope in streams.extractNetwork.',
                fatal=False)

    er = water_balance.get('et_ratio')
    if er is not None and np.isfinite(er):
        r.check(0.0 <= er <= 1.2,
                f'ET ratio {er:.3f} is plausible',
                f'ET ratio {er:.3f} is implausible.', fatal=False)

    # DHSVM's own basin balance, from MassBalance.c:
    #
    #   Input  = Precip + SnowVaporFlux + CanopySnowVaporFlux + CulvertReturn
    #   Output = ChannelInt + RoadInt + ET
    #   Error  = dStorage + Output - Input
    #
    # Note ChannelInt -- not outlet discharge -- is the hillslope output
    # term: it is water leaving the land surface INTO the channel network.
    # Outlet discharge is what leaves the channel network, and the two differ
    # by channel storage.  Checking P - Q - ET - dSoil would therefore
    # "fail" on a perfectly closed run, which is why the closure reported
    # here is DHSVM's own.
    err = water_balance.get('dhsvm_closure_error_mm')
    p = water_balance.get('precip_mm')
    if err is not None and np.isfinite(err):
        r.note('DHSVM closure error (MassBalance.c)', f'{err:.3e} mm')
        r.check(abs(err) < max(0.001 * abs(p or 1.0), 0.01),
                f"DHSVM's own basin mass balance closes to {err:.2e} mm",
                f"DHSVM's basin mass balance leaves {err:.3f} mm unaccounted; "
                f'this is a model-side problem, not an input one.')

    ci = water_balance.get('channel_int_mm')
    q = water_balance.get('streamflow_mm')
    if ci and q is not None and np.isfinite(q):
        r.note('hillslope -> channel (ChannelInt)', f'{ci:.1f} mm')
        r.note('channel -> outlet (discharge)', f'{q:.1f} mm')
        r.note('retained as channel storage', f'{ci - q:.1f} mm')
        r.check(q <= ci * 1.02,
                'Outlet discharge does not exceed what entered the channels',
                f'Outlet discharge {q:.1f} mm exceeds channel inflow {ci:.1f} mm.',
                fatal=False)

    return r
