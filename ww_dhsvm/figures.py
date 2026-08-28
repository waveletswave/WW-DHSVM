"""Composed diagnostic figures for a WW-DHSVM case.

:mod:`ww_dhsvm.plot` supplies the primitives -- a map, a time series, a
stat tile -- with the styling rules.  This module composes them into the
specific multi-panel figures a modeller wants to see at each stage of a
setup, so a notebook cell can be one call rather than thirty lines of
matplotlib.

Every figure here answers a question:

===============================  =========================================
figure                           question it answers
===============================  =========================================
:func:`domainOverview`           Is this the basin I meant to build?
:func:`terrainPanel`             Is the DEM sane, and what did
                                 conditioning change?
:func:`flowRoutingPanel`         Does water route the way the map says?
:func:`channelPanel`             Is the channel network the right density
                                 and topology?
:func:`soilPanel`                What soils, how deep, how much
                                 plant-available water?
:func:`vegetationPanel`          What cover, how tall, what seasonality?
:func:`forcingPanel`             Is the forcing physically plausible?
:func:`forcingComparison`        Do two met products agree?
:func:`inputDashboard`           One screen: is this case ready to run?
:func:`outputPanel`              Did the run behave like a watershed?
===============================  =========================================
"""

from typing import Optional, Dict, Any, List, Sequence
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap

import ww_dhsvm.plot as P
import ww_dhsvm.vegetation as _veg


#: Height, inches, of every figure whose left-hand panel is a basin map.
#: One number for all of them so the maps come out the same size when the
#: figures are stacked in a document -- which is how they are read.
MAP_ROW_HEIGHT = 7.6


def _mask(a, grid):
    return np.where(grid.mask != 0, a, np.nan) if grid.mask is not None else a


def _stats(a, grid):
    v = _mask(np.asarray(a, dtype='float64'), grid)
    return v[np.isfinite(v)]



def _toKm(gdf, crs):
    """Rescale a GeoDataFrame's coordinates from metres to kilometres.

    The raster panels are drawn with their extent in km so the tick
    labels stay readable, so any vector overlay must be rescaled to match
    -- otherwise it lands thousands of units off-axis and matplotlib
    silently expands the limits until the raster is a speck.
    """
    import shapely.affinity
    g = gdf.to_crs(crs).copy()
    g[g.geometry.name] = g.geometry.apply(
        lambda geom: shapely.affinity.scale(geom, xfact=1e-3, yfact=1e-3,
                                            origin=(0, 0)))
    return g


# ---------------------------------------------------------------------------

def domainOverview(grid, watershed, dem=None, reaches=None,
                   height=MAP_ROW_HEIGHT):
    """The basin, its mask, and where on Earth it is.

    Column widths come from the basin's own aspect ratio, so a long thin
    catchment does not end up as a sliver in the middle of an empty
    panel.
    """
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    text_w = 3.4
    fig = plt.figure(figsize=(2 * mw + text_w, height))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[mw, mw, text_w], wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    if dem is not None:
        P.mapRaster(dem, grid, ax=ax, cmap=P.HYPSOMETRIC, label='elevation (m)',
                    title='Model domain',
                    subtitle=f'{grid.nrows} x {grid.ncols} cells at '
                             f'{grid.cellsize:g} m',
                    hillshade=P.hillshade(dem, grid.cellsize))
    ws = _toKm(watershed, grid.crs)
    ws.boundary.plot(ax=ax, color=P.INK, linewidth=1.4, zorder=5)
    if reaches is not None and len(reaches):
        _toKm(reaches, grid.crs).plot(ax=ax, color=P.SERIES[0], linewidth=0.45,
                                      alpha=0.75, zorder=4)
    ax.set_xlim(*P._extent(grid)[:2]); ax.set_ylim(*P._extent(grid)[2:])

    ax2 = fig.add_subplot(gs[0, 1])
    active = 100.0 * grid.n_active / (grid.nrows * grid.ncols)
    # Two classes named in the subtitle need no legend box, and the box
    # would sit on top of the summary column beside it.
    P.mapCategorical(grid.mask, grid, labels={0: 'outside basin', 1: 'inside basin'},
                     colors={0: '#e8e7e3', 1: P.SERIES[0]}, ax=ax2,
                     mask=np.ones(grid.shape, 'uint8'), legend=False,
                     title='Basin mask',
                     subtitle=f'blue inside, grey outside (DHSVM value 0)')
    ws.boundary.plot(ax=ax2, color=P.INK, linewidth=1.2, zorder=5)
    ax2.set_xlim(*P._extent(grid)[:2]); ax2.set_ylim(*P._extent(grid)[2:])

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    lat, lon = grid.centerLatLon()
    rows = [
        ('Basin area', f'{grid.basin_area_km2:,.0f} km²'),
        ('Active cells', f'{grid.n_active:,}'),
        ('Grid', f'{grid.nrows} × {grid.ncols} @ {grid.cellsize:g} m'),
        ('CRS', f'EPSG:{grid.crs.to_epsg()}'),
        ('Mask coverage', f'{100.0 * grid.n_active / (grid.nrows * grid.ncols):.0f}'
                          f'% of the array'),
        ('Centre', f'{lat:.3f}°N, {abs(lon):.3f}°W'),
    ]
    if dem is not None:
        d = _stats(dem, grid)
        rows += [('Elevation', f'{d.min():,.0f} – {d.max():,.0f} m'),
                 ('Mean elevation', f'{d.mean():,.0f} m')]
    # The heading goes through set_title, like every other panel's, so it
    # sits on the same line as the map titles instead of one title-pad lower.
    P._finish(ax3, 'Domain summary')
    for i, (k, v) in enumerate(rows):
        y = 0.97 - i * 0.105
        ax3.text(0.0, y, k, fontsize=9.5, color=P.INK_MUTED, va='center')
        ax3.text(0.62, y, v, fontsize=9.5, color=P.INK, va='center',
                 weight='semibold')
    return fig


def terrainPanel(grid, terrain, height=MAP_ROW_HEIGHT):
    """Elevation, what depression filling changed, and the hypsometry."""
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    hist_w = 5.0
    fig = plt.figure(figsize=(2 * mw + hist_w, height))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[mw, mw, hist_w], wspace=0.42)

    dem = terrain['dem']
    hs = P.hillshade(dem, grid.cellsize)

    ax = fig.add_subplot(gs[0, 0])
    P.mapRaster(dem, grid, ax=ax, cmap=P.HYPSOMETRIC, label='elevation (m)',
                title='Elevation',
                subtitle='3DEP, depression-filled', hillshade=hs)

    ax2 = fig.add_subplot(gs[0, 1])
    fill = terrain.get('fill_amount')
    fill = np.where(fill > 1e-6, fill, np.nan) if fill is not None else None
    if fill is not None and np.isfinite(fill).any():
        # Filled cells are a few percent of the basin, so on their own they
        # read as scattered marks on white with no idea where the basin is.
        # A flat silhouette underneath gives them somewhere to be.
        ax2.imshow(np.where(grid.mask != 0, 1.0, np.nan), extent=P._extent(grid),
                   cmap=ListedColormap(['#e8e7e3']), interpolation='nearest')
        P.mapRaster(fill, grid, ax=ax2, cmap=P.CMAP_SEQ_2,
                    label='fill depth (m)', percentile_clip=(0, 99),
                    title='Depressions filled',
                    subtitle=f"{terrain['n_pits_filled']:,} cells raised, "
                             f"{terrain['fill_volume']/1e9:.2f} km³")
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'No depressions found', ha='center', va='center',
                 color=P.INK_MUTED)

    ax3 = fig.add_subplot(gs[0, 2])
    d = _stats(dem, grid)
    ax3.hist(d, bins=60, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=0.4)
    ax3.set_xlabel('elevation (m)')
    ax3.set_ylabel('cells')
    ax3.axvline(d.mean(), color=P.SERIES[1], linewidth=1.8, zorder=5)
    # Beside the rule it labels, at the top of the axes where the
    # distribution has no mass, rather than adrift in the far corner.
    ax3.annotate(f'mean {d.mean():.0f} m', xy=(d.mean(), 1.0),
                 xycoords=('data', 'axes fraction'), xytext=(5, -6),
                 textcoords='offset points', ha='left', va='top',
                 color=P.SERIES[1], fontsize=9, weight='semibold')
    ax3.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    # Routed through _finish so its title shares the row's baseline.
    P._finish(ax3, 'Hypsometry',
              f'{d.min():,.0f}–{d.max():,.0f} m over the basin')
    return fig


def _dilateMax(a: np.ndarray) -> np.ndarray:
    """3x3 maximum filter, for display of one-cell-wide features."""
    out = np.array(a, dtype='float64')
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out = np.maximum(out, np.roll(np.roll(a, dr, axis=0), dc, axis=1))
    return out


def flowRoutingPanel(grid, terrain, height=MAP_ROW_HEIGHT):
    """Slope, aspect and upstream area -- the three routing controls."""
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(3 * mw, height))
    plt.subplots_adjust(wspace=0.34)

    P.mapRaster(np.degrees(terrain['slope']), grid, ax=axes[0], cmap=P.CMAP_SEQ,
                label='slope (°)', title='Slope',
                subtitle="Horn's finite difference")

    # Aspect is circular: a cyclic colormap is the only honest encoding, and
    # the legend states the convention explicitly.
    P.mapRaster(terrain['aspect'], grid, ax=axes[1], cmap='twilight',
                vmin=0, vmax=360, percentile_clip=None,
                label='aspect (° from N)', title='Aspect',
                subtitle='° clockwise from north; cyclic scale')

    # Flow accumulation is extremely right-skewed -- almost every cell is a
    # hillslope with one cell of upstream area, and the structure worth seeing
    # lives in the top few percent.  A log transform plus a median-anchored
    # stretch puts the channel network in the dark end of the ramp instead of
    # compressing it into the last 5% of the colour range.
    ua = np.log10(np.maximum(terrain['uparea'], grid.cell_area))
    # A 3019-row basin drawn three inches wide is three grid rows to the
    # pixel, and a channel one cell wide is what disappears first: sampled
    # at nearest neighbour it breaks into speckle that looks like noise
    # rather than a network.  Dilating with a 3x3 maximum before drawing
    # keeps every channel continuous.  On a field this smooth it moves no
    # cell more than one cell, it can only ever raise a value, and the
    # subtitle says it was done -- the numbers quoted are the real ones.
    P.mapRaster(_dilateMax(ua), grid, ax=axes[2], cmap=P.CMAP_SEQ,
                percentile_clip=(50, 99.5),
                label='log₁₀ upstream area (m²)', title='Upstream contributing area',
                subtitle=f"D8, confined to the basin; max "
                         f"{np.nanmax(terrain['uparea'])/1e6:,.0f} km²; "
                         f"3×3 max filter for display")
    return fig


def channelPanel(grid, network, terrain, comparison=None,
                 height=MAP_ROW_HEIGHT):
    """The delineated network, its order distribution and its topology."""
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    bar_w, text_w = 4.4, 3.6
    fig = plt.figure(figsize=(mw + bar_w + text_w, height))
    # A wider gutter than the other panels use: the map carries a colour bar
    # with a rotated label on its right and the histogram carries five-figure
    # tick labels on its left, and both spill into the space between them.
    gs = GridSpec(1, 3, figure=fig, width_ratios=[mw, bar_w, text_w], wspace=0.52)

    ax = fig.add_subplot(gs[0, 0])
    P.mapStreams(network, grid, ax=ax,
                 background=P.hillshade(terrain['dem'], grid.cellsize),
                 title='Channel network',
                 subtitle=f"{len(network['segments']):,} segments above a "
                          f"{network['channel_threshold_km2']:g} km² threshold")

    segs = network['segments']
    ax2 = fig.add_subplot(gs[0, 1])
    ocol = 'strahler_order' if 'strahler_order' in segs else 'order'
    counts = segs[ocol].value_counts().sort_index()
    ax2.bar(counts.index, counts.values, color=P.SERIES[0],
            edgecolor=P.SURFACE, linewidth=1.6, width=0.68)
    for o, n in counts.items():
        ax2.annotate(f'{n:,}', xy=(o, n), xytext=(0, 3),
                     textcoords='offset points', ha='center',
                     fontsize=8.5, color=P.INK_2)
    ax2.set_xlabel('Strahler order')
    ax2.set_ylabel('segments')
    ax2.set_xticks(counts.index)
    ax2.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax2.margins(y=0.10)
    P._finish(ax2, 'Segments by stream order')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    rows = [
        ('Segments', f"{len(segs):,}"),
        ('Channel cells', f"{int(network['channel_mask'].sum()):,}"),
        ('Drainage density', f"{network['drainage_density']:.3f} km/km²"),
        ('Max Strahler order', f"{int(segs[ocol].max())}"),
        ('Max routing rank', f"{int(segs['order'].max())}"),
        ('Outlet segments', f"{int((segs['outlet'] == 0).sum())}"),
        ('Total channel length', f"{segs['length'].sum()/1000:,.0f} km"),
        ('Slope range', f"{segs['slope'].min():.1e} – {segs['slope'].max():.3f}"),
        ('Width range', f"{segs['width'].min():.1f} – {segs['width'].max():.1f} m"),
    ]
    if comparison:
        rows.append(('NHD density',
                     f"{comparison['drainage_density_reference']:.3f} km/km²"))
        rows.append(('Overlap with NHD',
                     f"{100*comparison['overlap_fraction']:.0f}%"))
    P._finish(ax3, 'Network summary')
    for i, (k, v) in enumerate(rows):
        y = 0.97 - i * 0.092
        ax3.text(0.0, y, k, fontsize=9.2, color=P.INK_MUTED, va='center')
        ax3.text(0.66, y, v, fontsize=9.2, color=P.INK, va='center',
                 weight='semibold')
    return fig


def soilPanel(grid, soil_map, soil_table, soil_depth, awc=None,
              height=MAP_ROW_HEIGHT):
    """Soil classes, column depth and plant-available water."""
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    fig = plt.figure(figsize=(3 * mw + 2.2, height))
    # The class legend hangs off the right of the first map, so the gutter
    # after it has to clear the next panel's axis label as well.
    gs = GridSpec(1, 3, figure=fig, width_ratios=[mw + 2.2, mw, mw], wspace=0.56)

    labels = {int(r['dhsvm_id']): r['name'] for _, r in soil_table.iterrows()}
    ax = fig.add_subplot(gs[0, 0])
    P.mapCategorical(soil_map, grid, labels, ax=ax, title='Soil texture',
                     subtitle='USDA triangle, from POLARIS fractions')

    ax2 = fig.add_subplot(gs[0, 1])
    P.mapRaster(soil_depth, grid, ax=ax2, cmap=P.CMAP_SEQ,
                label='soil depth (m)', title='Soil depth',
                subtitle='terrain index, capped by channels and roots')

    ax3 = fig.add_subplot(gs[0, 2])
    if awc is not None:
        P.mapRaster(awc, grid, ax=ax3, cmap=P.CMAP_SEQ_2,
                    label='available water (mm)',
                    title='Available water',
                    subtitle='(field cap. − wilting pt) × depth')
    else:
        d = _stats(soil_depth, grid)
        ax3.hist(d, bins=50, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=0.4)
        ax3.set_xlabel('soil depth (m)'); ax3.set_ylabel('cells')
        ax3.set_title('Soil depth distribution', color=P.INK)
    return fig


def vegetationPanel(grid, veg_map, veg_table, veg_blocks,
                    height=MAP_ROW_HEIGHT):
    """Land cover, its composition, and the canopy seasonality it implies."""
    P.useStyle()
    mw = P.mapPanelWidth(grid, height - 1.0)
    bar_w, lai_w = 4.6, 4.6
    fig = plt.figure(figsize=(mw + bar_w + lai_w, height))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[mw, bar_w, lai_w], wspace=0.30)

    labels = {int(r['dhsvm_id']): r['name'] for _, r in veg_table.iterrows()}
    colors = {int(r['dhsvm_id']): P.NLCD_COLORS.get(int(r['nlcd_code']),
                                                    P.SERIES[i % 8])
              for i, (_, r) in enumerate(veg_table.iterrows())}
    # The composition bars beside this map name and colour every class, so
    # the map's own legend would only duplicate them -- and collide with
    # them.  Identity still never rests on colour alone.
    ax = fig.add_subplot(gs[0, 0])
    P.mapCategorical(veg_map, grid, labels, colors=colors, ax=ax, legend=False,
                     title='Land cover',
                     subtitle='NLCD → DHSVM types')

    ax2 = fig.add_subplot(gs[0, 1])
    t = veg_table.sort_values('n_cells', ascending=True)
    frac = 100.0 * t['n_cells'] / max(t['n_cells'].sum(), 1)
    ax2.barh(range(len(t)), frac.values,
             color=[colors[int(i)] for i in t['dhsvm_id']],
             edgecolor=P.SURFACE, linewidth=1.6, height=0.72)
    ax2.set_yticks(range(len(t)))
    ax2.set_yticklabels([n if len(n) < 26 else n[:24] + '…' for n in t['name']],
                        fontsize=8.5)
    ax2.set_xlabel('% of basin')
    ax2.set_title('Cover composition', color=P.INK)
    for i, v in enumerate(frac.values):
        if v > 1.5:
            ax2.annotate(f'{v:.0f}%', xy=(v, i), xytext=(4, 0),
                         textcoords='offset points', va='center',
                         fontsize=8.2, color=P.INK_2)

    # Monthly LAI for the four most extensive types -- four series is the
    # limit for direct labelling, and enough to show the phenology contrast.
    ax3 = fig.add_subplot(gs[0, 2])
    top = veg_table.sort_values('n_cells', ascending=False).head(4)
    months = np.arange(1, 13)
    for i, (_, r) in enumerate(top.iterrows()):
        b = next(b for b in veg_blocks if b['id'] == int(r['dhsvm_id']))
        lai = np.array(b['overstory_monthly_lai']) + np.array(b['understory_monthly_lai'])
        ax3.plot(months, lai, color=P.SERIES[i], linewidth=2.0,
                 label=r['name'][:20], marker='o', markersize=3.4)
    ax3.set_xlabel('month')
    ax3.set_ylabel('total LAI (m² m⁻²)')
    ax3.set_xticks(months)
    ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    P._finish(ax3, 'Canopy seasonality')
    lo, hi = ax3.get_ylim()
    ax3.set_ylim(lo, hi + 0.26 * (hi - lo))
    ax3.legend(fontsize=8, labelcolor=P.INK_2, loc='upper left')
    return fig


def forcingPanel(met, stations, grid, dem, timestep_hours,
                 sample_days=14, height=11.4):
    """Station placement, a sample of every forcing variable, and totals.

    The station map spans the full height of the figure rather than
    sitting in one cell of a 3x3 grid.  A basin three times taller than
    it is wide, drawn into a short wide cell, comes out under an inch
    across -- too small to see whether the stations actually cover it,
    which is the only question the panel exists to answer.
    """
    P.useStyle()
    # The map column is sized from the basin's own aspect against the
    # full figure height; the two chart columns are fixed.
    mw = P.mapPanelWidth(grid, height * 0.86, pad_in=1.5)
    chart_w = 5.4
    fig = plt.figure(figsize=(mw + 2 * chart_w, height))
    gs = GridSpec(4, 3, figure=fig, hspace=0.62, wspace=0.34,
                  width_ratios=[mw, chart_w, chart_w],
                  height_ratios=[1.0, 1.0, 1.0, 1.0],
                  left=0.045, right=0.99, top=0.93, bottom=0.055)

    import ww_dhsvm.meteorology as _M
    t = _M.toDatetimeIndex(met['time'].values)
    spatial = [d for d in met['Tair'].dims if d != 'time']

    # -- station map, full height -------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    P.mapRaster(dem, grid, ax=ax, cmap=P.HYPSOMETRIC, label='elevation (m)',
                title='Forcing stations',
                subtitle=f'{len(stations)} stations, one DHSVM met file each',
                hillshade=P.hillshade(dem, grid.cellsize))
    ax.scatter([s.easting / 1000 for s in stations],
               [s.northing / 1000 for s in stations],
               s=30, c=P.SERIES[1], edgecolors=P.SURFACE, linewidths=1.1, zorder=6)

    # -- station elevation distribution -------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    elevs = np.array([s.elevation for s in stations])
    demv = _stats(dem, grid)
    ax2.hist(demv, bins=45, color='#d8d7d2', edgecolor=P.SURFACE, linewidth=0.4,
             density=True, label='all basin cells')
    ax2.hist(elevs, bins=20, color=P.SERIES[1], alpha=0.85, edgecolor=P.SURFACE,
             linewidth=0.6, density=True, label='stations')
    ax2.set_xlabel('elevation (m)'); ax2.set_ylabel('density')
    ax2.legend(labelcolor=P.INK_2, fontsize=8.5)
    P._finish(ax2, "Do stations span the basin's relief?",
              'station elevations against every basin cell')

    # -- precipitation totals -----------------------------------------
    pm = met['Precip'].mean(dim=spatial).to_series() * timestep_hours * 1000.0
    span_days = int((t[-1] - t[0]).days) + 1

    ax3 = fig.add_subplot(gs[0, 2])
    if span_days >= 120:
        monthly_p = pm.groupby(pm.index.month).sum()
        ax3.bar(monthly_p.index, monthly_p.values, color=P.SERIES[0],
                edgecolor=P.SURFACE, linewidth=1.6, width=0.7)
        ax3.set_xlabel('month'); ax3.set_ylabel('precipitation (mm)')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        P._finish(ax3, 'Monthly precipitation',
                  f'basin mean, total {monthly_p.sum():,.0f} mm')
    else:
        # A fortnight of forcing has no monthly structure to show, and a
        # bar chart of one filled month beside eleven empty ones says
        # nothing about the weather -- only about the download.
        daily_p = pm.resample('1D').sum()
        ax3.bar(daily_p.index, daily_p.values, color=P.SERIES[0],
                edgecolor=P.SURFACE, linewidth=1.0, width=0.78)
        ax3.set_ylabel('precipitation (mm)')
        P.formatTimeAxis(ax3, span_days)
        P._finish(ax3, 'Daily precipitation',
                  f'basin mean, total {daily_p.sum():,.0f} mm')

    # -- a sample window of every driving variable --------------------
    n = min(int(sample_days * 24 / timestep_hours), len(t))
    i0 = max(0, len(t) // 2 - n // 2)
    sl = slice(i0, i0 + n)
    ts = t[sl]

    panels = [
        (gs[1, 1], 'Tair', 'air temperature (°C)', 0),
        (gs[1, 2], 'Precip', 'precipitation (mm per step)', 0),
        (gs[2, 1], 'Wind', 'wind speed (m s⁻¹)', 2),
        (gs[2, 2], 'RH', 'relative humidity (%)', 2),
        (gs[3, 1], 'Sin', 'shortwave in (W m⁻²)', 3),
        (gs[3, 2], 'Lin', 'longwave in (W m⁻²)', 3),
    ]
    for cell, var, ylab, ci in panels:
        axx = fig.add_subplot(cell)
        v = met[var].mean(dim=spatial).values[sl]
        if var == 'Precip':
            v = v * timestep_hours * 1000.0
            axx.bar(ts, v, width=timestep_hours / 24.0, color=P.SERIES[ci],
                    linewidth=0)
        else:
            axx.plot(ts, v, color=P.SERIES[ci], linewidth=1.5)
            axx.fill_between(ts, np.nanmin(v), v, color=P.SERIES[ci],
                             alpha=0.10, linewidth=0)
        axx.set_ylabel(ylab, fontsize=8.8)
        # One window, one x-range: matplotlib pads bar charts more than line
        # charts, so without this the precipitation panel would silently show
        # a wider period than the five panels around it.
        axx.set_xlim(ts[0], ts[-1])
        P._finish(axx, var, fontsize=10)
        P.formatTimeAxis(axx, sample_days)
        axx.tick_params(axis='x', labelsize=8)

    fig.suptitle(f'Meteorological forcing — {met.attrs.get("met_source", "")}',
                 x=0.008, y=0.995, ha='left', fontsize=13, weight='semibold',
                 color=P.INK)
    return fig


def forcingComparison(met_a, met_b, name_a, name_b, timestep_hours,
                      figsize=(14, 7.4)):
    """Compare two meteorological products over the same basin and period.

    Two products, so exactly two series per panel -- well inside the
    all-pairs colour-separation limit, and both are directly labelled.
    """
    P.useStyle()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    plt.subplots_adjust(hspace=0.55, wspace=0.32, left=0.05, right=0.99,
                        top=0.88, bottom=0.09)

    sa = [d for d in met_a['Tair'].dims if d != 'time']
    sb = [d for d in met_b['Tair'].dims if d != 'time']

    variables = [('Tair', 'air temperature (°C)', 1.0),
                 ('Precip', 'precipitation (mm/day)', 24000.0),
                 ('Sin', 'shortwave in (W m⁻²)', 1.0),
                 ('RH', 'relative humidity (%)', 1.0),
                 ('Wind', 'wind speed (m s⁻¹)', 1.0),
                 ('Lin', 'longwave in (W m⁻²)', 1.0)]

    for ax, (var, ylab, scale) in zip(axes.ravel(), variables):
        a = met_a[var].mean(dim=sa).to_series().resample('1D').mean() * scale
        b = met_b[var].mean(dim=sb).to_series().resample('1D').mean() * scale
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]

        ax.plot(a.index, a.values, color=P.SERIES[0], linewidth=1.4, label=name_a)
        ax.plot(b.index, b.values, color=P.SERIES[1], linewidth=1.4, label=name_b)
        ax.set_ylabel(ylab, fontsize=8.8)

        bias = float((b - a).mean())
        r = float(np.corrcoef(a.values, b.values)[0, 1]) if len(a) > 2 else np.nan
        P._finish(ax, var,
                  f'{name_b} − {name_a}: bias {bias:+.2f}, r = {r:.3f}',
                  fontsize=10)
        span = (common[-1] - common[0]).days if len(common) > 1 else 1
        P.formatTimeAxis(ax, span)
        ax.tick_params(axis='x', labelsize=8)

    # One legend for the whole figure, placed under the title where it
    # cannot collide with any panel's bias/correlation subtitle.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.995, 1.005),
               ncol=2, fontsize=9, labelcolor=P.INK_2, frameon=False)
    fig.suptitle(f'Meteorological products compared — {name_a} vs {name_b}',
                 x=0.008, y=0.995, ha='left', fontsize=13, weight='semibold',
                 color=P.INK)
    return fig


def inputDashboard(case, figsize=(14, 2.1)):
    """One row of headline numbers describing the finished case."""
    P.useStyle()
    grid = case['grid']
    fig, axes = plt.subplots(1, 6, figsize=figsize)
    plt.subplots_adjust(wspace=0.12)

    segs = case['network']['segments']
    met = case.get('met_summary', {})
    report = case.get('report')
    n_err = len(report.errors) if report is not None else 0

    tiles = [
        (f"{grid.basin_area_km2:,.0f}", 'basin area (km²)',
         f'{grid.n_active:,} active cells', None),
        (f"{grid.cellsize:g} m", 'grid resolution',
         f'{grid.nrows} × {grid.ncols}', None),
        (f"{len(segs):,}", 'channel segments',
         f"{case['network']['drainage_density']:.2f} km/km² density", None),
        (f"{len(case['soil_blocks'])}/{len(case['veg_blocks'])}",
         'soil / vegetation types', 'DHSVM parameter blocks', None),
        (f"{met.get('annual_precip_mm', float('nan')):,.0f}",
         'precipitation (mm yr⁻¹)',
         f"{len(case.get('stations', []))} forcing stations", None),
        (f"{n_err}", 'structural errors',
         'ready to run' if n_err == 0 else 'must be fixed',
         P.STATUS_GOOD if n_err == 0 else P.STATUS_BAD),
    ]
    for ax, (v, lab, sub, col) in zip(axes, tiles):
        P.statTile(ax, v, lab, sub, color=col)
    return fig


def _waterBalanceWaterfall(ax, wb):
    """DHSVM's own water balance, drawn as a waterfall.

    ``MassBalance.c`` closes the basin on

    ``Precip + vapour fluxes = ET + ChannelInt + RoadInt + Delta storage``

    so those are the terms shown -- not a balance of my own devising, which
    is how a plausible-looking 150 mm of "unaccounted" water gets invented.
    Routed discharge at the outlet is deliberately absent: what leaves the
    hillslopes is ``ChannelInt``, and the two differ by whatever the channel
    network is still holding when the run stops.

    Sign is encoded twice -- by colour and by the signed number on every
    bar -- so the chart still reads correctly in greyscale.
    """
    def g(k):
        v = wb.get(k, np.nan)
        return float(v) if v is not None and np.isfinite(v) else 0.0

    vapour = g('snow_vapor_flux_mm') + g('canopy_snow_vapor_flux_mm')
    to_channel = g('channel_int_mm') + g('road_int_mm')
    steps = [('precip', g('precip_mm')),
             ('vapour', vapour),
             ('ET', -g('et_mm')),
             ('channels', -to_channel),
             ('Δ soil', -g('soil_storage_change_mm')),
             ('Δ snow', -g('snow_storage_change_mm'))]

    run = 0.0
    bottoms, heights, colors, tops = [], [], [], []
    for _, d in steps:
        bottoms.append(min(run, run + d))
        heights.append(abs(d))
        colors.append(P.SERIES[0] if d >= 0 else P.SERIES[1])
        run += d
        tops.append(run)

    names = [n for n, _ in steps] + ['residual']
    bottoms.append(min(0.0, run))
    heights.append(abs(run))
    colors.append(P.INK_MUTED)

    x = np.arange(len(names))
    ax.bar(x, heights, bottom=bottoms, color=colors, edgecolor=P.SURFACE,
           linewidth=1.6, width=0.66)
    # Connectors carry the running total from one step to the next; without
    # them a waterfall is just a bar chart with confusing baselines.
    for i in range(len(steps) - 1):
        ax.plot([i + 0.33, i + 1 - 0.33], [tops[i]] * 2, color=P.INK_MUTED,
                linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.axhline(0, color=P.GRID, linewidth=1.0, zorder=0)

    for i, (_, d) in enumerate(steps):
        y = max(tops[i], tops[i] - d)
        ax.annotate(f'{d:+,.0f}', xy=(i, y), xytext=(0, 4),
                    textcoords='offset points', ha='center', fontsize=8.2,
                    color=P.INK_2)
    ax.annotate(f'{run:+,.2f}', xy=(len(steps), max(run, 0.0)), xytext=(0, 4),
                textcoords='offset points', ha='center', fontsize=8.2,
                color=P.INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.2, rotation=26, ha='right')
    ax.set_ylabel('depth (mm)')
    ax.margins(y=0.14)
    P._finish(ax, 'Water balance over the run',
              'MassBalance.c terms; + adds water, − removes it', fontsize=10)
    return ax


def outputPanel(results, grid, timestep_hours, water_balance=None,
                figsize=(14.5, 11.2)):
    """The behaviour of a completed run, in one figure."""
    P.useStyle()
    fig = plt.figure(figsize=figsize)
    outer = GridSpec(3, 3, figure=fig, hspace=0.62, wspace=0.30,
                     height_ratios=[1.42, 1.0, 1.0],
                     left=0.05, right=0.99, top=0.92, bottom=0.06)

    agg = results.get('aggregated')
    mb = results.get('mass_balance')
    sf = results.get('streamflow')

    # -- hydrograph with precipitation above --------------------------
    tots = results.get('stream_totals')
    if tots is not None and len(tots):
        q = tots['outflow']          # authoritative basin outflow
    elif sf is not None and len(sf.columns):
        outlets = [c for c in sf.columns if c.upper().startswith('OUTLET')]
        q = sf[outlets or list(sf.columns)].sum(axis=1)
    else:
        q = None

    if q is not None:
        # Hyetograph and hydrograph are one chart read downwards, so they
        # share an x-axis *and* sit against each other.  A full row gap
        # between them -- which is what a single GridSpec hspace gives --
        # breaks the timing link that is the entire point of the pairing.
        inner = outer[0, :].subgridspec(2, 1, height_ratios=[0.40, 1.0],
                                        hspace=0.06)
        ax_p = fig.add_subplot(inner[0])
        ax_q = fig.add_subplot(inner[1], sharex=ax_p)
        precip = None
        if mb is not None and 'Precip(m)' in mb.columns:
            precip = mb['Precip(m)'].reindex(q.index).fillna(0) * 1000.0
        P.hydrograph(ax_q, q.index, q.values,
                     precip=precip.values if precip is not None else None,
                     ax_p=ax_p,
                     title='Simulated basin outflow',
                     subtitle='precipitation above (inverted), discharge below; '
                              'outflow summed over every channel outlet')
        span = (q.index[-1] - q.index[0]).days
        P.formatTimeAxis(ax_q, span)
        ax_q.set_xlim(q.index[0], q.index[-1])

        # Two numbers a reader would otherwise have to measure off the axis.
        qv = np.asarray(q.values, dtype='float64')
        qm = float(np.nanmean(qv))
        ax_q.axhline(qm, color=P.INK_MUTED, linewidth=0.9,
                     linestyle=(0, (4, 3)), zorder=2)
        ax_q.annotate(f'mean {qm:,.0f} m³ s⁻¹', xy=(0.004, qm),
                      xycoords=('axes fraction', 'data'), xytext=(0, 4),
                      textcoords='offset points', fontsize=8.4,
                      color=P.INK_MUTED)
        ipk = int(np.nanargmax(qv))
        ax_q.annotate(f'peak {qv[ipk]:,.0f} m³ s⁻¹', xy=(q.index[ipk], qv[ipk]),
                      xytext=(6, 1), textcoords='offset points',
                      fontsize=8.4, color=P.INK_2, va='bottom')
        ax_q.margins(y=0.12)
        ax_q.set_ylim(bottom=0)

    # -- snow ---------------------------------------------------------
    ax = fig.add_subplot(outer[1, 0])
    if agg is not None and 'Swq' in agg.columns:
        swe = agg['Swq'] * 1000.0
        ax.plot(swe.index, swe.values, color=P.SERIES[0], linewidth=1.7)
        ax.fill_between(swe.index, 0, swe.values, color=P.SERIES[0],
                        alpha=0.15, linewidth=0)
        ax.set_ylabel('SWE (mm)')
        P._finish(ax, 'Basin-mean snow water equivalent', fontsize=10)
        ax.set_xlim(swe.index[0], swe.index[-1])
        P.formatTimeAxis(ax, (swe.index[-1] - swe.index[0]).days)
        ax.tick_params(axis='x', labelsize=8)

    # -- evapotranspiration -------------------------------------------
    ax2 = fig.add_subplot(outer[1, 1])
    if agg is not None and 'TotalET' in agg.columns:
        et = agg['TotalET'] * 1000.0
        daily = et.resample('1D').sum()
        ax2.plot(daily.index, daily.values, color=P.SERIES[2], linewidth=1.5)
        ax2.fill_between(daily.index, 0, daily.values, color=P.SERIES[2],
                         alpha=0.15, linewidth=0)
        ax2.set_ylabel('ET (mm day⁻¹)')
        P._finish(ax2, 'Evapotranspiration', fontsize=10)
        ax2.set_xlim(daily.index[0], daily.index[-1])
        P.formatTimeAxis(ax2, (daily.index[-1] - daily.index[0]).days)
        ax2.tick_params(axis='x', labelsize=8)

    # -- soil moisture ------------------------------------------------
    ax3 = fig.add_subplot(outer[1, 2])
    if agg is not None:
        cols = [c for c in agg.columns if c.startswith('SoilMoist')]
        for i, c in enumerate(cols[:3]):
            ax3.plot(agg.index, agg[c], color=P.SERIES[i], linewidth=1.5,
                     label=f'layer {i+1}')
        ax3.set_ylabel('soil moisture (vol/vol)')
        P._finish(ax3, 'Soil moisture by layer', fontsize=10)
        # Headroom first, then the legend: a legend placed into a full
        # panel sits on the series it is naming.
        if cols:
            lo, hi = ax3.get_ylim()
            ax3.set_ylim(lo, hi + 0.30 * (hi - lo))
            ax3.legend(fontsize=8, labelcolor=P.INK_2, ncol=3, loc='upper left')
        ax3.set_xlim(agg.index[0], agg.index[-1])
        P.formatTimeAxis(ax3, (agg.index[-1] - agg.index[0]).days)
        ax3.tick_params(axis='x', labelsize=8)

    # -- flow duration, or saturated fraction if the run wrote it ------
    ax4 = fig.add_subplot(outer[2, 0])
    sat = results.get('saturation')
    if sat is not None and len(sat) and float(np.nanmax(sat.values)) > 0:
        ax4.plot(sat.index, 100 * sat.values, color=P.SERIES[4], linewidth=1.5)
        ax4.fill_between(sat.index, 0, 100 * sat.values, color=P.SERIES[4],
                         alpha=0.15, linewidth=0)
        ax4.set_ylabel('saturated area (%)')
        P._finish(ax4, 'Saturated fraction of basin', fontsize=10)
        P.formatTimeAxis(ax4, (sat.index[-1] - sat.index[0]).days)
        ax4.tick_params(axis='x', labelsize=8)
    elif q is not None:
        # The flow duration curve is the one summary that says whether the
        # whole flow regime is plausible, not just the peaks: a model with
        # no baseflow shows it here as a cliff at the dry end long before
        # the hydrograph looks wrong.
        qd = q.resample('1D').mean().dropna()
        vals = np.sort(qd.values)[::-1]
        exc = 100.0 * np.arange(1, len(vals) + 1) / (len(vals) + 1)
        ax4.semilogy(exc, vals, color=P.SERIES[0], linewidth=1.9)
        for pct in (10, 50, 90):
            v = float(np.interp(pct, exc, vals))
            ax4.plot([pct], [v], marker='o', markersize=5.5,
                     color=P.SERIES[0], markeredgecolor=P.SURFACE,
                     markeredgewidth=1.3, zorder=5)
            ax4.annotate(f'Q{pct} = {v:,.0f}', xy=(pct, v), xytext=(5, 6),
                         textcoords='offset points', fontsize=8.2,
                         color=P.INK_2)
        ax4.set_xlabel('exceedance probability (%)')
        ax4.set_ylabel('discharge (m³ s⁻¹)')
        ax4.set_xlim(0, 100)
        P._finish(ax4, 'Flow duration curve',
                  'daily means, log scale', fontsize=10)
    else:
        ax4.axis('off')

    # -- mass balance residual ----------------------------------------
    ax5 = fig.add_subplot(outer[2, 1])
    if mb is not None and 'Error' in mb.columns:
        # Deliberately not red: this series is good news, and the reserved
        # alarm colour on a residual at machine precision would say the
        # opposite of what the numbers say.
        ax5.plot(mb.index, mb['Error'].values, color=P.INK_2, linewidth=0.9)
        ax5.set_ylabel('closure error (mm)')
        P._finish(ax5, 'Per-timestep mass-balance residual',
                  f"max |error| = {mb['Error'].abs().max():.1e} mm — "
                  f"machine precision", fontsize=10)
        ax5.set_xlim(mb.index[0], mb.index[-1])
        P.formatTimeAxis(ax5, (mb.index[-1] - mb.index[0]).days)
        ax5.tick_params(axis='x', labelsize=8)

    # -- water balance waterfall --------------------------------------
    ax6 = fig.add_subplot(outer[2, 2])
    if water_balance:
        _waterBalanceWaterfall(ax6, water_balance)
    else:
        ax6.axis('off')

    fig.suptitle('DHSVM simulation results', x=0.008, y=0.997, ha='left',
                 fontsize=13.5, weight='semibold', color=P.INK)
    return fig


def scalingPanel(results, n_physical_cores=None, figsize=(15.0, 8.6)):
    """Strong and weak scaling of parallel DHSVM.

    Four panels, because speedup and efficiency answer different
    questions and squashing them onto one pair of axes would need a dual
    y-axis -- which lets the choice of scales manufacture any impression.

    The ideal curves are drawn as recessive dashed grey references, not
    as data series: they are a reference frame, and giving them a
    categorical hue would imply they were measured.

    Parameters
    ----------
    results : dict
        ``{'strong': [...], 'weak': [...]}`` from
        :mod:`demo.scaling`; each entry has ``nproc``, ``t_long``,
        ``per_step`` and ``init``.
    n_physical_cores : int, optional
        Marked with a rule, since scaling past the physical core count
        measures hyperthreading rather than parallelism.
    """
    P.useStyle()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.52, wspace=0.24, left=0.05, right=0.99,
                        top=0.90, bottom=0.08)

    strong = sorted(results.get('strong', []), key=lambda r: r['nproc'])
    weak = sorted(results.get('weak', []), key=lambda r: r['nproc'])

    def _mark_cores(ax, p):
        # The rule is the hardware limit, and the axis stops just past it:
        # left to autoscale, matplotlib pads a further two ranks of empty
        # space beyond the rule and the measured curve drifts left.
        if n_physical_cores:
            ax.axvline(n_physical_cores, color=P.INK_MUTED, linewidth=1.0,
                       linestyle=':', zorder=1)
            ax.annotate(f'{n_physical_cores} physical cores',
                        xy=(n_physical_cores, ax.get_ylim()[1]),
                        xytext=(-4, -12), textcoords='offset points',
                        ha='right', va='top', fontsize=8, color=P.INK_MUTED,
                        rotation=90)
            ax.set_xlim(0.0, max(float(np.max(p)), n_physical_cores) + 0.55)
        else:
            ax.set_xlim(0.0, float(np.max(p)) + 0.55)

    def _amdahlFit(p, t):
        """Fit ``T(p) = W/p + c*p`` -- perfectly divisible work plus a
        communication cost that grows with the rank count.

        Two free parameters, both linear, so it is an ordinary least
        squares solve rather than an optimisation.  Returned only when the
        communication term is positive; a negative one would mean the fit
        has no physical reading and drawing it would be decoration.
        """
        A = np.column_stack([1.0 / p, p])
        try:
            (W, c), *_ = np.linalg.lstsq(A, t, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not (np.isfinite(W) and np.isfinite(c)) or c <= 0 or W <= 0:
            return None
        return float(W), float(c)

    # -- strong: speedup ---------------------------------------------
    ax = axes[0, 0]
    if strong:
        p = np.array([r['nproc'] for r in strong], dtype=float)
        # Speedup on the *stepping* cost, which is the part that can
        # parallelize; total-time speedup is shown alongside it.
        step = np.array([r['per_step'] for r in strong])
        tot = np.array([r['t_long'] for r in strong])
        s_step = step[0] / step
        ax.plot(p, p, color=P.INK_MUTED, linestyle='--', linewidth=1.3,
                label='ideal', zorder=2)
        # The gap between ideal and measured is the result, so it is drawn
        # rather than left as white space for the reader to estimate.
        ax.fill_between(p, s_step, p, color=P.INK_MUTED, alpha=0.09,
                        linewidth=0, zorder=1)
        fit = _amdahlFit(p, step)
        if fit is not None:
            W, c = fit
            pp = np.linspace(p.min(), max(p.max(), 1.0), 200)
            ax.plot(pp, step[0] / (W / pp + c * pp), color=P.SERIES[0],
                    linewidth=1.1, linestyle=(0, (5, 3)), alpha=0.85,
                    zorder=3, label='W/p + c·p fit')
        ax.plot(p, s_step, color=P.SERIES[0], linewidth=2.0,
                marker='o', markersize=5, label='time stepping', zorder=5)
        ax.plot(p, tot[0] / tot, color=P.SERIES[1], linewidth=2.0,
                marker='s', markersize=4.5, label='total wall time', zorder=4)
        ax.set_xlabel('MPI ranks'); ax.set_ylabel('speedup')
        ax.set_xticks(p)
        ax.set_ylim(0, float(p.max()) * 1.04)
        ax.annotate('speedup lost to communication\nand serial work',
                    xy=(0.30, 0.72), xycoords='axes fraction',
                    fontsize=8.4, color=P.INK_MUTED, ha='left', va='center')
        ibest = int(np.argmax(s_step))
        ax.annotate(f'plateau at {s_step[ibest]:.1f}×',
                    xy=(p[ibest], s_step[ibest]), xytext=(0, -14),
                    textcoords='offset points', ha='center', va='top',
                    fontsize=8.4, color=P.SERIES[0], weight='semibold')
        ax.legend(loc='upper left', labelcolor=P.INK_2, fontsize=8.4)
        _mark_cores(ax, p)
    P._finish(ax, 'Strong scaling — speedup',
              'one fixed 150 m problem, more ranks')

    # -- strong: efficiency ------------------------------------------
    ax = axes[0, 1]
    if strong:
        p = np.array([r['nproc'] for r in strong], dtype=float)
        step = np.array([r['per_step'] for r in strong])
        tot = np.array([r['t_long'] for r in strong])
        ax.axhline(100, color=P.INK_MUTED, linestyle='--', linewidth=1.3, zorder=2)
        ax.plot(p, 100 * (step[0] / step) / p, color=P.SERIES[0], linewidth=2.0,
                marker='o', markersize=5, label='time stepping', zorder=4)
        ax.plot(p, 100 * (tot[0] / tot) / p, color=P.SERIES[1], linewidth=2.0,
                marker='s', markersize=4.5, label='total wall time', zorder=3)
        ax.set_xlabel('MPI ranks'); ax.set_ylabel('parallel efficiency (%)')
        ax.set_xticks(p); ax.set_ylim(0, 115)
        ax.legend(loc='lower left', labelcolor=P.INK_2)
        _mark_cores(ax, p)
    P._finish(ax, 'Strong scaling — efficiency', 'E = S(p) / p; 100% is ideal')

    # -- weak: time --------------------------------------------------
    ax = axes[1, 0]
    if weak:
        p = np.array([r['nproc'] for r in weak], dtype=float)
        step = np.array([r['per_step'] for r in weak])
        ax.axhline(step[0], color=P.INK_MUTED, linestyle='--', linewidth=1.3,
                   label='ideal (constant)', zorder=2)
        ax.plot(p, step, color=P.SERIES[2], linewidth=2.0, marker='o',
                markersize=5, label='time stepping', zorder=4)
        ax.set_xlabel('MPI ranks'); ax.set_ylabel('seconds per timestep')
        ax.set_xticks(p); ax.set_ylim(bottom=0)
        ax.legend(loc='upper left', labelcolor=P.INK_2)
        _mark_cores(ax, p)
        for j, r in enumerate(weak[::2]):
            # The first point sits on the y-axis, so its label is hung to
            # the right of the marker instead of centred over it.
            first = (j == 0)
            ax.annotate(f"{r.get('cellsize', 0):.0f} m",
                        xy=(r['nproc'], r['per_step']),
                        xytext=(9, 7) if first else (0, 8),
                        textcoords='offset points',
                        ha='left' if first else 'center',
                        fontsize=7.5, color=P.INK_MUTED)
    P._finish(ax, 'Weak scaling — cost per timestep',
              'work per rank held constant; grid spacing shown')

    # -- weak: efficiency --------------------------------------------
    ax = axes[1, 1]
    if weak:
        p = np.array([r['nproc'] for r in weak], dtype=float)
        step = np.array([r['per_step'] for r in weak])
        ax.axhline(100, color=P.INK_MUTED, linestyle='--', linewidth=1.3, zorder=2)
        ax.plot(p, 100 * step[0] / step, color=P.SERIES[2], linewidth=2.0,
                marker='o', markersize=5, zorder=4)
        ax.set_xlabel('MPI ranks'); ax.set_ylabel('weak efficiency (%)')
        ax.set_xticks(p); ax.set_ylim(0, 115)
        _mark_cores(ax, p)
    P._finish(ax, 'Weak scaling — efficiency',
              'E_w = T(1) / T(p); 100% means cost per rank is unchanged')

    fig.suptitle('Parallel DHSVM scaling — full Connecticut River Basin',
                 x=0.008, y=0.998, ha='left', fontsize=13.5, weight='semibold',
                 color=P.INK)
    return fig
