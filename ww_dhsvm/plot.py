"""Plotting for WW-DHSVM inputs, forcing and outputs.

Every figure this module makes is meant to be *read*, not admired: the
job is to let a modeller see whether an input is defensible before
spending compute on it, and whether an output is believable afterwards.

Design rules followed throughout
--------------------------------
* **Form follows the data's job.**  Magnitude -> sequential ramp;
  identity -> categorical hues with a labelled legend; polarity ->
  diverging with a neutral midpoint; change over time -> line.
* **Sequential ramps are single-hue and monotonic in lightness**, so
  that a greyscale print or a colour-blind reader still orders them
  correctly.  The one multi-hue ramp here, :data:`HYPSOMETRIC` for
  elevation, is built to be monotonic in CIE L* as well -- it looks like
  a topographic map without being a rainbow.
* **No rainbow ramps** (``jet``, ``rainbow``, matplotlib's ``terrain``):
  they invent visual edges where the data is smooth and destroy ordering
  under colour-vision deficiency.
* **Identity is never colour-alone.**  Categorical maps carry a labelled
  legend; multi-series plots carry both a legend and, where there are
  four or fewer series, direct labels.
* **One axis per plot.**  Two quantities of different scale get two
  panels, never two y-axes -- a dual axis lets the author manufacture
  any apparent correlation by choosing the scales.
* Grid and axes are recessive; the data is the darkest thing in the
  frame.

The categorical hues are the validated eight-slot order documented in
this project's design reference; they clear the colour-vision-deficiency
separation gates on the adjacent-pair list used by lines, bars and
stacks.  Where a plot needs all-pairs separation (scatter), it uses at
most the first three slots.
"""

from typing import Optional, Dict, List, Tuple, Any, Sequence
import logging

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

#: Validated categorical hue order.  Assign in this order, never cycled.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']

#: Single-hue sequential ramp (blue), light -> dark.
SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
            '#0d366b']

#: Second sequential ramp (orange), for when two magnitude fields coexist.
SEQ_ORANGE = ['#fde3d4', '#fbcdb2', '#f8b48e', '#f59c6b', '#f2854c', '#eb6834',
              '#d95926', '#c04b1d', '#a43e17', '#872f10', '#6b230b']

#: Constant space above every axes for the title, so titles align across
#: a row whether or not each panel has a subtitle beneath it.
TITLE_PAD = 20

#: Reserved status colours.  These are never used as a categorical series:
#: a reader who has learnt that this green means "fine" must not meet it
#: again as "catchment 3".  They always ship with a label, never alone.
STATUS_GOOD = '#0f7a4e'
STATUS_BAD = '#c0392c'

INK = '#0b0b0b'
INK_2 = '#52514e'
INK_MUTED = '#8a8880'
SURFACE = '#fcfcfb'
GRID = '#e5e4e0'

CMAP_SEQ = LinearSegmentedColormap.from_list('ww_blue', SEQ_BLUE)
CMAP_SEQ_2 = LinearSegmentedColormap.from_list('ww_orange', SEQ_ORANGE)

#: Diverging blue <-> red with a neutral grey midpoint.
CMAP_DIV = LinearSegmentedColormap.from_list(
    'ww_div', ['#0d366b', '#2a78d6', '#9ec5f4', '#f0efec',
               '#f5a3a2', '#e34948', '#8f2020'])

#: Hypsometric elevation ramp -- multi-hue but monotonic in lightness, so
#: it orders correctly in greyscale and under colour-vision deficiency.
#: Constructed in CIELAB with L* forced to rise linearly from 26 to 95 while
#: a*/b* trace green -> olive -> tan -> brown -> warm grey -> white.  Verified
#: strictly monotonic in L* (minimum step 6.6), so it survives greyscale
#: printing and colour-vision deficiency as a true magnitude ramp.
HYPSOMETRIC = LinearSegmentedColormap.from_list(
    'ww_hypso',
    ['#244524', '#3c5429', '#586330', '#77703b', '#947d4c', '#ae8c63',
     '#c49d7d', '#d4af98', '#dfc5b6', '#e8dbd2', '#f2f0ed'])

#: Official NLCD legend colours, so land-cover maps look the way every
#: hydrologist already expects them to.
NLCD_COLORS = {
    11: '#466b9f', 12: '#d1def8', 21: '#dec5c5', 22: '#d99282',
    23: '#eb0000', 24: '#ab0000', 31: '#b3ac9f', 41: '#68ab5f',
    42: '#1c5f2c', 43: '#b5c58f', 51: '#af963c', 52: '#ccb879',
    71: '#dfdfc2', 72: '#d1d182', 73: '#a3cc51', 74: '#82ba9e',
    81: '#dcd939', 82: '#ab6c28', 90: '#b8d9eb', 95: '#6c9fb8',
}


def useStyle() -> None:
    """Apply the WW-DHSVM matplotlib style.

    Recessive axes and grid, generous whitespace, and a type scale that
    stays legible when a figure is dropped into a report at half size.
    """
    mpl.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'savefig.bbox': 'tight',
        'savefig.dpi': 150,
        'figure.dpi': 110,
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.titlesize': 11.5,
        'axes.titleweight': 'semibold',
        'axes.titlelocation': 'left',
        'axes.titlepad': 9,
        'axes.labelsize': 9.5,
        'axes.labelcolor': INK_2,
        'axes.edgecolor': GRID,
        'axes.linewidth': 0.9,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': GRID,
        'grid.linewidth': 0.7,
        'grid.alpha': 1.0,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': 9,
        'lines.linewidth': 2.0,
        'lines.solid_capstyle': 'round',
        'text.color': INK,
    })


def _finish(ax, title=None, subtitle=None, xlabel=None, ylabel=None,
            fontsize=None):
    """Apply the common title/label treatment.

    When a subtitle is present the title is lifted so the two stack
    cleanly -- a subtitle drawn at the default title position would sit
    on top of it.
    """
    # The title pad is constant whether or not a subtitle follows, so that
    # titles sit on one line across a row of panels even when only some of
    # them carry a subtitle.
    if title:
        ax.set_title(title, color=INK, pad=TITLE_PAD, fontsize=fontsize)
    if subtitle:
        # Offset in points, not axes fractions: a fraction of a short panel
        # is a smaller gap than the same fraction of a tall one, and the
        # subtitle then reads as touching the title in the short panel.
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(0, 3), textcoords='offset points',
                    fontsize=8.6, color=INK_MUTED, ha='left', va='bottom',
                    annotation_clip=False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


# ---------------------------------------------------------------------------
# Raster maps
# ---------------------------------------------------------------------------

def _extent(grid):
    return (grid.xllcorner / 1000.0, grid.xurcorner / 1000.0,
            grid.yllcorner / 1000.0, grid.yurcorner / 1000.0)



def mapPanelWidth(grid, height_in: float = 5.6, pad_in: float = 2.05) -> float:
    """Axes width, inches, that lets a basin fill its panel.

    A map axes is drawn with ``set_aspect('equal')``, so a tall narrow
    basin plotted in a square panel leaves most of the panel empty.  The
    Connecticut is 161 km wide by 453 km tall -- an aspect of 0.35 -- and
    in equal-width columns the map occupies about a third of its cell
    while its colour bar drifts into the neighbouring panel.

    Sizing each map column from the domain's own aspect keeps the data
    large and the figure honest at any basin shape.

    Parameters
    ----------
    grid : ModelGrid
        Supplies the bounding-box aspect.
    height_in : float, optional
        Height available for the axes, inches.
    pad_in : float, optional
        Extra width for the y-axis labels and the colour bar.  Matplotlib's
        ``colorbar`` steals width from the axes it is attached to and
        places its label at the far right, so this must cover the axis
        labels, the bar and its label, or the label lands on the next
        panel's y-axis.

    Returns
    -------
    float
        Suggested axes width in inches.
    """
    xmin, ymin, xmax, ymax = grid.bounds
    aspect = (xmax - xmin) / max(ymax - ymin, 1e-9)
    return max(height_in * aspect, 1.1) + pad_in


def mapRaster(array: np.ndarray,
              grid,
              ax=None,
              cmap=None,
              mask: Optional[np.ndarray] = None,
              title: Optional[str] = None,
              subtitle: Optional[str] = None,
              label: Optional[str] = None,
              vmin: Optional[float] = None,
              vmax: Optional[float] = None,
              percentile_clip: Optional[Tuple[float, float]] = (1, 99),
              diverging: bool = False,
              colorbar: bool = True,
              hillshade: Optional[np.ndarray] = None):
    """Plot a continuous field on the model grid.

    Parameters
    ----------
    array : np.ndarray
        The field, shaped like the grid.
    grid : ModelGrid
        Supplies the extent and, if ``mask`` is None, the basin mask.
    cmap : Colormap, optional
        Defaults to the single-hue blue ramp, or the diverging ramp when
        ``diverging`` is True.
    percentile_clip : tuple, optional
        Clip the colour range to these percentiles of the in-basin data,
        so a handful of outliers cannot flatten the whole map.  Pass
        ``None`` to use the full range.
    hillshade : np.ndarray, optional
        A hillshade to multiply under the colour, adding relief without
        changing the encoded values.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.6))

    m = mask if mask is not None else grid.mask
    data = np.array(array, dtype='float64')
    if m is not None:
        data = np.where(m != 0, data, np.nan)

    finite = data[np.isfinite(data)]
    if vmin is None or vmax is None:
        if finite.size and percentile_clip:
            lo, hi = np.percentile(finite, percentile_clip)
        elif finite.size:
            lo, hi = finite.min(), finite.max()
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if vmin == vmax:
        vmax = vmin + 1e-6

    if cmap is None:
        cmap = CMAP_DIV if diverging else CMAP_SEQ
    if diverging:
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim

    if hillshade is not None:
        hs = np.where(m != 0, hillshade, np.nan) if m is not None else hillshade
        ax.imshow(hs, extent=_extent(grid), cmap='Greys_r',
                  vmin=0, vmax=1, interpolation='nearest')
        im = ax.imshow(data, extent=_extent(grid), cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', alpha=0.75)
    else:
        im = ax.imshow(data, extent=_extent(grid), cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest')

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('easting (km)')
    ax.set_ylabel('northing (km)')
    _finish(ax, title, subtitle)

    if colorbar:
        cb = _attachColorbar(ax, im, label)
    return ax


def _attachColorbar(ax, mappable, label=None, ticks=None):
    """Put a colour bar against a map, matching its drawn height.

    ``plt.colorbar(..., ax=ax)`` sizes the bar from a fixed aspect ratio
    and positions it from the axes' *allotted* cell.  On a map with
    ``set_aspect('equal')`` the drawn axes is usually much smaller than
    its cell, so the bar ends up floating in space beside the map and a
    third of its height.  An axes divider tracks the drawn box instead,
    so the bar is always exactly as tall as the map and right against it.
    """
    cax = make_axes_locatable(ax).append_axes('right', size='3.5%', pad=0.10,
                                              axes_class=plt.Axes)
    cb = plt.colorbar(mappable, cax=cax, ticks=ticks)
    cb.outline.set_visible(False)
    cax.grid(False)
    cb.ax.tick_params(length=0, labelsize=8.5, colors=INK_MUTED)
    if label:
        cb.set_label(label, fontsize=8.6, color=INK_2, labelpad=3)
    return cb


def mapCategorical(array: np.ndarray,
                   grid,
                   labels: Dict[int, str],
                   colors: Optional[Dict[int, str]] = None,
                   ax=None,
                   title: Optional[str] = None,
                   subtitle: Optional[str] = None,
                   mask: Optional[np.ndarray] = None,
                   legend_cols: int = 1,
                   legend: bool = True):
    """Plot a class map with a labelled legend.

    Identity is carried by the legend text as well as the colour, so the
    map is readable under any colour-vision deficiency.

    Parameters
    ----------
    labels : dict
        ``{class_value: display name}``.  Only these classes are drawn.
    colors : dict, optional
        ``{class_value: hex}``.  Defaults to the categorical order.
    legend : bool, optional
        Draw the labelled legend.  Set ``False`` only when an adjacent
        panel in the same figure already names every class -- identity
        must never rest on colour alone.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.6, 5.6))

    m = mask if mask is not None else grid.mask
    keys = sorted(labels)
    if colors is None:
        colors = {k: SERIES[i % len(SERIES)] for i, k in enumerate(keys)}

    lut = {k: i for i, k in enumerate(keys)}
    idx = np.full(array.shape, np.nan)
    for k, i in lut.items():
        idx[array == k] = i
    if m is not None:
        idx = np.where(m != 0, idx, np.nan)

    cmap = ListedColormap([colors[k] for k in keys])
    ax.imshow(idx, extent=_extent(grid), cmap=cmap,
              vmin=-0.5, vmax=len(keys) - 0.5, interpolation='nearest')
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlabel('easting (km)')
    ax.set_ylabel('northing (km)')
    _finish(ax, title, subtitle)

    if not legend:
        return ax

    total = int(np.count_nonzero(np.isfinite(idx)))
    handles = []
    for k in keys:
        n = int(np.count_nonzero(idx == lut[k]))
        pct = 100.0 * n / max(total, 1)
        handles.append(Patch(facecolor=colors[k], edgecolor=SURFACE, linewidth=1.5,
                             label=f'{labels[k]}  ({pct:.1f}%)'))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              ncol=legend_cols, fontsize=8.5, labelcolor=INK_2,
              handlelength=1.1, handleheight=1.1, borderpad=0)
    return ax


def hillshade(dem: np.ndarray, cellsize: float,
              azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Standard hillshade, in ``[0, 1]``, for use as a relief underlay."""
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    dy, dx = np.gradient(np.asarray(dem, dtype='float64'), cellsize)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (np.sin(alt) * np.sin(slope) +
              np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip((shaded + 1) / 2.0, 0, 1)


def mapStreams(network: Dict[str, Any], grid, ax=None,
               title: Optional[str] = None, subtitle: Optional[str] = None,
               background: Optional[np.ndarray] = None,
               color_by: str = 'strahler_order'):
    """Plot the derived channel network, coloured by Strahler order.

    Stream order is ordinal, so it gets a single-hue ramp that darkens
    and thickens downstream -- the visual weight then matches the
    hydrologic weight.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.6))

    if background is not None:
        bg = np.where(grid.mask != 0, background, np.nan) if grid.mask is not None \
            else background
        ax.imshow(bg, extent=_extent(grid), cmap='Greys', alpha=0.35,
                  interpolation='nearest')

    seg_grid = network['segment_id_grid']
    segs = network['segments'].set_index('ID')
    if color_by not in segs.columns:
        color_by = 'order'
    orders = np.zeros(grid.shape)
    for sid in np.unique(seg_grid):
        if sid <= 0 or sid not in segs.index:
            continue
        orders[seg_grid == sid] = segs.loc[sid, color_by]

    omax = int(orders.max()) if orders.max() > 0 else 1
    drawn, shades = [], []
    for o in range(1, omax + 1):
        sel = (orders == o)
        if not sel.any():
            continue
        rr, cc = np.nonzero(sel)
        xs = (grid.xllcorner + (cc + 0.5) * grid.cellsize) / 1000.0
        ys = (grid.yurcorner - (rr + 0.5) * grid.cellsize) / 1000.0
        shade = SEQ_BLUE[min(3 + 2 * (o - 1), len(SEQ_BLUE) - 1)]
        ax.scatter(xs, ys, s=1.0 + 2.2 * o, c=shade, marker='s', linewidths=0)
        drawn.append(o)
        shades.append(shade)

    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(_extent(grid)[0], _extent(grid)[1])
    ax.set_ylim(_extent(grid)[2], _extent(grid)[3])
    ax.set_xlabel('easting (km)')
    ax.set_ylabel('northing (km)')
    _finish(ax, title, subtitle)

    # A framed legend inside the axes hides the very network it explains,
    # and a long thin basin leaves no corner large enough for seven
    # entries.  Stream order is ordinal, so it belongs on a discrete
    # colour bar beside the map -- the same treatment every other map in
    # this module gives its scale.
    if drawn:
        dcmap = ListedColormap(shades)
        dnorm = BoundaryNorm(np.arange(len(drawn) + 1) - 0.5, len(drawn))
        sm = mpl.cm.ScalarMappable(cmap=dcmap, norm=dnorm)
        cb = _attachColorbar(ax, sm, 'Strahler order',
                             ticks=range(len(drawn)))
        cb.ax.set_yticklabels([str(o) for o in drawn])
    return ax


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def timeSeries(ax, x, series: Dict[str, np.ndarray],
               title=None, subtitle=None, ylabel=None,
               colors: Optional[List[str]] = None,
               direct_label: bool = True,
               fill_first: bool = False):
    """Plot one or more time series on a single axis.

    A legend appears for two or more series; with four or fewer, each
    line is also labelled directly at its right-hand end so the reader
    never has to move between legend and line.
    """
    colors = colors or SERIES
    keys = list(series)
    for i, k in enumerate(keys):
        c = colors[i % len(colors)]
        ax.plot(x, series[k], color=c, label=k, linewidth=1.8, zorder=3 + i)
        if fill_first and i == 0:
            ax.fill_between(x, 0, series[k], color=c, alpha=0.13, linewidth=0)
        if direct_label and len(keys) <= 4:
            y = np.asarray(series[k], dtype='float64')
            good = np.flatnonzero(np.isfinite(y))
            if good.size:
                ax.annotate(k, xy=(x[good[-1]], y[good[-1]]),
                            xytext=(5, 0), textcoords='offset points',
                            color=c, fontsize=8.5, va='center', weight='semibold')
    if len(keys) >= 2:
        ax.legend(loc='upper left', ncol=min(len(keys), 4), labelcolor=INK_2)
    _finish(ax, title, subtitle, ylabel=ylabel)
    return ax


def hydrograph(ax_q, time, discharge, precip=None, ax_p=None,
               title=None, subtitle=None, label='simulated',
               observed=None, observed_label='observed'):
    """The conventional hydrograph: discharge below, precipitation above.

    Precipitation is drawn in its *own panel*, inverted from the top, not
    on a second y-axis of the discharge plot.  A dual axis would let the
    choice of scales manufacture any apparent rainfall-runoff
    relationship; two aligned panels sharing an x-axis show the same
    timing information honestly.
    """
    if precip is not None and ax_p is not None:
        ax_p.bar(time, precip, width=(time[1] - time[0]) if len(time) > 1 else 1,
                 color=SERIES[0], linewidth=0, alpha=0.85)
        ax_p.invert_yaxis()
        ax_p.set_ylabel('precip\n(mm)')
        # The two panels are drawn against each other, so the upper one's
        # tick marks would otherwise poke into the lower one's title space.
        ax_p.tick_params(labelbottom=False, axis='x', length=0)
        ax_p.spines['bottom'].set_visible(False)
        # The pair reads as one figure, so the title and subtitle belong to
        # the top panel only -- repeating them below looks like two charts.
        _finish(ax_p, title, subtitle)
        title = subtitle = None

    ax_q.plot(time, discharge, color=SERIES[0], linewidth=1.9,
              label=label, zorder=4)
    ax_q.fill_between(time, 0, discharge, color=SERIES[0], alpha=0.13, linewidth=0)
    if observed is not None:
        ax_q.plot(time, observed, color=SERIES[1], linewidth=1.6,
                  label=observed_label, zorder=5)
        ax_q.legend(loc='upper left', labelcolor=INK_2)
    ax_q.set_ylabel('discharge (m$^3$ s$^{-1}$)')
    ax_q.set_ylim(bottom=0)
    _finish(ax_q, title, subtitle)
    return ax_q


def formatTimeAxis(ax, span_days: float, max_ticks: int = 5):
    """Choose date ticks that fit, for the span being shown.

    Tick *density* is capped rather than fixed by interval: a panel four
    inches wide holds about five ``21 Jun``-style labels before they
    collide, and colliding labels are worse than coarser ticks.
    """
    if span_days <= 3:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    elif span_days <= 35:
        step = max(1, int(np.ceil(span_days / max_ticks)))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    elif span_days <= 400:
        step = max(1, int(np.ceil(span_days / 30.0 / max_ticks)))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    return ax


def statTile(ax, value: str, label: str, sublabel: str = '',
             color: Optional[str] = None):
    """A single headline number.

    Some quantities -- basin area, annual precipitation, mass-balance
    error -- are one number.  A number set large and labelled is a better
    chart than a bar of length one.
    """
    ax.axis('off')
    ax.text(0.0, 0.74, value, fontsize=24, weight='bold',
            color=color or INK, ha='left', va='center')
    ax.text(0.0, 0.38, label, fontsize=9.5, color=INK_2, ha='left', va='center')
    if sublabel:
        ax.text(0.0, 0.15, sublabel, fontsize=8.5, color=INK_MUTED,
                ha='left', va='center')
    return ax
