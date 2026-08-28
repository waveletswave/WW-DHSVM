"""Soil classification and DHSVM soil parameters.

DHSVM discretizes the subsurface as a small number of **soil types**,
each described by ~20 parameters in the ``[SOILS]`` section of the
configuration file, plus two spatial maps: a soil-type map (``uint8``,
values ``1..NTypes``) and a total soil-depth map (``float32``, metres).
Within a cell the column is split into ``Number of Soil Layers`` root
zones plus a saturated zone, and several parameters are given per layer.

The DHSVM soil parameterization is *Brooks--Corey*, not van Genuchten:

* ``Porosity``               -- saturated water content, phi
* ``Pore Size Distribution`` -- Brooks--Corey lambda (= 1/b in the
  Clapp & Hornberger formulation)
* ``Bubbling Pressure``      -- air-entry head psi_b, **metres**
* ``Field Capacity``         -- theta at -33 kPa
* ``Wilting Point``          -- theta at -1500 kPa
* ``Residual Water Content`` -- theta_r
* ``Vertical Conductivity``  -- K_sat vertical, m/s
* ``Lateral Conductivity``   -- K_sat lateral at the surface, m/s
* ``Exponential Decrease``   -- f, the decay of lateral K with depth

That last pair deserves emphasis, because it is the single most
influential subsurface control in DHSVM.  Saturated subsurface flow --
the mechanism that generates most of the baseflow and much of the storm
response -- moves at a transmissivity that decays exponentially with
depth below the surface::

    T(z) = K_lat * D / f * (exp(-f * z / D) ... )

so ``Lateral Conductivity`` and ``Exponential Decrease`` together set
recession behaviour.  They are routinely *calibrated*, and the values
this module assigns from texture are a defensible starting point, not an
answer.  ``ww_dhsvm`` sets lateral K to 10x vertical K by default, which
is a common anisotropy assumption for structured soils.

Sources of soil texture, in the order this module prefers them:

1. **NRCS SSURGO** (:class:`ww_dhsvm.sources.ManagerNRCS`) -- polygon
   map units at 1:12,000--1:63,360; the standard for CONUS work.
2. **POLARIS** -- 30 m probabilistic remapping of SSURGO.
3. **SoilGrids** -- 250 m global, the fallback outside the US.

Watershed Workflow's ``soil_properties`` module serves the analogous role
for ATS, but converts to van Genuchten parameters and permeability in
m^2; the DHSVM formulation is different enough that this module is a
rewrite rather than a port.
"""

from typing import Optional, Dict, Tuple, List, Any
import os
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import ww_dhsvm.config
from ww_dhsvm.grid import ModelGrid


_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

#: USDA texture triangle class names in DHSVM class order (1-12).
USDA_CLASSES = ['SAND', 'LOAMY SAND', 'SANDY LOAM', 'SILT LOAM', 'SILT',
                'LOAM', 'SANDY CLAY LOAM', 'SILTY CLAY LOAM', 'CLAY LOAM',
                'SANDY CLAY', 'SILTY CLAY', 'CLAY']


def loadSoilPropertyTable() -> pd.DataFrame:
    """Load the built-in DHSVM soil-parameter table, keyed by USDA class.

    Values are compiled from the standard soil-physics literature --
    Clapp & Hornberger (1978) for the Brooks--Corey exponent and
    air-entry pressure, Rawls et al. (1982) for porosity, field capacity
    and wilting point, and Cosby et al. (1984) for saturated hydraulic
    conductivity -- and converted to DHSVM's units (metres, m/s,
    kg/m^3, W/m/K, J/m^3/K).

    Returns
    -------
    pd.DataFrame
        Indexed by ``class_id`` (1-12).
    """
    path = os.path.join(_DATA_DIR, 'dhsvm_soil_properties.csv')
    df = pd.read_csv(path)
    return df.set_index('class_id', drop=False)


def textureToUSDAClass(sand: np.ndarray,
                       silt: np.ndarray,
                       clay: np.ndarray) -> np.ndarray:
    """Classify sand/silt/clay percentages into the USDA texture triangle.

    Implements the standard USDA boundaries.  Inputs are renormalized to
    sum to 100 first, so fractions or percentages both work.

    Parameters
    ----------
    sand, silt, clay : np.ndarray
        Percentages (or any consistent unit) of each separate.

    Returns
    -------
    np.ndarray
        Integer class IDs ``1..12`` matching :data:`USDA_CLASSES`;
        ``0`` where the inputs were NaN.
    """
    sand = np.asarray(sand, dtype='float64')
    silt = np.asarray(silt, dtype='float64')
    clay = np.asarray(clay, dtype='float64')

    total = sand + silt + clay
    valid = np.isfinite(total) & (total > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        sa = np.where(valid, 100.0 * sand / total, np.nan)
        si = np.where(valid, 100.0 * silt / total, np.nan)
        cl = np.where(valid, 100.0 * clay / total, np.nan)

    out = np.zeros(sa.shape, dtype='uint8')

    def _set(cond, cls):
        np.copyto(out, cls, where=(cond & (out == 0) & valid))

    # Order matters: the triangle is evaluated from the corners inward,
    # exactly as in the USDA textural classification key.
    _set((cl >= 40) & (si < 40) & (sa < 45), 12)                    # CLAY
    _set((cl >= 40) & (si >= 40), 11)                               # SILTY CLAY
    _set((cl >= 35) & (sa >= 45), 10)                               # SANDY CLAY
    _set((cl >= 27) & (cl < 40) & (sa > 20) & (sa <= 45), 9)        # CLAY LOAM
    _set((cl >= 27) & (cl < 40) & (sa <= 20), 8)                    # SILTY CLAY LOAM
    _set((cl >= 20) & (cl < 35) & (si < 28) & (sa > 45), 7)         # SANDY CLAY LOAM
    _set((si >= 80) & (cl < 12), 5)                                 # SILT
    _set((si >= 50) & (cl < 27), 4)                                 # SILT LOAM
    _set((cl < 27) & (si >= 28) & (si < 50) & (sa <= 52), 6)        # LOAM
    _set((sa >= 85) & ((si + 1.5 * cl) < 15), 1)                    # SAND
    _set((sa >= 70) & ((si + 1.5 * cl) >= 15) & ((si + 2.0 * cl) < 30), 2)  # LOAMY SAND
    _set(((cl >= 7) & (cl < 20) & (sa > 52) & ((si + 2.0 * cl) >= 30))
         | ((cl < 7) & (si < 50) & ((si + 2.0 * cl) >= 30)), 3)     # SANDY LOAM
    # Anything left inside the triangle falls to LOAM.
    _set(np.ones_like(out, dtype=bool), 6)

    return out


def classifyFromFractions(sand: np.ndarray, silt: np.ndarray, clay: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          default_class: int = 6) -> np.ndarray:
    """USDA classification with holes filled, ready for a DHSVM soil map.

    DHSVM aborts if any in-basin cell holds a soil class greater than
    ``Number of Soil Types``, and treats class 0 as invalid, so every
    in-basin cell must carry a class in ``1..NTypes``.

    Parameters
    ----------
    sand, silt, clay : np.ndarray
        Texture fractions on the model grid.
    mask : np.ndarray, optional
        Basin mask; only in-basin cells are required to be valid.
    default_class : int, optional
        Class assigned where texture is missing.  Default 6 (LOAM), the
        centroid of the texture triangle.

    Returns
    -------
    np.ndarray
        ``uint8`` soil-class map.
    """
    cls = textureToUSDAClass(sand, silt, clay)
    missing = (cls == 0)
    if mask is not None:
        missing &= (mask != 0)
    n_missing = int(missing.sum())
    if n_missing:
        logging.info(f'  {n_missing} in-basin cells lack texture data; '
                     f'assigning class {default_class} '
                     f'({USDA_CLASSES[default_class-1]})')
    cls = np.where(cls == 0, default_class, cls).astype('uint8')

    counts = {USDA_CLASSES[c-1]: int(((cls == c) & ((mask != 0) if mask is not None else True)).sum())
              for c in range(1, 13)}
    total = max(sum(counts.values()), 1)
    logging.info('  soil texture composition:')
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if n:
            logging.info(f'      {name:<18s} {n:>8,d} cells ({100.0*n/total:5.1f}%)')
    return cls


def compactClasses(soil_class: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Renumber a soil map to a dense ``1..N`` range.

    DHSVM requires soil classes to be a contiguous run starting at 1, and
    every class up to ``Number of Soil Types`` must have a parameter
    block.  A texture map of a real basin typically uses only 4--8 of the
    12 USDA classes, so renumbering keeps the config file small and
    avoids declaring parameters for absent soils.

    Returns
    -------
    remapped : np.ndarray
        ``uint8`` map with values ``1..N``.
    table : pd.DataFrame
        The parameter table for the classes actually present, with a
        ``dhsvm_id`` column giving the new numbering.
    """
    props = loadSoilPropertyTable()

    active = soil_class if mask is None else soil_class[mask != 0]
    present = sorted(int(c) for c in np.unique(active) if c > 0)

    remap = np.zeros(256, dtype='uint8')
    for new_id, old_id in enumerate(present, start=1):
        remap[old_id] = new_id
    remapped = remap[soil_class]

    # Cells outside the basin still need a valid class: give them class 1.
    if mask is not None:
        remapped = np.where((mask == 0) & (remapped == 0), 1, remapped)
    remapped = np.where(remapped == 0, 1, remapped).astype('uint8')

    table = props.loc[present].copy()
    table['dhsvm_id'] = range(1, len(present) + 1)
    table = table.set_index('dhsvm_id', drop=False)

    logging.info(f'  compacted to {len(present)} DHSVM soil types: '
                 f'{", ".join(table["name"])}')
    return remapped, table


def buildSoilParameterBlocks(table: pd.DataFrame,
                             n_layers: int = 3,
                             lateral_anisotropy: Optional[float] = None
                             ) -> List[Dict[str, Any]]:
    """Expand the soil table into per-type DHSVM ``[SOILS]`` blocks.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compactClasses`.
    n_layers : int, optional
        Number of root-zone soil layers.  DHSVM's convention is 3, and
        the vegetation table's ``Root Zone Depths`` must have the same
        length.  Default 3.
    lateral_anisotropy : float, optional
        If given, lateral conductivity is set to this multiple of the
        vertical conductivity, overriding the table.  The built-in table
        already uses 10x.

    Returns
    -------
    list of dict
        One dict per soil type, with scalar and per-layer entries keyed
        exactly as :mod:`ww_dhsvm.config_writer` expects.
    """
    blocks = []
    for _, r in table.iterrows():
        klat = (float(r['vertical_conductivity_ms']) * lateral_anisotropy
                if lateral_anisotropy is not None
                else float(r['lateral_conductivity_ms']))
        blocks.append(dict(
            id=int(r['dhsvm_id']),
            description=str(r['name']),
            lateral_conductivity=klat,
            exponential_decrease=float(r['exponential_decrease']),
            depth_threshold=float(r['depth_threshold_m']),
            max_infiltration=float(r['max_infiltration_ms']),
            capillary_drive=float(r['capillary_drive_m']),
            surface_albedo=float(r['surface_albedo']),
            n_layers=n_layers,
            porosity=[float(r['porosity'])] * n_layers,
            pore_size_distribution=[float(r['pore_size_distribution'])] * n_layers,
            bubbling_pressure=[float(r['bubbling_pressure_m'])] * n_layers,
            field_capacity=[float(r['field_capacity'])] * n_layers,
            wilting_point=[float(r['wilting_point'])] * n_layers,
            bulk_density=[float(r['bulk_density_kgm3'])] * n_layers,
            vertical_conductivity=[float(r['vertical_conductivity_ms'])] * n_layers,
            thermal_conductivity=[float(r['thermal_conductivity_wmk'])] * n_layers,
            thermal_capacity=[float(r['thermal_capacity_jm3k'])] * n_layers,
            residual_water_content=[float(r['residual_water_content'])] * n_layers,
            mannings_n=float(r['mannings_n']),
        ))
    return blocks


def checkSoilConsistency(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check soil parameters for physical consistency.

    DHSVM does not validate these, and an inconsistent set produces
    plausible-looking but wrong output -- for instance a wilting point
    above field capacity gives a permanently zero available water
    capacity and therefore zero transpiration.

    Returns
    -------
    dict
        ``'ok'`` plus lists of ``'errors'`` and ``'warnings'``.
    """
    errors, warnings = [], []
    for b in blocks:
        nm = f"soil {b['id']} ({b['description']})"
        for layer in range(b['n_layers']):
            phi = b['porosity'][layer]
            fc = b['field_capacity'][layer]
            wp = b['wilting_point'][layer]
            tr = b['residual_water_content'][layer]
            if not (0 < phi < 1):
                errors.append(f'{nm} layer {layer}: porosity {phi} outside (0,1)')
            if not (tr <= wp <= fc <= phi):
                errors.append(f'{nm} layer {layer}: expected residual <= wilting <= '
                              f'field capacity <= porosity, got '
                              f'{tr} / {wp} / {fc} / {phi}')
            if b['vertical_conductivity'][layer] <= 0:
                errors.append(f'{nm} layer {layer}: vertical conductivity must be > 0')
        if b['lateral_conductivity'] <= 0:
            errors.append(f'{nm}: lateral conductivity must be > 0')
        if b['lateral_conductivity'] < b['vertical_conductivity'][0]:
            warnings.append(f'{nm}: lateral K is below vertical K, which inverts the '
                            f'usual anisotropy of structured soils')
        if b['exponential_decrease'] <= 0:
            errors.append(f'{nm}: exponential decrease must be > 0')

    ok = len(errors) == 0
    if ok:
        logging.info(f'  soil parameters consistent across {len(blocks)} types')
    for e in errors:
        logging.error(f'  SOIL ERROR: {e}')
    for w in warnings:
        logging.warning(f'  soil warning: {w}')
    return dict(ok=ok, errors=errors, warnings=warnings)


def availableWaterCapacity(blocks: List[Dict[str, Any]],
                           soil_class: np.ndarray,
                           soil_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Plant-available water capacity per cell, mm.

    ``AWC = (field capacity - wilting point) * soil depth``.  A useful
    diagnostic: values below ~30 mm mean the basin will dry out and stop
    transpiring within days of a storm, which usually signals a soil
    depth that is too thin rather than a real soil.

    Returns
    -------
    np.ndarray
        AWC in mm.
    """
    awc_frac = np.zeros(256, dtype='float64')
    for b in blocks:
        awc_frac[b['id']] = np.mean(
            [fc - wp for fc, wp in zip(b['field_capacity'], b['wilting_point'])])
    out = awc_frac[soil_class] * soil_depth * 1000.0
    if mask is not None:
        stats = out[mask != 0]
        logging.info(f'  available water capacity: {stats.min():.0f}-{stats.max():.0f} mm, '
                     f'mean {stats.mean():.0f} mm')
    return out


#: POLARIS depth-layer boundaries, cm.  SoilGrids uses the same set.
POLARIS_DEPTHS_CM = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]


def aggregateDepthLayers(da,
                         depths_cm=None,
                         max_depth_cm: float = 100.0,
                         depth_dim: Optional[str] = None) -> np.ndarray:
    """Collapse a layered soil property to one depth-weighted value per cell.

    Gridded soil products (POLARIS, SoilGrids) report properties for a
    stack of standard depth intervals -- 0--5, 5--15, 15--30, 30--60,
    60--100 and 100--200 cm.  DHSVM, by contrast, wants **one texture
    class per cell**, because the soil type is a single map and the
    per-layer parameters come from the type's parameter block.

    The right reduction is a **thickness-weighted mean over the depth
    DHSVM actually simulates**.  That is the root zone plus the saturated
    zone above bedrock -- for the 0.10/0.40/0.90 m root zones and 0.8--3 m
    soil columns this package generates, the top metre carries essentially
    all of it, so ``max_depth_cm`` defaults to 100.  Including the
    100--200 cm layer at equal weight would let deep subsoil clay, which
    the model never sees, dominate the classification.

    Parameters
    ----------
    da : xr.DataArray or np.ndarray
        Layered property, with the depth axis first (or named by
        ``depth_dim``).
    depths_cm : list of tuple, optional
        ``(top, bottom)`` of each layer, cm.  Defaults to
        :data:`POLARIS_DEPTHS_CM`.
    max_depth_cm : float, optional
        Ignore layers whose top lies at or below this depth, and truncate
        the layer that straddles it.  Default 100 cm.
    depth_dim : str, optional
        Name of the depth dimension when ``da`` is a DataArray.

    Returns
    -------
    np.ndarray
        2D thickness-weighted mean.
    """
    values = np.asarray(getattr(da, 'values', da), dtype='float64')

    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(f'aggregateDepthLayers expects a 2D or 3D array, '
                         f'got shape {values.shape}')

    if depth_dim is not None and hasattr(da, 'dims'):
        axis = list(da.dims).index(depth_dim)
        values = np.moveaxis(values, axis, 0)

    n = values.shape[0]
    depths = depths_cm if depths_cm is not None else POLARIS_DEPTHS_CM
    if len(depths) < n:
        raise ValueError(f'Array has {n} depth layers but only {len(depths)} '
                         f'depth intervals were supplied.')
    depths = depths[:n]

    weights = np.array([max(min(b, max_depth_cm) - t, 0.0) for t, b in depths],
                       dtype='float64')
    if weights.sum() <= 0:
        raise ValueError(f'No soil layers lie above max_depth_cm={max_depth_cm}')

    used = [f'{t}-{b}' for (t, b), w in zip(depths, weights) if w > 0]
    logging.info(f'  aggregating {n} depth layers over the top {max_depth_cm:g} cm '
                 f'using {len(used)} layer(s) [{", ".join(used)} cm], '
                 f'thickness-weighted')

    w = weights[:, None, None]
    masked = np.where(np.isfinite(values), values, np.nan)
    num = np.nansum(masked * w, axis=0)
    den = np.nansum(np.where(np.isfinite(masked), 1.0, 0.0) * w, axis=0)
    out = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out
