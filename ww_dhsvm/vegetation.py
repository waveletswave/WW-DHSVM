"""Vegetation classification and DHSVM vegetation parameters.

DHSVM's canopy is a **two-layer** representation: an optional overstory
and an optional understory, each with its own height, LAI, albedo,
stomatal-resistance parameters and root distribution.  Interception,
snow interception and unloading, radiation attenuation, aerodynamic
resistance and transpiration are all computed per layer, which is what
makes DHSVM well suited to forest-management questions -- and which is
why the vegetation table is the largest block in a DHSVM config file
(roughly 35 keys per type).

This module crosswalks **NLCD land cover** to DHSVM vegetation types and
emits those parameter blocks.  The crosswalk lives in
``data/nlcd_to_dhsvm_vegetation.json`` so it can be edited without
touching code -- and it *should* be edited: canopy height, fractional
coverage and minimum stomatal resistance are all regionally variable and
are among the most sensitive parameters in the model.

Seasonality
-----------
``Overstory Monthly LAI``, ``Understory Monthly LAI``, the two monthly
albedo series and ``Monthly Light Extinction`` are 12-element arrays.
The built-in table carries phenologically realistic Northern-Hemisphere
temperate profiles: deciduous forest swings from LAI 0.2 in winter to
5.5 at midsummer, evergreen stays near 4.5--5.5 year round.  Where MODIS
LAI is available, :func:`applyObservedLAI` will replace these with
observed climatologies -- the same MODIS product Watershed Workflow uses
for ATS, but aggregated to monthly means per land-cover class rather
than written as a transient HDF5 time series, because DHSVM's vegetation
table is climatological by construction.

Impervious area
---------------
NLCD's developed classes carry an impervious fraction, which DHSVM uses
to partition rainfall directly to overland flow.  ``ww_dhsvm`` assigns
each developed class its NLCD-typical mean; where the NLCD *impervious
surface* product is fetched, :func:`applyImperviousFraction` refines the
class-mean into a per-cell value and splits the class if needed.
"""

from typing import Optional, Dict, List, Tuple, Any
import os
import json
import logging

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

#: NLCD class codes and display names, matching the MRLC legend.
NLCD_NAMES = {
    11: 'Open Water', 12: 'Perennial Ice/Snow',
    21: 'Developed, Open Space', 22: 'Developed, Low Intensity',
    23: 'Developed, Medium Intensity', 24: 'Developed, High Intensity',
    31: 'Barren Land', 41: 'Deciduous Forest', 42: 'Evergreen Forest',
    43: 'Mixed Forest', 51: 'Dwarf Scrub', 52: 'Shrub/Scrub',
    71: 'Grassland/Herbaceous', 72: 'Sedge/Herbaceous', 73: 'Lichens',
    74: 'Moss', 81: 'Pasture/Hay', 82: 'Cultivated Crops',
    90: 'Woody Wetlands', 95: 'Emergent Herbaceous Wetlands',
}

#: NLCD classes without their own DHSVM block are folded into these.
NLCD_FALLBACK = {51: 52, 72: 71, 73: 71, 74: 71}


def loadVegetationTable() -> Dict[str, Dict[str, Any]]:
    """Load the NLCD-to-DHSVM vegetation crosswalk."""
    path = os.path.join(_DATA_DIR, 'nlcd_to_dhsvm_vegetation.json')
    with open(path) as fid:
        return json.load(fid)


def classifyLandCover(nlcd: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      default_class: int = 71) -> np.ndarray:
    """Normalize an NLCD raster into classes the DHSVM table covers.

    Rare classes (dwarf scrub, sedge, lichens, moss) are folded into
    their nearest represented relative, and nodata is replaced by
    ``default_class``.

    Parameters
    ----------
    nlcd : np.ndarray
        NLCD class codes on the model grid.
    mask : np.ndarray, optional
        Basin mask.
    default_class : int, optional
        Class assigned where NLCD is missing.  Default 71 (grassland).

    Returns
    -------
    np.ndarray
        ``uint8`` NLCD codes, all present in the vegetation table.
    """
    table = loadVegetationTable()
    known = set(int(k) for k in table)

    out = np.asarray(nlcd).astype('int32', copy=True)
    out = np.where((out < 0) | (out > 95), 0, out)

    for src, dst in NLCD_FALLBACK.items():
        n = int(np.count_nonzero(out == src))
        if n:
            logging.info(f'  folding NLCD {src} ({NLCD_NAMES.get(src, "?")}) '
                         f'into {dst} ({NLCD_NAMES.get(dst, "?")}): {n} cells')
            out = np.where(out == src, dst, out)

    unknown = ~np.isin(out, list(known))
    if mask is not None:
        unknown &= (mask != 0)
    n_unknown = int(unknown.sum())
    if n_unknown:
        logging.info(f'  {n_unknown} in-basin cells have no NLCD class; '
                     f'assigning {default_class} ({NLCD_NAMES[default_class]})')
    out = np.where(~np.isin(out, list(known)), default_class, out)
    return out.astype('uint8')


def compactClasses(nlcd: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Renumber NLCD codes into DHSVM's dense ``1..N`` vegetation IDs.

    DHSVM indexes vegetation types from 1 and requires a parameter block
    for every value up to ``Number of Vegetation Types``; NLCD codes are
    sparse (11, 21, 41, 90, ...), so they must be remapped.

    Returns
    -------
    remapped : np.ndarray
        ``uint8`` map with values ``1..N``.
    table : pd.DataFrame
        One row per DHSVM vegetation type, with ``dhsvm_id``,
        ``nlcd_code``, ``name`` and ``n_cells``.
    """
    active = nlcd if mask is None else nlcd[mask != 0]
    present = sorted(int(c) for c in np.unique(active) if c > 0)

    remap = np.zeros(256, dtype='uint8')
    rows = []
    for new_id, code in enumerate(present, start=1):
        remap[code] = new_id
        n = int(np.count_nonzero(active == code))
        rows.append(dict(dhsvm_id=new_id, nlcd_code=code,
                         name=NLCD_NAMES.get(code, str(code)), n_cells=n))

    remapped = remap[nlcd]
    if mask is not None:
        remapped = np.where((mask == 0) & (remapped == 0), 1, remapped)
    remapped = np.where(remapped == 0, 1, remapped).astype('uint8')

    table = pd.DataFrame(rows).set_index('dhsvm_id', drop=False)
    total = max(int(table['n_cells'].sum()), 1)
    logging.info(f'  {len(table)} DHSVM vegetation types:')
    for _, r in table.sort_values('n_cells', ascending=False).iterrows():
        logging.info(f"      {r['dhsvm_id']:>2d}  NLCD {r['nlcd_code']:<3d} "
                     f"{r['name']:<30s} {r['n_cells']:>8,d} cells "
                     f"({100.0*r['n_cells']/total:5.1f}%)")
    return remapped, table


def buildVegetationBlocks(table: pd.DataFrame) -> List[Dict[str, Any]]:
    """Expand the crosswalk into per-type DHSVM ``[VEGETATION]`` blocks.

    Parameters
    ----------
    table : pd.DataFrame
        Output of :func:`compactClasses`.

    Returns
    -------
    list of dict
        One dict per vegetation type, keyed as
        :mod:`ww_dhsvm.config_writer` expects.
    """
    crosswalk = loadVegetationTable()
    blocks = []
    for _, r in table.iterrows():
        code = str(int(r['nlcd_code']))
        if code not in crosswalk:
            raise KeyError(f'NLCD class {code} has no entry in the DHSVM '
                           f'vegetation crosswalk.')
        b = dict(crosswalk[code])
        b['id'] = int(r['dhsvm_id'])
        b['nlcd_code'] = int(r['nlcd_code'])
        blocks.append(b)
    return blocks


def applyImperviousFraction(blocks: List[Dict[str, Any]],
                            impervious: Optional[np.ndarray],
                            veg_class: np.ndarray,
                            mask: Optional[np.ndarray] = None
                            ) -> List[Dict[str, Any]]:
    """Refine each class's impervious fraction from the NLCD product.

    DHSVM's ``Impervious Fraction`` is a *per-vegetation-type* constant,
    not a map, so the best available refinement is to replace each
    developed class's literature default with the mean of the NLCD
    percent-impervious raster over the cells of that class in this
    particular basin.

    Parameters
    ----------
    blocks : list of dict
        Output of :func:`buildVegetationBlocks`; modified in place.
    impervious : np.ndarray or None
        NLCD percent-impervious (0-100) on the model grid.  ``None``
        leaves the defaults untouched.
    veg_class : np.ndarray
        DHSVM vegetation IDs on the model grid.
    mask : np.ndarray, optional
        Basin mask.

    Returns
    -------
    list of dict
        ``blocks``, for chaining.
    """
    if impervious is None:
        return blocks

    imp = np.asarray(impervious, dtype='float64')
    if np.nanmax(imp) > 1.5:
        imp = imp / 100.0

    for b in blocks:
        sel = (veg_class == b['id'])
        if mask is not None:
            sel &= (mask != 0)
        vals = imp[sel]
        vals = vals[np.isfinite(vals)]
        if vals.size < 5:
            continue
        new = float(np.clip(vals.mean(), 0.0, 1.0))
        old = b['impervious_fraction']
        if abs(new - old) > 0.01:
            logging.info(f"  {b['description']}: impervious fraction "
                         f'{old:.2f} -> {new:.2f} (NLCD basin mean)')
        b['impervious_fraction'] = new
    return blocks


def applyObservedLAI(blocks: List[Dict[str, Any]],
                     lai_monthly: Optional[Dict[int, List[float]]],
                     layer: str = 'overstory') -> List[Dict[str, Any]]:
    """Replace tabulated monthly LAI with an observed climatology.

    Parameters
    ----------
    blocks : list of dict
        Vegetation blocks; modified in place.
    lai_monthly : dict or None
        ``{dhsvm_id: [12 monthly LAI values]}``, typically derived from
        MODIS.  ``None`` leaves the table untouched.
    layer : str, optional
        ``'overstory'`` or ``'understory'``.

    Returns
    -------
    list of dict
        ``blocks``.
    """
    if not lai_monthly:
        return blocks
    key = f'{layer}_monthly_lai'
    for b in blocks:
        if b['id'] in lai_monthly:
            vals = list(lai_monthly[b['id']])
            if len(vals) != 12:
                raise ValueError(f'Monthly LAI for type {b["id"]} must have 12 '
                                 f'values, got {len(vals)}')
            # An overstory-free type must keep LAI at zero, or DHSVM will
            # intercept precipitation in a canopy that does not exist.
            if layer == 'overstory' and not b['overstory']:
                continue
            logging.info(f"  {b['description']}: {layer} LAI "
                         f"{np.mean(b[key]):.2f} -> {np.mean(vals):.2f} (observed mean)")
            b[key] = vals
    return blocks


def checkVegetationConsistency(blocks: List[Dict[str, Any]],
                               n_soil_layers: int = 3) -> Dict[str, Any]:
    """Check vegetation parameters against DHSVM's structural requirements.

    The checks encode constraints that ``InitTables.c`` and the canopy
    routines assume but do not all verify:

    * a type with ``Overstory Present = TRUE`` needs a positive height,
      a fractional coverage in (0, 1], and non-zero LAI;
    * a type with no overstory must not carry overstory LAI, or DHSVM
      will intercept water in a non-existent canopy;
    * ``Number of Root Zones`` must equal the soil's number of layers;
    * root fractions must sum to 1 for each present layer;
    * ``Trunk Space`` is a fraction of canopy height and must lie in
      (0, 1);
    * monthly arrays must have exactly 12 entries.
    """
    errors, warnings = [], []
    for b in blocks:
        nm = f"veg {b['id']} ({b['description']})"

        for key in ('overstory_monthly_lai', 'understory_monthly_lai',
                    'overstory_monthly_alb', 'understory_monthly_alb',
                    'monthly_light_extinction'):
            if len(b[key]) != 12:
                errors.append(f'{nm}: {key} has {len(b[key])} entries, expected 12')

        if b['n_root_zones'] != n_soil_layers:
            errors.append(f"{nm}: {b['n_root_zones']} root zones but the soil has "
                          f'{n_soil_layers} layers; DHSVM requires them to match')
        if len(b['root_zone_depths']) != b['n_root_zones']:
            errors.append(f'{nm}: root_zone_depths length does not match n_root_zones')

        if b['overstory']:
            if b['height'][0] <= 0:
                errors.append(f'{nm}: overstory present but height is {b["height"][0]}')
            if not (0 < b['fractional_coverage'] <= 1):
                errors.append(f"{nm}: fractional coverage {b['fractional_coverage']} "
                              f'outside (0, 1]')
            if max(b['overstory_monthly_lai']) <= 0:
                errors.append(f'{nm}: overstory present but all monthly LAI are zero')
            s = sum(b['overstory_root_fraction'])
            if abs(s - 1.0) > 1e-3:
                errors.append(f'{nm}: overstory root fractions sum to {s:.3f}, not 1')
            if not (0 < b['trunk_space'] < 1):
                errors.append(f"{nm}: trunk space {b['trunk_space']} outside (0, 1)")
        else:
            if max(b['overstory_monthly_lai']) > 0:
                errors.append(f'{nm}: no overstory, but overstory LAI is non-zero')

        if b['understory']:
            if max(b['understory_monthly_lai']) <= 0:
                warnings.append(f'{nm}: understory present but all monthly LAI are zero')
            s = sum(b['understory_root_fraction'])
            if abs(s - 1.0) > 1e-3:
                errors.append(f'{nm}: understory root fractions sum to {s:.3f}, not 1')

        if not (0 <= b['impervious_fraction'] <= 1):
            errors.append(f"{nm}: impervious fraction {b['impervious_fraction']} "
                          f'outside [0, 1]')

    ok = len(errors) == 0
    if ok:
        logging.info(f'  vegetation parameters consistent across {len(blocks)} types')
    for e in errors:
        logging.error(f'  VEGETATION ERROR: {e}')
    for w in warnings:
        logging.warning(f'  vegetation warning: {w}')
    return dict(ok=ok, errors=errors, warnings=warnings)


def maxVegLayers(blocks: List[Dict[str, Any]]) -> int:
    """Maximum number of canopy layers across all types.

    Sets the number of stacked matrices in ``Interception.State``; see
    :mod:`ww_dhsvm.states`.
    """
    return max(int(b['overstory']) + int(b['understory']) for b in blocks)


def canopyGapMap(veg_class: np.ndarray,
                 blocks: List[Dict[str, Any]],
                 gap_fraction: float = 0.0) -> np.ndarray:
    """Build the canopy-gap diameter map.

    DHSVM 3.2's canopy-gap module (Sun et al., 2018) represents a
    circular forest opening within a cell and solves a separate energy
    balance inside it.  The map holds the gap **diameter in metres**, and
    zero disables gapping for that cell.

    Parameters
    ----------
    veg_class : np.ndarray
        DHSVM vegetation IDs.
    blocks : list of dict
        Vegetation blocks carrying ``canopy_gap_diameter``.
    gap_fraction : float, optional
        Fraction of forested cells given a gap, chosen at random with a
        fixed seed for reproducibility.  Default 0 (no gaps), which also
        means ``Canopy Gapping = FALSE`` in the config.

    Returns
    -------
    np.ndarray
        ``float32`` gap-diameter map.
    """
    out = np.zeros(veg_class.shape, dtype='float32')
    if gap_fraction <= 0:
        return out

    rng = np.random.default_rng(20250827)
    for b in blocks:
        if not b['overstory'] or b['canopy_gap_diameter'] <= 0:
            continue
        sel = np.flatnonzero((veg_class == b['id']).ravel())
        if sel.size == 0:
            continue
        n = int(round(gap_fraction * sel.size))
        chosen = rng.choice(sel, size=n, replace=False)
        flat = out.ravel()
        flat[chosen] = b['canopy_gap_diameter']
    logging.info(f'  canopy gaps assigned to {int((out>0).sum())} cells '
                 f'({100*gap_fraction:.0f}% of forested cells)')
    return out
