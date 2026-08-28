"""Initial model-state files for DHSVM.

DHSVM can either start from a cold, internally-generated state or
*restore* a saved one.  Restoring is the normal mode -- ``Initial State
Directory`` in ``[OUTPUT]`` points at a directory, and DHSVM looks there
for four files stamped with the model start time::

    Interception.State.MM.DD.YYYY.HH.MM.SS.bin
    Snow.State.MM.DD.YYYY.HH.MM.SS.bin
    Soil.State.MM.DD.YYYY.HH.MM.SS.bin
    Channel.State.MM.DD.YYYY.HH.MM.SS          (plain text, no .bin)

The three ``.bin`` files are **stacks of same-shaped float32 matrices**
written back to back in the binary map layout of
:mod:`ww_dhsvm.binary`.  The stacking order is fixed by the sequence of
``Read2DMatrix`` calls in ``InitModelState.c``, and was verified against
the file sizes distributed with DHSVM's own Chiwawa test case
(425 x 300 cells, so 510,000 bytes per matrix):

===========================  ==========================  ==========  =======
file                         layout                      Chiwawa     matrices
===========================  ==========================  ==========  =======
``Interception.State``       ``2*L_veg + 1``             2,550,000   5
``Snow.State``               8, fixed                    4,080,000   8
``Soil.State``               ``2*L_soil + 4``            5,100,000   10
===========================  ==========================  ==========  =======

with ``L_veg`` the maximum number of canopy layers (2) and ``L_soil``
the number of soil layers (3).

Contents, in order:

``Interception.State``
    ``IntRain`` per canopy layer (m), ``IntSnow`` per canopy layer (m),
    then ``TempIntStorage`` (m).

``Snow.State``
    ``HasSnow`` flag, ``LastSnow`` (days since snowfall), ``Swq`` (m),
    ``PackWater`` (m), ``TPack`` (C), ``SurfWater`` (m), ``TSurf`` (C),
    ``ColdContent`` (J).

``Soil.State``
    ``Moist`` for ``L_soil + 1`` layers (the extra one is the zone below
    the root zone), ``TSurf`` (C), ``Temp`` per soil layer (C), ``Qst``
    (W/m^2), ``Runoff`` (m).

``Channel.State`` is plain text, one ``segment_id  storage`` pair per
line, read by ``fscanf(InFile, "%hu %f", ...)`` in ``ChannelState.c``.

Spin-up
-------
The states this module writes are a *reasonable* cold start, not an
equilibrated one: soil moisture is set to a specified fraction of field
capacity, snow is absent, and channels hold a small baseflow storage.
A DHSVM basin typically needs 1--3 years of spin-up before subsurface
storage stops drifting, so the recommended pattern is to run a spin-up
period, let DHSVM dump its own state with ``Number of Model States``,
and restart the analysis run from that.  :func:`writeInitialStates`
logs this reminder.
"""

from typing import Optional, Dict, List, Any
import os
import logging
import datetime

import numpy as np

import ww_dhsvm.binary as binary


def stateTimestamp(when: datetime.datetime) -> str:
    """Format a datetime as DHSVM's state-file stamp, ``MM.DD.YYYY.HH.MM.SS``."""
    return when.strftime('%m.%d.%Y.%H.%M.%S')


def writeInterceptionState(outdir: str,
                           when: datetime.datetime,
                           shape,
                           n_veg_layers: int = 2,
                           int_rain: float = 0.0,
                           int_snow: float = 0.0) -> str:
    """Write ``Interception.State``: ``2*L_veg + 1`` float32 matrices.

    A dry canopy (all zeros) is the right cold start -- interception
    storage equilibrates within a single storm, so there is nothing to
    gain from guessing.
    """
    stamp = stateTimestamp(when)
    path = os.path.join(outdir, f'Interception.State.{stamp}.bin')
    arrays = ([np.full(shape, int_rain, dtype='float32') for _ in range(n_veg_layers)]
              + [np.full(shape, int_snow, dtype='float32') for _ in range(n_veg_layers)]
              + [np.zeros(shape, dtype='float32')])
    binary.writeStack(path, arrays, 'dem')
    return path


def writeSnowState(outdir: str,
                   when: datetime.datetime,
                   shape,
                   swq: Optional[np.ndarray] = None,
                   pack_temp: float = 0.0,
                   surf_temp: float = 0.0) -> str:
    """Write ``Snow.State``: 8 float32 matrices.

    Parameters
    ----------
    swq : np.ndarray, optional
        Initial snow water equivalent, metres.  ``None`` means no snow,
        which is the correct cold start for an autumn (1 October) water-
        year beginning in a temperate basin.
    """
    stamp = stateTimestamp(when)
    path = os.path.join(outdir, f'Snow.State.{stamp}.bin')

    if swq is None:
        swq_arr = np.zeros(shape, dtype='float32')
    else:
        swq_arr = np.asarray(swq, dtype='float32')

    has_snow = (swq_arr > 0).astype('float32')
    last_snow = np.full(shape, 365.0, dtype='float32')   # days since snowfall
    pack_water = np.zeros(shape, dtype='float32')
    t_pack = np.full(shape, pack_temp, dtype='float32')
    surf_water = np.zeros(shape, dtype='float32')
    t_surf = np.full(shape, surf_temp, dtype='float32')
    cold_content = np.zeros(shape, dtype='float32')

    binary.writeStack(path, [has_snow, last_snow, swq_arr, pack_water,
                             t_pack, surf_water, t_surf, cold_content], 'dem')
    return path


def writeSoilState(outdir: str,
                   when: datetime.datetime,
                   shape,
                   soil_moisture: np.ndarray,
                   n_soil_layers: int = 3,
                   soil_temp_c: float = 10.0,
                   surface_temp_c: float = 10.0,
                   ponding_m: float = 0.0) -> str:
    """Write ``Soil.State``: ``2*L_soil + 4`` float32 matrices.

    Parameters
    ----------
    soil_moisture : np.ndarray
        Volumetric water content to initialize every layer with, shaped
        like the grid.  Build it with :func:`initialSoilMoisture`.
    soil_temp_c : float, optional
        Initial soil temperature.  Only used when ``Sensible Heat Flux``
        is on, but the matrices must be present regardless.
    """
    stamp = stateTimestamp(when)
    path = os.path.join(outdir, f'Soil.State.{stamp}.bin')

    moist = np.asarray(soil_moisture, dtype='float32')
    arrays = [moist.copy() for _ in range(n_soil_layers + 1)]
    arrays.append(np.full(shape, surface_temp_c, dtype='float32'))       # TSurf
    arrays.extend(np.full(shape, soil_temp_c, dtype='float32')
                  for _ in range(n_soil_layers))                         # Temp
    arrays.append(np.zeros(shape, dtype='float32'))                      # Qst
    arrays.append(np.full(shape, ponding_m, dtype='float32'))            # Runoff

    binary.writeStack(path, arrays, 'dem')
    return path


def writeChannelState(outdir: str,
                      when: datetime.datetime,
                      segment_ids,
                      storage: Optional[Dict[int, float]] = None,
                      default_storage: float = 0.0) -> str:
    """Write ``Channel.State``, a plain-text ``id storage`` table.

    Note the filename has **no** ``.bin`` extension -- ``ChannelState.c``
    builds it without ``fileext``, unlike the three binary state files.

    Parameters
    ----------
    segment_ids : sequence of int
        Every segment in ``stream.network.dat``.  DHSVM reads exactly
        this many records, so the two files must agree.
    storage : dict, optional
        ``{segment_id: storage_m3}``.  Missing IDs get
        ``default_storage``.
    """
    stamp = stateTimestamp(when)
    path = os.path.join(outdir, f'Channel.State.{stamp}')
    os.makedirs(outdir, exist_ok=True)
    storage = storage or {}
    with open(path, 'w') as fid:
        for sid in segment_ids:
            fid.write(f'{int(sid):12d} {storage.get(int(sid), default_storage):12.6f}\n')
    logging.info(f'  wrote {len(list(segment_ids))} channel states -> {path}')
    return path


def initialSoilMoisture(soil_class: np.ndarray,
                        soil_blocks: List[Dict[str, Any]],
                        fraction_of_field_capacity: float = 0.9,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Build an initial soil-moisture field from the soil parameters.

    Initializing at a fraction of field capacity is the standard DHSVM
    cold start: moist enough that the basin is not artificially
    water-limited from the first timestep, dry enough that it does not
    generate a spurious initial flood.  Values are clipped to lie
    strictly between the wilting point and porosity, since DHSVM's
    unsaturated-flow routines assume that.

    Parameters
    ----------
    soil_class : np.ndarray
        DHSVM soil-type IDs on the grid.
    soil_blocks : list of dict
        From :func:`ww_dhsvm.soils.buildSoilParameterBlocks`.
    fraction_of_field_capacity : float, optional
        Default 0.9.
    mask : np.ndarray, optional
        Basin mask, used only for the log message.

    Returns
    -------
    np.ndarray
        Volumetric water content.
    """
    theta = np.zeros(256, dtype='float64')
    lo = np.zeros(256, dtype='float64')
    hi = np.ones(256, dtype='float64')
    for b in soil_blocks:
        fc = float(np.mean(b['field_capacity']))
        wp = float(np.mean(b['wilting_point']))
        phi = float(np.mean(b['porosity']))
        theta[b['id']] = np.clip(fraction_of_field_capacity * fc,
                                 wp * 1.01, phi * 0.99)
        lo[b['id']], hi[b['id']] = wp, phi

    out = theta[soil_class]
    sel = out[mask != 0] if mask is not None else out
    logging.info(f'  initial soil moisture: {sel.min():.3f}-{sel.max():.3f} '
                 f'(vol/vol), {fraction_of_field_capacity:.0%} of field capacity')
    return out


def writeInitialStates(outdir: str,
                       when: datetime.datetime,
                       grid,
                       soil_class: np.ndarray,
                       soil_blocks: List[Dict[str, Any]],
                       segment_ids,
                       n_veg_layers: int = 2,
                       n_soil_layers: int = 3,
                       fraction_of_field_capacity: float = 0.9,
                       initial_swq: Optional[np.ndarray] = None,
                       soil_temp_c: float = 10.0,
                       channel_storage: float = 0.0) -> Dict[str, str]:
    """Write all four DHSVM initial-state files.

    Returns
    -------
    dict
        Paths keyed ``'interception'``, ``'snow'``, ``'soil'`` and
        ``'channel'``.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(f'Writing initial model state for {when:%Y-%m-%d %H:%M} '
                 f'-> {outdir}')

    moisture = initialSoilMoisture(soil_class, soil_blocks,
                                   fraction_of_field_capacity, grid.mask)

    files = {
        'interception': writeInterceptionState(outdir, when, grid.shape, n_veg_layers),
        'snow': writeSnowState(outdir, when, grid.shape, initial_swq),
        'soil': writeSoilState(outdir, when, grid.shape, moisture,
                               n_soil_layers, soil_temp_c),
        'channel': writeChannelState(outdir, when, segment_ids,
                                     default_storage=channel_storage),
    }

    logging.info('  NOTE: these are a cold start, not an equilibrated state.  '
                 'Subsurface storage in a DHSVM basin typically needs 1-3 years '
                 'to stop drifting; run a spin-up period, dump a state with '
                 '"Number of Model States", and restart the analysis run from it.')
    return files


def verifyStateFiles(files: Dict[str, str],
                     grid,
                     n_veg_layers: int = 2,
                     n_soil_layers: int = 3) -> Dict[str, Any]:
    """Check each state file has exactly the number of matrices DHSVM expects.

    A wrong count is the single most common cause of a DHSVM restart
    reading garbage: it does not check, it simply seeks to an offset and
    reads whatever is there.
    """
    expected = {
        'interception': 2 * n_veg_layers + 1,
        'snow': 8,
        'soil': 2 * n_soil_layers + 4,
    }
    errors = []
    for key, n_exp in expected.items():
        path = files.get(key)
        if not path or not os.path.exists(path):
            errors.append(f'{key} state file missing: {path}')
            continue
        n_got = binary.countSets(path, grid.nrows, grid.ncols, 'dem')
        if n_got != n_exp:
            errors.append(f'{key}: file holds {n_got} matrices, DHSVM expects {n_exp}')
        else:
            logging.info(f'  {key:<13s} {n_got:>2d} matrices  OK')

    ok = len(errors) == 0
    for e in errors:
        logging.error(f'  STATE ERROR: {e}')
    return dict(ok=ok, errors=errors)
