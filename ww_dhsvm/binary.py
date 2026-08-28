"""Serialization of DHSVM's native binary map format.

DHSVM stores every spatial input as a *headerless* stream of values in
row-major order, starting at the **north-west** corner of the grid and
walking west-to-east, north-to-south::

    index = row * ncols + col      row 0 == northernmost row

There is no metadata in the file at all -- the number of rows, columns,
cell size and origin all come from the ``[AREA]`` section of the DHSVM
configuration file, and the data type comes from a compiled-in table in
DHSVM's ``VarID.c``.  Writing a map with the wrong dtype produces a file
of the wrong length, and DHSVM will read past the end of it, so the
dtype table below is transcribed directly from the model source and is
treated as authoritative throughout ``ww_dhsvm``.

The relevant entries of ``VarID.c`` are:

=======  ==========================  ==========  ==============================
Var ID   DHSVM name                  NetCDF type  numpy dtype
=======  ==========================  ==========  ==============================
001      ``Basin.DEM``               NC_FLOAT     float32
002      ``Basin.Mask``              NC_BYTE      uint8
003      ``Soil.Type``               NC_BYTE      uint8
004      ``Soil.Depth``              NC_FLOAT     float32
005      ``Veg.Type``                NC_BYTE      uint8
007      ``Veg.CanopyGap``           NC_FLOAT     float32
010      ``Veg.Fract``               NC_FLOAT     float32
011      ``Veg.LAI``                 NC_FLOAT     float32
012      ``Soil.KsLat``              NC_FLOAT     float32
=======  ==========================  ==========  ==============================

``Format`` in the ``[OPTIONS]`` section selects the byte order: ``BIN``
means the machine's native order and ``BYTESWAP`` means the opposite.
``ww_dhsvm`` writes native-order files and sets ``Format = BIN``, which
is what DHSVM expects when the model and the pre-processor run on the
same machine.

Adapted for DHSVM from Watershed Workflow's ``io`` module, which serves
the same role for ATS's HDF5 and ExodusII outputs.
"""

from typing import Optional, Tuple, Dict, Any
import os
import logging

import numpy as np


#: numpy dtypes for each DHSVM variable ID, transcribed from ``VarID.c``.
DHSVM_DTYPES: Dict[int, Any] = {
    1: np.float32,    # Basin.DEM
    2: np.uint8,      # Basin.Mask
    3: np.uint8,      # Soil.Type
    4: np.float32,    # Soil.Depth
    5: np.uint8,      # Veg.Type
    6: np.int16,      # Travel.Time  (NC_SHORT)
    7: np.float32,    # Veg.CanopyGap
    10: np.float32,   # Veg.Fract
    11: np.float32,   # Veg.LAI
    12: np.float32,   # Soil.KsLat
    13: np.float32,   # Soil.Porosity
    14: np.float32,   # Soil.FieldCapacity
}

#: Human-readable names, keyed the same way, used for logging.
DHSVM_VARNAMES: Dict[int, str] = {
    1: 'Basin.DEM',
    2: 'Basin.Mask',
    3: 'Soil.Type',
    4: 'Soil.Depth',
    5: 'Veg.Type',
    6: 'Travel.Time',
    7: 'Veg.CanopyGap',
    10: 'Veg.Fract',
    11: 'Veg.LAI',
    12: 'Soil.KsLat',
    13: 'Soil.Porosity',
    14: 'Soil.FieldCapacity',
}

#: Convenience aliases so callers can say ``writeMap(..., kind='dem')``.
KIND_TO_VARID: Dict[str, int] = {
    'dem': 1,
    'mask': 2,
    'soil': 3,
    'soil_type': 3,
    'soil_depth': 4,
    'veg': 5,
    'veg_type': 5,
    'canopy_gap': 7,
    'veg_fract': 10,
    'veg_lai': 11,
    'ks_lat': 12,
    'porosity': 13,
    'field_capacity': 14,
}


def dtypeFor(kind: str | int) -> Any:
    """Return the numpy dtype DHSVM expects for a map.

    Parameters
    ----------
    kind : str or int
        Either a DHSVM variable ID (e.g. ``1``) or one of the aliases in
        :data:`KIND_TO_VARID` (e.g. ``'dem'``).

    Returns
    -------
    numpy dtype
    """
    varid = kind if isinstance(kind, int) else KIND_TO_VARID[kind]
    if varid not in DHSVM_DTYPES:
        raise KeyError(f'No DHSVM dtype known for variable ID {varid}')
    return DHSVM_DTYPES[varid]


def writeMap(filename: str,
             array: np.ndarray,
             kind: str | int,
             nodata_fill: Optional[float] = None) -> str:
    """Write a 2D array as a DHSVM binary map.

    Parameters
    ----------
    filename : str
        Destination path.  Parent directories are created if needed.
    array : np.ndarray
        2D array shaped ``(nrows, ncols)`` with **row 0 in the north**.
    kind : str or int
        DHSVM variable ID or alias, used to select the output dtype.
    nodata_fill : float, optional
        Value substituted for any NaN in ``array`` before casting.  If
        ``None`` and NaNs are present, a ``ValueError`` is raised --
        DHSVM has no nodata convention for its binary maps, so silently
        writing NaN would poison the simulation.

    Returns
    -------
    str
        ``filename``, for convenient chaining.
    """
    if array.ndim != 2:
        raise ValueError(f'DHSVM maps must be 2D, got shape {array.shape}')

    dtype = dtypeFor(kind)
    varid = kind if isinstance(kind, int) else KIND_TO_VARID[kind]
    name = DHSVM_VARNAMES.get(varid, str(varid))

    out = np.asarray(array)
    if np.issubdtype(out.dtype, np.floating):
        n_nan = int(np.count_nonzero(np.isnan(out)))
        if n_nan:
            if nodata_fill is None:
                raise ValueError(
                    f'{name}: array contains {n_nan} NaN values and no '
                    f'nodata_fill was given.  DHSVM binary maps have no '
                    f'nodata convention -- fill them explicitly.')
            out = np.where(np.isnan(out), nodata_fill, out)

    # Integer maps must round rather than truncate: a soil class stored as
    # 3.9999 from a float pipeline should become 4, not 3.
    if np.issubdtype(dtype, np.integer) and np.issubdtype(out.dtype, np.floating):
        out = np.rint(out)

    out = np.ascontiguousarray(out, dtype=dtype)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'wb') as fid:
        out.tofile(fid)

    logging.info(f'  wrote {name:<18s} {out.shape[0]}x{out.shape[1]} '
                 f'{np.dtype(dtype).name:<8s} -> {filename} '
                 f'({os.path.getsize(filename)} bytes)')
    return filename


def readMap(filename: str,
            nrows: int,
            ncols: int,
            kind: str | int,
            nset: int = 0) -> np.ndarray:
    """Read a 2D array back out of a DHSVM binary map.

    Parameters
    ----------
    filename : str
        Path to the binary map.
    nrows, ncols : int
        Grid shape, from the ``[AREA]`` section of the config file.
    kind : str or int
        DHSVM variable ID or alias, used to select the dtype.
    nset : int, optional
        Which stacked matrix to read.  DHSVM state files concatenate
        several same-shaped matrices into one file; ``nset`` selects the
        zero-based index, mirroring ``Read2DMatrixBin``'s ``NDataSet``.

    Returns
    -------
    np.ndarray
        2D array shaped ``(nrows, ncols)``, row 0 in the north.
    """
    dtype = dtypeFor(kind)
    itemsize = np.dtype(dtype).itemsize
    offset = nrows * ncols * itemsize * nset

    expected = offset + nrows * ncols * itemsize
    actual = os.path.getsize(filename)
    if actual < expected:
        raise ValueError(
            f'{filename}: file is {actual} bytes but reading set {nset} of a '
            f'{nrows}x{ncols} {np.dtype(dtype).name} map needs {expected}.')

    with open(filename, 'rb') as fid:
        fid.seek(offset)
        flat = np.fromfile(fid, dtype=dtype, count=nrows * ncols)
    return flat.reshape((nrows, ncols))


def countSets(filename: str, nrows: int, ncols: int, kind: str | int) -> int:
    """Return how many stacked matrices a DHSVM binary file holds."""
    itemsize = np.dtype(dtypeFor(kind)).itemsize
    per_set = nrows * ncols * itemsize
    size = os.path.getsize(filename)
    if size % per_set:
        raise ValueError(
            f'{filename}: size {size} is not a whole multiple of one '
            f'{nrows}x{ncols} matrix ({per_set} bytes).')
    return size // per_set


def writeStack(filename: str,
               arrays,
               kind: str | int,
               nodata_fill: Optional[float] = None) -> str:
    """Write several same-shaped arrays as one stacked DHSVM binary file.

    DHSVM's model-state files use this layout: ``Interception.State`` is
    ``2 * MaxVegLayers + 1`` float32 matrices back to back, and so on.

    Parameters
    ----------
    filename : str
        Destination path.
    arrays : sequence of np.ndarray
        Arrays to concatenate, all shaped ``(nrows, ncols)``.
    kind : str or int
        DHSVM variable ID or alias selecting the dtype.
    nodata_fill : float, optional
        See :func:`writeMap`.

    Returns
    -------
    str
        ``filename``.
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError('writeStack called with no arrays')
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f'writeStack: arrays have inconsistent shapes {shapes}')

    dtype = dtypeFor(kind)
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'wb') as fid:
        for a in arrays:
            out = np.asarray(a)
            if np.issubdtype(out.dtype, np.floating) and np.any(np.isnan(out)):
                if nodata_fill is None:
                    raise ValueError(f'{filename}: NaN present and no nodata_fill')
                out = np.where(np.isnan(out), nodata_fill, out)
            np.ascontiguousarray(out, dtype=dtype).tofile(fid)

    logging.info(f'  wrote stack of {len(arrays)} x {arrays[0].shape} '
                 f'{np.dtype(dtype).name} -> {filename} '
                 f'({os.path.getsize(filename)} bytes)')
    return filename


def writeASCIIGrid(filename: str,
                   array: np.ndarray,
                   xllcorner: float,
                   yllcorner: float,
                   cellsize: float,
                   nodata: float = -9999.0,
                   fmt: str = '%.6g') -> str:
    """Write an ESRI ASCII grid alongside a binary map.

    DHSVM never reads these, but they make the binary maps inspectable in
    QGIS/ArcGIS and are the format the legacy PNNL ``myconvert`` utility
    consumes, so ``ww_dhsvm`` emits them next to every map it writes.

    Parameters
    ----------
    filename : str
        Destination ``.asc`` path.
    array : np.ndarray
        2D array, row 0 in the north.
    xllcorner, yllcorner : float
        Coordinates of the grid's south-west corner.
    cellsize : float
        Grid spacing, in the same units as the corner coordinates.
    nodata : float, optional
        Value written into the header and substituted for NaN.
    fmt : str, optional
        ``%``-format used for each value.

    Returns
    -------
    str
        ``filename``.
    """
    nrows, ncols = array.shape
    out = np.asarray(array, dtype=float)
    out = np.where(np.isnan(out), nodata, out)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, 'w') as fid:
        fid.write(f'ncols         {ncols}\n')
        fid.write(f'nrows         {nrows}\n')
        fid.write(f'xllcorner     {xllcorner:.8f}\n')
        fid.write(f'yllcorner     {yllcorner:.8f}\n')
        fid.write(f'cellsize      {cellsize:g}\n')
        fid.write(f'NODATA_value  {nodata:g}\n')
        np.savetxt(fid, out, fmt=fmt, delimiter=' ')
    return filename
