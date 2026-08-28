"""Unit tests for the DHSVM file-format conventions WW-DHSVM must honour.

These are the conventions that are easy to get subtly wrong and that
DHSVM does not check: binary dtypes, matrix stacking order, grid origin,
row/column indexing, and meteorological units.  Where possible each is
checked against DHSVM's **own** Chiwawa test case, which ships with the
model, rather than against ww_dhsvm's expectations of itself.

Run with:  python -m pytest tests/test_formats.py -v
"""
import os
import sys
import datetime

import numpy as np
import pytest

sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))

import ww_dhsvm.binary as B
import ww_dhsvm.grid as G
import ww_dhsvm.crs as C
import ww_dhsvm.soils as SO
import ww_dhsvm.states as STA
import ww_dhsvm.meteorology as M
import ww_dhsvm.streams as ST

CHIWAWA = os.path.expanduser('~/ww_dhsvm/external/DHSVM-PNNL/TestCase/Chiwawa')
HAS_CHIWAWA = os.path.isdir(CHIWAWA)
chiwawa_only = pytest.mark.skipif(not HAS_CHIWAWA,
                                  reason='DHSVM test case not present')

# Chiwawa's [AREA] section, verbatim from INPUT.Chiwawa.Baseline
CHI_ROWS, CHI_COLS, CHI_DX = 425, 300, 90.0
CHI_XLL, CHI_YLL = 651368.575710976, 5300372.090466      # from header.txt
CHI_EXTREME_NORTH = 5338622.09046602                     # from the config


# ---------------------------------------------------------------- grid
def test_extreme_north_matches_dhsvm_config():
    """Extreme North is the grid's north EDGE, not a cell centre."""
    grid = G.ModelGrid(CHI_ROWS, CHI_COLS, CHI_DX, CHI_XLL, CHI_YLL,
                       C.from_epsg(32610))
    assert grid.yurcorner == pytest.approx(CHI_EXTREME_NORTH, abs=1e-6)


def test_rowcol_matches_dhsvm_rounding():
    """rowcol/xy must round the way InitMetSources.c does."""
    grid = G.ModelGrid(CHI_ROWS, CHI_COLS, CHI_DX, CHI_XLL, CHI_YLL,
                       C.from_epsg(32610))
    for r, c in [(0, 0), (7, 13), (424, 299), (212, 150)]:
        assert grid.rowcol(*grid.xy(r, c)) == (r, c)


def test_row_zero_is_north():
    grid = G.ModelGrid(10, 10, 100.0, 0.0, 0.0, C.from_epsg(32618))
    assert grid.xy(0, 0)[1] > grid.xy(9, 0)[1]


# -------------------------------------------------------------- binary
@chiwawa_only
def test_binary_dtypes_match_chiwawa_file_sizes():
    """The dtype table must reproduce DHSVM's own file sizes exactly."""
    n = CHI_ROWS * CHI_COLS
    expected = {'dem.bin': n * 4, 'mask.bin': n, 'soil.bin': n,
                'soild.bin': n * 4, 'veg.bin': n}
    for fname, size in expected.items():
        path = os.path.join(CHIWAWA, 'input', fname)
        assert os.path.getsize(path) == size, fname


@chiwawa_only
def test_read_chiwawa_dem_is_physical():
    dem = B.readMap(os.path.join(CHIWAWA, 'input', 'dem.bin'),
                    CHI_ROWS, CHI_COLS, 'dem')
    mask = B.readMap(os.path.join(CHIWAWA, 'input', 'mask.bin'),
                     CHI_ROWS, CHI_COLS, 'mask')
    inside = dem[mask > 0]
    assert dem.dtype == np.float32 and mask.dtype == np.uint8
    # Chiwawa is in the Washington Cascades.
    assert 500 < inside.min() < 800
    assert 2500 < inside.max() < 3000


def test_binary_roundtrip(tmp_path):
    a = np.arange(12, dtype='float64').reshape(3, 4) * 1.5
    p = str(tmp_path / 'x.bin')
    B.writeMap(p, a, 'dem')
    assert np.allclose(B.readMap(p, 3, 4, 'dem'), a)


def test_nan_without_fill_is_rejected(tmp_path):
    """DHSVM binary maps have no nodata convention, so NaN must not pass."""
    a = np.zeros((3, 3)); a[1, 1] = np.nan
    with pytest.raises(ValueError, match='NaN'):
        B.writeMap(str(tmp_path / 'x.bin'), a, 'dem')


def test_integer_maps_round_not_truncate(tmp_path):
    a = np.full((2, 2), 3.9999)
    p = str(tmp_path / 's.bin')
    B.writeMap(p, a, 'soil')
    assert (B.readMap(p, 2, 2, 'soil') == 4).all()


# --------------------------------------------------------------- state
@chiwawa_only
@pytest.mark.parametrize('fname,expected_sets', [
    ('Interception.State.10.01.1970.00.00.00.bin', 5),   # 2*L_veg + 1, L_veg=2
    ('Snow.State.10.01.1970.00.00.00.bin', 8),           # fixed
    ('Soil.State.10.01.1970.00.00.00.bin', 10),          # 2*L_soil + 4, L_soil=3
])
def test_state_stacking_matches_chiwawa(fname, expected_sets):
    path = os.path.join(CHIWAWA, 'modelstate', fname)
    assert B.countSets(path, CHI_ROWS, CHI_COLS, 'dem') == expected_sets


def test_written_state_has_the_counts_dhsvm_expects(tmp_path):
    grid = G.ModelGrid(20, 15, 100.0, 0.0, 0.0, C.from_epsg(32618))
    grid.mask = np.ones(grid.shape, 'uint8')
    _, table = SO.compactClasses(np.full(grid.shape, 3, 'uint8'), grid.mask)
    blocks = SO.buildSoilParameterBlocks(table, n_layers=3)
    files = STA.writeInitialStates(str(tmp_path), datetime.datetime(2020, 1, 1),
                                   grid, np.ones(grid.shape, 'uint8'), blocks,
                                   [1, 2, 3])
    assert STA.verifyStateFiles(files, grid)['ok']
    # Channel.State carries no .bin extension -- ChannelState.c omits fileext.
    assert not files['channel'].endswith('.bin')


# ----------------------------------------------------------------- met
def test_precip_is_metres_per_timestep(tmp_path):
    """Column 7 is a DEPTH over one timestep, not a rate."""
    import pandas as pd
    idx = pd.date_range('2025-01-01', periods=4, freq='3h')
    df = pd.DataFrame({'Tair': 5.0, 'Wind': 2.0, 'RH': 80.0, 'Sin': 100.0,
                       'Lin': 300.0,
                       'Precip': 0.001},                # 1 mm/hour
                      index=idx)
    p = str(tmp_path / 'st.txt')
    M.writeStationFile(p, df, timestep_hours=3.0)
    first = open(p).readline().split()
    assert first[0] == '01/01/2025-00'
    assert len(first) == 7
    assert float(first[1]) == pytest.approx(5.0)     # Celsius, not Kelvin
    assert float(first[6]) == pytest.approx(0.003)   # 1 mm/h x 3 h = 3 mm


def test_saturation_vapor_pressure():
    """Bolton (1980) at 20 C is 2338 Pa."""
    assert M.saturationVaporPressure(np.array([20.0]))[0] == pytest.approx(2338, rel=0.01)


def test_overcast_longwave_is_blackbody():
    t = 10.0
    lw = M.estimateLongwave(np.array([t]), np.array([70.0]), np.array([1.0]))[0]
    assert lw == pytest.approx(5.670374419e-8 * (t + 273.15) ** 4, rel=1e-6)


# ------------------------------------------------------------- texture
@pytest.mark.parametrize('sand,silt,clay,name', [
    (92, 5, 3, 'SAND'), (82, 12, 6, 'LOAMY SAND'), (65, 25, 10, 'SANDY LOAM'),
    (20, 65, 15, 'SILT LOAM'), (8, 86, 6, 'SILT'), (40, 40, 20, 'LOAM'),
    (60, 15, 25, 'SANDY CLAY LOAM'), (10, 55, 35, 'SILTY CLAY LOAM'),
    (30, 35, 35, 'CLAY LOAM'), (50, 10, 40, 'SANDY CLAY'),
    (8, 47, 45, 'SILTY CLAY'), (25, 25, 50, 'CLAY'),
])
def test_usda_texture_triangle(sand, silt, clay, name):
    cid = int(SO.textureToUSDAClass(np.array([sand]), np.array([silt]),
                                    np.array([clay]))[0])
    assert SO.USDA_CLASSES[cid - 1] == name


def test_texture_renormalises():
    """Fractions and percentages must classify identically."""
    a = SO.textureToUSDAClass(np.array([0.65]), np.array([0.25]), np.array([0.10]))
    b = SO.textureToUSDAClass(np.array([65.0]), np.array([25.0]), np.array([10.0]))
    assert a[0] == b[0]


# ------------------------------------------------------------- streams
def test_hydraulic_geometry_is_regionally_sane():
    """Connecticut River at ~25,000 km^2 is roughly 230 m wide."""
    w, d = ST.hydraulicGeometry(np.array([25000e6]))
    assert 180 < w[0] < 280
    assert 4 < d[0] < 8


def test_stream_map_indices_are_zero_based(tmp_path):
    """channel_grid.c rejects col/row outside [0, n)."""
    import pandas as pd
    sm = pd.DataFrame({'col': [0], 'row': [0], 'seg_id': [1], 'length': [10.0],
                       'cut_height': [0.5], 'cut_width': [2.0], 'aspect': [90.0]})
    p = str(tmp_path / 'stream.map.dat')
    ST.writeStreamMapFile(p, sm)
    data = [l for l in open(p) if not l.startswith('#')]
    assert data[0].split()[:2] == ['0', '0']


def test_topology_rejects_a_cycle():
    import pandas as pd
    segs = pd.DataFrame({'ID': [1, 2], 'outlet': [2, 1], 'order': [1, 1],
                         'slope': [0.01, 0.01], 'length': [100.0, 100.0]})
    res = ST.checkTopology(segs)
    assert not res['ok']
    assert any('cycle' in e for e in res['errors'])


def test_topology_rejects_nonpositive_slope():
    import pandas as pd
    segs = pd.DataFrame({'ID': [1], 'outlet': [0], 'order': [1],
                         'slope': [0.0], 'length': [100.0]})
    assert not ST.checkTopology(segs)['ok']


# ------------------------------------------------------- forcing gaps
def test_no_nan_ever_reaches_a_station_file(tmp_path):
    """A single NaN in one station poisons the whole basin in DHSVM.

    AORC v1.1 really does have isolated missing hours: five of the 46
    forcing stations over the Connecticut River Basin each carry two
    missing values, and one such station is enough to turn the rest of a
    year-long run into NaN with no error from the model.
    """
    import pandas as pd
    idx = pd.date_range('2025-01-01', periods=6, freq='3h')
    df = pd.DataFrame({'Tair': 5.0, 'Wind': 2.0, 'RH': 80.0, 'Sin': 100.0,
                       'Lin': 300.0,
                       'Precip': [0.0, 0.001, np.nan, np.nan, 0.002, 0.0]},
                      index=idx)
    p = str(tmp_path / 'st.txt')
    M.writeStationFile(p, df, timestep_hours=3.0)
    text = open(p).read().lower()
    assert 'nan' not in text and 'inf' not in text
    # The gap is bridged, not zeroed.
    vals = [float(l.split()[6]) for l in open(p)]
    assert vals[2] > 0 and vals[3] > 0


def test_long_gap_is_refused():
    import pandas as pd
    idx = pd.date_range('2025-01-01', periods=20, freq='3h')
    df = pd.DataFrame({'Tair': 5.0, 'Wind': 2.0, 'RH': 80.0, 'Sin': 100.0,
                       'Lin': 300.0, 'Precip': 0.0}, index=idx)
    df.loc[df.index[2:15], 'Tair'] = np.nan
    with pytest.raises(ValueError, match='gap'):
        M.repairGaps(df, 'x', max_gap=8)


def test_channel_area_never_exceeds_cell_area():
    """DHSVM forms channel area as width x length and never clamps it.

    A channel wider than the grid spacing would then intercept
    precipitation over more ground than the cell has.  On the Connecticut
    at 150 m this produced channel interception of 1,153 mm against
    955 mm of precipitation and a runoff ratio of 1.19.
    """
    import pandas as pd
    import ww_dhsvm.grid as G
    grid = G.ModelGrid(10, 10, 150.0, 0.0, 0.0, C.from_epsg(32618))
    # A 230 m wide river crossing one 150 m cell: 230 x 150 = 1.53 x cell area.
    sm = pd.DataFrame({'col': [3], 'row': [4], 'seg_id': [1], 'length': [150.0],
                       'cut_height': [5.0], 'cut_width': [230.0], 'aspect': [0.0]})
    out = ST._capChannelAreaToCell(sm, grid, max_fraction=0.9)
    area = float(out['cut_width'].iloc[0] * out['length'].iloc[0])
    assert area <= 0.9 * grid.cell_area + 1e-6
    assert out['cut_width'].iloc[0] < 230.0


def test_narrow_channels_are_left_alone():
    """The cap must not touch channels that already fit."""
    import pandas as pd
    import ww_dhsvm.grid as G
    grid = G.ModelGrid(10, 10, 150.0, 0.0, 0.0, C.from_epsg(32618))
    sm = pd.DataFrame({'col': [3], 'row': [4], 'seg_id': [1], 'length': [150.0],
                       'cut_height': [0.5], 'cut_width': [8.0], 'aspect': [0.0]})
    out = ST._capChannelAreaToCell(sm, grid, max_fraction=0.9)
    assert out['cut_width'].iloc[0] == 8.0
