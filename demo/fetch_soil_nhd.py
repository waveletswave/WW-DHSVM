"""Fetch the soil texture and hydrography that the first static pass missed.

POLARIS defaults to van Genuchten parameters; DHSVM needs *texture*, so
sand/silt/clay are requested explicitly.  NHD comes back with more than
one geometry column (flowline plus catchment), which GeoPackage cannot
hold, so the extras are dropped before writing.
"""
import os, sys, time, logging, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

import pickle, geopandas as gpd
import ww_dhsvm, ww_dhsvm.sources as S, ww_dhsvm.grid as G, ww_dhsvm.crs as C
import ww_dhsvm.terrain as T

CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache'))
g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
ws = gpd.read_file(os.path.join(CACHE, 'watershed.gpkg'))

def save(name, arr):
    np.save(os.path.join(CACHE, name + '.npy'), arr)
    logging.info(f'  cached {name}.npy {arr.shape} {arr.dtype}')

# ------------------------------------------------------- soil texture
# POLARIS returns a 6-depth-layer cube per variable.  Over the full
# Connecticut basin each variable is a ~2.8 GB netCDF, so requesting
# sand+silt+clay in ONE getDataset call materialises ~8.4 GB plus
# resampling temporaries and gets the process OOM-killed on a 31 GB
# machine.  Fetching one variable at a time, and releasing it before the
# next, keeps peak memory to roughly one variable.
import gc
for sep in ('sand', 'silt', 'clay'):
    if os.path.exists(os.path.join(CACHE, sep + '.npy')):
        logging.info(f'  {sep}: already cached')
        continue
    logging.info(f'=== POLARIS {sep} ===')
    t = time.time()
    import xarray as xr
    import ww_dhsvm.soils as SOIL
    ds = S.soil_sources['POLARIS'].getDataset(grid.polygon().buffer(600),
                                              grid.crs, variables=[sep])
    cand = [v for v in ds.data_vars if sep in v.lower()]
    if not cand:
        logging.warning(f'  {sep} missing from the POLARIS response')
        ds.close(); del ds; gc.collect()
        continue
    layered = ds[cand[0]]
    # Collapse the depth layers to a thickness-weighted mean over the top
    # metre -- the depth DHSVM actually simulates.
    flat = SOIL.aggregateDepthLayers(
        layered, max_depth_cm=100.0,
        depth_dim=[d for d in layered.dims if d not in ('x', 'y')][0]
        if layered.ndim == 3 else None)
    da = xr.DataArray(flat, dims=('y', 'x'),
                      coords={'y': layered['y'], 'x': layered['x']})
    da = da.rio.write_crs(layered.rio.crs)
    save(sep, T.resampleToGrid(da, grid, 'bilinear').astype('float32'))
    ds.close()
    del ds, layered, flat, da
    gc.collect()
    logging.info(f'  {sep} OK in {time.time()-t:.0f}s')

# -------------------------------------------------------- hydrography
if not os.path.exists(os.path.join(CACHE, 'nhd.gpkg')):
    logging.info('=== NHD flowlines ===')
    t = time.time()
    reaches = S.hydrography_sources['NHDPlus MR v2.1'].getShapesByGeometry(
        ws.union_all(), ws.crs, out_crs=grid.crs)
    # Keep only the active geometry; GeoPackage allows exactly one.
    geom_cols = [c for c in reaches.columns
                 if reaches[c].dtype.name == 'geometry']
    keep = reaches.geometry.name
    drop = [c for c in geom_cols if c != keep]
    if drop:
        logging.info(f'  dropping extra geometry columns: {drop}')
        reaches = reaches.drop(columns=drop)
    # Keep a compact attribute set: GPKG chokes on list-valued columns.
    simple = [c for c in reaches.columns
              if c == keep or reaches[c].map(lambda v: isinstance(
                  v, (str, int, float, np.integer, np.floating, type(None)))).all()]
    reaches = reaches[simple]
    reaches.to_file(os.path.join(CACHE, 'nhd.gpkg'), driver='GPKG')
    logging.info(f'  NHD OK: {len(reaches)} reaches in {time.time()-t:.0f}s')

logging.info('=== SOIL/NHD FETCH COMPLETE ===')
for f in sorted(os.listdir(CACHE)):
    logging.info(f'  {f}')
