"""Fetch and grid the static (non-meteorological) inputs for the demo basin.

Run separately from the notebook so the slow national-dataset downloads
happen once and are cached as .npy arrays on the model grid.
"""
import os, sys, time, logging, pickle
import numpy as np

sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

import ww_dhsvm, ww_dhsvm.sources as S, ww_dhsvm.grid as G, ww_dhsvm.terrain as T

HUC = os.environ.get('WWD_HUC', '0108')
CELL = float(os.environ.get('WWD_CELL', '150'))
CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache'))
os.makedirs(CACHE, exist_ok=True)

def save(name, arr):
    np.save(os.path.join(CACHE, name + '.npy'), arr)
    logging.info(f'  cached {name}.npy {arr.shape} {arr.dtype}')

def have(name):
    return os.path.exists(os.path.join(CACHE, name + '.npy'))

t0 = time.time()
sources = S.getDefaultSources()

# ---------------------------------------------------------------- domain
ws = ww_dhsvm.getWatershed(HUC, sources)
grid = G.ModelGrid.fromShape(ws, cellsize=CELL, buffer_cells=3)
grid.setMaskFromShape(ws)
with open(os.path.join(CACHE, 'grid.pkl'), 'wb') as f:
    pickle.dump(dict(nrows=grid.nrows, ncols=grid.ncols, cellsize=grid.cellsize,
                     xllcorner=grid.xllcorner, yllcorner=grid.yllcorner,
                     crs=grid.crs.to_wkt(), mask=grid.mask), f)
ws.to_file(os.path.join(CACHE, 'watershed.gpkg'), driver='GPKG')
logging.info(grid.summary())

# ------------------------------------------------------------------- DEM
if not have('dem_raw'):
    logging.info('=== 3DEP DEM ===')
    t = time.time()
    for res in (30, 60):
        try:
            dem_src = S.dem_sources[f'3DEP {res}m']
            ds = dem_src.getDataset(grid.polygon().buffer(600), grid.crs)
            var = list(ds.data_vars)[0]
            dem = T.resampleToGrid(ds[var], grid, 'bilinear')
            save('dem_raw', dem.astype('float32'))
            logging.info(f'  3DEP {res}m OK in {time.time()-t:.0f}s')
            break
        except Exception as exc:
            logging.warning(f'  3DEP {res}m failed: {type(exc).__name__}: {exc}')
    else:
        raise SystemExit('DEM acquisition failed at every resolution')

# ------------------------------------------------------------ land cover
if not have('nlcd'):
    logging.info('=== NLCD land cover ===')
    t = time.time()
    ds = sources['land cover'].getDataset(grid.polygon().buffer(600), grid.crs,
                                          variables=['cover'])
    nlcd = T.resampleToGrid(ds['cover'], grid, 'nearest', nodata=0)
    save('nlcd', np.nan_to_num(nlcd, nan=0).astype('uint8'))
    logging.info(f'  NLCD OK in {time.time()-t:.0f}s')

if not have('impervious'):
    try:
        logging.info('=== NLCD impervious ===')
        ds = sources['land cover'].getDataset(grid.polygon().buffer(600), grid.crs,
                                              variables=['impervious'])
        imp = T.resampleToGrid(ds['impervious'], grid, 'bilinear')
        save('impervious', imp.astype('float32'))
    except Exception as exc:
        logging.warning(f'  impervious failed: {type(exc).__name__}: {exc}')

# ----------------------------------------------------------- soil texture
if not have('sand'):
    logging.info('=== soil texture ===')
    for label, mgr, names in [
            ('POLARIS', S.soil_sources['POLARIS'], None),
            ('SoilGrids', S.soil_sources['SoilGrids'], None)]:
        try:
            t = time.time()
            ds = mgr.getDataset(grid.polygon().buffer(600), grid.crs)
            logging.info(f'  {label} variables: {list(ds.data_vars)}')
            got = {}
            for sep in ('sand', 'silt', 'clay'):
                cand = [v for v in ds.data_vars if sep in v.lower()]
                if cand:
                    got[sep] = T.resampleToGrid(ds[cand[0]], grid, 'bilinear')
            if len(got) >= 2:
                for k, v in got.items():
                    save(k, v.astype('float32'))
                logging.info(f'  {label} OK in {time.time()-t:.0f}s')
                break
        except Exception as exc:
            logging.warning(f'  {label} failed: {type(exc).__name__}: {exc}')
    else:
        logging.warning('  no soil texture source succeeded')

# ------------------------------------------------------- depth to bedrock
if not have('dtb'):
    try:
        logging.info('=== Pelletier depth to bedrock ===')
        ds = sources['depth to bedrock'].getDataset(grid.polygon().buffer(600), grid.crs)
        var = list(ds.data_vars)[0]
        dtb = T.resampleToGrid(ds[var], grid, 'bilinear')
        save('dtb', dtb.astype('float32'))
    except Exception as exc:
        logging.warning(f'  DTB failed: {type(exc).__name__}: {exc}')

# ----------------------------------------------------------- hydrography
if not os.path.exists(os.path.join(CACHE, 'nhd.gpkg')):
    try:
        logging.info('=== NHD flowlines ===')
        t = time.time()
        reaches = sources['hydrography'].getShapesByGeometry(
            ws.union_all(), ws.crs, out_crs=grid.crs)
        reaches.to_file(os.path.join(CACHE, 'nhd.gpkg'), driver='GPKG')
        logging.info(f'  NHD OK: {len(reaches)} reaches in {time.time()-t:.0f}s')
    except Exception as exc:
        logging.warning(f'  NHD failed: {type(exc).__name__}: {exc}')

logging.info(f'=== STATIC FETCH COMPLETE in {time.time()-t0:.0f}s ===')
for f in sorted(os.listdir(CACHE)):
    logging.info(f'  {f}')
