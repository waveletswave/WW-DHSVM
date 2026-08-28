"""Download and cache the full 2025 AORC forcing for the demo basin."""
import os, sys, logging, warnings, pickle, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.sources as S

CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache'))
g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
t0 = time.time()
# Decimate the 1 km AORC grid to ~8 km station spacing and resample to the
# 3 h DHSVM timestep.  DHSVM interpolates from a station list, so one station
# per 1 km cell over 13,000 km^2 is 25 GB of download for no added skill.
ds = S.met_sources['AORC'].getDataset(grid.polygon(), grid.crs,
                                      start='2025-01-01', end='2025-12-31',
                                      spatial_decimation=8,
                                      temporal_resampling='3h')
logging.info(f'AORC 2025 complete: {dict(ds.sizes)} in {(time.time()-t0)/60:.1f} min')
