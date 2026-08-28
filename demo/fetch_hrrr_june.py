"""Fetch one month of HRRR (June 2025) for the AORC comparison.

A full year is available by widening the date range; one month is enough
to validate the HRRR path end to end and to compare the two products,
without a multi-hour download.
"""
import os, sys, logging, warnings, pickle, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
os.environ.setdefault('HERBIE_SAVE_DIR', os.path.expanduser('~/ww_dhsvm/data/herbie'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.sources as S

CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache'))
g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
t0 = time.time()
ds = S.met_sources['HRRR'].getDataset(grid.polygon(), grid.crs,
                                      start='2025-06-01', end='2025-06-14')
logging.info(f'HRRR June 2025 complete: {dict(ds.sizes)} in {(time.time()-t0)/60:.1f} min')
