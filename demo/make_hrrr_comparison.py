"""Compare AORC and HRRR over the demo basin for June 2025."""
import os, sys, pickle, logging, warnings
warnings.filterwarnings('ignore')
import numpy as np
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
os.environ.setdefault('HERBIE_SAVE_DIR', os.path.expanduser('~/ww_dhsvm/data/herbie'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.sources as S
import ww_dhsvm.meteorology as M, ww_dhsvm.figures as FIG, ww_dhsvm.plot as PL

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)
PL.useStyle()
CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache_ct'))
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
TSTEP = 3.0

g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])

logging.info('loading HRRR June 2025 ...')
hrrr_raw = S.met_sources['HRRR'].getDataset(grid.polygon(), grid.crs,
                                            start='2025-06-01', end='2025-06-14')
hrrr = M.convertToDHSVM(hrrr_raw, 'HRRR').resample(time=f'{int(TSTEP)}h').mean()
logging.info(f'HRRR: {dict(hrrr.sizes)}')

logging.info('loading AORC 2025 ...')
aorc_raw = S.met_sources['AORC'].getDataset(grid.polygon(), grid.crs,
                                            start='2025-01-01', end='2025-12-31',
                                            spatial_decimation=8,
                                            temporal_resampling='3h')
aorc = M.convertToDHSVM(aorc_raw, 'AORC')

# Align on the HRRR window, in the same (real-calendar) time base.
import pandas as pd
aorc = aorc.assign_coords(time=M.toDatetimeIndex(aorc['time'].values))
hrrr = hrrr.assign_coords(time=M.toDatetimeIndex(hrrr['time'].values))
aorc_w = aorc.sel(time=slice('2025-06-01', '2025-06-14'))
logging.info(f'AORC window: {dict(aorc_w.sizes)}')

logging.info('')
logging.info('Basin-mean comparison over 1-14 June 2025')
logging.info('-' * 66)
logging.info(f'{"variable":<10s} {"AORC":>12s} {"HRRR":>12s} {"HRRR-AORC":>12s} {"r":>8s}')
sa = [d for d in aorc_w['Tair'].dims if d != 'time']
sh = [d for d in hrrr['Tair'].dims if d != 'time']
for v, unit, scale in [('Tair', 'deg C', 1.0), ('Precip', 'mm/day', 24000.0),
                       ('Sin', 'W/m2', 1.0), ('Lin', 'W/m2', 1.0),
                       ('RH', '%', 1.0), ('Wind', 'm/s', 1.0)]:
    a = aorc_w[v].mean(dim=sa).to_series().resample('1D').mean() * scale
    b = hrrr[v].mean(dim=sh).to_series().resample('1D').mean() * scale
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    r = float(np.corrcoef(a.values, b.values)[0, 1])
    logging.info(f'{v:<10s} {a.mean():12.3f} {b.mean():12.3f} '
                 f'{b.mean()-a.mean():+12.3f} {r:8.3f}   [{unit}]')

fig = FIG.forcingComparison(aorc_w, hrrr, 'AORC', 'HRRR', TSTEP)
p = os.path.join(FIGDIR, '10_aorc_vs_hrrr.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
logging.info(f'\n  figure -> {p}')

fig = FIG.forcingPanel(hrrr, M.placeStations(grid, np.load(
    os.path.join(CACHE, 'dem_raw.npy')).astype('float64'), hrrr,
    max_stations=48), grid,
    np.load(os.path.join(CACHE, 'dem_raw.npy')).astype('float64'),
    TSTEP, sample_days=10)
p = os.path.join(FIGDIR, '11_hrrr_forcing.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
logging.info(f'  figure -> {p}')
logging.info('HRRR COMPARISON COMPLETE')
