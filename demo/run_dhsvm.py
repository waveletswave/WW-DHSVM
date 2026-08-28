"""Run the Connecticut River Basin DHSVM case and plot the results.

Uses the parallel (Global Arrays + MPI) build on 8 ranks when it is
available, falling back to the serial build otherwise.
"""
import os, sys, time, pickle, logging, subprocess, warnings, datetime
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ww_dhsvm.grid as G, ww_dhsvm.crs as C
import ww_dhsvm.output as O, ww_dhsvm.diagnostics as D
import ww_dhsvm.figures as FIG, ww_dhsvm.plot as PL

NAME = os.environ.get('WWD_NAME', 'ConnecticutRiverBasin')
NPROC = int(os.environ.get('WWD_NPROC', '8'))
ROOT = os.path.expanduser(f'~/ww_dhsvm/demo/{NAME}')
CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache_ct'))
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
os.makedirs(FIGDIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)
PL.useStyle()

PAR = os.path.expanduser('~/ww_dhsvm/build/dhsvm-parallel/DHSVM/sourcecode/DHSVM')
SER = os.path.expanduser('~/ww_dhsvm/build/dhsvm-serial/DHSVM/sourcecode/DHSVM')

g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
meta = pickle.load(open(os.path.join(ROOT, 'case.pkl'), 'rb'))
TSTEP = meta['timestep_hours']
config_file = meta['config_file']

use_parallel = os.path.exists(PAR) and os.environ.get('WWD_SERIAL') != '1'
cmd = ([ 'mpirun', '-np', str(NPROC), PAR, config_file] if use_parallel
       else [SER, config_file])

logging.info('=' * 78)
logging.info(f'RUNNING DHSVM {"(parallel, %d ranks)" % NPROC if use_parallel else "(serial)"}')
logging.info(f'  {" ".join(cmd)}')
logging.info(f'  grid {grid.nrows} x {grid.ncols}, {grid.n_active:,} active cells, '
             f'{grid.basin_area_km2:,.0f} km^2')
logging.info('=' * 78)

t0 = time.time()
proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
elapsed = time.time() - t0

logging.info(f'exit code {proc.returncode}, wall time {elapsed/60:.2f} min')
with open(os.path.join(ROOT, 'dhsvm_run.log'), 'w') as f:
    f.write(proc.stdout)
with open(os.path.join(ROOT, 'dhsvm_run.err'), 'w') as f:
    f.write(proc.stderr)

for line in proc.stdout.splitlines()[-15:]:
    logging.info('  ' + line)
if proc.returncode != 0:
    logging.error('--- stderr tail ---')
    for line in proc.stderr.splitlines()[-30:]:
        logging.error('  ' + line)
    sys.exit(proc.returncode)

logging.info('--- final mass balance ---')
for line in proc.stderr.splitlines()[-24:]:
    logging.info('  ' + line)

# ------------------------------------------------------------- results
outdir = os.path.join(ROOT, 'output')
results = O.readAll(outdir, timestep_hours=TSTEP)
wb = O.waterBalanceSummary(results, grid.basin_area_km2, TSTEP)
rep = D.compareOutputToExpectation(wb, grid.basin_area_km2)
rep.log()

fig = FIG.outputPanel(results, grid, TSTEP, wb)
p = os.path.join(FIGDIR, '09_outputs.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
logging.info(f'  figure -> {p}')

with open(os.path.join(ROOT, 'results.pkl'), 'wb') as f:
    pickle.dump(dict(water_balance=wb, elapsed_s=elapsed, nproc=NPROC,
                     parallel=use_parallel), f)

logging.info('=' * 78)
logging.info(f'RUN COMPLETE in {elapsed/60:.2f} min on '
             f'{NPROC if use_parallel else 1} core(s)')
logging.info('=' * 78)
