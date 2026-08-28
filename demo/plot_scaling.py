"""Render the strong/weak scaling figure from demo/scaling/results.json."""
import os, sys, json, logging, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import ww_dhsvm.figures as FIG, ww_dhsvm.plot as PL

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
PL.useStyle()

RESULTS = os.path.expanduser('~/ww_dhsvm/demo/scaling/results.json')
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
N_PHYSICAL = 16

res = json.load(open(RESULTS))
fig = FIG.scalingPanel(res, n_physical_cores=N_PHYSICAL)
p = os.path.join(FIGDIR, '12_scaling.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
logging.info(f'figure -> {p}')

# A compact table for the README.
def table(rows, kind):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r['nproc'])
    t0, s0 = rows[0]['t_long'], rows[0]['per_step']
    logging.info('')
    logging.info(f'{kind} scaling')
    hdr = f"{'ranks':>6} {'grid':>12} {'total s':>9} {'init s':>8} {'s/step':>9} {'speedup':>8} {'eff %':>7}"
    logging.info(hdr); logging.info('-' * len(hdr))
    for r in rows:
        g = f"{r.get('nrows','')}x{r.get('ncols','')}" if 'nrows' in r else ''
        sp = s0 / r['per_step']
        eff = 100 * (sp / r['nproc'] if kind == 'Strong' else s0 / r['per_step'])
        logging.info(f"{r['nproc']:>6d} {g:>12} {r['t_long']:>9.1f} {r['init']:>8.1f} "
                     f"{r['per_step']:>9.3f} {sp:>8.2f} {eff:>7.1f}")

table(res.get('strong', []), 'Strong')
table(res.get('weak', []), 'Weak')
