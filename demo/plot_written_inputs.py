"""Read the DHSVM binary inputs back off disk and plot them.

Everything else in the demo plots arrays held in memory.  This reads the
five ``.bin`` files exactly as DHSVM will -- same dtype table, same
row-major north-west-origin layout -- so what is shown is what the model
sees.  A dtype or ordering error that survived every earlier check would
be obvious here.
"""
import os, sys, pickle, logging, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.binary as B
import ww_dhsvm.plot as P, ww_dhsvm.soils as SO, ww_dhsvm.vegetation as V

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)
P.useStyle()

NAME = os.environ.get('WWD_NAME', 'ConnecticutRiverBasin')
ROOT = os.path.expanduser(f'~/ww_dhsvm/demo/{NAME}')
CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache_ct'))
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
IP = os.path.join(ROOT, 'input')

g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
meta = pickle.load(open(os.path.join(ROOT, 'case.pkl'), 'rb'))

logging.info('Reading DHSVM binary inputs back off disk')
logging.info('-' * 70)
maps = {}
for key, fname, kind in [('DEM', 'dem.bin', 'dem'),
                         ('mask', 'mask.bin', 'mask'),
                         ('soil class', 'soil.bin', 'soil'),
                         ('soil depth', 'soild.bin', 'soil_depth'),
                         ('vegetation class', 'veg.bin', 'veg')]:
    path = os.path.join(IP, fname)
    a = B.readMap(path, grid.nrows, grid.ncols, kind)
    maps[key] = a
    logging.info(f'  {key:<18s} {fname:<12s} {str(a.shape):<14s} '
                 f'{str(a.dtype):<9s} {os.path.getsize(path):>10,d} B   '
                 f'range {a.min():.3f} .. {a.max():.3f}')

soil_table, veg_table = meta['soil_table'], meta['veg_table']

# Six maps of a basin three times taller than it is wide: the figure has
# to be tall, or every panel is a sliver in the middle of an empty cell.
fig = plt.figure(figsize=(15.0, 13.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.26, wspace=0.46)

hs = P.hillshade(maps['DEM'], grid.cellsize)
P.mapRaster(maps['DEM'], grid, ax=fig.add_subplot(gs[0, 0]), cmap=P.HYPSOMETRIC,
            label='elevation (m)', hillshade=hs,
            title='dem.bin', subtitle='float32, conditioned elevation')

P.mapCategorical(maps['mask'], grid, {0: 'outside', 1: 'inside basin'},
                 colors={0: '#e8e7e3', 1: P.SERIES[0]},
                 ax=fig.add_subplot(gs[0, 1]), mask=np.ones(grid.shape, 'uint8'),
                 title='mask.bin', subtitle='uint8, Outside Basin Value = 0')

P.mapRaster(maps['soil depth'], grid, ax=fig.add_subplot(gs[0, 2]),
            cmap=P.CMAP_SEQ, label='soil depth (m)',
            title='soild.bin', subtitle='float32, total soil column depth')

P.mapCategorical(maps['soil class'], grid,
                 {int(r['dhsvm_id']): r['name'] for _, r in soil_table.iterrows()},
                 ax=fig.add_subplot(gs[1, 0]),
                 title='soil.bin', subtitle='uint8, DHSVM soil type 1..N')

veg_colors = {int(r['dhsvm_id']): P.NLCD_COLORS.get(int(r['nlcd_code']),
                                                    P.SERIES[i % 8])
              for i, (_, r) in enumerate(veg_table.iterrows())}
P.mapCategorical(maps['vegetation class'], grid,
                 {int(r['dhsvm_id']): r['name'] for _, r in veg_table.iterrows()},
                 colors=veg_colors, ax=fig.add_subplot(gs[1, 1]), legend=False,
                 title='veg.bin',
                 subtitle='uint8, type 1..N; classes named in 06_vegetation')

# The stream files, drawn from the text DHSVM will parse.  Column 2 of
# stream.network.dat is the *routing rank*, not the Strahler order --
# DHSVM routes segments in ascending rank, so on this basin it runs 1 to
# 673.  Colouring by it would mean a 673-entry legend and would say
# nothing about the channels; the width each segment actually gets, via
# its class in stream.class.dat, is both readable and the number that
# changes the hydrology.
ax = fig.add_subplot(gs[1, 2])
smap = pd.read_csv(os.path.join(IP, 'stream.map.dat'), sep=r'\s+', comment='#',
                   header=None, engine='python',
                   names=['col', 'row', 'seg', 'len', 'cuth', 'cutw', 'aspect'])
net = pd.read_csv(os.path.join(IP, 'stream.network.dat'), sep=r'\s+',
                  header=None, engine='python', usecols=[0, 1, 2, 3, 4, 5],
                  names=['ID', 'rank', 'slope', 'length', 'class', 'outlet'])
cls = pd.read_csv(os.path.join(IP, 'stream.class.dat'), sep=r'\s+', comment='#',
                  header=None, engine='python',
                  names=['class', 'width', 'depth', 'n', 'infiltration'])
net['width'] = net['class'].map(dict(zip(cls['class'], cls['width'])))
smap['width'] = smap['seg'].map(dict(zip(net['ID'], net['width'])))
smap = smap.dropna(subset=['width'])

xs = (grid.xllcorner + (smap['col'] + 0.5) * grid.cellsize) / 1000.0
ys = (grid.yurcorner - (smap['row'] + 0.5) * grid.cellsize) / 1000.0
w = smap['width'].to_numpy(dtype='float64')
# Width spans two orders of magnitude, so the ramp is logarithmic and the
# marker grows with it -- visual weight matching hydrologic weight.
sc = ax.scatter(xs, ys, c=w, cmap=P.CMAP_SEQ,
                norm=LogNorm(vmin=max(w.min(), 1e-3), vmax=w.max()),
                s=0.6 + 2.4 * np.log2(w / max(w.min(), 1e-3) + 1.0),
                marker='s', linewidths=0)
ax.set_aspect('equal'); ax.grid(False)
ax.set_xlim(grid.xllcorner / 1000, grid.xurcorner / 1000)
ax.set_ylim(grid.yllcorner / 1000, grid.yurcorner / 1000)
ax.set_xlabel('easting (km)'); ax.set_ylabel('northing (km)')
P._finish(ax, 'stream.map.dat + .network.dat + .class.dat',
          f'{len(smap):,} cell records, {len(net):,} segments, '
          f'{len(cls)} classes')
P._attachColorbar(ax, sc, 'channel width (m)')

fig.suptitle('DHSVM inputs, read back from disk exactly as the model reads them',
             x=0.008, y=0.998, ha='left', fontsize=13.5, weight='semibold',
             color=P.INK)
p = os.path.join(FIGDIR, '00_written_inputs.png')
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
logging.info(f'\n  figure -> {p}')
logging.info('WRITTEN-INPUT VERIFICATION COMPLETE')
