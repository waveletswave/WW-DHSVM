"""End-to-end test: build a complete synthetic DHSVM case and run the model.

This exercises every writer in ww_dhsvm against the real DHSVM binary.
It is the fastest way to catch a format error, because DHSVM validates
far more of its input than any Python-side check can.
"""
import os, sys, shutil, logging, datetime, subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
logging.basicConfig(level=logging.INFO, format='%(message)s')

import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.terrain as T
import ww_dhsvm.streams as S, ww_dhsvm.soils as SO, ww_dhsvm.vegetation as V
import ww_dhsvm.binary as B, ww_dhsvm.states as ST, ww_dhsvm.config_writer as CW
import ww_dhsvm.meteorology as M

ROOT = os.path.expanduser('~/ww_dhsvm/tests/synthetic_case')
if os.path.exists(ROOT): shutil.rmtree(ROOT)
for d in ('input', 'met', 'state', 'output'):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# ---------------------------------------------------------------- grid
NY, NX, CELL = 60, 80, 150.0
grid = G.ModelGrid(NY, NX, CELL, 700000.0, 4600000.0, C.from_epsg(32618))

# A tilted, roughened surface draining to the south-east corner.
yy, xx = np.mgrid[0:NY, 0:NX]
dem = (900.0 - 4.0*yy - 2.5*xx + 35*np.sin(xx/7.0)*np.cos(yy/9.0)).astype('float32')

# Basin = everything except a border ring, so there is one clean outlet.
mask = np.zeros((NY, NX), 'uint8'); mask[2:-2, 2:-2] = 1
grid.mask = mask
print(grid.summary()); print()

# ------------------------------------------------------------- terrain
ter = T.conditionDEM(dem, grid)
soil_depth = T.estimateSoilDepth(ter['slope'], ter['dem'], ter['uparea'],
                                 min_depth=0.6, max_depth=2.5, mask=mask)

# ------------------------------------------------------------- streams
net = S.extractNetwork(ter, grid, channel_threshold_km2=0.5)
chk = S.checkTopology(net['segments']); assert chk['ok'], chk['errors']
soil_depth = T.enforceSoilDepthBelowChannels(soil_depth, net['cut_height_grid'])

# --------------------------------------------------------------- soils
sand = np.full((NY, NX), 55.0); silt = np.full((NY, NX), 30.0); clay = np.full((NY, NX), 15.0)
sand[:30] = 25.0; silt[:30] = 45.0; clay[:30] = 30.0     # two texture zones
soil_cls = SO.classifyFromFractions(sand, silt, clay, mask)
soil_map, soil_tbl = SO.compactClasses(soil_cls, mask)
soil_blocks = SO.buildSoilParameterBlocks(soil_tbl, n_layers=3)
assert SO.checkSoilConsistency(soil_blocks)['ok']

# ---------------------------------------------------------- vegetation
nlcd = np.full((NY, NX), 41, 'uint8'); nlcd[:20] = 42; nlcd[45:] = 71; nlcd[:, :10] = 21
nlcd = V.classifyLandCover(nlcd, mask)
veg_map, veg_tbl = V.compactClasses(nlcd, mask)
veg_blocks = V.buildVegetationBlocks(veg_tbl)
assert V.checkVegetationConsistency(veg_blocks, 3)['ok']

# DHSVM (CheckOut.c) requires soil depth > total rooting depth at every cell.
soil_depth = T.enforceSoilDepthBelowRootZone(soil_depth, veg_map, veg_blocks, mask)

# ------------------------------------------------------ binary writers
ip = os.path.join(ROOT, 'input')
paths = {
    'dem':        B.writeMap(os.path.join(ip, 'dem.bin'), ter['dem'], 'dem'),
    'mask':       B.writeMap(os.path.join(ip, 'mask.bin'), mask, 'mask'),
    'soil':       B.writeMap(os.path.join(ip, 'soil.bin'), soil_map, 'soil'),
    'soil_depth': B.writeMap(os.path.join(ip, 'soild.bin'), soil_depth, 'soil_depth'),
    'veg':        B.writeMap(os.path.join(ip, 'veg.bin'), veg_map, 'veg'),
}
sf = S.writeAll(ip, net)
paths.update(stream_map=sf['map'], stream_network=sf['network'],
             stream_class=sf['class'])
paths['impervious_routing'] = S.writeImperviousRoutingFile(
    os.path.join(ip, 'impervious.routing.txt'), net['channel_mask'], grid)

# ------------------------------------------------------------ forcing
START = datetime.datetime(2020, 10, 1, 0)
END   = datetime.datetime(2020, 10, 31, 0)
TSTEP = 3.0
times = pd.date_range(START, END, freq=f'{int(TSTEP)}h')

# Three stations at differing elevations, so lapse rates are exercised.
stations = []
for k, (r, c) in enumerate([(10, 15), (30, 40), (50, 65)]):
    e, n = grid.xy(r, c)
    st = M.MetStation(f'synth_{k}', e, n, 41.5, -72.5, float(ter['dem'][r, c]), r, c)
    rng = np.random.default_rng(100 + k)
    doy = times.dayofyear.values; hod = times.hour.values
    df = pd.DataFrame({
        'Tair':   12 - 8*np.cos(2*np.pi*doy/365) + 5*np.sin(2*np.pi*(hod-6)/24),
        'Wind':   np.clip(2.5 + rng.normal(0, 0.8, len(times)), 0.3, None),
        'RH':     np.clip(72 + rng.normal(0, 10, len(times)), 15, 100),
        'Sin':    np.clip(600*np.sin(np.pi*np.clip((hod-6)/12, 0, 1)), 0, None),
        'Lin':    300 + rng.normal(0, 15, len(times)),
        # m/hr; ~3 mm/hr bursts about 8% of the time
        'Precip': np.where(rng.random(len(times)) < 0.08,
                           rng.exponential(0.003, len(times)), 0.0),
    }, index=times)
    st.filename = M.writeStationFile(os.path.join(ROOT, 'met', f'{st.name}.txt'),
                                     df, TSTEP)
    stations.append(st)
print(f'\n  wrote {len(stations)} synthetic station files, '
      f'{len(times)} steps of {TSTEP:g} h')

# -------------------------------------------------------------- states
ST.writeInitialStates(os.path.join(ROOT, 'state'), START, grid, soil_map,
                      soil_blocks, net['segments']['ID'].tolist())

# -------------------------------------------------------------- config
cfg = CW.buildConfig('synthetic', grid, START, END, TSTEP, paths,
                     soil_blocks, veg_blocks, stations,
                     output_dir=os.path.join(ROOT, 'output'),
                     state_dir=os.path.join(ROOT, 'state'))
val = CW.validateConfig(cfg, grid, soil_blocks, veg_blocks, soil_map, veg_map, TSTEP)
assert val['ok'], val['errors']
cfgfile = cfg.write(os.path.join(ROOT, 'INPUT.synthetic'))

# ----------------------------------------------------------- run DHSVM
# DHSVM is not vendored in this repository -- build it from
# pnnl/DHSVM-PNNL (see the README) and point DHSVM_EXE at the binary.
exe = os.environ.get(
    'DHSVM_EXE',
    os.path.expanduser('~/ww_dhsvm/build/dhsvm-serial/DHSVM/sourcecode/DHSVM'))
if not os.path.exists(exe):
    print(f'\nDHSVM binary not found at {exe}')
    print('All inputs were written successfully; skipping the model run.')
    print('Build DHSVM (see README) and set DHSVM_EXE to run this end to end.')
    sys.exit(0)
print(f'\n=== running DHSVM ===')
p = subprocess.run([exe, cfgfile], cwd=ROOT, capture_output=True, text=True, timeout=1800)
print('exit code:', p.returncode)
tail = [l for l in p.stdout.splitlines() if l.strip()][-6:]
print('\n'.join('  ' + t for t in tail))
if p.returncode != 0:
    print('--- stdout tail ---'); print('\n'.join(p.stdout.splitlines()[-25:]))
    print('--- stderr tail ---'); print('\n'.join(p.stderr.splitlines()[-25:]))
    sys.exit(1)
print('\n--- final mass balance ---')
print('\n'.join(p.stderr.splitlines()[-20:]))
print('\n--- outputs ---')
for f in sorted(os.listdir(os.path.join(ROOT, 'output'))):
    print(f'  {f:<28s} {os.path.getsize(os.path.join(ROOT,"output",f)):>10,d} bytes')
print('\nEND-TO-END SYNTHETIC TEST PASSED')
