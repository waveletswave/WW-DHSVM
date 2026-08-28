"""Regenerate the input figures without touching any DHSVM input file.

build_demo.py both writes inputs and draws figures; this script only
draws, so it is safe to run while DHSVM is reading the station forcing.
"""
import os, sys, pickle, logging, warnings, datetime
warnings.filterwarnings('ignore')
import numpy as np, geopandas as gpd
sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.terrain as T
import ww_dhsvm.streams as ST, ww_dhsvm.soils as SO, ww_dhsvm.vegetation as V
import ww_dhsvm.meteorology as M, ww_dhsvm.sources as S
import ww_dhsvm.figures as FIG, ww_dhsvm.plot as PL, ww_dhsvm.diagnostics as D

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)
PL.useStyle()

CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache_ct'))
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
NAME = os.environ.get('WWD_NAME', 'ConnecticutRiverBasin')
ROOT = os.path.expanduser(f'~/ww_dhsvm/demo/{NAME}')
CELL_THRESH = float(os.environ.get('WWD_CHANNEL_KM2', '0.5'))
TSTEP, MAX_STATIONS = 3.0, 64

def savefig(fig, name):
    p = os.path.join(FIGDIR, name + '.png')
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    logging.info(f'  figure -> {p}')

g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
watershed = gpd.read_file(os.path.join(CACHE, 'watershed.gpkg'))
reaches = gpd.read_file(os.path.join(CACHE, 'nhd.gpkg'))

dem = np.load(os.path.join(CACHE, 'dem_raw.npy')).astype('float64')
dem = T.fillGaps(dem, grid.mask)
dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
terrain = T.conditionDEM(dem.astype('float32'), grid, streams=reaches, burn_depth=5.0)
dem = terrain['dem']

network = ST.extractNetwork(terrain, grid, channel_threshold_km2=CELL_THRESH)
comparison = ST.compareToReference(network['channel_mask'], reaches, grid)

sand = np.load(os.path.join(CACHE, 'sand.npy')).astype('float64')
silt = np.load(os.path.join(CACHE, 'silt.npy')).astype('float64')
clay = np.load(os.path.join(CACHE, 'clay.npy')).astype('float64')
soil_cls = SO.classifyFromFractions(sand, silt, clay, grid.mask)
soil_map, soil_table = SO.compactClasses(soil_cls, grid.mask)
soil_blocks = SO.buildSoilParameterBlocks(soil_table, n_layers=3)
soil_depth = T.estimateSoilDepth(terrain['slope'], terrain['dem'], terrain['uparea'],
                                 min_depth=0.8, max_depth=3.0, mask=grid.mask)
soil_depth = T.enforceSoilDepthBelowChannels(soil_depth, network['cut_height_grid'])

nlcd = np.load(os.path.join(CACHE, 'nlcd.npy'))
imperv = np.load(os.path.join(CACHE, 'impervious.npy')).astype('float64')
cover = V.classifyLandCover(nlcd, grid.mask)
veg_map, veg_table = V.compactClasses(cover, grid.mask)
veg_blocks = V.buildVegetationBlocks(veg_table)
V.applyImperviousFraction(veg_blocks, imperv, veg_map, grid.mask)
soil_depth = T.enforceSoilDepthBelowRootZone(soil_depth, veg_map, veg_blocks, grid.mask)
awc = SO.availableWaterCapacity(soil_blocks, soil_map, soil_depth, grid.mask)

met_raw = S.met_sources['AORC'].getDataset(
    grid.polygon(), grid.crs, start='2025-01-01', end='2025-12-31',
    spatial_decimation=8, temporal_resampling='3h')
met = M.convertToDHSVM(met_raw, 'AORC')
met_summary = M.annualWaterBalanceCheck(met, TSTEP)
stations = M.placeStations(grid, dem, met, max_stations=MAX_STATIONS)
for st in stations:
    st.filename = os.path.join(ROOT, 'met', f'{st.name}.txt')

savefig(FIG.domainOverview(grid, watershed, dem, reaches), '01_domain')
savefig(FIG.terrainPanel(grid, terrain), '02_terrain')
savefig(FIG.flowRoutingPanel(grid, terrain), '03_flow_routing')
savefig(FIG.channelPanel(grid, network, terrain, comparison), '04_channels')
savefig(FIG.soilPanel(grid, soil_map, soil_table, soil_depth, awc), '05_soils')
savefig(FIG.vegetationPanel(grid, veg_map, veg_table, veg_blocks), '06_vegetation')
savefig(FIG.forcingPanel(met, stations, grid, dem, TSTEP), '07_forcing')

report = D.checkInputs(grid, dem, soil_map, soil_depth, veg_map, soil_blocks,
                       veg_blocks, network=network, met_summary=met_summary,
                       expected_annual_precip_mm=1200.0)
case = dict(grid=grid, network=network, soil_blocks=soil_blocks,
            veg_blocks=veg_blocks, met_summary=met_summary,
            stations=stations, report=report)
savefig(FIG.inputDashboard(case), '08_dashboard')
report.log()

with open(os.path.join(CACHE, 'figcase.pkl'), 'wb') as f:
    pickle.dump(dict(met_summary=met_summary, n_stations=len(stations),
                     comparison=comparison,
                     n_segments=len(network['segments']),
                     drainage_density=network['drainage_density'],
                     n_errors=len(report.errors),
                     n_warnings=len(report.warnings)), f)
logging.info('FIGURES REGENERATED')
