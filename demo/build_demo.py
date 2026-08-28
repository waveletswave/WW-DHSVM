"""Build the Connecticut River Basin DHSVM case with WW-DHSVM.

This is the scripted twin of ``notebooks/lower_connecticut_dhsvm.ipynb``:
the same sequence of calls, without the narrative, so the case can be
rebuilt from a shell.  Static inputs come from the .npy cache written by
``fetch_static.py`` / ``fetch_soil_nhd.py``; meteorology comes from the
package cache written by ``fetch_aorc_2025.py``.
"""
import os, sys, time, pickle, logging, datetime, warnings
warnings.filterwarnings('ignore')
import numpy as np
import geopandas as gpd

sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))

import matplotlib
matplotlib.use('Agg')

import ww_dhsvm
import ww_dhsvm.sources as S
import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.terrain as T
import ww_dhsvm.streams as ST, ww_dhsvm.soils as SO, ww_dhsvm.vegetation as V
import ww_dhsvm.binary as B, ww_dhsvm.states as STA, ww_dhsvm.meteorology as M
import ww_dhsvm.config_writer as CW, ww_dhsvm.diagnostics as D
import ww_dhsvm.workflow as W, ww_dhsvm.figures as FIG, ww_dhsvm.plot as PL
import matplotlib.pyplot as plt

# ----------------------------------------------------------- parameters
NAME = os.environ.get('WWD_NAME', 'ConnecticutRiverBasin')
HUC = os.environ.get('WWD_HUC', '0108')
CELLSIZE = float(os.environ.get('WWD_CELL', '150'))
TIMESTEP_H = 3.0
START = datetime.datetime(2025, 1, 1, 0)
END = datetime.datetime(2025, 12, 31, 0)
MET_SOURCE = os.environ.get('WWD_MET', 'AORC')
CHANNEL_THRESHOLD_KM2 = float(os.environ.get('WWD_CHANNEL_KM2', '0.5'))
MAX_STATIONS = 64
SOIL_MIN_DEPTH, SOIL_MAX_DEPTH = 0.8, 3.0
EXPECTED_ANNUAL_PRECIP_MM = 1200.0     # New England climatology

CACHE = os.path.expanduser(os.environ.get('WWD_CACHE', '~/ww_dhsvm/demo/cache_ct'))
ROOT = os.path.expanduser(f'~/ww_dhsvm/demo/{NAME}')
FIGDIR = os.path.expanduser('~/ww_dhsvm/demo/figures')
os.makedirs(FIGDIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)
log = logging.getLogger()
PL.useStyle()

def savefig(fig, name):
    p = os.path.join(FIGDIR, name + '.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'  figure -> {p}')
    return p

t_all = time.time()
dirs = W.makeCaseDirectories(ROOT)

# ============================================================ STAGE 1
logging.info('=' * 78)
logging.info('STAGE 1  DOMAIN')
logging.info('=' * 78)
g = pickle.load(open(os.path.join(CACHE, 'grid.pkl'), 'rb'))
grid = G.ModelGrid(g['nrows'], g['ncols'], g['cellsize'], g['xllcorner'],
                   g['yllcorner'], C.from_wkt(g['crs']), g['mask'])
watershed = gpd.read_file(os.path.join(CACHE, 'watershed.gpkg'))
for line in grid.summary().splitlines():
    logging.info(line)

reaches = None
if os.path.exists(os.path.join(CACHE, 'nhd.gpkg')):
    reaches = gpd.read_file(os.path.join(CACHE, 'nhd.gpkg'))
    logging.info(f'  reference hydrography: {len(reaches)} NHD reaches')

case = dict(grid=grid, watershed=watershed, sources=S.getDefaultSources(),
            reaches=reaches, huc=HUC)

# ============================================================ STAGE 2
logging.info('=' * 78)
logging.info('STAGE 2  TERRAIN')
logging.info('=' * 78)
dem_raw = np.load(os.path.join(CACHE, 'dem_raw.npy')).astype('float64')
logging.info(f'  raw DEM: {np.nanmin(dem_raw):.1f} to {np.nanmax(dem_raw):.1f} m, '
             f'{int(np.isnan(dem_raw).sum()):,} nodata cells')
dem_raw = T.fillGaps(dem_raw, grid.mask)
dem_raw = np.nan_to_num(dem_raw, nan=float(np.nanmedian(dem_raw)))

terrain = T.conditionDEM(dem_raw.astype('float32'), grid,
                         streams=reaches, burn_depth=5.0)
case['terrain'] = terrain
case['dem'] = terrain['dem']

savefig(FIG.domainOverview(grid, watershed, terrain['dem'], reaches), '01_domain')
savefig(FIG.terrainPanel(grid, terrain), '02_terrain')
savefig(FIG.flowRoutingPanel(grid, terrain), '03_flow_routing')

# ============================================================ STAGE 3
logging.info('=' * 78)
logging.info('STAGE 3  CHANNEL NETWORK')
logging.info('=' * 78)
network = ST.extractNetwork(terrain, grid,
                            channel_threshold_km2=CHANNEL_THRESHOLD_KM2)
case['network'] = network
topo = ST.checkTopology(network['segments'])
comparison = None
if reaches is not None:
    try:
        comparison = ST.compareToReference(network['channel_mask'], reaches, grid)
    except Exception as exc:
        logging.warning(f'  network comparison failed: {exc}')
savefig(FIG.channelPanel(grid, network, terrain, comparison), '04_channels')

# ============================================================ STAGE 4
logging.info('=' * 78)
logging.info('STAGE 4  SOILS')
logging.info('=' * 78)
sand = np.load(os.path.join(CACHE, 'sand.npy')).astype('float64')
silt = np.load(os.path.join(CACHE, 'silt.npy')).astype('float64')
clay = np.load(os.path.join(CACHE, 'clay.npy')).astype('float64')
logging.info(f'  texture means over basin: sand {np.nanmean(sand[grid.mask!=0]):.1f}%, '
             f'silt {np.nanmean(silt[grid.mask!=0]):.1f}%, '
             f'clay {np.nanmean(clay[grid.mask!=0]):.1f}%')

soil_cls = SO.classifyFromFractions(sand, silt, clay, grid.mask)
soil_map, soil_table = SO.compactClasses(soil_cls, grid.mask)
soil_blocks = SO.buildSoilParameterBlocks(soil_table, n_layers=3)
SO.checkSoilConsistency(soil_blocks)

soil_depth = T.estimateSoilDepth(terrain['slope'], terrain['dem'], terrain['uparea'],
                                 min_depth=SOIL_MIN_DEPTH, max_depth=SOIL_MAX_DEPTH,
                                 mask=grid.mask)
soil_depth = T.enforceSoilDepthBelowChannels(soil_depth, network['cut_height_grid'])
case.update(soil_map=soil_map, soil_table=soil_table, soil_blocks=soil_blocks,
            soil_depth=soil_depth)

# ============================================================ STAGE 5
logging.info('=' * 78)
logging.info('STAGE 5  VEGETATION')
logging.info('=' * 78)
nlcd = np.load(os.path.join(CACHE, 'nlcd.npy'))
imperv = None
p_imp = os.path.join(CACHE, 'impervious.npy')
if os.path.exists(p_imp):
    imperv = np.load(p_imp).astype('float64')

cover = V.classifyLandCover(nlcd, grid.mask)
veg_map, veg_table = V.compactClasses(cover, grid.mask)
veg_blocks = V.buildVegetationBlocks(veg_table)
V.applyImperviousFraction(veg_blocks, imperv, veg_map, grid.mask)
V.checkVegetationConsistency(veg_blocks, 3)

soil_depth = T.enforceSoilDepthBelowRootZone(soil_depth, veg_map, veg_blocks,
                                             grid.mask)
case.update(veg_map=veg_map, veg_table=veg_table, veg_blocks=veg_blocks,
            soil_depth=soil_depth, canopy_gapping=False)

awc = SO.availableWaterCapacity(soil_blocks, soil_map, soil_depth, grid.mask)
savefig(FIG.soilPanel(grid, soil_map, soil_table, soil_depth, awc), '05_soils')
savefig(FIG.vegetationPanel(grid, veg_map, veg_table, veg_blocks), '06_vegetation')

# ================================================= WRITE INPUT MAPS
logging.info('=' * 78)
logging.info('WRITING DHSVM INPUT MAPS')
logging.info('=' * 78)
paths = W.writeMaps(case, dirs)

# ============================================================ STAGE 6
logging.info('=' * 78)
logging.info(f'STAGE 6  METEOROLOGICAL FORCING ({MET_SOURCE})')
logging.info('=' * 78)
mgr = S.met_sources[MET_SOURCE]
kw = {}
if MET_SOURCE == 'AORC':
    kw = dict(spatial_decimation=8, temporal_resampling='3h')
met_raw = mgr.getDataset(grid.polygon(), grid.crs,
                         start=START.strftime('%Y-%m-%d'),
                         end=END.strftime('%Y-%m-%d'), **kw)
logging.info(f'  raw met dataset: {dict(met_raw.sizes)}')
met = M.convertToDHSVM(met_raw, MET_SOURCE)

if MET_SOURCE != 'AORC':
    met = met.resample(time=f'{int(TIMESTEP_H)}h').mean()
    logging.info(f'  resampled to the {TIMESTEP_H:g} h model timestep')

M.summarize(met)
met_summary = M.annualWaterBalanceCheck(met, TIMESTEP_H)

stations = M.placeStations(grid, terrain['dem'], met, max_stations=MAX_STATIONS)
stations = M.writeAllStations(dirs['met'], met, stations, TIMESTEP_H)
case.update(met=met, stations=stations, timestep_hours=TIMESTEP_H,
            met_summary=met_summary, paths=paths)

savefig(FIG.forcingPanel(met, stations, grid, terrain['dem'], TIMESTEP_H),
        '07_forcing')

# ====================================================== STAGES 7-8
logging.info('=' * 78)
logging.info('STAGES 7-8  INITIAL STATE AND CONFIGURATION')
logging.info('=' * 78)
case = W.buildStatesAndConfig(case, dirs, NAME, START, END)

# ============================================================ STAGE 9
report = D.checkInputs(grid, terrain['dem'], soil_map, soil_depth, veg_map,
                       soil_blocks, veg_blocks, network=network,
                       met_summary=met_summary,
                       expected_annual_precip_mm=EXPECTED_ANNUAL_PRECIP_MM)
case['report'] = report
report.log()
savefig(FIG.inputDashboard(case), '08_dashboard')

with open(os.path.join(ROOT, 'case.pkl'), 'wb') as f:
    pickle.dump({k: case[k] for k in
                 ('soil_table', 'veg_table', 'soil_blocks', 'veg_blocks',
                  'timestep_hours', 'met_summary', 'paths', 'config_file')}, f)

logging.info('=' * 78)
logging.info(f'CASE BUILT in {(time.time()-t_all)/60:.1f} min -> {ROOT}')
logging.info(f'  configuration: {case["config_file"]}')
logging.info(f'  structural errors: {len(report.errors)}, warnings: {len(report.warnings)}')
logging.info('=' * 78)
