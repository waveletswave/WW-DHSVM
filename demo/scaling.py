"""Strong and weak scaling benchmarks for parallel DHSVM.

Two experiments, both on the full Connecticut River Basin:

**Strong scaling** -- one fixed problem (the 150 m grid, 1.30 M active
cells), solved on an increasing number of MPI ranks.  Reports speedup
``S(p) = T(1)/T(p)`` against the ideal ``S = p``, and parallel efficiency
``E(p) = S(p)/p``.

**Weak scaling** -- the work *per rank* held constant, so the problem
grows with the rank count.  DHSVM's cost is dominated by per-cell work,
and cell count scales as ``1/cellsize^2``, so the grid spacing is set to
``cellsize(p) = cellsize_base / sqrt(p)``.  Each configuration is a
complete, independently built DHSVM case at its own resolution -- not a
resampling of one case -- so the comparison is of real model setups.
Reports ``E_w(p) = T(1)/T(p)``, ideally 1.

Separating initialization from time stepping
--------------------------------------------
DHSVM's own ``Runtime Summary`` uses ``clock()``, which is CPU time and
counts MPI busy-waiting, so it is useless for scaling.  Wall time is
measured externally instead -- but a raw wall time mixes a largely serial
initialization (reading maps, building the channel network, computing
interpolation weights) with the parallel time-stepping loop.

Each configuration is therefore run **twice**, at a short and a long
simulation length, and the two are differenced::

    T_step = (T_long - T_short) / (N_long - N_short)
    T_init =  T_short - N_short * T_step

That isolates the per-timestep cost that actually parallelizes, while
still reporting the total wall time a user experiences.  The
initialization term is itself a scaling result: it is the Amdahl serial
fraction that bounds achievable speedup.
"""
import os, sys, json, time, shutil, logging, datetime, subprocess
import numpy as np

sys.path.insert(0, os.path.expanduser('~/ww_dhsvm'))
os.environ.setdefault('WW_DHSVM_DATA_DIR', os.path.expanduser('~/ww_dhsvm/data'))

DHSVM = os.path.expanduser('~/ww_dhsvm/build/dhsvm-parallel/DHSVM/sourcecode/DHSVM')
ROOT = os.path.expanduser('~/ww_dhsvm/demo/scaling')
RESULTS = os.path.join(ROOT, 'results.json')

#: MPI rank counts to sweep.  16 physical cores are available; 90% of
#: them caps the sweep at 14.
RANKS = [1, 2, 4, 6, 8, 10, 12, 14]

#: Simulation lengths, in timesteps, used to separate init from stepping.
N_SHORT, N_LONG = 8, 240        # 1 day and 30 days at a 3 h timestep

START = datetime.datetime(2025, 4, 1, 0)
TIMESTEP_H = 3.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s',
                    datefmt='%H:%M:%S', force=True)


def runCase(case_dir, config_name, nsteps, nproc, tag):
    """Run DHSVM for ``nsteps`` timesteps on ``nproc`` ranks; return wall seconds."""
    src = os.path.join(case_dir, config_name)
    end = START + datetime.timedelta(hours=TIMESTEP_H * nsteps)
    outdir = os.path.join(case_dir, f'out_{tag}')
    os.makedirs(outdir, exist_ok=True)

    cfg = os.path.join(case_dir, f'INPUT.{tag}')
    with open(src) as f:
        text = f.read()
    out = []
    for line in text.splitlines():
        k = line.split('=')[0].strip().lower()
        if k == 'model end':
            line = f'Model End   = {end:%m/%d/%Y-%H}'
        elif k == 'output directory':
            line = f'Output Directory = {outdir}{os.sep}'
        out.append(line)
    with open(cfg, 'w') as f:
        f.write('\n'.join(out) + '\n')

    cmd = ['mpirun', '-np', str(nproc), DHSVM, cfg]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        logging.error(f'    FAILED (rc={p.returncode}): '
                      f'{p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""}')
        return None
    shutil.rmtree(outdir, ignore_errors=True)
    return wall


def measure(case_dir, config_name, nproc, label):
    """Time one configuration at two lengths and split init from stepping."""
    t_short = runCase(case_dir, config_name, N_SHORT, nproc, f'{label}_p{nproc}_s')
    if t_short is None:
        return None
    t_long = runCase(case_dir, config_name, N_LONG, nproc, f'{label}_p{nproc}_l')
    if t_long is None:
        return None
    per_step = (t_long - t_short) / (N_LONG - N_SHORT)
    init = t_short - N_SHORT * per_step
    logging.info(f'    p={nproc:<3d} total {t_long:7.1f} s   init {init:6.1f} s   '
                 f'step {per_step:6.3f} s/step   ({N_LONG} steps)')
    return dict(nproc=nproc, t_short=t_short, t_long=t_long,
                per_step=per_step, init=max(init, 0.0), n_short=N_SHORT,
                n_long=N_LONG)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def buildCase(cellsize, out_root, cache, label):
    """Build a complete DHSVM case at one grid resolution.

    Every weak-scaling point is a genuine, independently parameterized
    model -- its own grid, conditioned DEM, channel network, soil and
    vegetation classes, forcing stations and initial state -- rather than
    a resampled copy of one case.  That is what makes the comparison a
    scaling test of DHSVM rather than of interpolation.
    """
    import pickle, warnings
    warnings.filterwarnings('ignore')
    import geopandas as gpd
    import ww_dhsvm.grid as G, ww_dhsvm.crs as C, ww_dhsvm.terrain as T
    import ww_dhsvm.streams as ST, ww_dhsvm.soils as SO, ww_dhsvm.vegetation as V
    import ww_dhsvm.meteorology as M, ww_dhsvm.sources as S
    import ww_dhsvm.states as STA, ww_dhsvm.config_writer as CW
    import ww_dhsvm.workflow as W

    case_dir = os.path.join(out_root, label)
    if os.path.exists(os.path.join(case_dir, f'INPUT.{label}')):
        logging.info(f'  {label}: reusing existing case')
        return case_dir, f'INPUT.{label}'

    dirs = W.makeCaseDirectories(case_dir)
    ws = gpd.read_file(os.path.join(cache, 'watershed.gpkg'))
    reaches = gpd.read_file(os.path.join(cache, 'nhd.gpkg'))

    grid = G.ModelGrid.fromShape(ws, cellsize=cellsize, buffer_cells=3)
    grid.setMaskFromShape(ws)

    # Resample the cached native-resolution rasters onto this grid.
    import rioxarray, xarray as xr
    base = pickle.load(open(os.path.join(cache, 'grid.pkl'), 'rb'))
    bg = G.ModelGrid(base['nrows'], base['ncols'], base['cellsize'],
                     base['xllcorner'], base['yllcorner'],
                     C.from_wkt(base['crs']), base['mask'])

    def regrid(name, resampling):
        a = np.load(os.path.join(cache, name + '.npy')).astype('float64')
        da = bg.toDataArray(a, name)
        return T.resampleToGrid(da, grid, resampling)

    dem = T.fillGaps(regrid('dem_raw', 'bilinear'), grid.mask)
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem)))
    terrain = T.conditionDEM(dem.astype('float32'), grid, streams=reaches,
                             burn_depth=5.0)

    # Channel-initiation threshold scales with cell area so the network
    # stays resolvable: a 0.5 km^2 threshold on a 600 m grid is under two
    # cells, which cannot form a network.
    thresh = max(0.5, 20.0 * (cellsize / 1000.0) ** 2)
    network = ST.extractNetwork(terrain, grid, channel_threshold_km2=thresh,
                                min_slope=1e-4)

    sand, silt, clay = (regrid(n, 'bilinear') for n in ('sand', 'silt', 'clay'))
    soil_map, soil_table = SO.compactClasses(
        SO.classifyFromFractions(sand, silt, clay, grid.mask), grid.mask)
    soil_blocks = SO.buildSoilParameterBlocks(soil_table, n_layers=3)
    depth = T.estimateSoilDepth(terrain['slope'], terrain['dem'], terrain['uparea'],
                                min_depth=0.8, max_depth=3.0, mask=grid.mask)
    depth = T.enforceSoilDepthBelowChannels(depth, network['cut_height_grid'])

    nlcd = np.rint(regrid('nlcd', 'nearest')).astype('uint8')
    veg_map, veg_table = V.compactClasses(
        V.classifyLandCover(nlcd, grid.mask), grid.mask)
    veg_blocks = V.buildVegetationBlocks(veg_table)
    V.applyImperviousFraction(veg_blocks, regrid('impervious', 'bilinear'),
                              veg_map, grid.mask)
    depth = T.enforceSoilDepthBelowRootZone(depth, veg_map, veg_blocks, grid.mask)

    case = dict(grid=grid, watershed=ws, sources=S.getDefaultSources(),
                reaches=reaches, terrain=terrain, dem=terrain['dem'],
                network=network, soil_map=soil_map, soil_table=soil_table,
                soil_blocks=soil_blocks, soil_depth=depth, veg_map=veg_map,
                veg_table=veg_table, veg_blocks=veg_blocks, canopy_gapping=False)
    paths = W.writeMaps(case, dirs)

    met_raw = S.met_sources['AORC'].getDataset(
        grid.polygon(), grid.crs, start='2025-01-01', end='2025-12-31',
        spatial_decimation=8, temporal_resampling='3h')
    met = M.convertToDHSVM(met_raw, 'AORC')
    stations = M.placeStations(grid, terrain['dem'], met, max_stations=64)
    stations = M.writeAllStations(dirs['met'], met, stations, TIMESTEP_H)

    case.update(paths=paths, stations=stations, timestep_hours=TIMESTEP_H)
    W.buildStatesAndConfig(case, dirs, label, START,
                           START + datetime.timedelta(days=31))

    logging.info(f'  {label}: {grid.nrows}x{grid.ncols}, '
                 f'{grid.n_active:,} active cells, '
                 f'{len(network["segments"]):,} segments, '
                 f'{len(stations)} stations')
    return case_dir, f'INPUT.{label}'


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def strongScaling(case_dir, config_name, ranks=RANKS):
    """One fixed problem, increasing rank count."""
    logging.info('=' * 78)
    logging.info('STRONG SCALING -- fixed 150 m grid, increasing ranks')
    logging.info('=' * 78)
    out = []
    for p in ranks:
        r = measure(case_dir, config_name, p, 'strong')
        if r:
            out.append(r)
    return out


def weakScaling(cache, out_root, ranks=RANKS, cellsize_base=600.0):
    """Work per rank held constant by scaling resolution as 1/sqrt(p)."""
    logging.info('=' * 78)
    logging.info('WEAK SCALING -- cells per rank held ~constant')
    logging.info(f'  cellsize(p) = {cellsize_base:g} / sqrt(p)')
    logging.info('=' * 78)
    out = []
    for p in ranks:
        cs = cellsize_base / np.sqrt(p)
        label = f'weak_p{p}'
        try:
            case_dir, cfg = buildCase(cs, out_root, cache, label)
        except Exception as exc:
            logging.error(f'  {label}: build failed: {type(exc).__name__}: {exc}')
            continue
        import pickle
        r = measure(case_dir, cfg, p, 'weak')
        if r:
            r['cellsize'] = cs
            # Record the realized problem size for the report.
            try:
                with open(os.path.join(case_dir, f'INPUT.{label}')) as f:
                    txt = f.read()
                rows = int([l for l in txt.splitlines()
                            if l.lower().startswith('number of rows')][0].split('=')[1])
                cols = int([l for l in txt.splitlines()
                            if l.lower().startswith('number of columns')][0].split('=')[1])
                r['nrows'], r['ncols'] = rows, cols
            except Exception:
                pass
            out.append(r)
    return out


def main():
    cache = os.path.expanduser('~/ww_dhsvm/demo/cache_ct')
    os.makedirs(ROOT, exist_ok=True)

    results = {}
    if os.path.exists(RESULTS):
        results = json.load(open(RESULTS))

    if 'strong' not in results:
        case_dir, cfg = buildCase(150.0, ROOT, cache, 'strong_150m')
        results['strong'] = strongScaling(case_dir, cfg)
        json.dump(results, open(RESULTS, 'w'), indent=1)

    if 'weak' not in results:
        results['weak'] = weakScaling(cache, ROOT)
        json.dump(results, open(RESULTS, 'w'), indent=1)

    logging.info(f'results -> {RESULTS}')
    return results


if __name__ == '__main__':
    main()
