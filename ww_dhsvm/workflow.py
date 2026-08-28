"""High-level orchestration: HUC in, DHSVM case out.

The notebooks in ``notebooks/`` walk through the setup step by step,
because that is how a modeller learns what the choices are and where
they can go wrong.  This module is the scripted counterpart: the same
sequence, wrapped so that a whole basin can be built in one call from a
command line or a batch job.

The pipeline, in order:

1. **Domain** -- resolve a HUC code or shapefile to a boundary polygon,
   choose a UTM zone, lay out the regular grid, rasterize the mask.
2. **Terrain** -- fetch and resample the DEM, burn the mapped stream
   network, fill depressions, derive D8 flow direction and accumulation,
   slope and aspect.
3. **Channels** -- delineate the network above a support-area threshold,
   split it into segments, classify them, and write DHSVM's three stream
   files.
4. **Soils** -- fetch texture, classify into USDA classes, build the
   parameter blocks, estimate soil depth from terrain, and reconcile it
   with both channel cut depth and rooting depth.
5. **Vegetation** -- fetch land cover, crosswalk NLCD to DHSVM types,
   build the parameter blocks.
6. **Forcing** -- fetch meteorology, convert to DHSVM's seven variables,
   place stations, write one file per station.
7. **State** -- write the four initial-state files.
8. **Configuration** -- assemble and validate the DHSVM config file.
9. **Diagnostics** -- run the QA report over everything.

Each stage returns its products in a single ``case`` dictionary, so a
caller can stop after any stage, inspect, adjust and resume.
"""

from typing import Optional, Dict, Any, List
import os
import logging
import datetime

import numpy as np

import ww_dhsvm
import ww_dhsvm.sources
import ww_dhsvm.grid as _grid
import ww_dhsvm.terrain as _terrain
import ww_dhsvm.streams as _streams
import ww_dhsvm.soils as _soils
import ww_dhsvm.vegetation as _veg
import ww_dhsvm.binary as _binary
import ww_dhsvm.states as _states
import ww_dhsvm.meteorology as _met
import ww_dhsvm.config_writer as _cfg
import ww_dhsvm.diagnostics as _diag


def makeCaseDirectories(root: str) -> Dict[str, str]:
    """Create the standard directory layout for a DHSVM case.

    DHSVM does not create any of these, and fails at initialization if
    the output directory is missing, so they are made up front::

        <root>/input/    binary maps and stream files
        <root>/met/      one forcing file per station
        <root>/state/    initial model state
        <root>/output/   DHSVM results
    """
    dirs = {k: os.path.join(root, k)
            for k in ('input', 'met', 'state', 'output')}
    dirs['root'] = root
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def buildDomain(huc: Optional[str] = None,
                shapefile: Optional[str] = None,
                cellsize: float = 150.0,
                sources: Optional[Dict[str, Any]] = None,
                crs=None,
                buffer_cells: int = 3) -> Dict[str, Any]:
    """Stage 1 -- resolve the watershed and lay out the model grid."""
    if sources is None:
        sources = ww_dhsvm.sources.getDefaultSources()

    logging.info('=' * 70)
    logging.info('STAGE 1: DOMAIN')
    logging.info('=' * 70)

    if huc is not None:
        watershed = ww_dhsvm.getWatershed(huc, sources)
    elif shapefile is not None:
        watershed = ww_dhsvm.getWatershedFromShapefile(shapefile)
    else:
        raise ValueError('buildDomain needs either a huc or a shapefile.')

    grid = _grid.ModelGrid.fromShape(watershed, cellsize, crs=crs,
                                     buffer_cells=buffer_cells)
    grid.setMaskFromShape(watershed)
    logging.info(grid.summary())
    return dict(watershed=watershed, grid=grid, sources=sources, huc=huc)


def buildTerrain(case: Dict[str, Any],
                 dem: Optional[np.ndarray] = None,
                 burn_streams: bool = True,
                 burn_depth: float = 5.0) -> Dict[str, Any]:
    """Stage 2 -- DEM acquisition, conditioning and flow routing."""
    logging.info('=' * 70)
    logging.info('STAGE 2: TERRAIN')
    logging.info('=' * 70)

    grid, sources = case['grid'], case['sources']

    if dem is None:
        ds = sources['DEM'].getDataset(grid.polygon().buffer(4 * grid.cellsize),
                                       grid.crs)
        var = list(ds.data_vars)[0]
        dem = _terrain.resampleToGrid(ds[var], grid, 'bilinear')

    dem = _terrain.fillGaps(dem, grid.mask)

    reaches = case.get('reaches')
    if burn_streams and reaches is None:
        try:
            reaches = sources['hydrography'].getShapesByGeometry(
                case['watershed'].union_all(), case['watershed'].crs,
                out_crs=grid.crs)
            case['reaches'] = reaches
        except Exception as exc:
            logging.warning(f'  could not fetch reference hydrography: {exc}')
            reaches = None

    terrain = _terrain.conditionDEM(dem, grid,
                                    streams=reaches if burn_streams else None,
                                    burn_depth=burn_depth)
    case['terrain'] = terrain
    case['dem'] = terrain['dem']
    return case


def buildChannels(case: Dict[str, Any],
                  channel_threshold_km2: float = 1.0,
                  **kwargs) -> Dict[str, Any]:
    """Stage 3 -- channel delineation and the DHSVM stream files."""
    logging.info('=' * 70)
    logging.info('STAGE 3: CHANNEL NETWORK')
    logging.info('=' * 70)

    network = _streams.extractNetwork(case['terrain'], case['grid'],
                                      channel_threshold_km2=channel_threshold_km2,
                                      **kwargs)
    case['network'] = network
    case['topology'] = _streams.checkTopology(network['segments'])
    if case.get('reaches') is not None:
        try:
            case['network_comparison'] = _streams.compareToReference(
                network['channel_mask'], case['reaches'], case['grid'])
        except Exception as exc:
            logging.warning(f'  network comparison failed: {exc}')
    return case


def buildSoils(case: Dict[str, Any],
               sand: Optional[np.ndarray] = None,
               silt: Optional[np.ndarray] = None,
               clay: Optional[np.ndarray] = None,
               observed_depth: Optional[np.ndarray] = None,
               min_depth: float = 0.6,
               max_depth: float = 3.0,
               weight_observed: float = 0.0,
               n_layers: int = 3) -> Dict[str, Any]:
    """Stage 4 -- soil texture, parameters and depth."""
    logging.info('=' * 70)
    logging.info('STAGE 4: SOILS')
    logging.info('=' * 70)

    grid, terrain = case['grid'], case['terrain']

    if sand is None:
        raise ValueError('buildSoils needs sand/silt/clay arrays on the model grid; '
                         'fetch them from sources["soil"] first.')

    soil_cls = _soils.classifyFromFractions(sand, silt, clay, grid.mask)
    soil_map, soil_table = _soils.compactClasses(soil_cls, grid.mask)
    soil_blocks = _soils.buildSoilParameterBlocks(soil_table, n_layers=n_layers)
    _soils.checkSoilConsistency(soil_blocks)

    depth = _terrain.estimateSoilDepth(terrain['slope'], terrain['dem'],
                                       terrain['uparea'], min_depth=min_depth,
                                       max_depth=max_depth, mask=grid.mask)
    if observed_depth is not None and weight_observed > 0:
        depth = _terrain.blendSoilDepth(depth, observed_depth, weight_observed,
                                        min_depth, max_depth)
    if 'network' in case:
        depth = _terrain.enforceSoilDepthBelowChannels(
            depth, case['network']['cut_height_grid'])

    case.update(soil_map=soil_map, soil_table=soil_table,
                soil_blocks=soil_blocks, soil_depth=depth)
    return case


def buildVegetation(case: Dict[str, Any],
                    nlcd: np.ndarray,
                    impervious: Optional[np.ndarray] = None,
                    canopy_gap_fraction: float = 0.0) -> Dict[str, Any]:
    """Stage 5 -- land cover crosswalk and vegetation parameters."""
    logging.info('=' * 70)
    logging.info('STAGE 5: VEGETATION')
    logging.info('=' * 70)

    grid = case['grid']
    cover = _veg.classifyLandCover(nlcd, grid.mask)
    veg_map, veg_table = _veg.compactClasses(cover, grid.mask)
    veg_blocks = _veg.buildVegetationBlocks(veg_table)
    _veg.applyImperviousFraction(veg_blocks, impervious, veg_map, grid.mask)
    _veg.checkVegetationConsistency(veg_blocks,
                                    case['soil_blocks'][0]['n_layers']
                                    if case.get('soil_blocks') else 3)

    # Reconcile soil depth with rooting depth now that vegetation is known.
    if 'soil_depth' in case:
        case['soil_depth'] = _terrain.enforceSoilDepthBelowRootZone(
            case['soil_depth'], veg_map, veg_blocks, grid.mask)

    gap = _veg.canopyGapMap(veg_map, veg_blocks, canopy_gap_fraction)
    case.update(veg_map=veg_map, veg_table=veg_table, veg_blocks=veg_blocks,
                canopy_gap=gap, canopy_gapping=canopy_gap_fraction > 0)
    return case


def writeMaps(case: Dict[str, Any], dirs: Dict[str, str],
              also_ascii: bool = False) -> Dict[str, str]:
    """Write every binary map and stream file DHSVM will read."""
    logging.info('=' * 70)
    logging.info('WRITING DHSVM INPUT MAPS')
    logging.info('=' * 70)

    grid = case['grid']
    ip = dirs['input']
    paths = {
        'dem': _binary.writeMap(os.path.join(ip, 'dem.bin'), case['dem'], 'dem'),
        'mask': _binary.writeMap(os.path.join(ip, 'mask.bin'), grid.mask, 'mask'),
        'soil': _binary.writeMap(os.path.join(ip, 'soil.bin'), case['soil_map'], 'soil'),
        'soil_depth': _binary.writeMap(os.path.join(ip, 'soild.bin'),
                                       case['soil_depth'], 'soil_depth'),
        'veg': _binary.writeMap(os.path.join(ip, 'veg.bin'), case['veg_map'], 'veg'),
    }
    if case.get('canopy_gapping'):
        paths['canopy_gap'] = _binary.writeMap(os.path.join(ip, 'CanopyGap.bin'),
                                               case['canopy_gap'], 'canopy_gap')

    sf = _streams.writeAll(ip, case['network'])
    paths.update(stream_map=sf['map'], stream_network=sf['network'],
                 stream_class=sf['class'])

    if any(b['impervious_fraction'] > 0 for b in case['veg_blocks']):
        paths['impervious_routing'] = _streams.writeImperviousRoutingFile(
            os.path.join(ip, 'impervious.routing.txt'),
            case['network']['channel_mask'], grid)

    if also_ascii:
        for key, arr in (('dem', case['dem']), ('mask', grid.mask),
                         ('soil', case['soil_map']),
                         ('soild', case['soil_depth']), ('veg', case['veg_map'])):
            _binary.writeASCIIGrid(os.path.join(ip, f'{key}.asc'), arr,
                                   grid.xllcorner, grid.yllcorner, grid.cellsize)

    case['paths'] = paths
    return paths


def buildForcing(case: Dict[str, Any], dirs: Dict[str, str],
                 start: datetime.datetime, end: datetime.datetime,
                 timestep_hours: float = 3.0,
                 met_source: str = 'HRRR',
                 max_stations: int = 60,
                 met_dataset=None) -> Dict[str, Any]:
    """Stage 6 -- meteorological forcing and station files."""
    logging.info('=' * 70)
    logging.info('STAGE 6: METEOROLOGICAL FORCING')
    logging.info('=' * 70)

    grid = case['grid']
    if met_dataset is None:
        mgr = ww_dhsvm.sources.met_sources[met_source]
        met_dataset = mgr.getDataset(grid.polygon(), grid.crs,
                                     start=start.strftime('%Y-%m-%d'),
                                     end=end.strftime('%Y-%m-%d'))
    met = _met.convertToDHSVM(met_dataset, met_source)

    if timestep_hours != 1.0:
        logging.info(f'  resampling forcing to the {timestep_hours:g} h model timestep')
        met = met.resample(time=f'{int(timestep_hours)}h').mean()

    stations = _met.placeStations(grid, case['dem'], met,
                                  max_stations=max_stations)
    stations = _met.writeAllStations(dirs['met'], met, stations, timestep_hours)

    case.update(met=met, stations=stations, timestep_hours=timestep_hours,
                met_summary=_met.annualWaterBalanceCheck(met, timestep_hours))
    return case


def buildStatesAndConfig(case: Dict[str, Any], dirs: Dict[str, str],
                         name: str,
                         start: datetime.datetime, end: datetime.datetime,
                         **cfg_kwargs) -> Dict[str, Any]:
    """Stages 7-8 -- initial states and the DHSVM configuration file."""
    logging.info('=' * 70)
    logging.info('STAGES 7-8: INITIAL STATE AND CONFIGURATION')
    logging.info('=' * 70)

    grid = case['grid']
    state_files = _states.writeInitialStates(
        dirs['state'], start, grid, case['soil_map'], case['soil_blocks'],
        case['network']['segments']['ID'].tolist(),
        n_veg_layers=_veg.maxVegLayers(case['veg_blocks']),
        n_soil_layers=case['soil_blocks'][0]['n_layers'])
    _states.verifyStateFiles(state_files, grid,
                             n_veg_layers=_veg.maxVegLayers(case['veg_blocks']),
                             n_soil_layers=case['soil_blocks'][0]['n_layers'])

    cfg = _cfg.buildConfig(
        name, grid, start, end, case['timestep_hours'], case['paths'],
        case['soil_blocks'], case['veg_blocks'], case['stations'],
        output_dir=dirs['output'], state_dir=dirs['state'],
        n_soil_layers=case['soil_blocks'][0]['n_layers'],
        canopy_gapping=case.get('canopy_gapping', False), **cfg_kwargs)

    validation = _cfg.validateConfig(cfg, grid, case['soil_blocks'],
                                     case['veg_blocks'], case['soil_map'],
                                     case['veg_map'], case['timestep_hours'])
    cfgfile = cfg.write(os.path.join(dirs['root'], f'INPUT.{name}'))

    case.update(state_files=state_files, config=cfg, config_file=cfgfile,
                config_validation=validation)
    return case


def runDiagnostics(case: Dict[str, Any],
                   expected_annual_precip_mm: Optional[float] = None) -> Any:
    """Stage 9 -- the full input-QA report."""
    report = _diag.checkInputs(
        case['grid'], case['dem'], case['soil_map'], case['soil_depth'],
        case['veg_map'], case['soil_blocks'], case['veg_blocks'],
        network=case.get('network'), met_summary=case.get('met_summary'),
        expected_annual_precip_mm=expected_annual_precip_mm)
    case['report'] = report
    return report
