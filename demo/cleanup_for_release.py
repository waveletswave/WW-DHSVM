"""Prepare the working directory for publication.

Removes third-party source and build trees that must not be redistributed,
every artefact of the superseded Lower Connecticut runs, and all transient
caches -- while keeping the downloaded data the Connecticut River Basin
demo needs in order to run without re-downloading anything.
"""
import os, shutil, sys, glob

ROOT = os.path.expanduser('~/ww_dhsvm')
DRY = '--dry-run' in sys.argv

# Third-party code we build against but must not host.
THIRD_PARTY = ['upstream-ww', 'external', 'build']

# Everything belonging to the superseded Lower Connecticut domain.
SUPERSEDED = ['demo/LowerConnecticut', 'demo/cache', 'tests/synthetic_case',
              'demo/scaling/strong_150m'] + [f'demo/scaling/weak_p{p}'
                                             for p in (1, 2, 4, 6, 8, 10, 12, 14)]

# Transient caches: regenerated on demand, never needed by a user.
TRANSIENT = ['cache', 'data/herbie', '.pytest_cache', 'notebooks/data']

def rm(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return 0
    size = sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(p) for f in fs
               if os.path.exists(os.path.join(d, f))) if os.path.isdir(p) \
        else os.path.getsize(p)
    print(f'  {"would remove" if DRY else "removing"}  {rel:<40s} {size/1e6:9.1f} MB')
    if not DRY:
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    return size

total = 0
print('Third-party source and builds (documented in README, not hosted):')
for r in THIRD_PARTY:
    total += rm(r)
print('\nSuperseded Lower Connecticut domain:')
for r in SUPERSEDED:
    total += rm(r)
print('\nTransient caches and scaling case directories:')
for r in TRANSIENT:
    total += rm(r)

print('\nGenerated Python/Jupyter caches:')
for pat in ('**/__pycache__', '**/.ipynb_checkpoints', '**/*.pyc'):
    for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
        rel = os.path.relpath(p, ROOT)
        total += rm(rel)

print('\nLog files:')
for p in sorted(glob.glob(os.path.join(ROOT, 'demo', '*.log'))):
    total += rm(os.path.relpath(p, ROOT))

# Meteorology cached for the superseded domain only.  The Connecticut
# basin's own bounding box is wider, so its files are kept.
print('\nMeteorology cached for the superseded (smaller) bounding box:')
for p in glob.glob(os.path.join(ROOT, 'data/meteorology/*/*.nc')):
    # Keep only the Connecticut basin's own (wider) bounding box.
    KEEP = ('_-73.2583_41.2000_-71.0250_45.3833',   # AORC, the grid in use
            '_-73.3200_41.1300_-70.9500_45.4500')     # HRRR, the grid in use
    if not any(k in p for k in KEEP):
        total += rm(os.path.relpath(p, ROOT))
for p in glob.glob(os.path.join(ROOT, 'data/meteorology/*/_parts')):
    total += rm(os.path.relpath(p, ROOT))

# POLARIS keeps a full-resolution netCDF per variable per bounding box.
# The Connecticut ones are 2.8 GB each and are only needed to regenerate
# demo/cache_ct/*.npy, which are already cached.
print('\nPOLARIS full-resolution intermediates (the gridded .npy are kept):')
for p in glob.glob(os.path.join(ROOT, 'data/soil_structure/POLARIS/*.nc')):
    total += rm(os.path.relpath(p, ROOT))

print(f'\n{"would free" if DRY else "freed"}: {total/1e9:.2f} GB')
