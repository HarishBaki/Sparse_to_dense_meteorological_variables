# %%
import numpy as np
import pandas as pd
import xarray as xr
import dask
import os, sys
import glob
import zarr
from joblib import Parallel, delayed
import os
import dask.array as da
import os, sys, time, glob, re

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from metpy.interpolate import interpolate_to_points

'''
var_name: the variable name inside each individual files. 
i10fg for wind gust
d2m for dew point temperature
sp for pressure
sh2 for specific humidity
t2m for temperature
wdir10 for wind direction
si10 for wind speed
'''

def int_or_none(v):
    return None if v.lower() == 'none' else int(v)

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__') or 'ipykernel' in sys.argv[0]

if is_interactive() or len(sys.argv) == 1:
    sys.argv = ["", "i10fg", "50", "w"]
var_name = sys.argv[1]
n_stations = int_or_none(sys.argv[2])
mode = sys.argv[3]

stations_seed = 42
dates = pd.date_range(start='2023-01-01T00', end='2023-12-31T23', freq='h')
yyyymmdd = pd.Series(dates.year*10000 + dates.month*100 + dates.day).unique()
data_dir = 'data' #'/data/harish/Sparse_to_dense_meteorological_variables'
source_zarr_store = f'{data_dir}/RTMA.zarr'

if n_stations is not None:
    target_zarr_store = f"{data_dir}/Barnes_interpolated/{stations_seed}/{n_stations}-random-stations/NYSM_test.zarr"
else:
    target_zarr_store = f"{data_dir}/Barnes_interpolated/{stations_seed}/all-stations/NYSM_test.zarr"
os.makedirs(target_zarr_store, exist_ok=True)

# %%
def init_zarr_store(zarr_store, dates,var_name,mode):
    orography = xr.open_dataset('orography.nc')
    orography.attrs = {}
    template = xr.full_like(orography.orog.expand_dims(time=dates),fill_value=np.nan,dtype='float32')
    template['time'] = dates
    template = template.chunk({'time': 24})
    template = template.transpose('time','y','x')
    template = template.assign_coords({
        'latitude': orography.latitude,
        'longitude': orography.longitude
    })
    template.to_dataset(name = var_name).to_zarr(zarr_store, compute=False, mode=mode)

# %%
# === Loading some topography and masking data ===
orography = xr.open_dataset('orography.nc')
RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
orography = orography.orog.values

# Load NYSM station data
nysm = pd.read_csv('nysm.csv')
# NYSM station lat/lon
nysm_latlon = np.stack([
    nysm['lat [degrees]'].values,
    (nysm['lon [degrees]'].values + 360) % 360
], axis=-1) # shape: (N, 2)

# Precompute grid KDTree
grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
tree = cKDTree(grid_points)
# Query the station locations
_, indices_flat = tree.query(nysm_latlon)
# Convert flat indices to 2D (y, x)
y_indices, x_indices = np.unravel_index(indices_flat, orography.shape)

# %%
# Loading the NYSM station data
NYSM = xr.open_dataset('data/NYSM.nc')
NYSM['longitude'] = (NYSM['longitude']+360) % 360   # this is needed to match the RTMA lon
NYSM_lat = NYSM.latitude.values
NYSM_lon = NYSM.longitude.values
# Precompute grid KDTree
station_points = np.stack([NYSM_lat.ravel(), NYSM_lon.ravel()], axis=-1)
tree = cKDTree(station_points)
# Query the station locations
_, station_indices = tree.query(nysm_latlon)
NYSM = NYSM.isel(station=station_indices)  # this is needed to match the nysm_latlon order
NYSM = NYSM.resample(time='1h').nearest()
# %%
# Check for the n_random_stations
if n_stations is not None:
    # Randomly select n_random_stations from the NYSM stations
    rng = np.random.default_rng(stations_seed) # The dataset index will always pick the same random stations for that seed, regardless of which worker, GPU, or process loads it.
    perm = rng.permutation(len(nysm_latlon))
    #random_indices = rng.choice(len(self.nysm_latlon), self.n_random_stations, replace=False)
    random_indices = perm[:n_stations]  # This will give same first n random indices for n_random_stations = 40, 50, 60, ...
    # Update the y_indices and x_indices to the random stations
    y_indices = y_indices[random_indices]
    x_indices = x_indices[random_indices]
    nysm_latlon = nysm_latlon[random_indices]
    station_indices = station_indices[random_indices]

# %%
'''
ds = NYSM[var_name].sel(time=dates)
sample = ds.isel(time=11)
print(sample.isel(station=station_indices).isnull().any().values)
local_station_indices = station_indices.copy()
local_nysm_latlon = nysm_latlon.copy()
missing_mask = sample.isel(station=local_station_indices).isnull()
missing_station_indices = np.argwhere(missing_mask.values)
local_station_indices = np.delete(local_station_indices, missing_station_indices)
local_nysm_latlon = np.delete(local_nysm_latlon, missing_station_indices, axis=0)

station_values = sample.values[local_station_indices]
interp_flat = interpolate_to_points(
    local_nysm_latlon, station_values, grid_points, interp_type='barnes', 
    #gamma=0.1, minimum_neighbors=1,kappa_star=10,
)
interp = interp_flat.reshape(RTMA_lat.shape).astype(np.float32)
interp = xr.DataArray(
    interp,
    dims=['y', 'x'],
    coords={
        'latitude': (['y', 'x'], RTMA_lat),
        'longitude': (['y', 'x'], RTMA_lon)
    }
)
interp.plot()
# examined the influence of having nans and eliminating nans, using local_station_values and local_nysm_latlon.
'''

# %%
# Get the best gamma and kappa_star for Barnes
scores_df = pd.read_csv('Barnes_parameter_search.csv')
gamma = scores_df[scores_df['idx'] == 14]['gamma'].iloc[0]
kappa_star = scores_df[scores_df['idx'] == 14]['kappa_star'].iloc[0]

# %%
chunk_size = 24
n_jobs = 60  # 2 threads per chunk, parallel across 60 chunks
ds = NYSM[var_name].sel(time=dates)

# Initialize the Zarr store
init_zarr_store(target_zarr_store, dates, var_name, mode)

# Open Zarr for writing
zarr_write = zarr.open(target_zarr_store, mode='a')
zarr_variable = zarr_write[var_name]

# Interpolation function
def interpolate_and_write_block(start_idx):
    success = 0
    end_idx = min(start_idx + chunk_size, len(ds.time))
    for t_idx in range(start_idx, end_idx):
        try:
            sample = ds.isel(time=t_idx)
            # Already the station_indices and nysm_latlon are randomly selected. 
            # But, we need to go through another layer of filtering missing stations. 
            local_station_indices = station_indices.copy()
            local_nysm_latlon = nysm_latlon.copy()
            missing_mask = sample.isel(station=local_station_indices).isnull()
            missing_station_indices = np.argwhere(missing_mask.values)
            local_station_indices = np.delete(local_station_indices, missing_station_indices)
            local_nysm_latlon = np.delete(local_nysm_latlon, missing_station_indices, axis=0)

            station_values = sample.values[local_station_indices]
            interp_flat = interpolate_to_points(
                local_nysm_latlon, station_values, grid_points, interp_type='barnes',
                gamma=gamma, kappa_star=kappa_star
            )
            interp = interp_flat.reshape(RTMA_lat.shape).astype(np.float32)
            zarr_variable[t_idx, :, :] = interp
            success += 1
        except Exception as e:
            print(f"Time index {t_idx} failed: {e}")
    return success

# Parallel execution
from tqdm import tqdm
chunk_starts = list(range(0, len(ds.time), chunk_size))
results = Parallel(n_jobs=n_jobs)(
    delayed(interpolate_and_write_block)(i) for i in tqdm(chunk_starts)
)

# Optional: print summary
total_success = sum(results)
print(f"Interpolation completed: {total_success}/{len(ds.time)} time steps written.")
