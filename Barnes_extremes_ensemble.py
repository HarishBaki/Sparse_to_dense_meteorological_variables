# %%
import os

# Prevent OpenBLAS and NumExpr from using too many threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# %%
import time
print("Current time is:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# %%
import torch 
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR

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
from scipy.spatial import cKDTree

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import DateFormatter

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
import gc

from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import joblib
import pathlib

from metpy.interpolate import interpolate_to_points

from scipy.ndimage import gaussian_filter1d

from data_loader import RTMA_sparse_to_dense_Dataset, Transform
from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedPSNR, MaskedSSIM

import os
import wandb, argparse, sys
from tqdm import tqdm

from data_loader import RTMA_sparse_to_dense_Dataset, Transform, NYSM_sparse_to_dense_Dataset
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet
from models.util import initialize_weights_xavier,initialize_weights_he

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss

from sampler import DistributedEvalSampler

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store

FIG_DIR = '/home/harish/Dropbox/Apps/Overleaf/AIES_DL_based_Spatial_reconstruction_of_meteorological_variables_measured_by_sparse_stations'
if not os.path.exists(FIG_DIR):
    FIG_DIR = 'Figures'
# %%
GRC_FIG_DIR = '/home/harish/OneDrive_me16d412/Presentations/GRC'

# %%
# === Loading some topography and masking data ===
orography = xr.open_dataset('orography.nc')
RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
orography = orography.set_coords(['latitude', 'longitude'])
orography = orography.orog

mask = xr.open_dataset('mask_2d.nc').mask
mask_tensor = torch.tensor(mask.values.astype(np.float32))

# Initiate the metrics
metrics = {
    'masked_rmse': MaskedRMSELoss(mask_tensor), 
    'masked_psnr': MaskedPSNR(mask_tensor),
    'masked_ssim': MaskedSSIM(mask_tensor),
}

# Load NYSM station data
nysm = pd.read_csv('nysm.csv')
# NYSM station lat/lon
nysm_latlon = np.stack([
    nysm['lat [degrees]'].values,
    (nysm['lon [degrees]'].values + 360) % 360
], axis=-1) # shape: (N, 2)

exclude_indices = [65, 102] # Exclude these indices, since they are falling outside the NYS mask region. 

# Precompute grid KDTree
grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
tree = cKDTree(grid_points)
# Query the station locations
_, indices_flat = tree.query(nysm_latlon)
# Convert flat indices to 2D (y, x)
y_indices, x_indices = np.unravel_index(indices_flat, RTMA_lat.shape)

station_mask = np.zeros_like(RTMA_lat, dtype=np.uint8)
station_mask[y_indices, x_indices] = 1  # Set 1 at the station locations

ref_zarr_store = 'data/RTMA.zarr'
test_dates_range = ['2023-01-01T00','2023-12-31T23']

# Load RTMA and NYSM once
RTMA_zarr_store = 'data/RTMA.zarr'
RTMA = xr.open_zarr(RTMA_zarr_store)
NYSM = xr.open_dataset('data/NYSM.nc')
NYSM['longitude'] = (NYSM['longitude'] + 360) % 360
NYSM_lat = NYSM.latitude.values
NYSM_lon = NYSM.longitude.values
station_points = np.stack([NYSM_lat.ravel(), NYSM_lon.ravel()], axis=-1)
tree = cKDTree(station_points)
_, station_indices = tree.query(nysm_latlon)
NYSM = NYSM.isel(station=station_indices)

missing_times = xr.open_dataset(f'nan_times_i10fg.nc').time
freq = 60  # Frequency in minutes, can be changed if needed
# if the additional input variables is not none, add the missing times of the additional input variables also. 
for var in ['si10','t2m','sh2']:
    missing_times = xr.concat([missing_times, xr.open_dataset(f'nan_times_{var}.nc').time], dim='time')
# remove duplicates
missing_times = missing_times.drop_duplicates('time')

# %%
NYSM_resampled = NYSM['i10fg'].sel(time=slice(*test_dates_range))
NYSM_resampled.rolling(time=12, min_periods = 1, center=True).max()
NYSM_resampled = NYSM_resampled.sel(time=NYSM_resampled.time.dt.minute==0)

condition = NYSM_resampled >= 17.5
# Count how many stations meet the condition at each time
count = condition.sum(dim='station')

# Threshold for majority
majority = 6  # or len(local_station_indices) // 2 + 1

# Get boolean mask for times where condition is met by majority
majority_time_mask = count >= majority

# Extract the times
extreme_times = NYSM_resampled.time.where(majority_time_mask, drop=True)
print(f"{len(extreme_times)} Times with {majority} stations exceedance:\n{extreme_times.values}")

extreme_days = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in np.unique(extreme_times.dt.date)]
print("Extreme days (YYYY-MM-DD):",extreme_days)

# Now: extract which stations satisfy the condition during those times

extreme_time_station_lists = []

for extreme_time in extreme_times:
    stations_true = condition.sel(time=extreme_time)
    station_ids_local = station_indices[stations_true.values]
    # Excluding the indices that fall outside the NYS mask region
    station_ids_local = station_ids_local[~np.isin(station_ids_local, exclude_indices)]
    # Store station indices (as a list or array) per time
    extreme_time_station_lists.append(station_ids_local)

print(f"Extreme stations lists: {extreme_time_station_lists}")


extreme_station_lists = []

for extreme_day in extreme_days:
    stations_true = condition.sel(time=extreme_day).any(dim='time')
    station_ids_local = station_indices[stations_true.values]
    # Excluding the indices that fall outside the NYS mask region
    station_ids_local = station_ids_local[~np.isin(station_ids_local, exclude_indices)]
    # Store station indices (as a list or array) per time
    extreme_station_lists.append(station_ids_local)

print(f"Extreme stations lists: {extreme_station_lists}")
# %%
WRF_start_dates=['2023-02-02T23:30:00', '2023-03-25T11:30:00', '2023-04-01T11:30:00', '2023-12-17T17:30:00']
WRF_end_dates=['2023-02-04T00:30:00', '2023-03-26T12:30:00', '2023-04-02T12:30:00', '2023-12-18T18:30:00']
event_titles = [
    '2023-02-03 to 04',
    '2023-03-25 to 26',
    '2023-04-01 to 02',
    '2023-12-17 to 18'
]
# %%
def NYSM_Barnes(variable,local_nysm_latlon,local_y_indices,local_x_indices,local_station_indices,start_time,end_time):
    freq = 5 # Frequency of the data in minutes
    dates = pd.date_range(start=start_time, end=end_time, freq=f'{freq}min')
    chunk_size = 24  # 24 hours in minutes divided by frequency

    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.orog.values
    mask = xr.open_dataset('mask_2d.nc').mask
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
    NYSM = NYSM.resample(time=f'{freq}min').nearest()

    # Get the best gamma and kappa_star for Barnes
    scores_df = pd.read_csv('Barnes_parameter_search.csv')
    gamma = scores_df[scores_df['idx'] == 14]['gamma'].iloc[0]
    kappa_star = scores_df[scores_df['idx'] == 14]['kappa_star'].iloc[0]

    # Copy the input indice information to local variables
    station_indices = local_station_indices
    nysm_latlon = local_nysm_latlon
    y_indices = local_y_indices
    x_indices = local_x_indices

    n_jobs = 24  # 2 threads per chunk, parallel across 60 chunks
    ds = NYSM[variable].sel(time=dates)

    def interpolate_one_time(t_idx):
        try:
            sample = ds.isel(time=t_idx)
            station_values = sample.values[local_station_indices]

            # Filter out NaNs
            valid_mask = ~np.isnan(station_values)
            if not np.any(valid_mask):
                return None  # skip empty time step

            valid_locs = local_nysm_latlon[valid_mask]
            valid_values = station_values[valid_mask]

            interp_flat = interpolate_to_points(
                valid_locs, valid_values, grid_points,
                interp_type='barnes', gamma=gamma,
                minimum_neighbors=1, kappa_star=kappa_star
            )
            interp_2d = interp_flat.reshape(RTMA_lat.shape).astype(np.float32)

            da_interp = xr.DataArray(
                interp_2d,
                dims=('y', 'x'),
                coords={'latitude': (('y', 'x'), RTMA_lat), 'longitude': (('y', 'x'), RTMA_lon)},
                name=variable
            )
            da_interp = da_interp.expand_dims(time=[ds.time.values[t_idx]])
            return da_interp
        except Exception as e:
            print(f"Failed at time index {t_idx}: {e}")
            return None

    # Run in serial or parallel
    n_jobs = min(120, os.cpu_count())
    from joblib import Parallel, delayed, parallel_backend
    from tqdm import tqdm
    with parallel_backend('loky'):
        results = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(interpolate_one_time)(i) for i in tqdm(range(len(ds.time)))
        )
    print("Interpolation completed.")
    # Remove None entries
    results_clean = [r for r in results if r is not None]
    print(f"Interpolated {len(results_clean)} time steps out of {len(ds.time)} total.")
    # Combine
    final_ds = xr.concat(results_clean, dim='time').to_dataset(name=variable)
    print(f"Final dataset shape: {final_ds[variable].shape}")
    return final_ds
# %%
def run_Barnes_interpolation_task(i, infer_val, inference_stations_seed, variable, exclude_station_indices, include_station_indices, start_time, end_time,
                       test_inputs):

    rng = np.random.default_rng(inference_stations_seed)
    perm = rng.permutation(len(nysm_latlon))

    exclude_set = set(exclude_station_indices) | set(exclude_indices)
    include_set = set(include_station_indices)

    include_perm = [idx for idx in perm if idx in include_set]
    neutral_perm = [idx for idx in perm if idx not in exclude_set and idx not in include_set]
    exclude_perm = [idx for idx in perm if idx in exclude_set]
    perm_cleaned = np.array(include_perm + neutral_perm + exclude_perm)

    inference_indices = perm_cleaned[:infer_val]
    unseen_indices = perm_cleaned[infer_val:]

    local_y_indices = y_indices[inference_indices]
    local_x_indices = x_indices[inference_indices]
    local_station_indices = station_indices[inference_indices]
    local_nysm_latlon = nysm_latlon[inference_indices]
    inference_fn = NYSM_Barnes

    output_ds = inference_fn(
        variable=variable,
        local_nysm_latlon=local_nysm_latlon,
        local_y_indices=local_y_indices,
        local_x_indices=local_x_indices,
        local_station_indices=local_station_indices,
        start_time=start_time,
        end_time=end_time
    )

    # Define output folder and filename
    base_dir = "Extreme_cases"
    folder = os.path.join(base_dir, 'Barnes', test_inputs, variable, f"seed-{inference_stations_seed}")
    os.makedirs(folder, exist_ok=True)

    start_str = start_time
    end_str = end_time
    file_name = f"{start_str}_to_{end_str}.zarr"
    target_path = os.path.join(folder, file_name)

    output_ds.to_zarr(target_path, mode='w')
    print('output dataset saved to:', target_path)
    del output_ds
    gc.collect()

    return (i, inference_stations_seed, target_path, inference_indices, unseen_indices)
# %%
variable = 'i10fg'

inference_stations_seed = int(sys.argv[1])
i = int(sys.argv[2]) # indice of the extreme event

infer_val = 100
#for data_dict_name, fn in [('test_data_dict.pkl', NYSM_inference), ('test_rtma_data_dict.pkl', RTMA_inference)]:
for test_inputs in ['NYSM']:
    test_data_dict = {}
    #extreme_day = pd.to_datetime(extreme_days[i])
    extreme_station_indices = extreme_station_lists[i]
    n = len(extreme_station_indices)
    half_plus = (n + 1) // 2

    exclude_station_indices = extreme_station_indices[:half_plus]  # To exclude
    include_station_indices = extreme_station_indices[half_plus:]  # To include

    '''
    print(f"Extreme day: {extreme_day}, Station: {exclude_station_indices}")
    start_time = extreme_day + pd.Timedelta(minutes=-30)
    end_time   = extreme_day + pd.Timedelta(days=1, minutes=30)
    '''
    start_time = WRF_start_dates[i]
    end_time = WRF_end_dates[i]
    print(f"Extreme day: {start_time}, Station: {exclude_station_indices}, Include: {include_station_indices}")

    # Generate full permutation with a fixed seed
    i_case, seed, target_path, inf_idx, un_idx = run_Barnes_interpolation_task(
        i, infer_val, inference_stations_seed, variable,
        exclude_station_indices, include_station_indices,
        start_time, end_time, test_inputs
    )
# %%
