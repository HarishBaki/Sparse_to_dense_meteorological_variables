# %%
import torch 
import numpy as np
import pandas as pd
import xarray as xr
import dask
import os, sys
import glob
import zarr
from joblib import Parallel, delayed
import os, argparse
import dask.array as da
from scipy.spatial import cKDTree

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.colors import TwoSlopeNorm

from joblib import Parallel, delayed
from tqdm import tqdm

from data_loader import RTMA_sparse_to_dense_Dataset, Transform
from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedPSNR, MaskedSSIM
from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store


# %%
if __name__ == "__main__":   
    # %%   
    # This is the main entry point of the script. 
    # === Args for interactive debugging ===
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__') or 'ipykernel' in sys.argv[0]

    # If run interactively, inject some sample arguments
    if is_interactive() or len(sys.argv) == 1:
        sys.argv = [
            "",  # Script name placeholder
            "--variable", "i10fg",
            "--inference_stations_seed", "42",
            "--n_inference_stations", "none", 
            "--data_type","NYSM",
        ]
        print("DEBUG: Using injected args:", sys.argv)

    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Train with DDP")
    parser.add_argument("--variable", type=str, default="i10fg", 
                        help="Target variable to train on ('i10fg','d2m','t2m','si10','sh2','sp')")
    parser.add_argument("--inference_stations_seed", type=int, default=42,
                        help="Seed for inference stations, used to select random stations if n_inference_stations is not None")
    parser.add_argument("--n_inference_stations", type=int_or_none, default=None,
                        help="Number of inference stations to use, if None, all stations will be used") 
    parser.add_argument("--data_type", type=str, default="RTMA")

    args, unknown = parser.parse_known_args()

    # %%
    #
    variable = args.variable
    n_inference_stations = args.n_inference_stations
    inference_stations_seed = args.inference_stations_seed
    if n_inference_stations is not None:
        test_dir = f'data/Barnes_interpolated/{inference_stations_seed}/{n_inference_stations}-inference-stations'
    else:
        test_dir = f'data/Barnes_interpolated/all-stations'
    data_type = args.data_type
    test_zarr_store = f'{test_dir}/{data_type}_test.zarr'

    # %%
    # ==== Print the parsed and converted arguments along with the device ====
    print(
    f"Args:\n"
    f"  variable: {variable}\n"
    f"  test_dir: {test_dir}\n"
    f"  test_zarr_store: {test_zarr_store}"
    )

    # %%
    # # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.orog.values

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

    # Precompute grid KDTree
    grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
    tree = cKDTree(grid_points)
    # Query the station locations
    _, indices_flat = tree.query(nysm_latlon)
    # Convert flat indices to 2D (y, x)
    y_indices, x_indices = np.unravel_index(indices_flat, orography.shape)

    station_mask = np.zeros_like(RTMA_lat, dtype=np.uint8)
    station_mask[y_indices, x_indices] = 1  # Set 1 at the station locations

    ref_zarr_store = 'data/RTMA.zarr'
    test_dates_range = ['2023-01-01T00','2023-12-31T23']

    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time


    # Load NYSM once
    NYSM = xr.open_dataset('data/NYSM.nc')
    NYSM['longitude'] = (NYSM['longitude'] + 360) % 360
    NYSM_lat = NYSM.latitude.values
    NYSM_lon = NYSM.longitude.values
    station_points = np.stack([NYSM_lat.ravel(), NYSM_lon.ravel()], axis=-1)
    tree = cKDTree(station_points)
    _, station_indices = tree.query(nysm_latlon)
    NYSM = NYSM.isel(station=station_indices).sel(time=slice(*test_dates_range)) #.resample(time='1h').max()
        
    # %%
    # ========= Actual metric computation starts here ===
    ds_all_vars = {}

    # Load predictions and targets
    target_ds = xr.open_zarr(ref_zarr_store)[variable].sel(time=slice(*test_dates_range))
    test_ds = xr.open_zarr(test_zarr_store)[variable].sel(time=slice(*test_dates_range))
    NYSM_var_data = NYSM[variable].sel(time=slice(*test_dates_range))

    if n_inference_stations is not None:
        rng = np.random.default_rng(inference_stations_seed)
        perm = rng.permutation(len(nysm_latlon))
        random_indices = perm[:n_inference_stations]
        y_indices = y_indices[random_indices]
        x_indices = x_indices[random_indices]

    T = len(target_ds.time)
    H, W = RTMA_lat.shape

    if data_type == 'RTMA':
        station_mask = np.zeros((T, 1, H, W), dtype=np.uint8)
        station_mask[:, 0, y_indices, x_indices] = 1

    elif data_type == 'NYSM':
        if variable == 'i10fg':
            # Convert to pandas datetime index and infer frequency
            test_ds_freq = pd.infer_freq(pd.DatetimeIndex(test_ds.time.values))
            if test_ds_freq == '5min':
                test_ds = test_ds.rolling(time=12, min_periods = 1, center=True).max()
                test_ds = test_ds.sel(time=test_ds.time.dt.minute==0)
            else:
                test_ds = test_ds.resample(time='1h').nearest()
            NYSM_var_data = NYSM_var_data.rolling(time=12, min_periods = 1, center=True).max()
            NYSM_var_data = NYSM_var_data.sel(time=NYSM_var_data.time.dt.minute==0)
        else:
            test_ds = test_ds.resample(time='1h').nearest()
            NYSM_var_data = NYSM[variable].resample(time='1h').nearest()
        missing_mask = NYSM_var_data.T.isnull().values  # shape [T, N_stations], this is crucial to make sure since the trianing process excludes the missing stations
        station_mask = np.zeros((T, 1, H, W), dtype=np.uint8)
        for s in range(len(y_indices)):
            y, x = y_indices[s], x_indices[s]
            present_mask = ~missing_mask[:, s]
            station_mask[present_mask, 0, y, x] = 1
    # %%
    target_tensor = torch.tensor(target_ds.values, dtype=torch.float32).unsqueeze(1)
    test_tensor = torch.tensor(test_ds.values, dtype=torch.float32).unsqueeze(1)
    nan_mask = torch.isnan(target_tensor) | torch.isnan(test_tensor)
    target_tensor = target_tensor.masked_fill(nan_mask, 0.0)
    test_tensor = test_tensor.masked_fill(nan_mask, 0.0)
    station_mask_tensor = torch.tensor(station_mask, dtype=torch.float32)

    # Compute per-time metrics
    metric_names = list(metrics.keys())
    results = [metrics[k](test_tensor, target_tensor, station_mask_tensor, reduction='none') for k in metric_names]
    data_array = torch.stack(results, dim=0).T  # [time, metric]
    da = xr.DataArray(
        data=data_array.numpy(),
        dims=["time", "metric"],
        coords={"time": target_ds.time.values, "metric": metric_names},
        name=variable
    )

    # Compute overall metrics
    # Exclude the missing time instance from both the data, when computing the overall metrics.
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time

    # === Filter time indices ===
    valid_time_mask = ~target_ds.time.isin(missing_times)
    valid_time_indices = np.where(valid_time_mask.values)[0]

    # === Subset all time-dependent tensors ===
    target_tensor = torch.tensor(target_ds.isel(time=valid_time_indices).values, dtype=torch.float32).unsqueeze(1)
    test_tensor = torch.tensor(test_ds.isel(time=valid_time_indices).values, dtype=torch.float32).unsqueeze(1)
    station_mask_tensor = torch.tensor(station_mask[valid_time_indices], dtype=torch.float32)
    nan_mask = torch.isnan(target_tensor) | torch.isnan(test_tensor)
    target_tensor = target_tensor.masked_fill(nan_mask, 0.0)
    test_tensor = test_tensor.masked_fill(nan_mask, 0.0)
    # %%
    summary_values = []
    for key in metric_names:
        reduction = 'global' if key == 'masked_rmse' else 'elementwise_mean'
        overall_metric = metrics[key](test_tensor, target_tensor, station_mask_tensor, reduction=reduction)
        summary_values.append(overall_metric.item())

    summary_da = xr.DataArray(
        data=np.array(summary_values),
        dims=["metric"],
        coords={"metric": metric_names},
        name=f"{variable}_summary"
    )
    print(summary_da)

    ds_all_vars[variable] = da
    ds_all_vars[f"{variable}_summary"] = summary_da
    # %%
    combined_ds = xr.Dataset(ds_all_vars)
    combined_ds.to_netcdf(f'{test_dir}/{data_type}_test_metrics.nc')

# %%
