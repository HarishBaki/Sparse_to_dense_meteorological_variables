# %%
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
import os, sys, time, glob, re
import pickle

from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from metpy.interpolate import interpolate_to_points
from util import str_or_none, int_or_none

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

# %%
if __name__ == "__main__": 
    # %% Check if running interactively
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__') or 'ipykernel' in sys.argv[0]

    # If run interactively, inject some sample arguments
    if is_interactive() or len(sys.argv) == 1:
        sys.argv = [
            "",  # Script name placeholder
            "--prediction_dir", "Predictions",
            "--variable", "i10fg",
            "--n_inference_stations", "none", 
            "--fold","0",
            "--mode", "w"
        ]
        print("DEBUG: Using injected args:", sys.argv)

    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Train with DDP")
    parser.add_argument("--prediction_dir", type=str, default="Predictions", help="Directory to save predictions")
    parser.add_argument("--variable", type=str, default="i10fg", 
                        help="Target variable to train on ('i10fg','d2m','t2m','si10','sh2','sp')")
    parser.add_argument("--n_inference_stations", type=str_or_none, default="none",
                        help="Number of inference stations to use (e.g., 'none', '93', '62')")
    parser.add_argument("--fold", type=int_or_none, default=0, help="Fold number for cross-validation")
    parser.add_argument("--mode", type=str, default="w", choices=["w", "a"], help="Zarr mode: 'w' for write, 'a' for append")
    args, unknown = parser.parse_known_args()

    # %%
    prediction_dir = args.prediction_dir
    var_name = args.variable
    n_inference_stations = args.n_inference_stations
    fold = args.fold
    mode = args.mode

    # %%
    dates = pd.date_range(start='2023-01-01T00', end='2023-12-31T23', freq='h')
    yyyymmdd = pd.Series(dates.year*10000 + dates.month*100 + dates.day).unique()
    source_zarr_store = f'data/RTMA.zarr'
    prediction_dir = 'Predictions' #'/data/harish/Sparse_to_dense_meteorological_variables'

    if n_inference_stations is not None:
        target_zarr_store = f"{prediction_dir}/Barnes_interpolated/{n_inference_stations}-inference-stations/fold_{fold}/RTMA_test.zarr"
    else:
        target_zarr_store = f"{prediction_dir}/Barnes_interpolated/all-stations/RTMA_test.zarr"
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
    orography = orography.set_coords(['latitude', 'longitude'])
    orography = orography.orog.values
    mask = xr.open_dataset('mask_2d.nc').mask
    # Load NYSM station data
    nysm = pd.read_csv('nysm.csv')
    # NYSM station lat/lon
    nysm_latlon = np.stack([
        nysm['lat [degrees]'].values,
        (nysm['lon [degrees]'].values + 360) % 360
    ], axis=-1) # shape: (N, 2)

    exclude_indices = [65, 102]
    nysm_latlon = np.delete(nysm_latlon, exclude_indices, axis=0)
    nysm = nysm.drop(index=exclude_indices).reset_index(drop=True)  # Drop the excluded indices from the DataFrame

    # Precompute grid KDTree
    grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
    tree = cKDTree(grid_points)
    # Query the station locations
    _, indices_flat = tree.query(nysm_latlon)
    # Convert flat indices to 2D (y, x)
    y_indices, x_indices = np.unravel_index(indices_flat, orography.shape)

    # %%
    # Check for the n_random_stations
    if n_inference_stations is not None and fold is not None:
        with open(f"inference_stations_list.pkl", "rb") as f:
            inference_stations_list = pickle.load(f)
        random_indices = inference_stations_list[f'n_infer_{n_inference_stations}'][f'fold_{fold}']['inference']
        # Update the y_indices and x_indices to the random stations
        y_indices = y_indices[random_indices]
        x_indices = x_indices[random_indices]
        nysm_latlon = nysm_latlon[random_indices]

    # %%
    # Get the best gamma and kappa_star for Barnes
    scores_df = pd.read_csv('Barnes_parameter_search.csv')
    gamma = scores_df[scores_df['idx'] == 14]['gamma'].iloc[0]
    kappa_star = scores_df[scores_df['idx'] == 14]['kappa_star'].iloc[0]

    # %%
    '''
    ds = xr.open_zarr(source_zarr_store, chunks={'time': 24})[var_name].sel(time=dates)
    sample = ds.isel(time=2000)
    station_values = sample.values[y_indices, x_indices]
    interp_flat = interpolate_to_points(
        nysm_latlon, station_values, grid_points, interp_type='barnes', 
        gamma=gamma,minimum_neighbors=1,kappa_star=kappa_star,
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
    interp.where(mask,0).plot()
    '''

    # %%
    chunk_size = 24
    n_jobs = 60  # 2 threads per chunk, parallel across 60 chunks
    ds = xr.open_zarr(source_zarr_store, chunks={'time': chunk_size})[var_name].sel(time=dates)

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
                station_values = sample.values[y_indices, x_indices]
                interp_flat = interpolate_to_points(
                    nysm_latlon, station_values, grid_points, interp_type='barnes',
                    gamma=gamma, minimum_neighbors=1,kappa_star=kappa_star
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
