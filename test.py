# %%
# === Top level imports ===
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import zarr
import pandas as pd
import time

import os
import wandb, argparse
from tqdm import tqdm

from data_loader import RTMA_sparse_to_dense_Dataset
from models.Google_Unet import UNet

# %%
def restore_model_checkpoint(model, optimizer, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    # Handle DDP 'module.' prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Restored checkpoint from: {path} (epoch {checkpoint['epoch']})")
    return model, optimizer, start_epoch

# %%
def init_zarr_store(zarr_store, dates,variable,mode):
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
    template.to_dataset(name = variable).to_zarr(zarr_store, compute=False, mode=mode)

# %%
if __name__ == "__main__":
    # %%
    # === Argument Parser ===
    parser = argparse.ArgumentParser(description="Sparse-to-dense model testing and Zarr writing")

    parser.add_argument('--variable', type=str, required=True,
                        help='Variable name (e.g., i10fg, u10, v10)')
    parser.add_argument('--mode', type=str, default='w', choices=['w', 'a'],
                        help='Mode to write to Zarr store: w (write) or a (append)')

    args = parser.parse_args()

    variable = args.variable
    mode = args.mode

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc').orog
    RTMA_lat = orography.latitude.values
    RTMA_lon = orography.longitude.values
    orography = orography.values

    mask = xr.open_dataset('mask_2d.nc').mask
    # Load NYSM station data
    nysm = pd.read_csv('nysm.csv')
    # NYSM station lat/lon
    nysm_latlon = np.stack([
        nysm['lat [degrees]'].values,
        (nysm['lon [degrees]'].values + 360) % 360
    ], axis=-1)

    # Precompute grid KDTree
    grid_points = np.stack([RTMA_lat.ravel(), RTMA_lon.ravel()], axis=-1)
    tree = cKDTree(grid_points)
    # Query the station locations
    _, indices_flat = tree.query(nysm_latlon)
    # Convert flat indices to 2D (y, x)
    y_indices, x_indices = np.unravel_index(indices_flat, orography.shape)

    # %%
    # === Loading the RTMA data ===
    variable = 'i10fg'
    mode = 'w' # only for 1st variable, for the rest 'a'
    zarr_store = 'data/RTMA.zarr'
    train_dates_range = ['2018-01-01T00', '2021-12-31T23']
    validation_dates_range = ['2022-01-01T00', '2022-12-31T23']
    test_dates_range = ['2023-01-01T00', '2023-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time
    batch_size = 64

    test_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        variable,
        test_dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=batch_size
    )
    print("Data loaded successfully.")
    print(f"Test dataset size: {len(test_dataset)}")

    # %%
    # === Set up device, model, loss, optimizer ===
    in_channels = 2
    out_channels = 1
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Model created and moved to device.")

    # %%
    # === Optional resume ===
    checkpoint_dir = "checkpoints"+'/'+variable
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    model, optimizer, start_epoch = restore_model_checkpoint(model, optimizer, ckpt_path, device)

    # %%
    # === Creating a zarr for test data ===
    dates = pd.date_range(start=test_dates_range[0], end=test_dates_range[1], freq='h')
    zarr_store = 'data/RTMA_test.zarr'
    init_zarr_store(zarr_store, dates, variable, mode)

    # %%
    # === Step 2: Evaluate and write predictions using matched time indices ===
    ds = xr.open_zarr(zarr_store, consolidated=False)
    zarr_time = ds['time'].values  # dtype=datetime64[ns]
    time_to_idx = {t: i for i, t in enumerate(zarr_time)}

    # Use low-level Zarr for writing directly
    zarr_write = zarr.open(zarr_store, mode='a')
    zarr_variable = zarr_write[variable]

    # %%
    # === Validation and saving into test zarr===
    model.eval()
    val_loss_total = 0.0
    val_bar = tqdm(test_dataloader, desc=f"Test", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            input_tensor, target_tensor, time_value = batch
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)

            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            val_loss_total += loss.item()
            val_bar.set_postfix(loss=loss.item())

            # create an xarray dataset from the output
            output_np = output.squeeze(1).cpu().numpy()
            time_np = np.array(time_value, dtype='datetime64[ns]')

            # Match and write to correct time indices
            for i, t in enumerate(time_np):
                idx = time_to_idx.get(t)
                if idx is not None:
                    zarr_variable[idx] = output_np[i]
                else:
                    print(f"Warning: Time {t} not found in time axis.")
            
    avg_val_loss = val_loss_total / len(test_dataloader)
    print(f"Test Loss: {avg_val_loss:.4f}")
    # %%
