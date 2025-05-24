# %%
# === Top level imports ===
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ExponentialLR


import xarray as xr
import zarr
import dask.array as da
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

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

# %%
# === Creating an initial zarr with the full dataset ===
# This call will be run for once, just to create the zarr store.
# The NYSM data also have exactly 631008 time instances, which is at 5min interval. 
dates = pd.date_range(start='2018-01-01T00:00', end='2023-12-31T23:59', freq='5min')
zarr_store = '/data/harish/Gust_field_nowcasting_from_Sparse_stations/NYSM.zarr'    #This is not the path of Sparse_to_Dense, but the path of the Gust Nowcasting zarr store
os.makedirs(zarr_store, exist_ok=True)
# === Computing the outputs on test data and saving them to zarr ===
def init_zarr_store(zarr_store, dates, variable, chunk_size=24):
    orography = xr.open_dataset('orography.nc')
    orography.attrs = {}
    shape = (len(dates),) + orography.orog.shape  # (time, y, x)

    # Create a lazy Dask array filled with NaNs
    data = da.full(shape, np.nan, chunks=(chunk_size, -1, -1), dtype='float32')

    template = xr.DataArray(
        data,
        dims=('time', 'y', 'x'),
        coords={
            'time': dates,
            'latitude': orography.latitude,
            'longitude': orography.longitude
        },
        name=variable,
        attrs={}
    )

    ds = template.to_dataset()
    ds.to_zarr(zarr_store, mode='w')
# Execute the below line only once to create the zarr store
#init_zarr_store(zarr_store, dates, 'i10fg',chunk_size=288)
# === End of zarr creation ===

# %%
def run_test(model, test_dataloader, device,
               best_ckpt_path, variable , target_transform=None):
    # load the best model Handle DDP 'module.' prefix
    model, _, _, _ = restore_model_checkpoint(model, optimizer, scheduler, best_ckpt_path, device)

    # === Step 2: Evaluate and write predictions using matched time indices ===
    ds = xr.open_zarr(zarr_store, consolidated=False)
    zarr_time = ds['time'].values  # dtype=datetime64[ns]
    time_to_idx = {t: i for i, t in enumerate(zarr_time)}

    # Use low-level Zarr for writing directly
    zarr_write = zarr.open(zarr_store, mode='a')
    zarr_variable = zarr_write[variable]

    # === testing and saving into test zarr===
    model.eval()
    show_progress = True
    test_bar = tqdm(test_dataloader, desc=f"[Test]", leave=False) if show_progress else test_dataloader
    with torch.no_grad():
        for batch in test_bar:
            input_tensor, time_value = batch
            input_tensor = input_tensor.to(device, non_blocking=True)

            output = model(input_tensor)    # [B, 1, H, W]

            # === Optional: Apply inverse transform if needed ===
            if target_transform is not None:
                output = target_transform.inverse(output)
            # Clamping the output to be non-negative
            output = torch.clamp(output, min=0.0)
            # create an xarray dataset from the output
            output_np = output.cpu().numpy()    # [B, 1, H, W]
            time_np = np.array(time_value, dtype='datetime64[ns]')

            # Collect all indices and outputs, then write in batch after loop
            idx_list, data_list = [], []
            # Match and write to correct time indices
            for i, t in enumerate(time_np):
                idx = time_to_idx.get(t)
                if idx is not None:
                    idx_list.append(idx)
                    data_list.append(output_np[i].squeeze(0))
                else:
                    print(f"Warning: Time {t} not found in time axis.")
            
            # Now write once
            if idx_list:
                zarr_variable.oindex[idx_list] = np.stack(data_list)

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
            "--year", "2021",
        ]
        print("DEBUG: Using injected args:", sys.argv)

    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Run daily inference")
    parser.add_argument("--year", type=str, required=True, help="Year in YYYY format")
    args = parser.parse_args()

    # %%
    # Required time slice from arguments
    year = args.year
    dates_range = [f"{year}-01-01T00:00", f"{year}-12-31T23:59"]

    # === Default parameters for internal use ===
    best_ckpt_path = "Sparse_to_Dense_best_model.pt"
    variable = "i10fg"
    model_name = "UNet"
    orography_as_channel = True
    additional_input_variables = ["si10", "t2m", "sh2"]
    additional_channels = ["orography"] + additional_input_variables
    stations_seed = 42
    n_random_stations = None
    randomize_stations_persample = True
    loss_name = "MaskedCharbonnierLoss"
    transform = "standard"
    activation_layer = "gelu"
    weights_seed = 42
    n_inference_stations = None
    num_epochs = 2
    batch_size = 24
    num_workers = 24

    # %%
    # ==================== Distributed setup ====================
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        # Get rank and set device as before
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Fallback: single GPU (non-DDP) for debugging or interactive use
        print("Running without distributed setup (no torchrun detected)")
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def is_distributed():
        return dist.is_available() and dist.is_initialized()       # useful for checking if we are in a distributed environment

    # %%
    # ==== Print the parsed and converted arguments along with the device ====
    print(
    f"Args:\n"
    f"  best_ckpt_path: {best_ckpt_path}\n"
    f"  variable: {variable}\n"
    f"  model_name: {model_name}\n"
    f"  orography_as_channel: {orography_as_channel}\n"
    f"  additional_input_variables: {additional_input_variables}\n"
    f"  years_range: {dates_range}\n"
    f"  stations_seed: {stations_seed}\n"
    f"  n_random_stations: {n_random_stations}\n"
    f" randomize_stations_persample: {randomize_stations_persample}\n"
    f"  loss_name: {loss_name}\n"
    f"  transform: {transform}\n"
    f"  num_epochs: {num_epochs}\n"
    f"  batch_size: {batch_size}\n"
    f"  num_workers: {num_workers}\n"
    f"  device: {device}\n"
    f"  weights_seed: {weights_seed}\n"
    f"  activation_layer: {activation_layer}\n"
    f"  Training on distrbuuted: {is_distributed()}\n"
    )
    
    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc')
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.orog.values

    mask = xr.open_dataset('mask_2d.nc').mask
    mask_tensor = torch.tensor(mask.values.astype(np.float32), device=device)  # [H, W], defnitely send it to device
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

    missing_times = (NYSM[variable].isnull()).all(dim='station')    #We need to check if all the stations have missing values for a given time
    # if the additional input variables is not none, add the missing times of the additional input variables also. 
    if additional_input_variables is not None:
        for var in additional_input_variables:
            missing_times |= (NYSM[var].isnull()).all(dim='station')
    missing_times = NYSM.time.where(missing_times).dropna(dim='time')
    print(f"Missing times shape: {missing_times.shape}")
    # %%
    # Read stats of RTMA data
    RTMA_stats = xr.open_dataset('RTMA_variable_stats.nc')
    input_variables_in_order = [variable] if additional_input_variables is None else [variable]+additional_input_variables  
    target_variables_in_order = [variable]
    input_stats = RTMA_stats.sel(variable=input_variables_in_order+['orography']) if orography_as_channel else RTMA_stats.sel(variable=input_variables_in_order)    
    input_channnel_indices = list(range(len(input_variables_in_order+['orography']))) if orography_as_channel else list(range(len(input_variables_in_order)))
    target_stats = RTMA_stats.sel(variable=target_variables_in_order)  
    target_channnel_indices = list(range(len(target_variables_in_order)))

    if not dist.is_initialized() or dist.get_rank() == 0:  
        print(f"Input stats: {input_stats}", input_channnel_indices)
        print(f"Target stats: {target_stats}", target_channnel_indices)

    if transform.lower() == 'none':
        input_transform = None
        target_transform = None
    else:
        input_transform = Transform(
            mode=transform.lower(),  # 'standard' or 'minmax'
            stats=input_stats,  # So, no need to pass the channel indices, since the transformation will happen on channels from 0 to -1 
            channel_indices=input_channnel_indices
        )
        target_transform = Transform(
            mode=transform.lower(),  # 'standard' or 'minmax'
            stats=target_stats,
            channel_indices=target_channnel_indices
        )

    # %%
    # === Set up device, model, loss, optimizer ===
    input_resolution = (orography.shape[0], orography.shape[1])
    if orography_as_channel:
        in_channels = len(input_variables_in_order) + 2  # input variables + orography + station mask
    else:
        in_channels = len(input_variables_in_order) + 1 # input variables + station mask
    out_channels = 1

    if activation_layer == 'gelu':
        act_layer = nn.GELU
    elif activation_layer == 'relu':
        act_layer = nn.ReLU
    elif activation_layer == 'leakyrelu':
        act_layer = nn.LeakyReLU

    if model_name == "DCNN":
        C = 48
        kernel = (7, 7)
        final_kernel = (3, 3)
        n_layers = 7
        model = DCNN(in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, 
                        kernel=kernel,
                        final_kernel=final_kernel, 
                        n_layers=n_layers,
                        act_layer=act_layer,
                        hard_enforce_stations=True).to(device)
    elif model_name == "UNet":
        C = 32
        n_layers = 4
        dropout_prob=0.2
        drop_path_prob=0.2
        model = UNet(in_channels=in_channels, 
                        out_channels=out_channels,
                        C=C, 
                        dropout_prob=dropout_prob,
                        drop_path_prob=drop_path_prob,
                        act_layer=act_layer,
                        n_layers=n_layers,
                        hard_enforce_stations=True).to(device)
    
    elif model_name == "SwinT2UNet":
        C = 32
        n_layers = 4
        window_sizes = [8, 8, 4, 4, 2]
        head_dim = 32
        attn_drop = 0.2
        proj_drop = 0.2
        mlp_ratio = 4.0
        model = SwinT2UNet(input_resolution=input_resolution, 
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, n_layers=n_layers, 
                        window_sizes=window_sizes,
                            head_dim=head_dim,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            mlp_ratio=mlp_ratio,
                            act_layer=act_layer,
                            hard_enforce_stations=True).to(device)
    
    if act_layer == nn.GELU:
            initialize_weights_xavier(model,seed = weights_seed)
    elif act_layer == nn.ReLU:
        initialize_weights_he(model,seed = weights_seed)
    elif act_layer == nn.LeakyReLU:
        initialize_weights_he(model,seed = weights_seed)

    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define the loss criterion and metric here, based on input loss name. The functions are sent to the GPU inside
    if loss_name == "MaskedMSELoss":
        criterion = MaskedMSELoss(mask_tensor)
    elif loss_name == "MaskedRMSELoss":
        criterion = MaskedRMSELoss(mask_tensor)
    elif loss_name == "MaskedTVLoss":
        criterion = MaskedTVLoss(mask_tensor,tv_loss_weight=0.001, beta=0.5)    
    elif loss_name == "MaskedCharbonnierLoss":
        criterion = MaskedCharbonnierLoss(mask_tensor,eps=1e-3)
    metric = MaskedRMSELoss(mask_tensor)

    # === Optimizer, scheduler, and early stopping ===
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    early_stopping = EarlyStopping(patience=20, min_delta=0.0)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Model created and moved to device.")
    
    # %%
    # === Run the test and save the outputs to zarr ===
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("Running the test and saving the outputs to zarr.")
        test_dataset = NYSM_sparse_to_dense_Dataset(
            NYSM,
            input_variables_in_order,
            orography_as_channel,
            dates_range,
            orography,
            RTMA_lat,
            RTMA_lon,
            nysm_latlon,
            y_indices,
            x_indices,
            station_indices,
            mask,
            None,   # We won't pass the missing time instances, since the NYSM data  handles it inside. 
            input_transform=input_transform,
            target_transform=target_transform,
            stations_seed=stations_seed,
            n_random_stations=n_inference_stations,     # This is a key change, since we may use different number of stations during inference.
        )
        test_sampler = None
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False, # shuffle if not using DDP
            sampler=test_sampler,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=False
        )

        print("Test data loaded successfully.")
        print(f"Test dataset size: {len(test_dataset)}")

        print("Starting Testing...")
        start_time = time.time()
        run_test(
            model = model, 
            test_dataloader = test_dataloader,
            device = device,
            best_ckpt_path = best_ckpt_path,
            variable = variable, 
            target_transform = target_transform
        )
        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60
        print(f"Testing completed in {elapsed_minutes:.2f} minutes.")
    
    # === Finish run and destroy process group ===
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print("Finished testing and cleaned up.")


# %%
