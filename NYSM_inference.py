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
def run_test(model, test_dataloader, test_dates_range, criterion, metric, device,
               checkpoint_dir, variable , target_transform=None):
    # load the best model Handle DDP 'module.' prefix
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    model, _, _, _ = restore_model_checkpoint(model, optimizer, scheduler, best_ckpt_path, device)

    # === Creating a zarr for test data ===
    dates = pd.date_range(start=test_dates_range[0], end=test_dates_range[1], freq='h')
    zarr_store = os.path.join(checkpoint_dir, "NYSM_test.zarr")
    init_zarr_store(zarr_store, dates, variable)
    print(f"Zarr store initialized at {zarr_store}.")

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

            # create an xarray dataset from the output
            output_np = output.cpu().numpy()    # [B, 1, H, W]
            time_np = np.array(time_value, dtype='datetime64[ns]')

            # Match and write to correct time indices
            for i, t in enumerate(time_np):
                idx = time_to_idx.get(t)
                if idx is not None:
                    # Write to zarr
                    zarr_variable[idx] = (output_np[i]).squeeze(0)
                else:
                    print(f"Warning: Time {t} not found in time axis.")

# %%
if __name__ == "__main__":      
    # This is the main entry point of the script. 

    # === Args for interactive debugging ===
    def is_interactive():
        import __main__ as main
        return not hasattr(main, '__file__') or 'ipykernel' in sys.argv[0]

    # If run interactively, inject some sample arguments
    if is_interactive() or len(sys.argv) == 1:
        sys.argv = [
            "",  # Script name placeholder
            "--checkpoint_dir", "checkpoints",
            "--variable", "i10fg",
            "--model", "UNet",
            "--orography_as_channel", "true",
            "--additional_input_variables", "none",
            "--train_years_range", "2018,2021",
            "--global_seed", "42",
            "--n_random_stations", "none",
            "--loss", "MaskedCharbonnierLoss",
            "--transform", "standard",
            "--epochs", "2",
            "--batch_size", "16",
            "--num_workers", "32",
            "--wandb_id", "none",
            # "--resume",  # Optional flag — include if you want to resume
            "--weights_seed", "42",
            "--activation_layer", "gelu"
        ]
        print("DEBUG: Using injected args:", sys.argv)

    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Train with DDP")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--variable", type=str, default="i10fg", 
                        help="Target variable to train on ('i10fg','d2m','t2m','si10','sh2','sp')")
    parser.add_argument("--model", type=str, default="DCNN", 
                        help="Model architecture to use ('DCNN', 'GoogleUNet', 'UNet', 'SwinT2UNet')")
    parser.add_argument("--orography_as_channel", type=bool_from_str, default=False,
                    help="Use orography as input channel: True or False")
    parser.add_argument("--additional_input_variables", type=str_or_none, default=None, 
                        help="Additional input variables to train on seperated by comma ('si10,t2m,,sh2'), else pass None")
    parser.add_argument("--train_years_range", type=str, default="2018,2021",
                    help="Comma-separated training years range, e.g., '2018,2019' for 2018 to 2019")
    parser.add_argument("--global_seed", type=int, default=42, help="Global seed for reproducibility")
    parser.add_argument("--n_random_stations", type=int_or_none, default=None, help="Number of random stations in each sample")
    parser.add_argument("--loss", type=str, default="MaskedCharbonnierLoss", 
                        help="Loss function to use ('MaskedMSELoss', 'MaskedRMSELoss', 'MaskedTVLoss', 'MaskedCharbonnierLoss')")
    parser.add_argument("--transform", type=str, default="standard", 
                        help="Transform to apply to the data ('none', 'minmax', 'standard')")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--wandb_id", type=str_or_none, default=None, help="WandB run ID for resuming, not passing will create a new run")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from latest checkpoint (just passing --resume is enough for resume)")
    parser.add_argument("--weights_seed", type=int, default=42, help="Seed for weight initialization")
    parser.add_argument("--activation_layer", type=str, default="gelu", 
                        help="Activation layer to use ('gelu', 'relu', 'leakyrelu')") 
    args, unknown = parser.parse_known_args()

    # %%
    #
    checkpoint_dir = args.checkpoint_dir

    variable = args.variable
    checkpoint_dir = checkpoint_dir+'/'+variable

    model_name = args.model
    checkpoint_dir = checkpoint_dir+'/'+model_name

    orography_as_channel = args.orography_as_channel
    additional_input_variables = args.additional_input_variables
    if additional_input_variables is not None:
        additional_input_variables = [v.strip() for v in additional_input_variables.split(",")]
    additional_channels = []
    if orography_as_channel:
        additional_channels.append('orography')
    if additional_input_variables is not None:
        additional_channels.extend(additional_input_variables)
    if len(additional_channels) > 0:
        checkpoint_dir = checkpoint_dir + '/' + "-".join(additional_channels)
    else:
        checkpoint_dir = checkpoint_dir + '/no-additional-channels'
    
    years = args.train_years_range.split(",")
    if len(years) == 1:
        start_year = end_year = years[0].strip()
        checkpoint_dir = checkpoint_dir+'/'+start_year
    else:
        start_year, end_year = [y.strip() for y in years[:2]]
        checkpoint_dir = checkpoint_dir+'/'+start_year+'-'+end_year
    # Compose the date strings for slicing
    train_dates_range = [f"{start_year}-01-01T00", f"{end_year}-12-31T23"] # ['2018-01-01T00', '2021-12-31T23']

    n_random_stations = args.n_random_stations
    global_seed = args.global_seed
    if n_random_stations is not None:
        checkpoint_dir = f"{checkpoint_dir}/{global_seed}/{n_random_stations}-random-stations"
    else:
        checkpoint_dir = f"{checkpoint_dir}/all-stations"

    loss_name = args.loss
    checkpoint_dir = checkpoint_dir+'/'+loss_name

    transform = args.transform
    checkpoint_dir = checkpoint_dir+'/'+transform

    activation_layer = args.activation_layer
    weights_seed = args.weights_seed

    checkpoint_dir = checkpoint_dir+'/'+activation_layer+'-'+str(weights_seed)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = args.epochs
    resume = args.resume
    batch_size = args.batch_size
    num_workers = args.num_workers  

    # %%
    # ==================== Distributed setup ====================
    if "RANK" in os.environ:
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
    f"  checkpoint_dir: {checkpoint_dir}\n"
    f"  variable: {variable}\n"
    f"  model_name: {model_name}\n"
    f"  orography_as_channel: {orography_as_channel}\n"
    f"  additional_input_variables: {additional_input_variables}\n"
    f"  train_years_range: {train_dates_range}\n"
    f"  global_seed: {global_seed}\n"
    f"  n_random_stations: {n_random_stations}\n"
    f"  loss_name: {loss_name}\n"
    f"  transform: {transform}\n"
    f"  num_epochs: {num_epochs}\n"
    f"  batch_size: {batch_size}\n"
    f"  num_workers: {num_workers}\n"
    f"  wandb_id: {args.wandb_id}\n"
    f"  device: {device}\n"
    f"  resume: {resume}\n"
    f"  weights_seed: {weights_seed}\n"
    f"  activation_layer: {activation_layer}\n"
    f"  Training on distrbuuted: {is_distributed()}\n"
    )
    
    # %%
    # === Loading some topography and masking data ===
    orography = xr.open_dataset('orography.nc').orog
    RTMA_lat = orography.latitude.values    # Nx, Ny 2D arrays
    RTMA_lon = orography.longitude.values   # Nx, Ny 2D arrays
    orography = orography.values

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
    NYSM = NYSM.resample(time='1h').nearest()

    missing_times = (NYSM[variable].isnull()).any(dim='station')
    # if the additional input variables is not none, add the missing times of the additional input variables also. 
    if additional_input_variables is not None:
        for var in additional_input_variables:
            missing_times |= (NYSM[var].isnull()).any(dim='station')
    missing_times = NYSM.time.where(missing_times).dropna(dim='time')
    print(f"Missing times shape: {missing_times.shape}")

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

    # === Initializing the wandb ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        if args.wandb_id is not None:
            wandb.init(
                project="NYSM inference",id=args.wandb_id,resume='allow',
            )
        else:
            wandb.init(
                project="NYSM inference",
                name=checkpoint_dir[len('checkpoints/'):].replace('/','_'),
                config={
                    "variable": variable,
                    "model": model_name,
                    "input channels": in_channels,
                    "output channels": out_channels,
                    "input_resolution": input_resolution,
                    "optimizer": "Adam",
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss_fn": loss_name,
                    "metric": "MaskedRMSELoss",      # or set from args if you add support
                    "epochs": num_epochs,
                    "batch_size": batch_size,
                    "transform": transform,
                    "train_dates_range": train_dates_range,
                    "scheduler": "ExponentialLR",
                    "additional_input_variables": additional_input_variables,
                    "global_seed": global_seed,
                    "n_random_stations": n_random_stations,
                    "orography_as_channel": orography_as_channel,
                    "activation_layer": args.activation_layer,
                    "weights_seed": args.weights_seed,
                }
            )
    
    # === Run the test and save the outputs to zarr ===
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print("Running the test and saving the outputs to zarr.")
        test_dates_range = ['2023-01-01T00:00', '2023-12-31T23:59']
        test_dataset = NYSM_sparse_to_dense_Dataset(
            NYSM,
            input_variables_in_order,
            orography_as_channel,
            test_dates_range,
            orography,
            RTMA_lat,
            RTMA_lon,
            nysm_latlon,
            y_indices,
            x_indices,
            station_indices,
            mask,
            None,
            input_transform=input_transform,
            target_transform=target_transform,
            global_seed=global_seed,
            n_random_stations=n_random_stations
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
        run_test(
            model = model, 
            test_dataloader = test_dataloader,
            test_dates_range = test_dates_range,
            criterion = criterion,
            metric = metric,
            device = device,
            checkpoint_dir = checkpoint_dir,
            variable = variable, 
            target_transform = target_transform
        )
    
    # === Finish run and destroy process group ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print("Finished testing and cleaned up.")

