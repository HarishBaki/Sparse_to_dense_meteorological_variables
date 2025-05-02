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

from data_loader import RTMA_sparse_to_dense_Dataset, Transform
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss

def str_or_none(v):
    return None if v.lower() == 'none' else v

def int_or_none(v):
    return None if v.lower() == 'none' else int(v)

def bool_from_str(v):
    return v.lower() == 'true'

# %%
# === Early stopping, and checkpointing functions ===
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
def save_model_checkpoint(model, optimizer,scheduler, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f" Model checkpoint saved at: {path}")

def restore_model_checkpoint(model, optimizer, scheduler, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Restored checkpoint from: {path} (epoch {checkpoint['epoch']})")
    return model, optimizer, scheduler, start_epoch

# %%
def run_epochs(model, train_dataloader, val_dataloader, optimizer, criterion, metric, device, num_epochs,
               checkpoint_dir, train_sampler, scheduler, early_stopping, target_transform=None,resume=False):

    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Optional resume ===
    start_epoch = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")

    if resume and os.path.exists(latest_ckpt_path):
        model, optimizer, scheduler, start_epoch = restore_model_checkpoint(model, optimizer, scheduler, latest_ckpt_path, device)

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # === Training ===
        model.train()
        train_loss_total = 0.0
        train_metric_total = 0.0
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False) if show_progress else train_dataloader
        for batch in train_bar:
            input_tensor, target_tensor,_ = batch
            # Here you can extract the station values and mask if needed
            input_tensor = input_tensor.to(device, non_blocking=True)   # [B, C, H, W]
            target_tensor = target_tensor.to(device, non_blocking=True)    # [B, C, H, W]
            station_mask = input_tensor[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor, station_mask)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

            # === Optional: Apply inverse transform if needed ===
            if target_transform is not None:
                output = target_transform.inverse(output)
                target_tensor = target_transform.inverse(target_tensor)

            # Compute the metric
            metric_value = metric(output, target_tensor, station_mask)
            train_metric_total += metric_value.item()

            if show_progress:
                train_bar.set_postfix(loss=loss.item(), metric=metric_value.item())

        avg_train_loss = train_loss_total / len(train_dataloader)
        avg_train_metric = train_metric_total / len(train_dataloader)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        val_metric_total = 0.0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False) if show_progress else val_dataloader
        with torch.no_grad():
            for batch in val_bar:
                input_tensor, target_tensor,_ = batch
                input_tensor = input_tensor.to(device, non_blocking=True)
                target_tensor = target_tensor.to(device, non_blocking=True)
                station_mask = input_tensor[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]

                output = model(input_tensor)
                loss = criterion(output, target_tensor,station_mask)
                val_loss_total += loss.item()

                # === Optional: Apply inverse transform if needed ===
                if target_transform is not None:
                    output = target_transform.inverse(output)
                    target_tensor = target_transform.inverse(target_tensor)
                
                # Compute the metric
                metric_value = metric(output, target_tensor, station_mask)
                val_metric_total += metric_value.item()

                if show_progress:
                    val_bar.set_postfix(loss=loss.item(), metric=metric_value.item())
        avg_val_loss = val_loss_total / len(val_dataloader)
        avg_val_metric = val_metric_total / len(val_dataloader)

        # === Scheduler step ===
        scheduler.step()

        # === Log to Weights & Biases ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_metric": avg_train_metric,
                "val_loss": avg_val_loss,
                "val_metric": avg_val_metric,
                "learning_rate": scheduler.get_last_lr()[0],   # log current LR
            })

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Epoch {epoch+1}]  Train Loss: {avg_train_loss:.4f} |  Val Loss: {avg_val_loss:.4f}  | LR: {scheduler.get_last_lr()[0]:.6f}")

        # === Save Checkpoints ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            # Update best model if needed
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_model_checkpoint(model, optimizer, scheduler, epoch, best_ckpt_path)

            # Always update latest checkpoint
            save_model_checkpoint(model, optimizer, scheduler, epoch, latest_ckpt_path)

        # === Early stopping check (in ALL ranks, after validation step) ===
        if early_stopping is not None:
            stop_flag = torch.tensor(
                int(early_stopping(avg_val_loss)), device=device, dtype=torch.int
            )
            if dist.is_initialized():
                dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
            if stop_flag.item() > 0:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

# === Computing the outputs on test data and saving them to zarr ===
def init_zarr_store(zarr_store, dates,variable):
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
    template.to_dataset(name = variable).to_zarr(zarr_store, compute=False, mode='w')

def run_test(model, test_dataloader, test_dates_range, criterion, metric, device,
               checkpoint_dir, variable , target_transform=None):
    # load the best model Handle DDP 'module.' prefix
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)

    # === Creating a zarr for test data ===
    dates = pd.date_range(start=test_dates_range[0], end=test_dates_range[1], freq='h')
    zarr_store = os.path.join(checkpoint_dir, "RTMA_test.zarr")
    if not dist.is_initialized() or dist.get_rank() == 0:   # Initialize only on rank 0 and wait for others
        init_zarr_store(zarr_store, dates, variable)
        print(f"Zarr store initialized at {zarr_store}.")
    if dist.is_initialized():
        dist.barrier()  # All other ranks wait here until rank 0 finishes

    # === Step 2: Evaluate and write predictions using matched time indices ===
    ds = xr.open_zarr(zarr_store, consolidated=False)
    zarr_time = ds['time'].values  # dtype=datetime64[ns]
    time_to_idx = {t: i for i, t in enumerate(zarr_time)}

    # Use low-level Zarr for writing directly
    zarr_write = zarr.open(zarr_store, mode='a')
    zarr_variable = zarr_write[variable]

    # === testing and saving into test zarr===
    model.eval()
    test_loss_total = 0.0
    test_metric_total = 0.0
    show_progress = not dist.is_initialized() or dist.get_rank() == 0
    test_bar = tqdm(test_dataloader, desc=f"[Test]", leave=False) if show_progress else test_dataloader
    with torch.no_grad():
        for batch in test_bar:
            input_tensor, target_tensor, time_value = batch
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.to(device, non_blocking=True)
            station_mask = input_tensor[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]

            output = model(input_tensor)    # [B, 1, H, W]

            # Compute the loss
            loss = criterion(output, target_tensor,station_mask)
            test_loss_total += loss.item()

            # === Optional: Apply inverse transform if needed ===
            if target_transform is not None:
                output = target_transform.inverse(output)
                target_tensor = target_transform.inverse(target_tensor)

            # Compute the metric
            metric_value = metric(output, target_tensor, station_mask)
            test_metric_total += metric_value.item()

            if show_progress:
                test_bar.set_postfix(loss=loss.item(), metric=metric_value.item())

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
        
    avg_test_loss = test_loss_total / len(test_dataloader)
    avg_test_metric = test_metric_total / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f} | Test Metric: {avg_test_metric:.4f}")
    
    # Log the test loss and metric to Weights & Biases
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.log({"Test Loss": avg_test_loss, "Test Metric": avg_test_metric})

# %%
if __name__ == "__main__":      
    # This is the main entry point of the script. 
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
    # === Loading the RTMA data ===
    zarr_store = 'data/RTMA.zarr'
    validation_dates_range = ['2022-01-01T00', '2022-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time
    # if the additional input variables is not none, add the missing times of the additional input variables also. 
    if additional_input_variables is not None:
        for var in additional_input_variables:
            missing_times = xr.concat([missing_times, xr.open_dataset(f'nan_times_{var}.nc').time], dim='time')
        # remove duplicates
        missing_times = missing_times.drop_duplicates('time')
    if not dist.is_initialized() or dist.get_rank() == 0:
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
    train_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        input_variables_in_order,
        orography_as_channel,
        train_dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times,
        input_transform=input_transform,
        target_transform=target_transform,
        n_random_stations=n_random_stations,
        global_seed=global_seed
    )
    if is_distributed():
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # shuffle if not using DDP
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )
    validation_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        input_variables_in_order,
        orography_as_channel,
        validation_dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times,
        input_transform=input_transform,
        target_transform=target_transform,
        n_random_stations=n_random_stations,
        global_seed=global_seed
    )
    if is_distributed():
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        validation_sampler = None
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=(validation_sampler is None), # shuffle if not using DDP
        sampler=validation_sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data loaded successfully.")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")

    # %%
    # === Set up device, model, loss, optimizer ===
    input_resolution = (orography.shape[0], orography.shape[1])
    if orography_as_channel:
        in_channels = len(input_variables_in_order) + 2  # input variables + orography + station mask
    else:
        in_channels = len(input_variables_in_order) + 1 # input variables + station mask
    out_channels = 1
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
                        hard_enforce_stations=True).to(device)
    elif model_name == "UNet":
        C = 32
        n_layers = 4
        model = UNet(in_channels=in_channels, 
                        out_channels=out_channels,
                        C=C, 
                        n_layers=n_layers,
                        hard_enforce_stations=True).to(device)
    elif model_name == "SwinT2UNet":
        C = 32
        n_layers = 4
        window_sizes = [8, 8, 4, 4, 2]
        head_dim = 32
        model = SwinT2UNet(input_resolution=input_resolution, 
                        in_channels=in_channels, 
                        out_channels=out_channels, 
                        C=C, n_layers=n_layers, 
                        window_sizes=window_sizes,
                            head_dim=head_dim,
                            hard_enforce_stations=True).to(device)
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
                project="sparse-to-dense",id=args.wandb_id,resume='allow',
            )
        else:
            wandb.init(
                project="Sparse-to-Dense",
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
                }
            )
    
    # === Run the training and validation ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Starting training and validation...")
    run_epochs(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        metric=metric,
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        train_sampler=train_sampler, 
        scheduler=scheduler,
        early_stopping=early_stopping,
        target_transform=target_transform,
        resume=resume
    )
    # === Barrier to ensure all ranks wait for checkpoint ===
    if dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Training and validation completed.")
    
    # === Run the test and save the outputs to zarr ===
    test_dates_range = ['2023-01-01T00', '2023-12-31T23']
    test_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        input_variables_in_order,
        orography_as_channel,
        test_dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times,
        input_transform=input_transform,
        target_transform=target_transform
    )
    if is_distributed():
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=(test_sampler is None), # shuffle if not using DDP
        sampler=test_sampler,
        pin_memory=True,
        num_workers=num_workers
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
            print("Test data loaded successfully.")
            print(f"Test dataset size: {len(test_dataset)}")

    if not dist.is_initialized() or dist.get_rank() == 0:
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


