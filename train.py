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
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

import os
import wandb, argparse
from tqdm import tqdm

from data_loader import RTMA_sparse_to_dense_Dataset, Transform
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss

# %%
# === Early stopping, and checkpointing functions ===
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
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
               checkpoint_dir, train_sampler, scheduler, early_stopping, resume=False):

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
            target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)    # [B, C, H, W]
            station_mask = input_tensor[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor, station_mask)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

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
                target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)
                station_mask = input_tensor[:, -1, ...].unsqueeze(1)  # [B, 1, H, W]

                output = model(input_tensor)
                loss = criterion(output, target_tensor,station_mask)
                val_loss_total += loss.item()

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

        # === Early stopping check ===
        if early_stopping is not None:
            if early_stopping(avg_val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

def main():
    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Train with DDP")

    parser.add_argument("--variable", type=str, default="i10fg", 
                        help="Variable to train on ('i10fg','d2m','t2m','si10','sh2','sp')")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume from latest checkpoint (just passing --resume is enough for resume)") 
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="MaskedRMSELoss", 
                        help="Loss function to use ('MaskedMSELoss', 'MaskedRMSELoss', 'MaskedTVLoss', 'MaskedCharbonnierLoss')")
    parser.add_argument("--model", type=str, default="DCNN", 
                        help="Model architecture to use ('DCNN', 'GoogleUNet', 'UNet', 'SwinT2UNet')")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--transform", type=str, default="none", 
                        help="Transform to apply to the data ('none', 'minmax', 'standard')")
    parser.add_argument("--train_years_range", type=str, default="2018,2021",
                    help="Comma-separated training years range, e.g., '2018,2019' for 2018 to 2019")

    args = parser.parse_args()
    variable = args.variable
    num_epochs = args.epochs
    resume = args.resume
    loss_name = args.loss
    model_name = args.model
    batch_size = args.batch_size
    transform = args.transform
    # Parse the input string into a list of years
    years = args.train_years_range.split(",")
    if len(years) == 1:
        start_year = end_year = years[0].strip()
        checkpoint_dir = args.checkpoint_dir+'/'+variable+'/'+model_name+'/'+loss_name+'/'+start_year
    else:
        start_year, end_year = [y.strip() for y in years[:2]]
        checkpoint_dir = args.checkpoint_dir+'/'+variable+'/'+model_name+'/'+loss_name+'/'+start_year+'-'+end_year
    # Compose the date strings for slicing
    train_dates_range = [f"{start_year}-01-01T00", f"{end_year}-12-31T23"] # ['2018-01-01T00', '2021-12-31T23']    

    # === Distributed setup ===
    dist.init_process_group(backend="nccl")

    # Get rank and local GPU id
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])  # comes from torchrun automatically

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

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
    test_dates_range = ['2023-01-01T00', '2023-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time

    # Read stats of RTMA data
    RTMA_stats = xr.open_dataset('RTMA_variable_stats.nc')
    input_variables_in_order = [variable,'orography']  # modify this to match when we work on multiple variabls as input
    target_variables_in_order = [variable]
    input_stats = RTMA_stats.sel(variable=input_variables_in_order)
    target_stats = RTMA_stats.sel(variable=target_variables_in_order)
    if transform.lower() == 'none':
        input_transform = None
        target_transform = None
    else:
        input_transform = Transform(
            mode=transform.lower(),  # 'standard' or 'minmax'
            stats=input_stats,
            channel_indices=[0, 1]
        )
        target_transform = Transform(
            mode=transform.lower(),  # 'standard' or 'minmax'
            stats=target_stats,
            channel_indices=[0]
        )

    train_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        variable,
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
        target_transform=target_transform
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        num_workers=batch_size
    )
    validation_dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        variable,
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
        target_transform=target_transform
    )
    validation_sampler = DistributedSampler(validation_dataset)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=validation_sampler,
        pin_memory=True,
        num_workers=batch_size
    )
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
        missing_times,
        input_transform=input_transform,
        target_transform=target_transform
    )
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=batch_size
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Data loaded successfully.")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

    # %%
    # === Set up device, model, loss, optimizer ===
    input_resolution = (orography.shape[0], orography.shape[1])
    in_channels = 3
    out_channels = 1
    if model_name == "DCNN":
        C = 48
        kernel = (7, 7)
        final_kernel = (3, 3)
        n_layers = 7
        model = DCNN(in_channels=in_channels, out_channels=out_channels, C=C, kernel=kernel, final_kernel=final_kernel, n_layers=n_layers).to(device)
    elif model_name == "UNet":
        C = 32
        n_layers = 4
        model = UNet(in_channels=in_channels, out_channels=out_channels,C=C, n_layers=n_layers).to(device)
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
                         head_dim=head_dim).to(device)
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
    early_stopping = EarlyStopping(patience=10, min_delta=0.0)
    print("Model created and moved to device.")

    # === Initializing the wandb ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="sparse-to-dense-RTMA",
            name=f"{variable}_{model_name}_{loss_name}",
            config={
                "variable": variable,
                "model": model_name,
                "optimizer": "Adam",
                "lr": optimizer.param_groups[0]["lr"],
                "loss_fn": loss_name,
                "metric": "MaskedRMSELoss",      # or set from args if you add support
                "epochs": num_epochs,
                "batch_size": batch_size,
                "transform": transform,
                "train_dates_range": train_dates_range,
                "scheduler": "ExponentialLR",
            }
        )

    # === Run the training and validation ===
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
        resume=resume
    )

    # === Finish run and destroy process group ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

# %%
if __name__ == "__main__":
    '''
    Sample command to run the script:
    torchrun --nproc_per_node=2 \
        train.py \
        --variable i10fg \
        --epochs 2 \
        --loss MaskedCharbonnierLoss \
        --model DCNN \
        --batch_size 64 \
        --transform standard
    '''
    main()

