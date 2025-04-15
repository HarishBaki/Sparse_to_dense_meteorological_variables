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
import pandas as pd
import time

import os
import wandb, argparse
from tqdm import tqdm

from data_loader import RTMA_sparse_to_dense_Dataset
from models.Google_Unet import UNet

# %%
# === training, validation, and checkpointing functions ===
def training_step(model, batch, criterion, optimizer, device):
    model.train()
    input_tensor, target_tensor = batch
    input_tensor = input_tensor.to(device, non_blocking=True)
    target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    return loss.item()

def validation_step(model, batch, criterion, device):
    model.eval()
    input_tensor, target_tensor = batch
    input_tensor = input_tensor.to(device, non_blocking=True)
    target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)

    with torch.no_grad():
        output = model(input_tensor)
        loss = criterion(output, target_tensor)

    return loss.item()

def save_model_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, path)
    print(f" Model checkpoint saved at: {path}")

def restore_model_checkpoint(model, optimizer, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Restored checkpoint from: {path} (epoch {checkpoint['epoch']})")
    return model, optimizer, start_epoch

# %%

def run_epochs(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs,
               checkpoint_dir, train_sampler, resume=False):

    os.makedirs(checkpoint_dir, exist_ok=True)

    # === Optional resume ===
    start_epoch = 0
    best_val_loss = float("inf")
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")

    if resume and os.path.exists(latest_ckpt_path):
        model, optimizer, start_epoch = restore_model_checkpoint(model, optimizer, latest_ckpt_path, device)

    for epoch in range(start_epoch, num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # === Training ===
        model.train()
        train_loss_total = 0.0
        show_progress = not dist.is_initialized() or dist.get_rank() == 0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False) if show_progress else train_dataloader
        for batch in train_bar:
            input_tensor, target_tensor,_ = batch
            input_tensor = input_tensor.to(device, non_blocking=True)
            target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            if show_progress:
                train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_total / len(train_dataloader)

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False) if show_progress else val_dataloader
        with torch.no_grad():
            for batch in val_bar:
                input_tensor, target_tensor,_ = batch
                input_tensor = input_tensor.to(device, non_blocking=True)
                target_tensor = target_tensor.unsqueeze(1).to(device, non_blocking=True)

                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                val_loss_total += loss.item()
                if show_progress:
                    val_bar.set_postfix(loss=loss.item())
        avg_val_loss = val_loss_total / len(val_dataloader)

        print(f"[Epoch {epoch+1}]  Train Loss: {avg_train_loss:.4f} |  Val Loss: {avg_val_loss:.4f}")

        # === Log to Weights & Biases ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })

        # === Save Checkpoints ===
        if not dist.is_initialized() or dist.get_rank() == 0:
            epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pt")
            save_model_checkpoint(model, optimizer, epoch, epoch_ckpt_path)

            # Update best model if needed
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_model_checkpoint(model, optimizer, epoch, best_ckpt_path)

            # Always update latest checkpoint
            save_model_checkpoint(model, optimizer, epoch, latest_ckpt_path)


def main():
    # === Argparse and DDP setup ===
    parser = argparse.ArgumentParser(description="Train with DDP")

    parser.add_argument("--variable", type=str, default="i10fg", help="Variable to train on")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    args = parser.parse_args()
    variable = args.variable
    num_epochs = args.epochs
    resume = args.resume
    checkpoint_dir = args.checkpoint_dir+'/'+variable

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
    zarr_store = 'data/RTMA.zarr'
    train_dates_range = ['2018-01-01T00', '2021-12-31T23']
    validation_dates_range = ['2022-01-01T00', '2022-12-31T23']
    test_dates_range = ['2023-01-01T00', '2023-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time
    batch_size = 32

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
        missing_times
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
        missing_times
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
        missing_times
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
    in_channels = 2
    out_channels = 1
    model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("Model created and moved to device.")

    # === Initializing the wandb ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.init(
            project="sparse-to-dense-RTMA",
            name=variable+'_rank'+str(rank),
            config={
                "model": "UNet",
                "optimizer": "Adam",
                "lr": optimizer.param_groups[0]["lr"],
                "loss_fn": "MSE",
                "epochs": num_epochs,
                "batch_size": train_dataloader.batch_size,
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
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        train_sampler=train_sampler
    )

    # === Finish run and destroy process group ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()

# %%
if __name__ == "__main__":
    main()