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

from data_loader import RTMA_sparse_to_dense_Dataset, Transform
from models.Google_Unet import GoogleUNet
from models.Deep_CNN import DCNN
from models.UNet import UNet
from models.SwinT2_UNet import SwinT2UNet
from models.util import initialize_weights_xavier,initialize_weights_he

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss

from sampler import DistributedEvalSampler

# %%
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

def strip_ddp_prefix(state_dict):
    """Remove 'module.' prefix if loading DDP model into non-DDP."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def restore_model_checkpoint(model, optimizer, scheduler, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try stripping 'module.' prefix
        print("Warning: DDP prefix found, attempting to strip 'module.' from keys...")
        state_dict = strip_ddp_prefix(state_dict)
        model.load_state_dict(state_dict)
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Restored checkpoint from: {path} (epoch {checkpoint['epoch']})")
    return model, optimizer, scheduler, start_epoch

# %%
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