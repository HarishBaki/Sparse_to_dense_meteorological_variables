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
from models.util import initialize_weights_xavier,initialize_weights_he

from losses import MaskedMSELoss, MaskedRMSELoss, MaskedTVLoss, MaskedCharbonnierLoss, MaskedCombinedMAEQuantileLoss

from sampler import DistributedEvalSampler

from util import str_or_none, int_or_none, bool_from_str, EarlyStopping, save_model_checkpoint, restore_model_checkpoint, init_zarr_store

# %%
def run_test(model, test_dataloader, test_dates_range, freq, mask, criterion, metric, device,
               checkpoint_dir, variable , target_zarr_store,target_transform=None):
    # load the best model Handle DDP 'module.' prefix
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    model, _, _, _ = restore_model_checkpoint(model, optimizer, scheduler, best_ckpt_path, device)

    # === Creating a zarr for test data ===
    dates = pd.date_range(start=test_dates_range[0], end=test_dates_range[1], freq=f'{freq}min')
    chunk_size = 24 * 60 // freq  # 24 hours in minutes divided by frequency

    init_zarr_store(target_zarr_store, dates, variable, chunk_size=chunk_size)
    print(f"Zarr store initialized at {target_zarr_store}.")
    
    # === Step 2: Evaluate and write predictions using matched time indices ===
    ds = xr.open_zarr(target_zarr_store, consolidated=False)
    zarr_time = ds['time'].values  # dtype=datetime64[ns]
    time_to_idx = {t: i for i, t in enumerate(zarr_time)}
    
    # Use low-level Zarr for writing directly
    zarr_write = zarr.open(target_zarr_store, mode='a')
    zarr_variable = zarr_write[variable]
    
    # === testing and saving into test zarr===
    model.eval()
    test_loss_total = 0.0
    test_metric_total = 0.0
    show_progress = True
    test_bar = tqdm(test_dataloader, desc=f"[Test]", leave=False) if show_progress else test_dataloader
    with torch.no_grad():
        for b,batch in enumerate(test_bar):
            #print(f"Processing batch {b+1}")
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
            # Apply masking, since we don't need the values outside NYS
            output = torch.where(torch.tensor(mask.values).to(device=device),output,0)
            # Clamping the output to be non-negative
            output = torch.clamp(output, min=0.0)
            target_tensor = torch.where(torch.tensor(mask.values).to(device=device),target_tensor,0)
            # Clamping the target tensor to be non-negative
            target_tensor = torch.clamp(target_tensor, min=0.0)

            # Compute the metric
            metric_value = metric(output, target_tensor, station_mask)
            test_metric_total += metric_value.item()

            if show_progress:
                test_bar.set_postfix(loss=loss.item(), metric=metric_value.item())

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
    avg_test_loss = test_loss_total / len(test_dataloader)
    avg_test_metric = test_metric_total / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f} | Test Metric: {avg_test_metric:.4f}")
    
    wandb.log({"Test Loss": avg_test_loss, "Test Metric": avg_test_metric})

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
            "--checkpoint_dir", "checkpoints",
            "--variable", "i10fg",
            "--model", "UNet",
            "--orography_as_channel", "true",
            "--additional_input_variables", "si10,t2m,sh2",
            "--train_years_range", "2018,2021",
            "--stations_seed", "42",
            "--n_random_stations", "none",
            "--randomize_stations_persample", "false",
            "--loss", "MaskedCharbonnierLoss",
            "--transform", "standard",
            "--epochs", "2",
            "--batch_size", "24",
            "--num_workers", "24",
            "--wandb_id", "none",
            # "--resume",  # Optional flag — include if you want to resume
            "--weights_seed", "42",
            "--activation_layer", "gelu",
            "--inference_stations_seed", "43",
            "--n_inference_stations", "100", 
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
    parser.add_argument("--stations_seed", type=int, default=42, help="Global seed for reproducibility")
    parser.add_argument("--n_random_stations", type=int_or_none, default=None, help="Number of random stations in each sample")
    parser.add_argument("--randomize_stations_persample", type=bool_from_str, default=False, 
                        help="Randomize stations for each sample: True or False")
    parser.add_argument("--loss", type=str, default="MaskedCharbonnierLoss", 
                        help="Loss function to use ('MaskedMSELoss', 'MaskedRMSELoss', 'MaskedTVLoss', 'MaskedCharbonnierLoss','MaskedCombinedMAEQuantileLoss')")
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
    parser.add_argument("--inference_stations_seed", type=int, default=42,
                        help="Seed for inference stations, used to select random stations if n_inference_stations is not None")
    parser.add_argument("--n_inference_stations", type=int_or_none, default=None,
                        help="Number of inference stations to use, if None, all stations will be used") 
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
    randomize_stations_persample = args.randomize_stations_persample
    stations_seed = args.stations_seed
    if not randomize_stations_persample:
        if n_random_stations is not None:
            checkpoint_dir = f"{checkpoint_dir}/{stations_seed}/{n_random_stations}-random-stations"
        else:
            checkpoint_dir = f"{checkpoint_dir}/all-stations"
    else:
        if n_random_stations is not None:
            checkpoint_dir = f"{checkpoint_dir}/{stations_seed}/{n_random_stations}-random-stations-per-sample"
        else:
            checkpoint_dir = f"{checkpoint_dir}/{stations_seed}/unknown-random-stations-per-sample"

    loss_name = args.loss
    checkpoint_dir = checkpoint_dir+'/'+loss_name

    transform = args.transform
    checkpoint_dir = checkpoint_dir+'/'+transform

    activation_layer = args.activation_layer
    weights_seed = args.weights_seed

    checkpoint_dir = checkpoint_dir+'/'+activation_layer+'-'+str(weights_seed)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    n_inference_stations = args.n_inference_stations
    inference_stations_seed = args.inference_stations_seed
    if n_inference_stations is not None:
        target_zarr_store = f'{checkpoint_dir}/{inference_stations_seed}/{n_inference_stations}-inference-stations/RTMA_test.zarr'
    else:
        target_zarr_store = f'{checkpoint_dir}/all-inference-stations/RTMA_test.zarr'
    os.makedirs(target_zarr_store, exist_ok=True)

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
    f"  stations_seed: {stations_seed}\n"
    f"  n_random_stations: {n_random_stations}\n"
    f" randomize_stations_persample: {randomize_stations_persample}\n"
    f" inference_stations_seed: {inference_stations_seed}\n"
    f"  n_inference_stations: {n_inference_stations}\n"
    f" target_zarr_store: {target_zarr_store}\n"
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
    # === Loading the RTMA data ===
    zarr_store = 'data/RTMA.zarr'
    validation_dates_range = ['2022-01-01T00', '2022-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time
    freq = 60  # Frequency in minutes, can be changed if needed
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
    elif loss_name == "MaskedCombinedMAEQuantileLoss":
        criterion = MaskedCombinedMAEQuantileLoss(mask_tensor, tau=0.95, mae_weight=0.5, quantile_weight=0.5)
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
                project="RTMA_inference",id=args.wandb_id,resume='allow',
            )
        else:
            wandb.init(
                project="RTMA_inference",
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
                    "stations_seed": stations_seed,
                    "n_random_stations": n_random_stations,
                    "orography_as_channel": orography_as_channel,
                    "activation_layer": args.activation_layer,
                    "weights_seed": args.weights_seed,
                    "inference_stations_seed": inference_stations_seed,
                    "n_inference_stations": n_inference_stations,
                    "randomize_stations_persample": randomize_stations_persample,
                }
            )
    # %%
    # === Run the test and save the outputs to zarr ===
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
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
            target_transform=target_transform,
            stations_seed=inference_stations_seed,  # This is a key change, since we may use different seed for inference stations.
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

        # %%
        print("Starting Testing...")
        run_test(
            model = model, 
            test_dataloader = test_dataloader,
            test_dates_range = test_dates_range,
            freq = freq,
            mask = mask,
            criterion = criterion,
            metric = metric,
            device = device,
            checkpoint_dir = checkpoint_dir,
            variable = variable, 
            target_zarr_store = target_zarr_store,
            target_transform = target_transform
        )
    
    # === Finish run and destroy process group ===
    if not dist.is_initialized() or dist.get_rank() == 0:
        wandb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    print("Finished testing and cleaned up.")

