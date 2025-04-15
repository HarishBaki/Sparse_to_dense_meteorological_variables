import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time

class RTMA_sparse_to_dense_Dataset(Dataset):
    def __init__(self, zarr_store,variable, dates_range, orography, RTMA_lat, RTMA_lon, nysm_latlon, y_indices, x_indices,mask,missing_times):
        self.zarr_store = zarr_store
        self.variable = variable
        self.dates_range = dates_range
        self.RTMA_lat = RTMA_lat
        self.RTMA_lon = RTMA_lon
        self.orography = orography
        self.mask = mask
        self.nysm_latlon = nysm_latlon
        self.y_indices = y_indices
        self.x_indices = x_indices

        # Pre-select the time range
        ds = xr.open_zarr(zarr_store)[variable]
        ds = ds.sel(time=slice(dates_range[0], dates_range[1]))
        #print(f"Total samples in the dataset: {len(ds.time)}")

        # Filter the valid indices based on ds.time and missing_times

        valid_times = ds["time"].where(~ds["time"].isin(missing_times))
        self.valid_indices = np.where(~pd.isnull(valid_times.values))[0]

        #print(f"Total valid samples: {len(self.valid_indices)}")
        self.ds = ds

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        output = self.ds.isel(time=real_idx)

        # Grab station values from output
        station_values = output.values[self.y_indices, self.x_indices]

        # Interpolate station values to full grid
        interp = griddata(
            self.nysm_latlon,
            station_values,
            (self.RTMA_lat, self.RTMA_lon),
            method='nearest'
        )

        # Combine inputs: interpolated + orography
        input_tensor = np.stack([interp, self.orography], axis=0)  # shape: [2, y, x]

        # Apply mask
        input_tensor = np.where(self.mask, input_tensor, 0)
        target_tensor = np.where(self.mask, output.values, 0)

        # Convert to torch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32)
        # Return input and target tensors
        return input_tensor, target_tensor, str(output.time.values)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Everything here only runs when you execute this file directly
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

    zarr_store = 'data/RTMA.zarr'
    variable = 'i10fg'
    dates_range = ['2018-01-01T00','2021-12-31T23']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time

    # compute time taken call dataset
    start_time = time.time()
    # Create dataset instance
    dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        variable,
        dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times
    )
    end_time = time.time()
    print(f"Dataset creation time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,num_workers=2, pin_memory=True)
    end_time = time.time()
    print(f"Dataloader time: {end_time - start_time:.2f} seconds")

    # Example usage
    for i, (input_tensor, target_tensor,_) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        print(f"Batch {i+1}:")
        print("Input tensor shape:", input_tensor.shape)
        print("Target tensor shape:", target_tensor.shape)
        # Break after first batch for demonstration
        break
    end_time = time.time()
    print(f"DataLoader iteration time: {end_time - start_time:.2f} seconds")

    # Check time taken for another iteration
    start_time = time.time()
    for i, (input_tensor, target_tensor,_) in enumerate(dataloader):
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        print(f"Batch {i+1}:")
        print("Input tensor shape:", input_tensor.shape)
        print("Target tensor shape:", target_tensor.shape)
        # Break after first batch for demonstration
        break
    end_time = time.time()
    print(f"DataLoader iteration time: {end_time - start_time:.2f} seconds")