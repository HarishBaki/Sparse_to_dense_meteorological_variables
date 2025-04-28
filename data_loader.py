# %%
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pandas as pd
import time
import os

class Transform:
    def __init__(self, mode, stats, channel_indices=None):
        """
        mode: 'standard' or 'minmax'
        stats: 
            if mode == 'standard': {'mean': [...], 'std': [...]}
            if mode == 'minmax': {'min': [...], 'max': [...]}
        channel_indices: list of channel indices to apply transform to. 
        If tranforming input channels, then the channel_indices will be [0,1] if only working with one variable + orography, otherwise it will be [0,1,2,...]
        If transforming target channels, then the channel_indices will be [0] always.
        """
        self.mode = mode
        self.channel_indices = channel_indices
        self.stats = stats

        if mode == "standard":
            self.mean = stats['mean'].values
            self.std = stats['std'].values
        elif mode == "minmax":
            self.min = stats['min'].values
            self.max = stats['max'].values
        else:
            raise ValueError("mode must be 'standard' or 'minmax'")

    def __call__(self, x):
        C = x.shape[0]
        indices = self.channel_indices or range(C)

        for i in indices:
            if self.mode == "standard":
                x[i] = (x[i] - self.mean[i]) / (self.std[i] + 1e-8)
            elif self.mode == "minmax":
                x[i] = (x[i] - self.min[i]) / (self.max[i] - self.min[i] + 1e-8)
        return x

    def inverse(self, x):
        C = x.shape[0]
        indices = self.channel_indices or range(C)

        for i in indices:
            if self.mode == "standard":
                x[i] = x[i] * self.std[i] + self.mean[i]
            elif self.mode == "minmax":
                x[i] = x[i] * (self.max[i] - self.min[i]) + self.min[i]
        return x


class RTMA_sparse_to_dense_Dataset(Dataset):
    def __init__(self, zarr_store,input_variables_in_order, dates_range, orography, 
                 RTMA_lat, RTMA_lon, nysm_latlon, y_indices, 
                 x_indices,mask,missing_times,input_transform=None,target_transform=None, n_random_stations=None,global_seed=42):
        '''
        Custom dataset class for loading RTMA data.
        This dataset is designed to work with the RTMA data stored in Zarr format.
        Parameters:
        - zarr_store (str): Path to the Zarr store containing RTMA data.
        - input_variables_in_order (list): Names of the variables to load from the Zarr store, in which the first one itself is the target. even for a single variable, it should be a list of length 1.
        - dates_range (list): List containing start and end dates for filtering the data.
        - orography (ndarray): Orography data for the region.
        - RTMA_lat (ndarray): Latitude values for the RTMA grid.
        - RTMA_lon (ndarray): Longitude values for the RTMA grid.
        - nysm_latlon (ndarray): Latitude and longitude values for the NYSM stations.
        - y_indices (ndarray): Y indices of the NYSM stations in the RTMA grid.
        - x_indices (ndarray): X indices of the NYSM stations in the RTMA grid.
        - mask (ndarray): Mask for the RTMA grid.
        - missing_times (ndarray): Array of missing times in the dataset.
        - input_transform (callable, optional): Transformation to apply to the input data.
        - target_transform (callable, optional): Transformation to apply to the target data.
        - n_random_stations (int, optional): Number of random stations to select from the NYSM stations.
        - global_seed (int, optional): Seed for random number generation.
        Returns:
        - input_tensor (Tensor): Input tensor containing interpolated station values, orography, and station locations mask. shape [batch, 3, y, x]
        - target_tensor (Tensor): Target tensor containing the RTMA data. shape [batch, y, x]
        - time_instance (str): Time instance of the data sample.
        '''
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.zarr_store = zarr_store
        self.input_variables_in_order = input_variables_in_order
        self.dates_range = dates_range
        self.RTMA_lat = RTMA_lat
        self.RTMA_lon = RTMA_lon
        self.orography = np.expand_dims(orography, axis=0)  # shape: [1, y, x], should be compatable with the input tensor
        self.mask = mask
        self.nysm_latlon = nysm_latlon
        self.y_indices = y_indices
        self.x_indices = x_indices
        self.global_seed = global_seed
        self.n_random_stations = n_random_stations

        # Pre-select the time range
        ds = xr.open_zarr(zarr_store)[input_variables_in_order] # this is needed, since we might be working with order of variables. 
        ds = ds.sel(time=slice(dates_range[0], dates_range[1]))
        #print(f"Total samples in the dataset: {len(ds.time)}")

        # Filter the valid indices based on ds.time and missing_times
        valid_times = ds["time"].where(~ds["time"].isin(missing_times))
        self.valid_indices = np.where(~pd.isnull(valid_times.values))[0]    # valid_indices are the actual indices in ds, but omitted the missing times

        #print(f"Total valid samples: {len(self.valid_indices)}")
        self.ds = ds

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        target = self.ds[self.input_variables_in_order[0]].isel(time=real_idx)    # an xarray DataArray, shape: [ y, x]. Can access the values directly.
        input = self.ds[self.input_variables_in_order].isel(time=real_idx)  # an xarray Dataset, shape: [ y, x] with number of input variables. Cannot directly extract the values. 

        y_indices = self.y_indices
        x_indices = self.x_indices
        nysm_latlon = self.nysm_latlon
        # Check for the n_random_stations
        if self.n_random_stations is not None:
            # Randomly select n_random_stations from the NYSM stations
            rng = np.random.default_rng(self.global_seed + idx) # The dataset index will always pick the same random stations for that idx, regardless of which worker, GPU, or process loads it.
            perm = rng.permutation(len(self.nysm_latlon))
            #random_indices = rng.choice(len(self.nysm_latlon), self.n_random_stations, replace=False)
            random_indices = perm[:self.n_random_stations]  # This will give same first n random indices for n_random_stations = 40, 50, 60, ...
            # Update the y_indices and x_indices to the random stations
            y_indices = self.y_indices[random_indices]
            x_indices = self.x_indices[random_indices]
            nysm_latlon = self.nysm_latlon[random_indices]
        print(f"{idx} First 2 random nysm latlon: {nysm_latlon[:2]}")
        print(f"{idx} First 2 y_indices: {y_indices[:2]}")
        print(f"{idx} First 2 x_indices: {x_indices[:2]}")

        station_mask = np.zeros_like(RTMA_lat, dtype=np.uint8)
        # Set 1 at the station locations
        station_mask[y_indices, x_indices] = 1
        station_mask = np.expand_dims(station_mask, axis=0)  # shape: [1, y, x], should be compatable with the input tensor

        # Grab station values from input for all input variables
        inputs_interp = []
        for i, var in enumerate(self.input_variables_in_order):
            station_values = input[var].values[y_indices, x_indices]
            # Interpolate station values to full grid
            interp = griddata(
                nysm_latlon,
                station_values,
                (self.RTMA_lat, self.RTMA_lon),
                method='nearest'
            )
            inputs_interp.append(interp)
        # Stack the interpolated inputs
        interp = np.stack(inputs_interp, axis=0) # shape: [num_input_variables, y, x] 
        
        # Combine inputs: interpolated + orography
        input_tensor = np.concatenate([interp, self.orography,station_mask], axis=0)  # shape: [num_input_variables+2, y, x]
        # Apply mask
        input_tensor = np.where(self.mask, input_tensor, 0)
        target_tensor = np.where(self.mask, target.values, 0)

        # Convert to torch tensors
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        target_tensor = torch.tensor(target_tensor, dtype=torch.float32).unsqueeze(0)  # shape: [1, H, W]

        # Apply transformations if provided
        if self.input_transform is not None:
            input_tensor = self.input_transform(input_tensor)
        if self.target_transform is not None:
            target_tensor = self.target_transform(target_tensor)

        # Return input and target tensors
        return input_tensor, target_tensor, str(target.time.values)

# %%
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
    additional_input_variables = ['t2m','d2m','si10','sh2']
    dates_range = ['2023-11-09T06','2023-11-10T13']
    missing_times = xr.open_dataset(f'nan_times_{variable}.nc').time
    # if the additional input variables is not none, add the missing times of the additional input variables also. 
    if additional_input_variables is not None:
        for var in additional_input_variables:
            missing_times = xr.concat([missing_times, xr.open_dataset(f'nan_times_{var}.nc').time], dim='time')
        # remove duplicates
        missing_times = missing_times.drop_duplicates('time')
    print(f"Missing times shape: {missing_times.shape}")

    # Read stats of RTMA data
    '''
    A key point, the stats are designed to extract based on variable names, in braces. Even a single variable name is treated as a list of length 1.
    While, the Transform class is designed to extract based on the channel indices. For target, it will be [0].
    On the other hand, the RTMA_sparse_to_dense_Dataset class is designed to work with variable names, such that the first variable is the target variable. 
    '''
    RTMA_stats = xr.open_dataset('RTMA_variable_stats.nc')
    input_variables_in_order = [variable] if additional_input_variables is None else [variable]+additional_input_variables  
    target_variables_in_order = [variable]
    input_stats = RTMA_stats.sel(variable=input_variables_in_order+['orography'])     
    input_channnel_indices = list(range(len(input_variables_in_order+['orography'])))
    target_stats = RTMA_stats.sel(variable=target_variables_in_order)  
    target_channnel_indices = list(range(len(target_variables_in_order)))
    # Standardization
    input_transform = Transform(
        mode="standard",  # 'standard' or 'minmax'
        stats=input_stats,
        channel_indices=input_channnel_indices
    )
    target_transform = Transform(
        mode="standard",  # 'standard' or 'minmax'
        stats=target_stats,
        channel_indices=target_channnel_indices
    )

    # Now, setting up the random seed for reproducibility
    global_seed = 42    
    n_random_stations = None    # If None, all the stations are taken without any randomness. Else, randomly n_random_stations are selected. 

    # %%
    # Examining the batches without transformations
    # compute time taken call dataset
    start_time = time.time()
    # Create dataset instance
    dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        input_variables_in_order,
        dates_range,
        orography,
        RTMA_lat,
        RTMA_lon,
        nysm_latlon,
        y_indices,
        x_indices,
        mask,
        missing_times,
        input_transform=None,
        target_transform=None,
        global_seed=global_seed,
        n_random_stations=n_random_stations
    )
    end_time = time.time()
    print(f"Dataset creation time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=2, pin_memory=True)
    end_time = time.time()
    print(f"Dataloader time: {end_time - start_time:.2f} seconds")

    iterator = iter(dataloader)
    # Example usage
    for b in range(3):
        start_time = time.time()
        batch = next(iterator, None)

        if batch is not None:
            input_tensor, target_tensor, time_instance = batch
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            print(f"\n Batch {b+1}")
            print("Input tensor shape:", input_tensor.shape)
            print("Target tensor shape:", target_tensor.shape)

            for i in range(input_tensor.shape[1]):
                print(f"Input Channel {i} ➜ max: {input_tensor[0, i].max().item():.4f}, min: {input_tensor[0, i].min().item():.4f}")
            print(f"Target ➜ max: {target_tensor[0,0].max().item():.4f}, min: {target_tensor[0,0].min().item():.4f}")

        else:
            print(f"Batch {b+1}: No data in this batch.")
        
        end_time = time.time()
        print(f" DataLoader iteration time: {end_time - start_time:.2f} seconds")
    # %%
    # Examining the batches with transformations
    # compute time taken call dataset
    start_time = time.time()
    # Create dataset instance
    dataset = RTMA_sparse_to_dense_Dataset(
        zarr_store,
        input_variables_in_order,
        dates_range,
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
        global_seed=global_seed,
        n_random_stations=n_random_stations
    )
    end_time = time.time()
    print(f"Dataset creation time: {end_time - start_time:.2f} seconds")

    start_time = time.time()
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=2, pin_memory=True)
    end_time = time.time()
    print(f"Dataloader time: {end_time - start_time:.2f} seconds")

    iterator = iter(dataloader)
    # Example usage
    for b in range(3):
        start_time = time.time()
        batch = next(iterator, None)

        if batch is not None:
            input_tensor, target_tensor, time_instance = batch
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            print(f"\n Batch {b+1}")
            print("Input tensor shape:", input_tensor.shape)
            print("Target tensor shape:", target_tensor.shape)

            for i in range(input_tensor.shape[1]):
                print(f"Input Channel {i} ➜ max: {input_tensor[0, i].max().item():.4f}, min: {input_tensor[0, i].min().item():.4f}")
            print(f"Target ➜ max: {target_tensor[0,0].max().item():.4f}, min: {target_tensor[0,0].min().item():.4f}")

            # Inverse transform for input and target
            input_tensor_inv = input_transform.inverse(input_tensor[0])     # shape: [3, y, x]
            target_tensor_inv = target_transform.inverse(target_tensor[0])  # shape: [1, y, x]
            print("Inverse transformed tensors shapes:", input_tensor_inv.shape, target_tensor_inv.shape)
            for i in range(input_tensor.shape[1]):
                print(f"Inverse Input Channel 0 ➜ max: {input_tensor_inv[i].max().item():.4f}, min: {input_tensor_inv[i].min().item():.4f}")
            print(f"Inverse Target ➜ max: {target_tensor_inv.max().item():.4f}, min: {target_tensor_inv.min().item():.4f}")

        else:
            print(f"Batch {b+1}: No data in this batch.")
        
        end_time = time.time()
        print(f" DataLoader iteration time: {end_time - start_time:.2f} seconds")
    # %%
