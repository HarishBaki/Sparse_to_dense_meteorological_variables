import numpy as np
import pandas as pd
import xarray as xr
import dask
import os, sys
import glob
import zarr
from joblib import Parallel, delayed
import os
import dask.array as da
import os, sys, time, glob, re


if len(sys.argv) > 1:
    variable = sys.argv[1]
    mode = sys.argv[2]
    var_name = sys.argv[3]
else:
    # Set a default value for debugging
    variable = 'i10fg'  # Replace 'i10fg' with any variable you want to debug
    mode = 'w'  # Replace 'w' with 'a' if you want to append
    var_name = 'i10fg'

dates = pd.date_range(start='2018-01-01T00', end='2023-12-31T23', freq='h')
yyyymmdd = pd.Series(dates.year*10000 + dates.month*100 + dates.day).unique()
zarr_store = '/data/harish/Sparse_to_dense_meteorological_variables/RTMA.zarr'
data_source_dir = '/data/harish/RTMA'

def init_zarr_store(zarr_store, dates,var_name,mode):
    orography = xr.open_dataset('orography.nc')
    orography.attrs = {}
    template = orography.orog.pipe(xr.zeros_like).expand_dims(time=len(dates))
    template['time'] = dates
    template = template.chunk({'time': 24})
    template = template.transpose('time','y','x')
    template = template.assign_coords({
        'latitude': orography.latitude,
        'longitude': orography.longitude
    })
    template.to_dataset(name = var_name).to_zarr(zarr_store, compute=False, consolidated=True, mode=mode)

init_zarr_store(zarr_store, dates, var_name, mode)

def daily_processing(variable,date):
    # define the region of interest
    nx, ny = 288, 256
    x_start, x_end = 1800, 1800 + nx
    y_start, y_end = 830, 830 + ny

    # read files in sorted order with keywords in the ascending order t00z, t01z, ... , t23z
    files = glob.glob(f'{data_source_dir}/{variable}/rtma/{date}/*')
    def extract_hour(file):
        # Match the pattern 'tXXz' where XX is the hour (e.g., t00z, t01z, etc.)
        match = re.search(r't(\d{2})z', file)
        if match:
            return int(match.group(1))  # Return the hour as an integer
        return 0  # Default in case no match is found (although unlikely here)

    # Sort the files by the extracted hour
    sorted_files = sorted(files, key=extract_hour)

    def preprocess(ds):
        return ds.isel(y=slice(y_start,y_end),x=slice(x_start,x_end))

    ds = xr.open_mfdataset(sorted_files,concat_dim='time',combine='nested', parallel=True, preprocess=preprocess,
                        engine="cfgrib", backend_kwargs={'indexpath': ''})
    
    # Convert date to string (yyyymmdd)
    date_str = str(date)
    # Get all 24 hours for that day
    full_times = pd.date_range(start=pd.to_datetime(date_str, format="%Y%m%d"), periods=24, freq="h")

    # Sometimes time values are missing or unordered
    if ds.time.size < 24 or not np.array_equal(
            ds.time.values.astype("datetime64[ns]").astype("int64"),
            full_times.values[:ds.time.size].astype("datetime64[ns]").astype("int64")
        ):
        ds = ds.reindex(time=full_times)        # pad missing hours with NaNs

    ds = ds.chunk({'time': 24, 'y': ny, 'x': nx})
    return ds

def write_chunk(ds_chunk, zarr_store, region):
    """
    Function to write a single chunk to the Zarr store.
    """
    ds_chunk.to_zarr(zarr_store, region=region, mode='a')

def process_and_write_single_day(date, variable, zarr_store):
    try:
        ds = daily_processing(variable, date)
        time_indices = np.searchsorted(dates.values, ds.time.values)
        region = {"time": slice(time_indices[0], time_indices[-1] + 1)}
        to_drop = ['time', 'valid_time', 'surface', 'heightAboveGround', 'step','latitude','longitude']
        existing_vars = [var for var in to_drop if var in ds.variables]
        ds_chunk = ds.drop_vars(existing_vars)
        write_chunk(ds_chunk, zarr_store, region)
    except Exception as e:
        print(f"Failed to process date {date}: {e}")

batch_size = 30
for i in range(0, len(yyyymmdd), batch_size):
    batch_dates = yyyymmdd[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}: {batch_dates[0]} to {batch_dates[-1]}")
    
    Parallel(n_jobs=os.cpu_count(), verbose=10)(
        delayed(process_and_write_single_day)(date, variable, zarr_store)
        for date in batch_dates
    )