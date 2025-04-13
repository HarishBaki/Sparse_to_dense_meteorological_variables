import xarray as xr
import numpy as np
import pandas as pd
import os, sys, time, glob, re

# set the directory to current directory
root_dir = '/data/harish'
os.chdir(root_dir)

# take yearmonthday folder name as input
date = sys.argv[1]
variable = sys.argv[2]

# read files in sorted order with keywords in the ascending order t00z, t01z, ... , t23z
files = glob.glob(f'{root_dir}/RTMA/{variable}/rtma/{date}/*')
def extract_hour(file):
    # Match the pattern 'tXXz' where XX is the hour (e.g., t00z, t01z, etc.)
    match = re.search(r't(\d{2})z', file)
    if match:
        return int(match.group(1))  # Return the hour as an integer
    return 0  # Default in case no match is found (although unlikely here)

# Sort the files by the extracted hour
sorted_files = sorted(files, key=extract_hour)

def preprocess(ds):
    return ds.isel(y=slice(830,830+256),x=slice(1740,1740+384))

ds = xr.open_mfdataset(sorted_files,concat_dim='time',combine='nested', parallel=True, preprocess=preprocess,
                       engine="cfgrib", backend_kwargs={'indexpath': ''})
ds.to_netcdf(f'{root_dir}/intermediate_files/{date}.nc')