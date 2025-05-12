# %%
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

import metpy.calc as mpcalc
from metpy.units import units

ds = xr.open_dataset('/data/NYSM/Meteo_5min/Meteo_5min_gapfilled.nc')
# %%
ds_subset = ds[['latitude','longitude',
         'temp_2m','relative_humidity','dewpoint','station_pressure',
         'avg_wind_speed_prop_sonic','max_wind_speed_merge']]

# Quantify variables with units
ds_q = ds_subset.metpy.quantify()

# Compute specific humidity
ds_q['sh2'] = mpcalc.specific_humidity_from_dewpoint(
    ds_q['station_pressure'],
    ds_q['dewpoint']
)

# Dequantify if you want to remove unit annotations for further use
ds_subset['sh2'] = ds_q['sh2'].metpy.dequantify()
# %%
ds_subset.to_netcdf('/data/harish/Sparse_to_dense_meteorological_variables/NYSM.nc')