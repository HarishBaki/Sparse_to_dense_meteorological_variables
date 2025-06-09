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
from scipy.spatial import cKDTree

# %%
# Get the WRF simulation duration
cases=['case_1','case_2','case_3','case_4']  # <-- Define this as per your actual simulation case names
start_dates=['2023-02-02T12:00:00', '2023-03-25T00:00:00', '2023-04-01T00:00:00', '2023-12-17T06:00:00']
end_dates=['2023-02-04T00:00:00', '2023-03-26T12:00:00', '2023-04-02T12:00:00', '2023-12-18T18:00:00']
"""
In this, the simulation time step is not uniform. 
So, we need to extract data at specific times, could be 30s, 1min, or 5min.
"""

# Load NYSM station data
nysm = pd.read_csv('nysm.csv')

# %%
for case, start_date, end_date in zip(cases[1:], start_dates[1:], end_dates[1:]):
    print(f"Processing {case} from {start_date} to {end_date}")
    
    # Define the time range with 5min frequency
    # Convert to timestamps
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    time_index = pd.date_range(start=start_ts, end=end_ts,freq='5min')
    time_index = time_index[time_index >= start_ts + pd.Timedelta(hours=12)]

    # Store si10 values for each station
    df_dict = {}

    for idx, row in nysm.iterrows():
        stid = row['stid']
        try:
            df = pd.read_csv(
                f"WRF_simulations/{case}/WRF_run_1/{stid}.d02.TS",
                delim_whitespace=True,
                skiprows=1,
                header=None,
                low_memory=False,
                usecols=[1, 7, 8]
            )
            df.columns = ['ts_hour', 'U10', 'V10']
            df.set_index('ts_hour', inplace=True)
            si10 = np.sqrt(df['U10']**2 + df['V10']**2)

            df = si10.copy()
            # Step 1 Define 5-minute interval in hours
            interval = 5 / 60  # 0.083333...

            # Step 2: Generate the window start times
            start = np.floor(df.index.min())
            end = np.ceil(df.index.max())
            window_starts = np.arange(start, end, interval)

            # Step 3: Rolling-window max
            new_index = []
            max_values = []

            for t0 in window_starts:
                t1 = t0 + interval
                window_df = df[(df.index >= t0) & (df.index < t1)]
                if not window_df.empty:
                    max_val = window_df.max()
                    new_index.append(t0 + interval)  # assign to interval end time
                    max_values.append(max_val)

            # Step 4: Create a new dataframe
            df_5min_max = pd.DataFrame({'si10_max': max_values}, index=new_index)
            df_5min_max.index.name = 'ts_hour'

            df_dict[stid] = df_5min_max[12:].values.ravel()
        except FileNotFoundError:
            print(f"Missing file for station: {stid}")
            continue

    # Convert to xarray DataArray
    max_len = max(len(v) for v in df_dict.values())
    station_names = list(df_dict.keys())

    # Pad arrays to same length
    data = np.full((len(station_names), max_len), np.nan)
    for i, stid in enumerate(station_names):
        v = df_dict[stid]
        data[i, :len(v)] = v

    # Create xarray
    ds = xr.DataArray(
        data,
        dims=["station", "time"],
        coords={"station": station_names, "time": time_index},
    )

    target_location = f'WRF_simulations/tslist_outputs/{case}/WRF_run_1'
    os.makedirs(target_location, exist_ok=True)

    ds.to_netcdf(
        f"{target_location}/i10fg.nc",
        mode='w',
        format='NETCDF4',
        unlimited_dims='time'
    )
# %%
