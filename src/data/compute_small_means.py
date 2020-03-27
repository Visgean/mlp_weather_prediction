import xarray as xr
import os


DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')

# Load the geo500 and temp850 data and merge
z = xr.open_mfdataset(
    f'{DATADIR}geopotential_500/*.nc',
    combine='by_coords'
)
t = xr.open_mfdataset(
    f'{DATADIR}temperature_850/*.nc',
    combine='by_coords'
)
ds = xr.merge([z, t], compat='override')


var_dict = {'z': [500], 't': [850]}

data = []
generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
for var, levels in var_dict.items():
    try:
        data.append(ds[var].sel(level=levels))
    except ValueError:
        data.append(ds[var].expand_dims({'level': generic_level}, 1))

data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

mean = data.mean(('time', 'lat', 'lon')).compute()
std = data.std('time').mean(('lat', 'lon')).compute()

mean.to_netcdf('baseline_mean.nc')
std.to_netcdf('baseline_std.nc')

import ipdb
ipdb.set_trace()
