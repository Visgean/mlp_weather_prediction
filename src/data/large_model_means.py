import xarray as xr
import os

DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')

# Load the geo500 and temp850 data and merge
# z = xr.open_mfdataset(
#     f'{DATADIR}geopotential/*.nc',
#     combine='by_coords',
#     parallel=True,
#     chunks={'time': 10}
#
# )
# t = xr.open_mfdataset(
#     f'{DATADIR}temperature/*.nc',
#     combine='by_coords',
#     parallel=True,
#     chunks={'time': 10}
#
# )
tp = xr.open_mfdataset(
    f'{DATADIR}total_precipitation/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 5}

)

wu = xr.open_mfdataset(
    f'{DATADIR}10m_u_component_of_wind/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 5}

)
wv = xr.open_mfdataset(
    f'{DATADIR}10m_v_component_of_wind/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 5}

)

ds = xr.merge([tp, wu, wv], compat='override')
# ds.load()

var_dict = {
    # 'z': [300, 400, 500, 600, 700],
    # 't': [500, 600, 700, 850, 1000],
    'tp': None,
    'u10': None,
    'v10': None,
}

data = []
generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
for var, levels in var_dict.items():
    if levels:
        data.append(ds[var].sel(level=levels))
    else:
        data.append(ds[var].expand_dims({'level': generic_level}, 1))

data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

mean = data.mean(('time', 'lat', 'lon')).compute()
std = data.std('time').mean(('lat', 'lon')).compute()

mean.to_netcdf('large_model_mean_full.nc')
std.to_netcdf('large_model_std_full.nc')
