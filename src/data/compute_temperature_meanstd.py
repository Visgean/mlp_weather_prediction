import xarray as xr

DATADIR = os.getenv('DATASET_DIR', '/home/s1652610/')
ds = xr.open_mfdataset(f'{DATADIR}temperature/*.nc',
                       combine='by_coords', parallel=True, chunks={'time': 10})

temp_levels = [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]
var_dict = {'t': temp_levels}

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

mean.to_netcdf('temperature_mean_full.nc')
std.to_netcdf('temperature_std_full.nc')
