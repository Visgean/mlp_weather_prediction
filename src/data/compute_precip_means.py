import xarray as xr
import os


DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')

precipitation = xr.open_mfdataset(
    f'{DATADIR}total_precipitation/*.nc',
    combine='by_coords'
)


mean = precipitation.mean(('time', 'lat', 'lon')).compute()
std = precipitation.std('time').mean(('lat', 'lon')).compute()

mean.to_netcdf('precipitation_mean.nc')
std.to_netcdf('precipitation_std.nc')
