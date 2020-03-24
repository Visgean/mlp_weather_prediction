import xarray as xr
import os


DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')

precipitation = xr.open_mfdataset(
    f'{DATADIR}10m_v_component_of_wind/*.nc',
    combine='by_coords'
)


mean = precipitation.mean(('time', 'lat', 'lon')).compute()
std = precipitation.std('time').mean(('lat', 'lon')).compute()

mean.to_netcdf('10m_v_component_of_wind_mean.nc')
std.to_netcdf('10m_v_component_of_wind_std.nc')
