import os
import train
import xarray as xr
import train_ltsm

DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')

print(DATADIR)

datasets = {
    'z': 'geopotential',
    't': 'temperature',
    'tp': 'total_precipitation',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
}

var_dict = {
    'z': [300, 400, 500, 600, 700],
    't': [500, 600, 700, 850, 1000],
    'tp': None,
    'u10': None,
    'v10': None,
}

generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])

data = []
means_all = []
stds_all = []

for var, name in datasets.items():
    print('loading', name)
    levels = var_dict[var]
    means = xr.load_dataarray(f'data/{name}_mean.nc')
    stds = xr.load_dataarray(f'data/{name}_std.nc')

    ds = xr.open_mfdataset(
        f'{DATADIR}{name}/*.nc',
        combine='by_coords',
        parallel=True,
        chunks={'time': 10}
    )

    if levels:
        means_all.extend(means.sel(level=levels).to_dict()['data'])
        stds_all.extend(stds.sel(level=levels).to_dict()['data'])
        data.append(ds[var].sel(level=levels))
    else:
        means_all.append(means.to_dict()['data'])
        stds_all.append(stds.to_dict()['data'])
        data.append(ds[var].expand_dims({'level': generic_level}, 1))

data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

# normalized = (data - means_all) / stds
# dp1 = data.sel(time='1989-01-01T23:00:00', lat=-87.1875, lon=11.25)
# dp_normalized = normalized.sel(time='1989-01-01T23:00:00', lat=-87.1875, lon=11.25)



if __name__ == '__main__':
    train_ltsm.train(
        ds=data,
        means=means_all,
        stds=stds_all,
        levels_per_variable=None,
        # levels_per_variable=var_dict,
        filters=[64, 64, 64, 64, 13],
        kernels=[5, 5, 5, 5, 5],
        lr=1e-4,
        activation='elu',
        dr=0,
        batch_size=64,
        patience=50,
        model_save_fn=OUT_DIR,
        pred_save_fn=os.path.join(OUT_DIR, 'predictions'),
        train_years=('1979', '2014'),
        valid_years=('2015', '2016'),
        test_years=('2017', '2018'),
        lead_time=3*24,
        seq_length=8,
        gpu=0,
        iterative=False,
        step_size=1
    )
