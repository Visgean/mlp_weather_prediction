import os
import train
import xarray as xr


DATADIR = '/home/visgean/Downloads/weather-small/'

means = xr.load_dataarray('geopotential_mean_full.nc')
stds = xr.load_dataarray('geopotential_std_full.nc')

ds = xr.open_mfdataset(
    f'{DATADIR}geopotential/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 10}
)


if __name__ == '__main__':
    train.train(
        ds=ds,
        means=means,
        stds=stds,
        filters=[64, 64, 64, 64, 11],
        kernels=[5, 5, 5, 5, 5],
        lr=1e-4,
        activation='elu',
        dr=0,
        batch_size=128,
        patience=3,
        model_save_fn='./models/',
        pred_save_fn='./predictions/',
        train_years=('1979', '2015'),
        valid_years=('2016', '2016'),
        test_years=('2017', '2018'),
        lead_time=72,
        gpu=0,
        iterative=False,
        RAM_DOWNLOADED_FULLY=True,
    )
