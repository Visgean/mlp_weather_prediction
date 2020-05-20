import os
import train
import xarray as xr


# import tensorflow as tf
# assert tf.test.is_gpu_available()


means = xr.load_dataarray('data/geopotential_mean_full.nc')
stds = xr.load_dataarray('data/geopotential_std_full.nc')


DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR')



# TODO: Flexible input data
ds = xr.open_mfdataset(
    f'{DATADIR}geopotential/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 10}
)

geo_levels = [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]
levels_per_variable = {'z': geo_levels}

if __name__ == '__main__':
    train.train(
        ds=ds,
        means=means,
        stds=stds,
        levels_per_variable=levels_per_variable,
        filters=[64, 64, 64, 64, 11],
        kernels=[5, 5, 5, 5, 5],
        lr=1e-4,
        activation='elu',
        dr=0,
        batch_size=128,
        patience=50,
        model_save_fn=OUT_DIR,
        pred_save_fn=os.path.join(OUT_DIR, 'predictions'),
        train_years=('1979', '2014'),
        valid_years=('2015', '2016'),
        test_years=('2017', '2018'),
        lead_time=72,
        gpu=0,
        iterative=False,
        RAM_DOWNLOADED_FULLY=False,
    )
