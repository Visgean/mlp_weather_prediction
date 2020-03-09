import os

import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr

from data_gen import SelectiveDataGenerator
from weatherbench.score import *
from weatherbench.train_nn import DataGenerator, create_iterative_predictions, build_cnn, create_predictions


def evaluate(datadir, pred_save_fn, valid_years, ):
    ds = xr.open_mfdataset(
        f'{datadir}geopotential/*.nc',
        combine='by_coords',
        parallel=True,
        chunks={'time': 10}
    )

    ds_valid = ds.sel(time=slice(*valid_years))

    pred = xr.open_mfdataset(pred_save_fn)

    levels = [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]

    for level in levels:
        level_pred = pred.sel(level=level).to_array()
        ds_valid_array = ds_valid.sel(level=level).to_array()

        rmse = compute_weighted_rmse(level_pred, ds_valid_array).load().to_dict()['data']
        print(f'level {level}, rmse: {rmse}')


if __name__ == '__main__':
    DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
    means = xr.load_dataarray('mean_full.nc')
    stds = xr.load_dataarray('std_full.nc')

    evaluate(
        datadir=DATADIR,
        pred_save_fn='../models/full-geopotential/predictions',
        valid_years=('2017', '2018'),
    )
