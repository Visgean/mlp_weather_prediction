import os

import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr

from data_gen import SelectiveDataGenerator
from weatherbench.score import *
from weatherbench.train_nn import DataGenerator, create_iterative_predictions, build_cnn, create_predictions


def evaluate(datadir, pred_filename, valid_years, ):
    # Load the geo500 and temp850 data and merge
    z = xr.open_mfdataset(
        f'{datadir}geopotential/*.nc',
        combine='by_coords'
    )

    pred = xr.open_mfdataset(pred_filename)

    z_rmse = compute_weighted_rmse(pred.z, z.to_array()).load().to_dict()['data']
    print(f'Geopotential at 500, rmse: {z_rmse}')


if __name__ == '__main__':
    DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')


    evaluate(
        datadir=DATADIR,
        pred_filename='/dropbox/Dropbox/2. mlp/trained-baseline/predictions',
        valid_years=('2017', '2018'),
    )
