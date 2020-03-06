from tensorflow.keras.layers import *
from weatherbench.score import *
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, Lambda
import tensorflow.keras.backend as K

from weatherbench.train_nn import DataGenerator, create_iterative_predictions, limit_mem, build_cnn, create_predictions


assert tf.test.is_gpu_available()



DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')


def train(datadir, filters, kernels, lr, activation, dr, batch_size,
          patience, model_save_fn, pred_save_fn, train_years, valid_years,
          test_years, lead_time, gpu, iterative, RAM_DOWNLOADED_FULLY=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Limit TF memory usage
    # limit_mem()

    # Open dataset and create data generators
    # TODO: Flexible input data
    ds = xr.open_mfdataset(
        f'{datadir}geopotential/*.nc',
        combine='by_coords',
        parallel=True,
        chunks={'time': 10}
    )

    # TODO: Flexible valid split
    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    geo_levels = [1, 10, 100, 200, 300, 400, 500, 600, 700, 850, 1000]

    levels_per_variable = {'z': geo_levels}

    print("Loading training data")
    dg_train = DataGenerator(
        ds_train,
        levels_per_variable,
        lead_time,
        batch_size=batch_size,
        load=RAM_DOWNLOADED_FULLY
    )

    print("Loading validation data")
    dg_valid = DataGenerator(
        ds_valid,
        levels_per_variable,
        lead_time,
        batch_size=batch_size,
        mean=dg_train.mean,
        std=dg_train.std,
        shuffle=False,
        load=RAM_DOWNLOADED_FULLY
    )
    print("Loading test data")
    dg_test = DataGenerator(
        ds_test,
        levels_per_variable,
        lead_time,
        batch_size=batch_size,
        mean=dg_train.mean,
        std=dg_train.std,
        shuffle=False,
        load=RAM_DOWNLOADED_FULLY
    )

    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    # Build model
    # TODO: Flexible input shapes and optimizer
    model = build_cnn(filters, kernels, input_shape=(32, 64, len(geo_levels)), activation=activation, dr=dr)
    model.compile(keras.optimizers.Adam(lr), 'mse')
    print(model.summary())

    # Train model
    # TODO: Learning rate schedule
    model.fit_generator(dg_train, epochs=100, validation_data=dg_valid,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=patience,
                            verbose=1,
                            mode='auto'
                        )]
                        )
    print(f'Saving model weights: {model_save_fn}')
    model.save_weights(model_save_fn)

    # Create predictions
    pred = create_iterative_predictions(model, dg_test) if iterative else create_predictions(model, dg_test)
    print(f'Saving predictions: {pred_save_fn}')
    pred.to_netcdf(pred_save_fn)

    ds_valid_array = ds_valid.to_array()

    if iterative:
        print(evaluate_iterative_forecast(pred, ds_valid).load())
    else:
        print(compute_weighted_rmse(pred, ds_valid_array).load())


if __name__ == '__main__':
    train(
        datadir=DATADIR,
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
        RAM_DOWNLOADED_FULLY=False,
    )
