import os

import tensorflow as tf
import tensorflow.keras as keras

from data_gen import SelectiveDataGenerator
from weatherbench.score import *
from weatherbench.train_nn import DataGenerator, create_iterative_predictions, build_cnn, create_predictions


def train(ds, filters, kernels, lr, activation, dr, batch_size,
          patience, model_save_fn, pred_save_fn, train_years, valid_years,
          test_years, lead_time, gpu, iterative, means, stds,
          levels_per_variable, RAM_DOWNLOADED_FULLY=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Limit TF memory usage
    # limit_mem()

    # TODO: Flexible valid split
    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    print("Loading training data")
    dg_train = SelectiveDataGenerator(
        ds_train,
        levels_per_variable,
        lead_time,
        mean=means,
        std=stds,
        batch_size=batch_size,
        load=RAM_DOWNLOADED_FULLY
    )

    print("Loading validation data")
    dg_valid = DataGenerator(
        ds_valid,
        levels_per_variable,
        lead_time,
        batch_size=batch_size,
        mean=means,
        std=stds,
        shuffle=False,
        load=True
    )

    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    # Build model
    # TODO: Flexible input shapes and optimizer
    # Get the number of levels from the first variable (should be only one)
    num_levels = len(list(levels_per_variable.values())[0])
    model = build_cnn(filters, kernels, input_shape=(32, 64, num_levels), activation=activation, dr=dr)
    model.compile(keras.optimizers.Adam(lr), 'mse')
    print(model.summary())

    # Train model
    # TODO: Learning rate schedule
    model.fit_generator(dg_train, epochs=500, validation_data=dg_valid,
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

    print("Loading test data")
    dg_test = DataGenerator(
        ds_test,
        levels_per_variable,
        lead_time,
        batch_size=batch_size,
        mean=means,
        std=stds,
        shuffle=False,
        load=True
    )

    pred = create_iterative_predictions(model, dg_test) if iterative else create_predictions(model, dg_test)
    print(f'Saving predictions: {pred_save_fn}')
    pred.to_netcdf(pred_save_fn)

    ds_valid_array = ds_valid.to_array()

    if iterative:
        print(evaluate_iterative_forecast(pred, ds_valid).load())
    else:
        print(compute_weighted_rmse(pred, ds_valid_array).load())

