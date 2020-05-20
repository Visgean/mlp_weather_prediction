import os

import tensorflow as tf
import tensorflow.keras as keras

from data_gen import SelectiveDataGenerator
from weatherbench.score import *
from weatherbench.train_nn import DataGenerator, create_iterative_predictions, build_cnn, create_predictions


def train(ds, filters, kernels, lr, activation, dr, batch_size,
          patience, model_save_fn, pred_save_fn, train_years, valid_years,
          test_years, lead_time, gpu, iterative, means, stds,
          levels_per_variable, RAM_DOWNLOADED_FULLY=True, weights=None, model_builder=build_cnn):
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
    first_var = list(levels_per_variable.values())[0]
    # Compatibility solution for baseline where {'z': None, 't': None}
    num_levels = len(first_var) if first_var is not None else len(list(levels_per_variable.keys()))
    model = model_builder(filters, kernels, input_shape=(32, 64, num_levels), activation=activation, dr=dr)

    model.compile(keras.optimizers.Adam(lr), 'mse')
    if weights:
        print(f'loading {weights}')
        model.load_weights(weights)
    print(model.summary())

    # Train model
    # TODO: Learning rate schedule
    model.fit_generator(dg_train, epochs=100, validation_data=dg_valid,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor='val_loss',
                                min_delta=0,
                                patience=patience,
                                verbose=1,
                                mode='auto'
                            ),
                            tf.keras.callbacks.ModelCheckpoint(
                                os.path.join(model_save_fn, 'models', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
                            )
                        ]
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
    return pred