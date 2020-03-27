# For Keras
import tensorflow as tf

from weatherbench.score import compute_weighted_rmse

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True

sess = tf.Session(config=config)

# set_session(sess)  # set this TensorFlow session as the default session for Keras

import os

import tensorflow.keras as keras

from build_network import build_cnn_ltsm
from data_gen import SelectiveLSTMDataGenerator
from weatherbench.train_nn import create_iterative_predictions, create_predictions
import xarray as xr

DATADIR = os.getenv('DATASET_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/datasets/')
OUT_DIR = os.getenv('SAVE_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/output_baseline_ltsm/')

# Load the geo500 and temp850 data and merge
z = xr.open_mfdataset(
    f'{DATADIR}geopotential_500/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 10}

)
t = xr.open_mfdataset(
    f'{DATADIR}temperature_850/*.nc',
    combine='by_coords',
    parallel=True,
    chunks={'time': 10}

)

def evaluate(ds, filters, kernels, lr, activation, dr, batch_size,
             patience, model_save_fn, pred_save_fn, train_years, valid_years,
             test_years, lead_time, gpu, iterative, means, stds,
             levels_per_variable, seq_length, weights=None, step_size=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    if levels_per_variable is not None:
        first_var = list(levels_per_variable.values())[0]
        # Compatibility solution for baseline where {'z': None, 't': None}
        num_levels = len(first_var) if first_var is not None else len(list(levels_per_variable.keys()))
    else:
        num_levels = len(ds.level)

    model = build_cnn_ltsm(filters, kernels, input_shape=(None, 32, 64, num_levels), activation=activation, dr=dr)
    model.compile(keras.optimizers.Adam(lr), 'mse')

    print(f'loading {weights}')
    model.load_weights(weights)

    print(model.summary())

    # Create predictions

    print("Loading test data")
    dg_test = SelectiveLSTMDataGenerator(
        ds=ds_test,
        var_dict=levels_per_variable,
        lead_time=lead_time,
        batch_size=batch_size,
        mean=means,
        std=stds,
        shuffle=False,
        load=True,
        seq_length=seq_length,
        years_per_epoch=None,
        step_size=step_size
    )




    print(weights)
    pred = create_predictions(model, dg_test)
    print(f'Saving predictions: {pred_save_fn}')
    pred.to_netcdf(pred_save_fn)
    print(weights)

    t_rmse = compute_weighted_rmse(pred.t, t.to_array()).load().to_dict()['data']
    print(f'Temperature at 850m, rmse: {t_rmse}')

    z_rmse = compute_weighted_rmse(pred.z, z.to_array()).load().to_dict()['data']
    print(f'Geopotential at 500, rmse: {z_rmse}')

    ds_valid_array = ds_valid.to_array()
    print(compute_weighted_rmse(pred, ds_valid_array).load())


if __name__ == '__main__':
    means = xr.load_dataarray('data/baseline_mean.nc')
    stds = xr.load_dataarray('data/baseline_std.nc')
    # means = None
    # stds = None

    # DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
    # OUT_DIR = os.getenv('SAVE_DIR', '/home/visgean/Downloads/test')

    print(DATADIR)


    ds = xr.merge([z, t], compat='override')

    levels_per_variable = {'z': None, 't': None}

    evaluate(
        ds=ds,
        means=means,
        stds=stds,
        levels_per_variable=levels_per_variable,
        filters=[64, 64, 64, 64, 2],
        kernels=[5, 5, 5, 5, 5],
        lr=1e-4,
        activation='elu',
        dr=0,
        batch_size=64,
        patience=50,
        model_save_fn=OUT_DIR,
        pred_save_fn=os.path.join(OUT_DIR, 'predictions-5days'),
        train_years=('1979', '2014'),
        valid_years=('2015', '2016'),
        test_years=('2017', '2018'),
        lead_time=5 * 24,
        seq_length=8,
        step_size=4,
        gpu=0,
        iterative=False,
        weights='/home/s1660124/output_ltsm_5days/models/weights.11-0.56.hdf5'
    )

    # evaluate(
    #     ds=ds,
    #     means=means,
    #     stds=stds,
    #     levels_per_variable=levels_per_variable,
    #     filters=[64, 64, 64, 64, 2],
    #     kernels=[5, 5, 5, 5, 5],
    #     lr=1e-4,
    #     activation='elu',
    #     dr=0,
    #     batch_size=64,
    #     patience=50,
    #     model_save_fn=OUT_DIR,
    #     pred_save_fn=os.path.join(OUT_DIR, 'predictions-3days'),
    #     train_years=('1979', '2014'),
    #     valid_years=('2015', '2016'),
    #     test_years=('2017', '2018'),
    #     lead_time=3 * 24,
    #     seq_length=8,
    #     step_size=4,
    #     gpu=0,
    #     iterative=False,
    #     weights='../models/ltsm-step-4-days-3-weights.61-0.47.hdf5'
    # )
