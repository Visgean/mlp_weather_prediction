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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

DATADIR = os.getenv('DATASET_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/datasets/')
OUT_DIR = os.getenv('SAVE_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/output_baseline_ltsm/')
valid_years = ('2015', '2016')

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

means = xr.load_dataarray('data/baseline_mean.nc')
stds = xr.load_dataarray('data/baseline_std.nc')
ds = xr.merge([z, t], compat='override')

levels_per_variable = {'z': None, 't': None}


ds_valid = ds.sel(time=slice(*valid_years))



def eval_file(path):

    pred = xr.open_mfdataset(path)

    t_rmse = compute_weighted_rmse(pred.t, t.to_array()).load().to_dict()['data']
    print(f'{path}: Temperature at 850m, rmse: {t_rmse}')

    z_rmse = compute_weighted_rmse(pred.z, z.to_array()).load().to_dict()['data']
    print(f'{path}: Geopotential at 500, rmse: {z_rmse}')




eval_file('/home/s1660124/output_unet/predictions')
eval_file('/home/s1660124/output_ltsm_6hours_large_model/predictions')
eval_file('/home/s1660124/output_ltsm_3days_large_model/predictions')

