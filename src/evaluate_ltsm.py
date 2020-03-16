import os

import tensorflow as tf
import tensorflow.keras as keras
import xarray as xr

from build_network import build_cnn_ltsm
from data_gen import SelectiveDataGenerator, LSTMDataGenerator
from weatherbench.score import *
from weatherbench.train_nn import DataGenerator, create_iterative_predictions, build_cnn, create_predictions

means = xr.load_dataarray('data/baseline_mean.nc')
stds = xr.load_dataarray('data/baseline_std.nc')
# means = None
# stds = None


DATADIR = os.getenv('DATASET_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/datasets/')
OUT_DIR = os.getenv('SAVE_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/output_baseline_ltsm/')

# Load the geo500 and temp850 data and merge
z = xr.open_mfdataset(
    f'{DATADIR}geopotential_500/*.nc',
    combine='by_coords',
)
t = xr.open_mfdataset(
    f'{DATADIR}temperature_850/*.nc',
    combine='by_coords',

)



ds = xr.merge([z, t], compat='override')
ds.load()

levels_per_variable = {'z': None, 't': None}

filters = [64, 64, 64, 64, 2]
kernels = [5, 5, 5, 5, 5]
lr = 1e-4
activation = 'elu'
dr = 0
batch_size = 64
patience = 50
model_save_fn = OUT_DIR
pred_save_fn = os.path.join(OUT_DIR, 'predictions')
train_years = ('1979', '2014')
valid_years = ('2015', '2016')
test_years = ('2017', '2018')
lead_time = 72
seq_length = 8
gpu = 0
iterative = False
weights = '../models/ltsm/weights.57-0.37.hdf5'

ds_train = ds.sel(time=slice(*train_years))
ds_valid = ds.sel(time=slice(*valid_years))
ds_test = ds.sel(time=slice(*test_years))

print("Loading test data")
dg_test = LSTMDataGenerator(
    ds=ds_test,
    var_dict=levels_per_variable,
    lead_time=lead_time,
    batch_size=batch_size,
    mean=means,
    std=stds,
    shuffle=False,
    load=True,
    seq_length=seq_length,
)

first_var = list(levels_per_variable.values())[0]
# Compatibility solution for baseline where {'z': None, 't': None}
num_levels = 2

model = build_cnn_ltsm(filters, kernels, input_shape=(None, 32, 64, num_levels), activation=activation, dr=dr)
model.compile(keras.optimizers.Adam(lr), 'mse')
if weights:
    print(f'loading {weights}')
    model.load_weights(weights)

print(model.summary())

pred = create_iterative_predictions(model, dg_test) if iterative else create_predictions(model, dg_test)
print(f'Saving predictions: {pred_save_fn}')
pred.to_netcdf(pred_save_fn)

t_rmse = compute_weighted_rmse(pred.t, t.to_array()).load().to_dict()['data']
print(f'Temperature at 850m, rmse: {t_rmse}')

z_rmse = compute_weighted_rmse(pred.z, z.to_array()).load().to_dict()['data']
print(f'Geopotential at 500, rmse: {z_rmse}')
