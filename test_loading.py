import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import seaborn as sns
import pickle
from src.weatherbench.score import *
from collections import OrderedDict

from src.weatherbench.train_nn import DataGenerator

DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')

z500_valid = load_test_data(f'{DATADIR}geopotential_500', 'z')
t850_valid = load_test_data(f'{DATADIR}temperature_850', 't')
valid = xr.merge([z500_valid, t850_valid], compat='override')

z = xr.open_mfdataset(f'{DATADIR}geopotential_500/*.nc', combine='by_coords')
t = xr.open_mfdataset(f'{DATADIR}temperature_850/*.nc', combine='by_coords')


datasets = [z, t]
ds = xr.merge(datasets, compat='override')

ds_train = ds.sel(time=slice('2015', '2016'))
ds_test = ds.sel(time=slice('2017', '2018'))

dic = OrderedDict({'z': None, 't': None})
bs = 32
lead_time = 6

dg_train = DataGenerator(
    ds_train.sel(time=slice('2015', '2015')),
    dic,
    lead_time,
    batch_size=bs,
    load=True
)

dg_valid = DataGenerator(
    ds_train.sel(time=slice('2016', '2016')),
    dic, lead_time,
    batch_size=bs,
    mean=dg_train.mean,
    std=dg_train.std,
    shuffle=False
)
