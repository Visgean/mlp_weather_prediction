from tensorflow.keras.layers import *
from src.score import *
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, Lambda
import tensorflow.keras.backend as K

from src.train_nn import DataGenerator

DATADIR = '/home/visgean/Downloads/weather/'



geopotential = xr.open_mfdataset(f'{DATADIR}geopotential/*.nc', combine='by_coords')
dg = DataGenerator(geopotential, var_dict={'z': [1, 1000]}, lead_time=72, load=False)
















