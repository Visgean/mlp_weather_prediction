from tensorflow.keras.layers import *
from src.score import *
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, Lambda
import tensorflow.keras.backend as K

DATADIR = '/home/visgean/Downloads/weather/'


class DataGenerator3D(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True, mean=None, std=None):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """

        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        # data = []
        # generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        # for var, levels in var_dict.items():
        #     try:
        #         data.append(ds[var].sel(level=levels))
        #     except ValueError:
        #         data.append(ds[var].expand_dims({'level': generic_level}, 1))

        # self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.data = ds.transpose('time', 'lat', 'lon', 'level')



        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'

        # self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        # self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        # self.data = (self.data - self.mean) / self.std


        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


geopotential = xr.open_mfdataset(f'{DATADIR}geopotential/*.nc', combine='by_coords')
dg = DataGenerator3D(geopotential, var_dict={'z': None}, lead_time=72, load=False)
















