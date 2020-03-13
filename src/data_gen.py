from weatherbench.score import *
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


def limit_mem():
    """Limit TF GPU mem usage"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)


class SelectiveDataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, years_per_epoch=5, batch_size=32, shuffle=True, load=True, mean=None,
                 std=None):
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
        self.years_per_epoch = years_per_epoch

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

        self.mean = data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = data.std('time').mean(('lat', 'lon')).compute() if std is None else std

        # Normalize
        self.data_full = (data - self.mean) / self.std

        self.data = None

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        y = self.data.isel(time=idxs + self.lead_time).values
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        del self.data
        current_year = np.random.randint(1979, 2016 - self.years_per_epoch)
        end_year = current_year + self.years_per_epoch

        print(f"Loading data from {current_year} to {end_year}")

        self.data = self.data_full.sel(time=slice(str(current_year), str(end_year)))
        self.data.load()
        self.data.compute()

        self.n_samples = self.data.isel(time=slice(0, -self.lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -self.lead_time)).time
        self.valid_time = self.data.isel(time=slice(self.lead_time, None)).time

        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)



class LSTMDataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, seq_length, lead_time, years_per_epoch=5, batch_size=32, shuffle=False, load=True, mean=None, std=None):
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
        self.seq_length = seq_length
        self.years_per_epoch = years_per_epoch


        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')

        self.mean = data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = data.std('time').mean(('lat', 'lon')).compute() if std is None else std

        # Normalize
        self.data_full = (data - self.mean) / self.std

        self.data = None

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        Xs = [self.data.isel(time=slice(idxs[i],idxs[i]+self.seq_length)).values for i in range(self.batch_size)]
        ys = [self.data.isel(time=idxs[i]+self.seq_length+self.lead_time).values for i in range(self.batch_size)]
        X = np.stack(Xs, axis = 0)
        y = np.stack(ys, axis = 0)
        #y = y.reshape(y.shape[0],1,y.shape[1],y.shape[2],y.shape[3])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        if self.years_per_epoch:
            del self.data
            current_year = np.random.randint(1979, 2016 - self.years_per_epoch)
            end_year = current_year + self.years_per_epoch

            print(f"Loading data from {current_year} to {end_year}")

            self.data = self.data_full.sel(time=slice(str(current_year), str(end_year)))

        else:
            self.data = self.data_full
        self.data.load()
        self.data.compute()

        self.n_samples = self.data.isel(time=slice(0, -self.lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -self.lead_time)).time
        self.valid_time = self.data.isel(time=slice(self.lead_time, None)).time

        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
