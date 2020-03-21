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


class SelectiveLSTMDataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, seq_length, lead_time, step_size, years_per_epoch=5, batch_size=32, shuffle=False, load=True,
                 mean=None, std=None):
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
        # self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.seq_length = seq_length
        self.years_per_epoch = years_per_epoch
        self.step_size = step_size

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
        self.data_full = data
        self.data = None

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data_full.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        try:
            Xs = np.stack(
                [self.data.isel(time=slice(sample_id, sample_id + self.seq_length, self.step_size)).values for sample_id in idxs],
                axis=0
            )
        except:
            print(f'X data empty, retrieving to previous batchw{i}')
            r_i =  np.random.randint(0, i-1)
            return self.__getitem__(r_i)

        Y = self.data.isel(time=idxs + self.seq_length + self.lead_time + ((self.step_size-1)*(self.seq_length-1))).values
        return Xs, Y

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

        self.data = (self.data - self.mean) / self.std
        self.data.load()
        self.data.compute()

        adjusted_lead_time = self.lead_time+self.seq_length+((self.step_size-1)*(self.seq_length-1))

        self.n_samples = self.data.isel(time=slice(0, -adjusted_lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -adjusted_lead_time)).time
        self.valid_time = self.data.isel(time=slice(adjusted_lead_time, None)).time

        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


class LSTMDataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, seq_length, lead_time, batch_size=32, shuffle=False, load=True, mean=None,
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
        self.seq_length = seq_length

        data = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(level=levels))
            except ValueError:
                data.append(ds[var].expand_dims({'level': generic_level}, 1))

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.mean = self.data.mean(('time', 'lat', 'lon')).compute() if mean is None else mean
        self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        # Normalize
        self.data = (self.data - self.mean) / self.std
        adjusted_lead_time = self.seq_length + lead_time

        self.n_samples = self.data.isel(time=slice(0, -adjusted_lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -adjusted_lead_time)).time
        self.valid_time = self.data.isel(time=slice(adjusted_lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load: print('Loading data into RAM'); self.data.load()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        try:
            Xs = np.stack(
                [self.data.isel(time=slice(sample_id, sample_id + self.seq_length)).values for sample_id in idxs],
                axis=0
            )
        except:
            print(f'X data empty, retrieving to previous batchw{i}')
            r_i =  np.random.randint(0, i-1)
            return self.__getitem__(r_i)

        Y = self.data.isel(time=idxs + self.seq_length + self.lead_time).values
        return Xs, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
