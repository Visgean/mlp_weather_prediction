import os
import train
import xarray as xr

# import tensorflow as tf
# assert tf.test.is_gpu_available()
import train_ltsm
from build_network_unet import build_cnn_ltsm_unet, get_unet_proper
from precip_models import get_vgg16
from weatherbench.score import compute_weighted_rmse

means = xr.load_dataarray('data/baseline_mean.nc')
stds = xr.load_dataarray('data/baseline_std.nc')
# means = None
# stds = None


DATADIR = os.getenv('DATASET_DIR', '/home/visgean/Downloads/weather/')
OUT_DIR = os.getenv('SAVE_DIR', '/home/visgean/Downloads/test')
# DATADIR = os.getenv('DATASET_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/datasets/')
# OUT_DIR = os.getenv('SAVE_DIR', '/afs/inf.ed.ac.uk/user/s16/s1660124/output_baseline_ltsm/')


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
ds = xr.merge([z, t], compat='override')

levels_per_variable = {'z': None, 't': None}

if __name__ == '__main__':
    pred = train.train(
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
        patience=5,
        model_save_fn=OUT_DIR,
        pred_save_fn=os.path.join(OUT_DIR, 'predictions'),
        train_years=('1979', '2014'),
        valid_years=('2015', '2016'),
        test_years=('2017', '2018'),
        lead_time=72,
        gpu=0,
        iterative=False,
        model_builder=get_unet_proper,
        weights='/home/s1660124/output_unet_proper/models/weights.18-0.47.hdf5'
    )

    t_rmse = compute_weighted_rmse(pred.t, t.to_array()).load().to_dict()['data']
    print(f'Temperature at 850m, rmse: {t_rmse}')

    z_rmse = compute_weighted_rmse(pred.z, z.to_array()).load().to_dict()['data']
    print(f'Geopotential at 500, rmse: {z_rmse}')

