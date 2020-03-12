from weatherbench.score import compute_weighted_rmse
import xarray as xr


def evaluate(datadir, pred_save_fn, valid_years, name):
    ds = xr.open_mfdataset(
        f'{datadir}*.nc',
        combine='by_coords',
    )

    ds_valid = ds.sel(time=slice(*valid_years))

    pred = xr.open_mfdataset(pred_save_fn)
    for level in pred.level:
        level_pred = pred.sel(level=level).to_array()
        ds_valid_array = ds_valid.sel(level=level).to_array()
        level_int = level.to_dict()['data']
        rmse = compute_weighted_rmse(level_pred, ds_valid_array).load().to_dict()['data']
        print(f'{name}, level {level_int}, rmse: {rmse}')


if __name__ == '__main__':
    evaluate(
        datadir='/home/visgean/Downloads/weather/geopotential/',
        pred_save_fn='/dropbox/Dropbox/2. mlp/output_geopotential/predictions',
        valid_years=('2017', '2018'),
        name='Geopotential'
    )

    evaluate(
        datadir='/home/visgean/Downloads/weather/temperature/',
        pred_save_fn='/dropbox/Dropbox/2. mlp/output_temperature/predictions',
        valid_years=('2017', '2018'),
        name='Temperature'
    )