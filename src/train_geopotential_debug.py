import os
import train



DATADIR = '/home/visgean/Downloads/weather-small/'

if __name__ == '__main__':
    train.train(
        datadir=DATADIR,
        filters=[64, 64, 64, 64, 11],
        kernels=[5, 5, 5, 5, 5],
        lr=1e-4,
        activation='elu',
        dr=0,
        batch_size=128,
        patience=3,
        model_save_fn='./models/',
        pred_save_fn='./predictions/',
        train_years=('1979', '2015'),
        valid_years=('2016', '2016'),
        test_years=('2017', '2018'),
        lead_time=72,
        gpu=0,
        iterative=False,
        RAM_DOWNLOADED_FULLY=True,
    )
