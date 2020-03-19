from ecmwfapi import ECMWFDataServer
import calendar
import os
from fire import Fire


def main(var, years=[2017, 2018], month_start=1, month_end=12, step_end=120,
         all_steps=True,
         path='/Users/fasand/Downloads/WeatherbenchDataset/tigge/raw/'):
    os.makedirs(path, exist_ok=True)
    server = ECMWFDataServer()
    months = range(month_start, month_end+1)
    if all_steps:
        steps = "/".join([str(i) for i in range(0, step_end+1, 6)])
    else:
        steps = str(step_end)
    for year in years:
        for month in months:
            days = calendar.monthrange(year, month)[1]
            month = str(month).zfill(2)
            if var == 'z':
                server.retrieve({
                    "class": "ti",
                    "dataset": "tigge",
                    "date": f"{year}-{month}-01/to/{year}-{month}-{days}",
                    "expver": "prod",
                    "grid": "0.703125/0.703125",
                    "levelist": "500",
                    "levtype": "pl",
                    "origin": "ecmf",
                    "param": "156",
                    "step": steps,
                    "time": "00:00:00/12:00:00",
                    "type": "cf",
                    "target": f"{path}/z500_{year}_{month}_raw.grib",
                })
            elif var == 't':
                server.retrieve({
                    "class": "ti",
                    "dataset": "tigge",
                    "date": f"{year}-{month}-01/to/{year}-{month}-{days}",
                    "expver": "prod",
                    "grid": "0.703125/0.703125",
                    "levelist": "850",
                    "levtype": "pl",
                    "origin": "ecmf",
                    "param": "130",
                    "step": steps,
                    "time": "00:00:00/12:00:00",
                    "type": "cf",
                    "target": f"{path}/t850_{year}_{month}_raw.grib",
                })


if __name__ == '__main__':
    Fire(main)
