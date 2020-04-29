import enum
import itertools
import json
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plot

import numpy as np

from . import data_sanitization


class DataSourceCovital(enum.Enum):
    community = 0
    clinical = 1


class DatasetStatistics(object):
    def __init__(self):
        self.spo2 = np.array([])
        self.hr = np.array([])
        self.date = list()
        self.type = list()

    def clear(self):
        self.spo2.clear()
        self.hr.clear()
        self.date.clear()
        self.type.clear()

    def get_data(self, folder: str) -> None:
        spo2_l = list()
        # spo2_l.append(100)
        hr_l = list()

        _, data_files = data_sanitization.run_fast_scandir_json(
            folder, dataformat=data_sanitization.DataFormatCovital.current)
        n = 1
        for file in data_files:
            print("Checking file", n, "of", len(data_files))
            n += 1
            with open(file) as json_file:
                data = json.load(json_file)
                spo2_l.append(data['survey']['spo2'])
                hr_l.append(data['survey']['hr'])
                datetimeobject = datetime.strptime(
                    data['survey']['date'].split(".")[0], "%Y-%m-%jT%H:%M:%S"
                    )
                # print(datetimeobject)
                self.date.append(datetimeobject.date())
                if "covital-data-clinical" in file:
                    self.type.append(DataSourceCovital.clinical)
                else:
                    self.type.append(DataSourceCovital.community)
        self.spo2 = np.array(spo2_l)
        self.hr = np.array(hr_l)

    def calculate(self):
        self.spo2_mean = self.spo2.mean()
        self.hr_mean = self.hr.mean()

        self.spo2_std = self.spo2.std()
        self.hr_std = self.hr.std()

    def print(self):
        plot.subplot(3, 1, 1)
        values_spo2 = np.arange(np.amin(self.spo2), np.amax(self.spo2)+1, 1)
        bins_nb_spo2 = len(values_spo2) - 1
        plot.hist(self.spo2, bins=bins_nb_spo2, rwidth=0.5, align="right")
        # print(bins)
        plot.xticks(np.arange(np.amin(self.spo2), 101, 1))
        plot.title("SpO2")

        plot.subplot(3, 1, 2)
        values_hr = np.arange(np.amin(self.hr), np.amax(self.hr)+1, 1)
        bins_nb_hr = len(values_hr) - 1
        # print(int(bins_nb))
        # print(self.hr)
        plot.hist(self.hr, bins=int(bins_nb_hr), align="right", rwidth=0.5)
        xticks = np.arange(np.amin(self.hr), np.amax(self.hr)+1, 5)
        plot.xticks(xticks)
        plot.title("heart rates")

        ax = plot.subplot(3, 1, 3)

        days = mdates.DayLocator(interval=1)
        h_fmt = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(h_fmt)

        mindate = min(self.date)
        maxdate = max(self.date)
        print(self.date)
        bins_days = (maxdate - mindate).days
        values_hr = np.arange(0, bins_days + 1, 1)

        mask = [el == DataSourceCovital.community for el in self.type]
        community_date = list(itertools.compress(self.date, mask))
        mask_clinical = [el == DataSourceCovital.clinical for el in self.type]
        clinical_date = list(itertools.compress(self.date, mask_clinical))

        plot.hist([community_date, clinical_date], bins=bins_days,
                  align="right", rwidth=0.1,
                  label=['community', 'clinical'])
        plot.title("samples per day and dataset")
        plot.legend(loc='upper left')

        plot.tight_layout()
        plot.savefig("spo2_bin.png")
        plot.show()
