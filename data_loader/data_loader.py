import numpy as np
import os
import json
import torch
import time
from pathlib import Path

import abc


class Spo2Dataset(abc.ABC):
    """Spo2Dataset.
        Base class to provide generic function needed to get the ppg
    """

    #@timing
    def reshape(self, frame):
        return frame.reshape(-1, 3)

    #@timing
    def rescale_frame(self, frame, percent=10):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    #@timing
    def crop_frame(self, frame, dims=(500, 500)):
        in_dims = (frame.shape[0], frame.shape[1])
        xlocs = (int(in_dims[0] / 2 - 0.5 * dims[0]),
                 int(in_dims[0] / 2 + 0.5 * dims[0]))
        ylocs = (int(in_dims[1] / 2 - 0.5 * dims[1]),
                 int(in_dims[1] / 2 + 0.5 * dims[1]))
        cropped_frame = frame[xlocs[0]:xlocs[1], ylocs[0]:ylocs[1]]

        return cropped_frame


    #@timing
    def mean_t(self, frame):
        return np.array([frame.mean(axis=0), frame.std(axis=0)]).T

    #@timing
    def transform(self, frame):
        frame = self.reshape(frame)
        ret = self.mean_t(frame)
        return ret

    #@timing
    def get_channels(self, frame, blue=0, green=1, red=2):
        blue_channel = frame[:, :, blue]
        green_channel = frame[:, :, green]
        red_channel = frame[:, :, red]

        return blue_channel, green_channel, red_channel

    #@timing
    def mean_fast(self, blue_channel, green_channel, red_channel):
        blue_channel_mean = blue_channel.mean()
        green_channel_mean = green_channel.mean()
        red_channel_mean = red_channel.mean()

        return blue_channel_mean, green_channel_mean, red_channel_mean

    #@timing
    def std_fast(self, blue_channel, green_channel, red_channel):
        blue_channel_mean = blue_channel.std()
        green_channel_mean = green_channel.std()
        red_channel_mean = red_channel.std()

        return blue_channel_mean, green_channel_mean, red_channel_mean

    #@timing
    def transform_faster(self, frame):
        blue_channel, green_channel, red_channel = self.get_channels(frame)
        blue_channel_mean, green_channel_mean, red_channel_mean = self.mean_fast(
            blue_channel, green_channel, red_channel)
        blue_channel_std, green_channel_std, red_channel_std = self.std_fast(
            blue_channel, green_channel, red_channel)

        return blue_channel_mean, blue_channel_std, green_channel_mean, green_channel_std, red_channel_mean, red_channel_std
