"""
Documentation, License etc.

@package lamonaca_2015
"""
import cv2
import scipy.signal as signal
import numpy as np
import math

from . import utils


class Camera_specs(object):
    def __init__(self):
        self.qe_green_940 = None
        self.qe_red_600 = None
        self.sum_qe_green_600_to_940 = None
        self.sum_qe_red_600_to_940 = None


class Lamonaca2015(object):
    def __init__(self):
        self.camera_specs = Camera_specs()
        print("Init")

    def ppg_lamonaca(self, frame, blue=0, green=1, red=2):
        blue_channel_mean, green_channel_mean, red_channel_mean = utils.getColorComponentsMean(
            frame, blue, green, red
        )

        vp_600_green = green_channel_mean * (
            self.camera_specs.qe_green_940 / self.camera_specs.sum_qe_green_600_to_940
        )
        vp_600_red = red_channel_mean * (
            self.camera_specs.qe_red_600 / self.camera_specs.sum_qe_red_600_to_940
        )

        return (vp_600_green, vp_600_red)

    def ppg_filtered(self, frames):
        camera_specs = Camera_specs()
        vps_940_green = list()
        vps_600_red = list()
        for frame in frames:
            vp_600_green, vp_600_red = self.ppg_lamonaca(frame, camera_specs)
            vps_940_green.append(vp_600_green)
            vps_600_red.append(vp_600_red)

        # First, design the Buterworth filter
        vps_940_green_filtered = utils.cutoff(vps_940_green)
        vps_600_red_filtered = utils.cutoff(vps_600_red)

        return (vps_940_green_filtered, vps_600_red_filtered)

    def ppg_filtered_from_video_file(self, video_file):
        camera_specs = Camera_specs()
        vps_940_green = list()
        vps_600_red = list()
        frame_count = 0

        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamps = list()

        while cap.isOpened():

            ret, frame = cap.read()
            if ret == True:

                if frame_count % 50 == 0:
                    print("Frame:", frame_count)

                timestamps.append(frame_count / fps)
                frame_count = frame_count + 1

                vp_600_green, vp_600_red = self.ppg_lamonaca(frame, camera_specs)
                vps_940_green.append(vp_600_green)
                vps_600_red.append(vp_600_red)
            else:
                break

        cap.release()

        duration = frame_count / fps

        # First, design the Buterworth filter
        vps_940_green_filtered = utils.cutoff(vps_940_green)
        vps_600_red_filtered = utils.cutoff(vps_600_red)

        return (vps_940_green_filtered, vps_600_red_filtered, duration, timestamps)

    def lacomana(self, video_frames, timestamps):
        vps_940_green_filtered, vps_600_red_filtered = self.ppg_filtered(video_frames)
        return utils.spo2_estimation(
            vps_940_green_filtered, vps_600_red_filtered, timestamps
        )

    def lacomana(self, video_file):
        print("lacoma")
        vps_940_green_filtered, vps_600_red_filtered, duration, timestamps = self.ppg_filtered_from_video_file(
            video_file
        )
        print("Duration:", duration)
