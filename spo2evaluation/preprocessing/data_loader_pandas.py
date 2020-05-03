# from torch.utils.data import Dataset, DataLoader
import json
import os

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from . import data_loader
from . import utils


class Spo2DatasetPandas(data_loader.Spo2Dataset):
    """Spo2Dataset dataset using pandas.
        It preprocess the data in order to create a Dataset with the average
        and std of each channel per frame, the ground truth labels,
        and the video fps.
        It is store in a pandas Dataframe wehre each row correspond to one of
        those element and video are stacked up.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the data folder.
        """
        self.data_path = data_path
        self.path_samples = os.path.join(self.data_path, "sample_data")
        self.path_gt = os.path.join(self.data_path, "ground_truths_sample")
        self.path_meta = os.path.join(self.data_path, "meta")
        if self.is_pickled():
            print("Data was already pickled. Loading previously pickled dataset.")
            # TODO: LOAD THE DATA FROM PICKLES
            self.load_pickle()
        else:

            self.video_folders = [folder for folder in os.listdir(data_path) if
                                  os.path.isdir(os.path.join(data_path, folder))]
            print("v", self.video_folders)
            # self.number_of_videos = len(self.video_folders)
            # print(self.number_of_videos)

            data_list_red_mean = []
            data_list_blue_mean = []
            data_list_green_mean = []
            data_list_red_std = []
            data_list_blue_std = []
            data_list_green_std = []
            data_frame_number = []
            data_video_id = []
            data_sample_source = []
            meta = []

            data_sample_label_id = []
            data_label_source = []
            data_hr = []
            data_spo2 = []

            # column = []
            nb_video = 0
            for video in self.video_folders:

                ppg_red_mean = list()
                ppg_green_mean = list()
                ppg_blue_mean = list()
                ppg_red_std = list()
                ppg_green_std = list()
                ppg_blue_std = list()
                frame_numbers = list()
                video_id = list()
                sample_source = list()
                print("Loading video:", nb_video)
                # print(data_list)
                ppg = []
                video_path = os.path.join(self.data_path, video)

                video_file = os.path.join(video_path,
                                          [file_name for file_name in
                                              os.listdir(video_path)
                                              if file_name.endswith('mp4')][0])

                vidcap = cv2.VideoCapture(video_file)

                meta.append(vidcap.get(cv2.CAP_PROP_FPS))

                (grabbed, frame) = vidcap.read()
                frame_count = 0
                while grabbed:
                    blue_channel_mean, blue_channel_std, green_channel_mean, \
                        green_channel_std, red_channel_mean, red_channel_std = \
                        self.transform_faster(frame)

                    ppg_blue_mean.append(blue_channel_mean)
                    ppg_green_mean.append(green_channel_mean)
                    ppg_red_mean.append(red_channel_mean)
                    ppg_blue_std.append(blue_channel_std)
                    ppg_green_std.append(green_channel_std)
                    ppg_red_std.append(red_channel_std)
                    frame_numbers.append(frame_count)
                    video_id.append(nb_video)
                    sample_source.append(video_file)

                    ppg.append(frame)
                    (grabbed, frame) = vidcap.read()
                    if frame_count % 100 == 0 and frame_count != 0:
                        print("Frame:", frame_count)
                        # break
                    frame_count += 1
                with open(os.path.join(video_path, 'data.json'), 'r') as f:
                    ground_truth = json.load(f)
                # print("b", data_list)

                data_list_blue_mean = data_list_blue_mean + ppg_blue_mean
                # print(data_list)
                data_list_blue_std = data_list_blue_std + ppg_blue_std
                data_list_green_mean = data_list_green_mean + ppg_green_mean
                data_list_green_std = data_list_green_std + ppg_green_std
                data_list_red_mean = data_list_red_mean + ppg_red_mean
                data_list_red_std = data_list_red_std + ppg_red_std
                data_frame_number = data_frame_number + frame_numbers
                data_video_id = data_video_id + video_id
                data_sample_source = data_sample_source + sample_source

                data_sample_label_id = data_sample_label_id + [nb_video]
                data_label_source = data_label_source + [video_path]
                data_hr = data_hr + [int(ground_truth['survey']['hr'])]
                data_spo2 = data_spo2 + [int(ground_truth['survey']['spo2'])]
                # data_list.append([int(ground_truth['SpO2'])])
                # data_list.append([int(ground_truth['HR'])])
                # data_list.append([meta['video_fps']])
                nb_video += 1

            # making the data
            column = ["mean_red", "std_red", "mean_green", "std_green",
                      "mean_blue", "std_blue", "frame", "sample_id",
                      "sample_source"]
            data_list = [data_list_red_mean, data_list_red_std,
                         data_list_green_mean, data_list_green_std,
                         data_list_blue_mean, data_list_blue_std,
                         data_frame_number, data_video_id, data_sample_source]
            print(column)
            self.sample_data = pd.DataFrame(data_list, column).T

            # Making the labels

            column_labels = ["sample_id", "path", "HR", "SpO2"]
            data_list_labels = [data_sample_label_id, data_label_source, data_hr,
                                data_spo2]
            self.ground_truths_sample = pd.DataFrame(data_list_labels,
                                                     column_labels).T

            self.meta = pd.DataFrame([data_sample_label_id, meta], ["sample_id", "fps"]).T

            self.number_of_videos = len(self.meta.index)

            # self.data = self.data.T
            print(self.sample_data)
            # self.data = self.data.T
            print(self.ground_truths_sample)
            print(self.meta)

    def print_pgg(self, video_id):
        plt.figure()
        # print("GO")
        # print(self.sample_data)

        video = self.sample_data.loc[self.sample_data['sample_id'] == video_id]

        print("video")
        print(video)

        red = video["mean_red"].to_numpy()
        red_n = utils.normalize_ppg(red)
        green = video["mean_green"].to_numpy()
        green_n = utils.normalize_ppg(green)
        blue = video["mean_blue"].to_numpy()
        blue_n = utils.normalize_ppg(blue)

        plt.plot(video["frame"].to_numpy(), red_n, "r", label="max: " +
                 str(np.amax(red)))
        plt.plot(video["frame"].to_numpy(), green_n, "g", label="max: " +
                 str(np.amax(green)))
        plt.plot(video["frame"].to_numpy(), blue_n, "b", label="max: " +
                 str(np.amax(blue)))
        # plt.plot(video["frame"].to_array(), video["mean_green"].to_array())
        # plt.plot(video["frame"].to_array(), video["mean_blue"].to_array())
        # plt.plot(timestamps, vps_red[1:], 'b')

        plt.legend()
        plt.xlabel("t")
        plt.grid(True)

        print(video["sample_source"].iloc[0])
        plt.title(video["sample_source"].iloc[0])

        # plt.figure(1)
        # plt.plot(timestamps, ppg_green_normalized[1:])
        # plt.plot(timestamps, ppg_green_filtered_normalized, "g")
        # plt.plot(timestamps, vps_green[1:], 'b')

        # plt.xlabel("t")
        # plt.grid(True)
        # plt.title("PPG signal for the color green")

    def get_video(self, index):
        return (self.sample_data.loc[self.sample_data['sample_id'] == index],
                self.ground_truths_sample.loc[self.ground_truths_sample['sample_id'] == index],
                self.meta.loc[self.meta['sample_id'] == index])

    def pickle_data(self):
        self.sample_data.to_pickle(self.path_samples)
        self.ground_truths_sample.to_pickle(self.path_gt)
        self.meta.to_pickle(self.path_meta)

    def load_pickle(self):
        print("LOAD")
        self.sample_data = pd.read_pickle(self.path_samples)
        self.ground_truths_sample = pd.read_pickle(self.path_gt)
        self.meta = pd.read_pickle(self.path_meta)
        # print("yo\n", self.sample_data.loc[self.sample_data['sample_id'] == 1])
        self.number_of_videos = len(self.meta.index)
        print("Loaded", self.number_of_videos)

    def is_pickled(self) -> bool:
        if os.path.isfile(self.path_samples) and os.access(self.path_samples, os.R_OK) \
                and os.path.isfile(self.path_gt) and os.access(self.path_gt, os.R_OK) \
                and os.path.isfile(self.path_meta) and os.access(self.path_meta, os.R_OK):
            return True
        return False