from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torchvision
import os
import json
import torch
from threading import Thread
from queue import Queue
import time
from pathlib import Path


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Spo2Dataset(Dataset):
    """Spo2Dataset dataset.
        It preprocess the data in order to create a Dataset with the average and std of each channel per frame.
        The process is slow so it may take a while to create the Dataset when first initated.
    """

    def __init__(self, data_path, file_type='mp4', rescale=True):
        """
        Args:
            data_path (string): Path to the data folder.
        """
        self.data_path = Path(data_path)
        # Recursively find all files of file_type in the data path
        self.video_paths = self.data_path.glob(f'**/*.{file_type}')
        self.videos_ppg = []
        self.labels_list = []
        self.meta_list = []

        for nb_video, video in enumerate(self.video_paths):
            print(f"Loading video {nb_video} from '{video}'")
            ppg = []
            meta = {}
            vidcap = cv2.VideoCapture(str(video))
            meta['video_fps'] = vidcap.get(cv2.CAP_PROP_FPS)
            (grabbed, frame) = vidcap.read()
            while grabbed:
                if rescale:
                    frame = self.rescale_frame(frame)
                frame = self.transform_faster(frame)
                ppg.append(frame)
                (grabbed, frame) = vidcap.read()

            with open(video.parent/'gt.json', 'r') as f:
                ground_truth = json.load(f)

            labels = torch.Tensor(
                [int(ground_truth['SpO2']), int(ground_truth['HR'])])
            self.videos_ppg.append(torch.Tensor(np.array(ppg)))
            self.meta_list.append(meta)
            self.labels_list.append(labels)

    #@timing
    def reshape(self, frame):
        return frame.reshape(-1, 3)

    #@timing
    def rescale_frame(self, frame, percent=50):
        width = int(frame.shape[1] * percent / 100)
        height = int(frame.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

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

        return np.array([[blue_channel_mean, blue_channel_std],
                         [green_channel_mean, green_channel_std],
                         [red_channel_mean, red_channel_std]])

    def __len__(self):
        return len(list(self.video_paths))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.videos_ppg[idx], self.meta_list[idx], self.labels_list[idx]]


def spo2_collate_fn(batch):
    videos_length = [element[0].shape[0] for element in batch]
    max_length = max(videos_length)
    videos_tensor = torch.FloatTensor(
        size=[len(videos_length), max_length, 3, 2])
    labels_tensor = torch.FloatTensor(size=[len(videos_length), 2])
    for i, element in enumerate(batch):
        padding = max_length-videos_length[i]
        if padding > 0:
            padding = torch.zeros([padding, 3, 2])
            video = torch.cat([element[0], padding])
        else:
            video = element[0]
        labels = element[2]
        videos_tensor[i] = video
        labels_tensor[i] = element[2]
    return videos_tensor, labels_tensor, torch.Tensor(videos_length)


if __name__ == "__main__":
    dataset = Spo2Dataset('sample_data')
    dataloader = DataLoader(
        dataset, batch_size=4, collate_fn=spo2_collate_fn)
    for tup in dataloader:
        print(tup)
        print('Padded video (length, color, (mean,std)): ',
              videos_batch[0].shape)
        print('Video original length: ', videos_lengths[0])
        print('Labels (so2, hr): ', labels_batch[0])
