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

    def __len__(self):
        return len(self.videos_ppg)

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
    DATADIR = Path('sample_data')
    PERSIST = False

    dataset = Spo2Dataset(DATADIR)

    if PERSIST:
        # Persist Spo2Dataset object to a pickle for quick reload
        persist_dir = DATADIR/'pickled'
        print(f"Persisting dataset to {persist_dir}")
        if not os.path.exists(persist_dir):
            os.makedirs(persist_dir)
        import pickle
        with open(persist_dir/'spo2dataset.pkl', 'wb') as pkl_file:
            pickle.dump(dataset, pkl_file)

    dataloader = DataLoader(
        dataset, batch_size=4, collate_fn=spo2_collate_fn)
    for videos_batch, labels_batch, videos_lengths in dataloader:
        print('Padded video (length, color, (mean,std)): ',
              videos_batch[0].shape)
        print('Video original length: ', videos_lengths[0])
        print('Labels (SpO2, HR): ', labels_batch[0])
>>>>>>> master