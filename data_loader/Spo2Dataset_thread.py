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

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    It stores the frames in a list.
    It adds new frames only when there is less than a 100 in the "stack" to help overcome memory issues.
    """

    def __init__(self, src=0, stack_size = 80):
        self.stream = cv2.VideoCapture(src)
        self.stack_size = stack_size
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frames = [self.frame]
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            elif len(self.frames) < self.stack_size:
                (self.grabbed, self.frame) = self.stream.read()
                self.frames.append(self.frame)

    def stop(self):
        self.stopped = True

class Spo2Dataset(Dataset):
    """Spo2Dataset dataset.
        It preprocess the data in order to create a Dataset with the average and std of each channel per frame.
        The process is slow so it may take a while to create the Dataset when first initated.
    """
    def transform(self,frame):
        frame = frame.reshape(len(frame),-1,3)
        return np.array([frame.mean(axis=1), frame.std(axis=1)]).reshape(len(frame),3,2)
    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the data folder.
        """
        self.data_path = data_path
        self.video_folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,folder))]
        self.videos_ppg = []
        self.labels_list = []
        self.meta_list = []

        for video in self.video_folders:
            ppg = []
            video_path = os.path.join(self.data_path, video)
            video_file = os.path.join(video_path, [file_name for file_name in os.listdir(video_path) if file_name.endswith('mp4')][0])
            vidcap = VideoGet(video_file).start()
            meta = {}
            meta['video_fps'] = vidcap.fps
            while True:
                if vidcap.stopped and len(vidcap.frames)==1:
                    vidcap.stop()
                    break
                if len(vidcap.frames)>1:
                    frames = np.array(vidcap.frames[:-1])
                    vidcap.frames = vidcap.frames[len(frames):]
                    frame = self.transform(frames)
                    ppg.extend(frame)

            with open(os.path.join(video_path, 'gt.json'), 'r') as f:
                ground_truth = json.load(f)

            labels = torch.Tensor([int(ground_truth['SpO2']), int(ground_truth['HR'])])
            self.videos_ppg.append(torch.Tensor(np.array(ppg)))
            self.meta_list.append(meta)
            self.labels_list.append(labels)
    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return [self.videos_ppg[idx],self.meta_list[idx],self.labels_list[idx]]

class Spo2DataLoader(DataLoader):
    def collate_fn(batch):
        videos_length = [element[0].shape[0] for element in batch]
        max_length = max(videos_length)
        videos_tensor = torch.FloatTensor(size=[len(videos_length),max_length, 3, 2])
        labels_tensor = torch.FloatTensor(size=[len(videos_length), 2])
        for i, element in enumerate(batch):
            padding = max_length-videos_length[i]
            if padding > 0:
                padding = torch.zeros([padding,3,2])
                video = torch.cat([element[0], padding])
            else:
                video = element[0]
            labels = element[2]
            videos_tensor[i] = video
            labels_tensor[i] = element[2]
        return videos_tensor, labels_tensor, torch.Tensor(videos_length)

if __name__== "__main__":
    dataset = Spo2Dataset('sample_data')
    dataloader = Spo2DataLoader(dataset, batch_size=4, collate_fn= Spo2DataLoader.collate_fn)
    for videos_batch, labels_batch, videos_lengths in dataloader:
        print('Padded video (length, color, (mean,std)): ', videos_batch[0].shape)
        print('Video original length: ', videos_lengths[0])
        print('Labels (so2, hr): ', labels_batch[0])