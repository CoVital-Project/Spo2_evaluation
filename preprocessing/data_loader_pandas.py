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

import pandas as pd

import data_loader

#def timing(f):
    #def wrap(*args):
        #time1 = time.time()
        #ret = f(*args)
        #time2 = time.time()
        #print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        #return ret
    #return wrap


class Spo2DatasetPandas(data_loader.Spo2Dataset):
    """Spo2Dataset dataset using pandas.
        It preprocess the data in order to create a Dataset with the average and std of each channel per frame, the ground truth labels, and the video fps.
        It is store in a pandas Dataframe wehre each row correspond to one of those element and video are stacked up.
    """    
    
    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the data folder.
        """
        self.data_path = data_path
        self.video_folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,folder))]
        print("v", self.video_folders)
        self.number_of_videos = len(self.video_folders)
        #print(self.number_of_videos)
        
        data_list = []
        column = []
        nb_video = 1
        for video in self.video_folders:
            
            ppg_red_mean = list()
            ppg_green_mean = list()
            ppg_blue_mean = list()
            ppg_red_std = list()
            ppg_green_std = list()
            ppg_blue_std = list()
            print("Loading video:", nb_video)
            
            #print(data_list)
            
            nb_video += 1
            ppg = []
            video_path = os.path.join(self.data_path, video)
            video_file = os.path.join(video_path, [file_name for file_name in os.listdir(video_path) if file_name.endswith('mp4')][0])
            vidcap = cv2.VideoCapture(video_file)
            meta = {}
            meta['video_fps'] = vidcap.get(cv2.CAP_PROP_FPS)
            (grabbed, frame) = vidcap.read()
            frame_count = 1
            while grabbed:
                blue_channel_mean, blue_channel_std, green_channel_mean, green_channel_std, red_channel_mean, red_channel_std = self.transform_faster(frame)
                
                ppg_blue_mean.append(blue_channel_mean)
                ppg_green_mean.append(green_channel_mean)
                ppg_red_mean.append(red_channel_mean)
                ppg_blue_std.append(blue_channel_std)
                ppg_green_std.append(green_channel_std)
                ppg_red_std.append(red_channel_std)
                
                ppg.append(frame)
                (grabbed, frame) = vidcap.read()
                if(frame_count % 50 == 0):
                    print("Frame:", frame_count)
                    break
                frame_count += 1
            with open(os.path.join(video_path, 'gt.json'), 'r') as f:
                ground_truth = json.load(f)
                
            #print("b", data_list)
                
            data_list.append(ppg_blue_mean)
            #print(data_list)
            data_list.append(ppg_blue_std)
            data_list.append(ppg_green_mean)
            data_list.append(ppg_green_std)
            data_list.append(ppg_red_mean)
            data_list.append(ppg_red_std)
            data_list.append([int(ground_truth['SpO2'])])
            data_list.append([int(ground_truth['HR'])])
            data_list.append([meta['video_fps']])
            
            column = column + ["blue", "blue std", "green", "green std", "red", "red std", "spo2_gt", "hr_gt", "fps"]
            
            #print(data_list)
                        
        print(column)
        self.data =  pd.DataFrame(data_list, column)
        #self.data = self.data.T
        print(self.data)
        
    def get_video(self, index):
        index_frame = 9 * index
        return self.data.iloc[index_frame:index_frame+9]


if __name__== "__main__":
    dataset = Spo2DatasetPandas('test_data')
    
