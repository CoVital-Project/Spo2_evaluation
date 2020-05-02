from torch.utils.data import Dataset
import cv2
import numpy as np
import json
import torch
import time
from . import data_loader


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(
            f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Spo2DatasetPyTorch(Dataset, data_loader.Spo2Dataset):
    """Spo2Dataset pytorch dataset.
        It preprocess the data in order to create a Dataset with the average and std of each channel per frame.
        The process is slow so it may take a while to create the Dataset when first initated.
    """

    def __init__(self, data_path, file_type='mp4', crop=False, rescale=True):
        """
        Args:
            data_path (string): Path to the data folder.
            file_type (string): File extentions to look for with videos (default: mp4)
            rescale (bool): Whether to downscale videos 50%
        """
        self.data_path = Path(data_path)
        # Recursively find all files of file_type in the data path
        self.video_folders = list(self.data_path.glob(f'**/*.{file_type}'))
        self.videos_ppg = []
        self.labels_list = []
        self.meta_list = []

        for nb_video, video in enumerate(self.video_folders):
            try:
                print(f"Loading video {nb_video} from '{video}'")
                ppg = []
                vidcap = cv2.VideoCapture(str(video))
                meta = {}
                meta['video_fps'] = vidcap.get(cv2.CAP_PROP_FPS)
                meta['video_source'] = str(video)
                (grabbed, frame) = vidcap.read()
                while grabbed:
                    if rescale:
                        frame = self.rescale_frame(frame)
                    if crop:
                        frame = self.crop_frame(frame)
                    blue_channel_mean, blue_channel_std, green_channel_mean,\
                        green_channel_std, red_channel_mean, red_channel_std\
                        = self.transform_faster(frame)
                    tmp_array = np.array([
                        [blue_channel_mean, blue_channel_std],
                        [green_channel_mean, green_channel_std],
                        [red_channel_mean, red_channel_std]])
                    
                    ppg.append(tmp_array)

                    (grabbed, frame) = vidcap.read()

                with open(video.parent/'gt.json', 'r') as f:
                    ground_truth = json.load(f)
                    if ground_truth['SpO2'] == 'Unkown':
                        ground_truth['SpO2'] = -1
                        continue

                labels = torch.Tensor(
                    [int(ground_truth['SpO2']), int(ground_truth['HR'])])
                self.videos_ppg.append(torch.Tensor(np.array(ppg)))
                self.meta_list.append(meta)
                self.labels_list.append(labels)
            except Exception as e:
                print(video, e)

    #def __init__(self, data_path, file_type='mp4', crop=False, rescale=True):
        #"""
        #Args:
            #data_path (string): Path to the data folder.
            #file_type (string): File extentions to look for with videos (default: mp4)
            #rescale (bool): Whether to downscale videos 50%
        #"""
        #self.data_path = Path(data_path)
        ## Recursively find all files of file_type in the data path
        #self.video_folders = list(self.data_path.glob(f'**/*.{file_type}'))
        #self.videos_ppg = []
        #self.labels_list = []
        #self.meta_list = []

        #for nb_video, video in enumerate(self.video_folders):
            #print(f"Loading video {nb_video} from '{video}'")
            #ppg = []
            #vidcap = cv2.VideoCapture(str(video))
            #meta = {}
            #meta['video_fps'] = vidcap.get(cv2.CAP_PROP_FPS)
            #(grabbed, frame) = vidcap.read()
            #while grabbed:
                #if rescale:
                    #frame = self.rescale_frame(frame)
                #if crop:
                    #frame = self.crop_frame(frame)
                #frame = self.transform_faster(frame)
                #ppg.append(frame)

                #(grabbed, frame) = vidcap.read()

            #with open(video.parent/'gt.json', 'r') as f:
                #ground_truth = json.load(f)
                #if ground_truth['SpO2'] == 'Unkown':
                    #ground_truth['SpO2'] = -1
                    #continue
            #labels = torch.Tensor(
                #[int(ground_truth['SpO2']), int(ground_truth['HR'])])
            #self.videos_ppg.append(torch.Tensor(np.array(ppg)))
            #self.meta_list.append(meta)
            #self.labels_list.append(labels)

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
        # labels = element[2]
        videos_tensor[i] = video
        labels_tensor[i] = element[2]
    return videos_tensor, labels_tensor, torch.Tensor(videos_length)



    #def __len__(self):
        #return len(list(self.videos_ppg))

    #def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()
        #return [self.videos_ppg[idx], self.meta_list[idx], self.labels_list[idx]]


#def spo2_collate_fn(batch):
    #videos_length = [element[0].shape[0] for element in batch]
    #max_length = max(videos_length)
    #videos_tensor = torch.FloatTensor(
        #size=[len(videos_length), max_length, 3, 2])
    #labels_tensor = torch.FloatTensor(size=[len(videos_length), 2])
    #for i, element in enumerate(batch):
        #padding = max_length-videos_length[i]
        #if padding > 0:
            #padding = torch.zeros([padding, 3, 2])
            #video = torch.cat([element[0], padding])
        #else:
            #video = element[0]
        #labels = element[2]
        #videos_tensor[i] = video
        #labels_tensor[i] = element[2]
    #return videos_tensor, labels_tensor, torch.Tensor(videos_length)


#if __name__ == "__main__":
    #dataset = Spo2Dataset('sample_data')
    #dataloader = DataLoader(
        #dataset, batch_size=4, collate_fn=spo2_collate_fn)
    #for videos_batch, labels_batch, videos_lengths in dataloader:
        #print('Padded video (length, color, (mean,std)): ',
              #videos_batch[0].shape)
        #print('Video original length: ', videos_lengths[0])
        #print('Labels (SpO2, HR): ', labels_batch[0])
