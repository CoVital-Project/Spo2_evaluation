from torch.utils.data import Dataset, DataLoader

import torchvision
import os
import json
import torch

class Spo2Dataset(Dataset):
    """Spo2Dataset dataset."""

    def __init__(self, data_path):
        """
        Args:
            data_path (string): Path to the data folder.
        """
        self.data_path = data_path
        self.video_folders = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path,folder))]

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = os.path.join(self.data_path, self.video_folders[idx])
        video_file = os.path.join(video_path, [file_name for file_name in os.listdir(video_path) if file_name.endswith('mp4')][0])
        with open(os.path.join(video_path, 'gt.json'), 'r') as f:
            ground_truth = json.load(f)
        v, _, meta = torchvision.io.read_video(video_file, pts_unit="sec")

        v = v.float()
        #v = v/255

        v.resize_(v.shape[0], v.shape[1]*v.shape[2], v.shape[3])
        '''if torch.cuda.is_available():
            v = v.to(device='cuda')'''
        mean = v.mean(dim=1)
        std = v.std(dim=1)
        v = torch.stack([mean, std]).reshape(-1,3,2)
        labels = torch.Tensor([int(ground_truth['Spo2']), int(ground_truth['heart_rate'])])
        return [v,meta,labels]

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

    dataset = Spo2Dataset('data')
    dataloader = Spo2DataLoader(dataset, batch_size=4, collate_fn= Spo2DataLoader.collate_fn)
    for videos_batch, labels_batch, videos_lengths in dataloader:
        print('Padded video (length, color, (mean,std)): ', videos_batch[0].shape)
        print('Video original length: ', videos_lengths[0])
        print('Labels (so2, hr): ', labels_batch[0])