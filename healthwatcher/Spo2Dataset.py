from torch.utils.data import Dataset
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
        v, _, meta = torchvision.io.read_video(video_file, pts_unit="sec")  ## assumes it's being run from `healthwatcher` directory

        return [v,meta,ground_truth]
