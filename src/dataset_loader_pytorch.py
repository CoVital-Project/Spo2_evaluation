import sys
sys.path.append('..')
import os
from pathlib import Path

from spo2evaluation.preprocessing import data_loader_pytorch

from torch.utils.data import DataLoader

if __name__ == "__main__":
    DATADIR = Path('data/nemcova_data')
    PERSIST = False

    dataset = data_loader_pytorch.Spo2DatasetPyTorch(DATADIR)

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
        dataset, batch_size=4, collate_fn=data_loader_pytorch.spo2_collate_fn)
    for videos_batch, labels_batch, videos_lengths in dataloader:
        print("Padded video (length, color, (mean,std)): ",
              videos_batch[0].shape)
        print("Video original length: ", videos_lengths[0])
        print("Labels (SpO2, HR): ", labels_batch[0])
