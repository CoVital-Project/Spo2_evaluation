import os
import sys
sys.path.append('..')
from argparse import ArgumentParser
import spo2evaluation
from spo2evaluation.modelling.healthwatcher import healthwatcher
from spo2evaluation.preprocessing import data_loader_pandas


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-directory",
        dest="data_folder",
        help="select folder holding the sanitized video data",
        metavar="FOLDER",
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    while data_folder is None:
        data_folder = input("Data folder> ")
    sample_path = os.path.join(data_folder, "sample_data")
    path_gt = os.path.join(data_folder, "sample_data")

    print("Loading dataset. this will take a while")
    dataset = data_loader_pandas.Spo2DatasetPandas(data_folder, read_pickle=False)

    if dataset.is_pickled():
        print("Data was already pickled")
        print(dataset.sample_data)
        print(dataset.ground_truths_sample)
        print(dataset.meta)
    else:
        print("Pickling the data")
        dataset.pickle_data()

    for i in range(dataset.number_of_videos):
        sample, gt, meta = dataset.get_video(i)
        fps = meta['fps'].iloc[0]

        spo2 = healthwatcher.predict_spo2_from_df(sample, fps=fps, smooth=False)
        print(f"Predicted SpO2: {spo2: .2f}")
