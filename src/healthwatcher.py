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
        help="select folder holding the sanitize",
        metavar="FOLDER",
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    while data_folder is None:
        data_folder = input("Data folder> ")
    sample_path = os.path.join(data_folder, "sample_data")
    path_gt = os.path.join(data_folder, "sample_data")

    print("Loading dataset. this will take a while")
    dataset = data_loader_pandas.Spo2DatasetPandas(data_folder)

    if dataset.is_pickled():
        print("Data was already pickled")
    else:
        print("Picling the data")
        dataset.pickle_data()

    # for i in range(dataset.number_of_videos):
    #     df = dataset.get_video(i)
    #
    #     df = df.T
    #
    #     blue = df['blue'].to_numpy()
    #     red = df['red'].to_numpy()
    #     green = df['green'].to_numpy()
    #     blue_std = df['blue std'].to_numpy()
    #     red_std = df['red std'].to_numpy()
    #     green_std = df['green std'].to_numpy()
    #     fps = df['fps'][0]
    #
    #     spo2 = healthwatcher.health_watcher(blue, blue_std, red, red_std, fps,
    #                                         smooth=False)
    #     print(spo2)
