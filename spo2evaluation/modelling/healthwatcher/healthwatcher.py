"""
method of Scully et al.: "Physiological Parameter monitoring from optical recordings with a mobile phone"
"""

import torchvision
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def predict_spo2_from_df(data: pd.DataFrame, A=100, B=5, fps=30.0, smooth=True):
    """Predict SpO2 values from a dataframe of mean and std values for each colour.

    A = 100  # From "determination of spo2 and heart-rate using smartphone camera
    B = 5
    """
    if (not isinstance(data, pd.DataFrame)) or ('mean_blue' not in data.columns):
        raise ValueError(
            "You need to pass in a dataframe with 'mean_<colour>' and 'std_<colour>' columns")

    ppg_blue = data['mean_blue']
    ppg_blue_std = data['std_blue']
    ppg_red = data['mean_red']
    ppg_red_std = data['std_red']


    #TODO curve fitting for A and B. If we have more data, we can do a linear regression that best fits all patients

    spo2 = (A - B*(ppg_red_std / ppg_red)/(ppg_blue_std / ppg_blue))

    if smooth:
        secs_to_smooth = 10
        frames_to_smooth = int(10*fps)
        spo2_smooth = [spo2[i:i+frames_to_smooth].mean()
                       for i in range(len(spo2)-frames_to_smooth)]
    else:
        spo2_smooth = spo2

    return np.mean(spo2_smooth)


def health_watcher_old(video, meta):
    #v, _, meta = torchvision.io.read_video('../data/S98T89.mp4', pts_unit="sec")  ## assumes it's being run from `healthwatcher` directory
    print(meta)
    fps = meta['video_fps']

    blue = 0
    green = 1
    red = 2

    print(meta)

    # smash width and height together
    video.resize_(video.shape[0], video.shape[1]
                  * video.shape[2], video.shape[3])

    bc, gc, rc = video[:, :, blue].float(), video[:, :, green].float(
    ), video[:, :, red].float()  # get separate channels

    bc_mean = bc.mean(dim=1)    # calc mean and std for each channel
    bc_std = bc.std(dim=1)

    rc_mean = rc.mean(dim=1)
    rc_std = rc.std(dim=1)

    A = 100  # From "determination of spo2 and heart-rate using smartphone camera
    B = 5

    #TODO curve fitting for A and B. If we have more data, we can do a linear regression that best fits all patients
    #TODO Add all mp4 from figshare
    #TODO Add all ground truth from figshare
    spo2 = (A - B*(rc_std / rc_mean)/(bc_std / bc_mean)).numpy()
    secs_to_smooth = 10
    frames_to_smooth = int(10*fps)
    spo2_smooth = [spo2[i:i+frames_to_smooth].mean()
                   for i in range(len(spo2)-frames_to_smooth)]

    x = [i for i in range(len(spo2_smooth))]

    plt.figure()
    plt.plot(x, spo2_smooth)
    plt.show()

