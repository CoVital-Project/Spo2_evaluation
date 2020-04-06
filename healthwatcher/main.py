
import sys
sys.path.append("../data_loader")
import data_loader_pandas

#method of Scully et al.: "Physiological Parameter monitoring from optical recordings with a mobile phone"

import torchvision
from matplotlib import pyplot as plt


def health_watcher(ppg_blue, ppg_blue_std, ppg_red, ppg_red_std, fps, smooth = True):
    #print(meta)
    #fps = meta['video_fps']
    
    A=100 # From "determination of spo2 and heart-rate using smartphone camera
    B=5

    #TODO curve fitting for A and B. If we have more data, we can do a linear regression that best fits all patients
    #TODO Add all mp4 from figshare
    #TODO Add all ground truth from figshare

    spo2 = (A - B*(ppg_red_std / ppg_red )/(ppg_blue_std / ppg_blue))
    
    if smooth:
        secs_to_smooth = 10
        frames_to_smooth = int(10*fps)
        spo2_smooth = [spo2[i:i+frames_to_smooth].mean() for i in range(len(spo2)-frames_to_smooth)]
    else:
        spo2_smooth = spo2

    x = [i for i in range(len(spo2_smooth))]
    
    plt.figure()
    plt.plot(x, spo2_smooth)
    plt.show()
    
    return spo2_smooth.mean()


def health_watcher_old(video, meta):
    #v, _, meta = torchvision.io.read_video('../data/S98T89.mp4', pts_unit="sec")  ## assumes it's being run from `healthwatcher` directory
    print(meta)
    fps = meta['video_fps']

    blue=0
    green=1
    red=2

    print(meta)

    video.resize_(video.shape[0], video.shape[1]*video.shape[2], video.shape[3])    # smash width and height together

    bc, gc, rc = video[:,:,blue].float(), video[:,:,green].float(), video[:,:,red].float()  # get separate channels

    bc_mean = bc.mean(dim=1)    # calc mean and std for each channel
    bc_std = bc.std(dim=1)

    rc_mean=rc.mean(dim=1)
    rc_std = rc.std(dim=1)

    A=100 # From "determination of spo2 and heart-rate using smartphone camera
    B=5


    #TODO curve fitting for A and B. If we have more data, we can do a linear regression that best fits all patients
    #TODO Add all mp4 from figshare
    #TODO Add all ground truth from figshare


    spo2 = (A - B*(rc_std / rc_mean )/(bc_std / bc_mean)).numpy()
    secs_to_smooth = 10
    frames_to_smooth = int(10*fps)
    spo2_smooth = [spo2[i:i+frames_to_smooth].mean() for i in range(len(spo2)-frames_to_smooth)]


    x = [i for i in range(len(spo2_smooth))]

    plt.figure()
    plt.plot(x, spo2_smooth)
    plt.show()



if __name__== "__main__":
    dataset = data_loader_pandas.Spo2DatasetPandas('test_data')
    
    for i in range(dataset.number_of_videos):
        df = dataset.get_video(i)
        
        df = df.T
        
        blue = df['blue'].to_numpy()
        red = df['red'].to_numpy()
        green = df['green'].to_numpy()
        blue_std = df['blue std'].to_numpy()
        red_std = df['red std'].to_numpy()
        green_std = df['green std'].to_numpy()
        fps = df['fps'][0]
        
        spo2 = health_watcher(blue, blue_std, red, red_std, fps, smooth=False)
        print(spo2)
    
    
    
    
