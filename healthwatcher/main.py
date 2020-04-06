#method of Scully et al.: "Physiological Parameter monitoring from optical recordings with a mobile phone"

import torchvision
from matplotlib import pyplot as plt

import DataLoader


def health_watcher(ppg_blue, ppg_blue_std, ppg_red, ppg_red_std, meta):
    print(meta)
    fps = meta['video_fps']
    
    A=100 # From "determination of spo2 and heart-rate using smartphone camera
    B=5

    #TODO curve fitting for A and B. If we have more data, we can do a linear regression that best fits all patients
    #TODO Add all mp4 from figshare
    #TODO Add all ground truth from figshare

    spo2 = (A - B*(ppg_red_std / ppg_red )/(ppg_blue_std / ppg_blue)).numpy()
    secs_to_smooth = 10
    frames_to_smooth = int(10*fps)
    spo2_smooth = [spo2[i:i+frames_to_smooth].mean() for i in range(len(spo2)-frames_to_smooth)]

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
    dataset = DataLoader.Spo2Dataset('test_data')
    dataloader = DataLoader.Spo2DataLoader(dataset, batch_size=4, collate_fn= DataLoader.Spo2DataLoader.collate_fn)
    for videos_batch, labels_batch, videos_lengths in dataloader:
        print('Padded video (length, color, (mean,std)): ', videos_batch[0].shape)
        print('Video original length: ', videos_lengths[0])
        print('Labels (so2, hr): ', labels_batch[0])
        
        # To numpy array of tensor
        numpy_ppg = list()
        for el in dataset.videos_ppg:
            numpy_ppg.append(el.numpy())
        
        
        
        print(health_watcher(dataset.))
