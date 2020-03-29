#method of Scully et al.: "Physiological Parameter monitoring from optical recordings with a mobile phone"

import torchvision
from matplotlib import pyplot as plt
from Spo2Dataset import Spo2Dataset

dataset = Spo2Dataset('../data')

v, meta, gt = dataset[0]# = torchvision.io.read_video('../data/S98T89.mp4', pts_unit="sec")  ## assumes it's being run from `healthwatcher` directory
v, meta, gt = torchvision.io.read_video('../data/S98T89.mp4', pts_unit="sec")  ## assumes it's being run from `healthwatcher` directory

print(gt)
print(meta)
fps = meta['video_fps']

blue=0
green=1
red=2

print(meta)

#v.resize_(v.shape[0], v.shape[1]*v.shape[2], v.shape[3])    # smash width and height together

#bc, gc, rc = v[:,:,blue].float(), v[:,:,green].float(), v[:,:,red].float()  # get separate channels

bc_mean = v[:,blue,0]    # calc mean and std for each channel
bc_std = v[:,blue,1]

rc_mean= v[:,red,0]
rc_std = v[:,red,1]

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



