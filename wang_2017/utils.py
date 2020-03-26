import cv2
import scipy.signal as signal
import numpy as np
import math
import scipy


import matplotlib.pyplot as plt 

def getColorComponentsMean(frame, blue = 0, green = 1, red = 2, normalize = False):
    
    blue_channel = frame[:,:,blue]
    green_channel = frame[:,:,green]
    red_channel = frame[:,:,red]
    
    #red_channel_tmp = frame.copy()
    #blue_channel_tmp = frame.copy()
    
    #red_channel_tmp[:,:,2] = np.zeros([red_channel_tmp.shape[0], red_channel_tmp.shape[1]])
    #blue_channel_tmp[:,:,0] = np.zeros([blue_channel_tmp.shape[0], blue_channel_tmp.shape[1]])
    
    
    #cv2.imshow("removed red", red_channel_tmp)
    #cv2.imshow("removed blue", blue_channel_tmp)
    ##cv2.imshow("green", green_channel)
    #cv2.waitKey(0)
    
    if normalize == True:
        
        blue_channel_normalized = cv2.normalize(blue_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        green_channel_normalized = cv2.normalize(green_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        red_channel_normalized = cv2.normalize(red_channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        blue_channel_mean = blue_channel_normalized.mean()
        green_channel_mean = green_channel_normalized.mean()
        red_channel_mean = red_channel_normalized.mean()
    else:
        blue_channel_mean = blue_channel.mean()
        green_channel_mean = green_channel.mean()
        red_channel_mean = red_channel.mean()
    
    return (blue_channel_mean, green_channel_mean, red_channel_mean)





def ppg(frames, timestamps):
    vps_green = list()
    vps_red = list()
    for frame in frames:
        green_channel_mean, red_channel_mean, = getColorComponentsMean(frame, blue = 0, green = 1, red = 2)
        vps_green.append(green_channel_mean)
        vps_red.append(red_channel_mean)
    
    # Spline interpolation
    green_channel_mean_interpolated = scipy.interpolate.CubicSpline(timestamps, green_channel_mean)
    red_channel_mean_interpolated = scipy.interpolate.CubicSpline(timestamps, red_channel_mean)
    #green_channel_mean_interpolated = scipy.interpolate.CubicSpline(timestamps, green_channel_mean)
    
    return (green_channel_mean_interpolated, red_channel_mean_interpolated)



