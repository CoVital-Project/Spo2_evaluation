import cv2
import numpy as np


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


def load_video(video_file):
    """
        Load a video and return the PPG signal on each color channel.
    """
    ppg_full_green = list()
    ppg_full_red = list()
    ppg_full_blue = list()
    frame_count = 0
    
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        fps = 30
    
    print("FPS", fps)
    
    timestamps_list = list()
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == True:
            
            if(frame_count % 50 == 0):
                print("Frame:", frame_count)
            #if frame_count >= 60:
                #break
            timestamps_list.append(frame_count / fps)
            frame_count = frame_count + 1
        
            blue_channel_mean, green_channel_mean, red_channel_mean = getColorComponentsMean(frame)
            
            ppg_full_green.append(green_channel_mean)
            ppg_full_red.append(red_channel_mean)
            ppg_full_blue.append(blue_channel_mean)
        else:
            break
    
    cap.release()
    
    print("Video length", frame_count / fps)
    print("timestamps", timestamps_list)
 
    #Slice ten second in the middle
    ppg_full_green = np.asarray(ppg_full_green)
    ppg_full_red = np.asarray(ppg_full_red)
    ppg_full_blue = np.asarray(ppg_full_blue)
    timestamps_list = np.asarray(timestamps_list)
    
    return ppg_full_green, ppg_full_blue, ppg_full_red, timestamps_list


def load_dataset(folder_dataset):
    pass


def main():
    load_video("Video/PC000001.mp4")
    print("Done")

if __name__== "__main__":
  main()
