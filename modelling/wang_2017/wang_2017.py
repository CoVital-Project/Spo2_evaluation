import cv2
import scipy.signal as signal
import numpy as np
import math

import utils
import matplotlib.pyplot as plt 
import scipy


class Wang2017(object):
    def __init__(self):
        pass
    
    
    def ppg_from_video_file(self, video_file):
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
                #if frame_count >= 100:
                    #break
                
                timestamps_list.append(frame_count / fps)
                
                frame_count = frame_count + 1
            
                blue_channel_mean, green_channel_mean, red_channel_mean = utils.getColorComponentsMean(frame)
                ppg_full_green.append(green_channel_mean)
                ppg_full_red.append(red_channel_mean)
                ppg_full_blue.append(blue_channel_mean)
            else:
                break
        
        cap.release()
        
        print("Video length", frame_count / fps)
        print("timestamps", timestamps_list)
        print("ppg len", len(ppg_full_green))
        
        ppg_full_green = np.asarray(ppg_full_green)
        ppg_full_red = np.asarray(ppg_full_red)
        ppg_full_blue = np.asarray(ppg_full_blue)
        timestamps= np.asarray(timestamps_list)
        
        assert(len(ppg_full_green) == len(ppg_full_red) == len(ppg_full_blue))        
        
        ppg_green_spline = scipy.interpolate.CubicSpline(timestamps, ppg_full_green)
        ppg_red_spline = scipy.interpolate.CubicSpline(timestamps, ppg_full_red)
        
        
        ## ATTENTION: I'm manually creating a sampling rate of 60 fps since the paper use a cubic spline but does not detail the inrease in sampling rate of the spline
        
        fps_spline = 60 
        video_length = frame_count / fps
        
        timestamps_spline = np.arange(0, video_length, 1/fps_spline)
        ppg_green_interpolated = ppg_green_spline(timestamps_spline)
        ppg_red_interpolated = ppg_red_spline(timestamps_spline)
        
        #Slice ten second in the middle
        
        five_sec = int(5 * fps_spline)
        print("five sec", five_sec)
        l = int(len(ppg_green_interpolated)/2)
        down = l - five_sec
        up = l + five_sec
        if down < 0:
            down = 0
        if up >= len(ppg_green_interpolated):
            up = len(ppg_green_interpolated) - 1
        print("Midddle", l, " from ", down , " to ", up)
        
        ppg_green_interpolated = ppg_green_interpolated[down : up]
        ppg_red_interpolated = ppg_red_interpolated[down : up]
        #ppg_full_blue = ppg_full_blue[down : up]
        timestamps_spline = timestamps_spline[down : up]
        
        # The phase delay of the filtered signal.
        #delay = 0.5 * (N-1) / sample_rate


        ##print("time", timestamps)
        ##print("signal", vps_red_normalized)
        #plt.figure(0)
        #plt.plot(timestamps_spline, ppg_red_interpolated)
        ##plt.plot(timestamps_spline, ppg_red_filtered_normalized, 'g')
        ##plt.plot(timestamps, vps_red[1:], 'b')

        #plt.xlabel('t')
        #plt.grid(True)
        
        #plt.title("PPG signal for the color red")
        
        #plt.figure(1)
        #plt.plot(timestamps_spline, ppg_green_interpolated)
        ##plt.plot(timestamps_spline, ppg_green_filtered_normalized, 'g')
        ##plt.plot(timestamps, vps_green[1:], 'b')

        #plt.xlabel('t')
        #plt.grid(True)
        #plt.title("PPG signal for the color green")

        #plt.show()
        
        
        
        return (ppg_green_interpolated, ppg_red_interpolated, video_length, timestamps_spline, fps_spline)
    
    def heart_rate(self, ppg_signal, fps_spline, timestamps):
        
        print("FPS", fps_spline)
        
        ppg_signal = ppg_signal - ppg_signal.mean()
        
        #plt.figure(3)
        #plt.plot(timestamps, ppg_signal)
        ###plt.plot(timestamps_spline, ppg_green_filtered_normalized, 'g')
        ###plt.plot(timestamps, vps_green[1:], 'b')

        #plt.xlabel('t')
        #plt.grid(True)
        #plt.title("PPG signal for the color green FFT input")

        #plt.show()
        
        
        temp_fft = np.fft.fft(ppg_signal)
        temp_psd = np.abs(temp_fft) ** 2
        #sampling_rate = 1 / fps_spline
        ## now 1 unit correspond to 1 sec since we give one second divided by the number of frame
        fftfreq = np.fft.fftfreq(len(temp_psd), 1 / fps_spline)
        positive_freq = fftfreq > 0
        
        #index_0_8 = 0.8 / fps_spline
        #index_2_0 = 7 / fps_spline
        
        temp_fft = np.abs(temp_fft)
        
        #positive_freq_0_8_2_0 = positive_freq[index_0_8:index_2_0]
        #temp_fft = temp_fft[index_0_8:index_2_0].real
        
        #print(temp_fft, "MAX", np.amax(temp_fft))
        
        
        hr = np.argmax(temp_fft[positive_freq])
        
        #plt.plot(fftfreq[positive_freq], temp_fft[positive_freq].real)
        #plt.show()
        
        sampling_freq = 1 / fps_spline
        print("sp", sampling_freq, "signal len", len(ppg_signal), "highest freq", hr, "or", fftfreq[hr], "value", temp_fft[hr].real, " the max ", np.amax(temp_fft[positive_freq].real))
        
        hr = fftfreq[hr]

        return hr
    
    
    def spo2_estimation(self, video_file, optimize = True):
        ppg_green_filtered_normalized, ppg_full_green, ppg_red_filtered_normalized, ppg_full_red, ppg_full_blue, duration, timestamps, fps = self.ppg_filtered_from_video_file(video_file)
        
        print("Estimation")
        spo2, hr = utils.spo2_estimation(ppg_green_filtered_normalized, ppg_red_filtered_normalized, timestamps, fps)
        
        spo2 = np.median(spo2)
        
        if optimize == True:
            print("Before optimization", spo2, hr)
            pr = np.amax(ppg_full_red) - np.amin(ppg_full_red)
            pg = np.amax(ppg_full_green) - np.amin(ppg_full_green)
            pb = np.amax(ppg_full_blue) - np.amin(ppg_full_blue)
            
            print(pr, " " , pg, " ", pb)
            
            spo2 = (100 * spo2 + (pr * 0.25) - (pg * 0.05) - (pb * 0.55)) - 25.9152
        
        return spo2, hr
            
            
            #(SpO 2 × 100 + (P R × 0.25) − (P G × 0.05) − (P B × 0.55)) − 25.9152
            
            
            
        
    
    
