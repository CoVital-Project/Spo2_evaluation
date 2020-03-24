import cv2
import scipy.signal as signal
import numpy as np
import math

import utils
import matplotlib.pyplot as plt 


class Nemcova2020(object):
    def __init__(self):
        pass
    
    def ppg_nemcova(self, frame, blue = 0, green = 1, red = 2):
        #blue_channel_mean, green_channel_mean, red_channel_mean = utils.getColorComponentsMean(frame, blue, green, red)
        blue_channel_mean, green_channel_mean, red_channel_mean = utils.getColorComponentsMean(frame, blue, green, red)
        
        return (green_channel_mean, red_channel_mean)

    def ppg_filtered(self, frames, sampling_rate):
        vps_green = list()
        vps_red = list()
        for frame in frames:
            green_channel_mean, red_channel_mean, = self.ppg_nemcova(frame)
            vps_green.append(green_channel_mean)
            vps_red.append(red_channel_mean)
            
            
        # Normalize signal
        vps_green_normalized = [float(i)/max(vps_green) for i in vps_green]
        vps_red_normalized = [float(i)/max(vps_red) for i in vps_red]
        
            
        # First, design the Buterworth filter
        vps_940_green_filtered_normalized = self.cutoff_fir(vps_green_normalized, 3.2, sampling_rate)
        vps_600_red_filtered_normalized = self.cutoff_fir(vps_red_normalized, 2.2, sampling_rate)
        
        return (vps_940_green_filtered_normalized, vps_600_red_filtered_normalized)
    
    
    def cutoff_fir(self, data, cutoff_freq, sampling_rate):
        
        # The Nyquist rate of the signal.
        nyq_rate = sampling_rate / 2.0

        # The desired width of the transition from pass to stop,
        # relative to the Nyquist rate.  We'll design the filter
        # with a 5 Hz transition width.
        #width = 5.0/nyq_rate

        # The desired attenuation in the stop band, in dB.
        #ripple_db = 60.0

        # Compute the order and Kaiser parameter for the FIR filter.
        #N, beta = signal.kaiserord(ripple_db, width)
        N = 2
        # The cutoff frequency of the filter.
        #cutoff_hz = 10.0

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = signal.firwin(N, cutoff_freq/nyq_rate)

        # Use lfilter to filter x with the FIR filter.
        filtered_data = signal.lfilter(taps, 1.0, data)
        
        return filtered_data
        
        #print(cutoff_freq, "and")
        #n = 61
        #b = signal.firwin(n, cutoff_freq, window = "hamming")
        #return signal.lfilter(b, [1.0], list_input)
    
    
    def ppg_filtered_from_video_file(self, video_file):
        vps_green = list()
        vps_red = list()
        frame_count = 0
        
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        timestamps_list = list()
        
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            if ret == True:
                
                if(frame_count % 50 == 0):
                    print("Frame:", frame_count)
                if frame_count >= 150:
                    break
                
                timestamps_list.append(frame_count / fps)
                
                frame_count = frame_count + 1
            
                green_channel_mean, red_channel_mean, = self.ppg_nemcova(frame)
                vps_green.append(green_channel_mean)
                vps_red.append(red_channel_mean)
            else:
                break
        
        cap.release()
        
        #vps_green_normalized = [(float(i) - min(vps_green))/(max(vps_green) - min(vps_green)) for i in vps_green]
        #vps_red_normalized = [(float(i) - min(vps_red))/(max(vps_red) - min(vps_red)) for i in vps_red]
        green_absolute_max = max([abs(i) for i in vps_green])
        red_absolute_max = max([abs(i) for i in vps_red])
        
        assert(green_absolute_max >= 0)
        assert(red_absolute_max >= 0)
        
        vps_green_normalized = [float(i)/green_absolute_max for i in vps_green]
        vps_red_normalized = [float(i)/red_absolute_max for i in vps_red]
        
        timestamps = np.array( timestamps_list )
        
        duration = frame_count/fps
        
        # First, design the Buterworth filter
        vps_green_filtered_normalized = self.cutoff_fir(vps_green_normalized, 3.2, fps)
        vps_red_filtered_normalized = self.cutoff_fir(vps_red_normalized, 2.2, fps)
        
        print(len(vps_green_filtered_normalized))
        vps_green_filtered_normalized = np.delete(vps_green_filtered_normalized, [0])
        print(len(vps_green_filtered_normalized))
        
        vps_red_filtered_normalized = np.delete(vps_red_filtered_normalized, [0])
        print(len(vps_red_filtered_normalized))
        
        timestamps = np.delete(timestamps, [0])
        print(len(timestamps))
        
        
        # The phase delay of the filtered signal.
        #delay = 0.5 * (N-1) / sample_rate


        #print("time", timestamps)
        #print("signal", vps_red_normalized)
        plt.figure(0)
        plt.plot(timestamps, vps_red_normalized[1:])
        plt.plot(timestamps, vps_red_filtered_normalized, 'g', linewidth=4)
        #plt.plot(timestamps, vps_red[1:], 'b')

        plt.xlabel('t')
        plt.grid(True)
        
        plt.figure(1)
        plt.plot(timestamps, vps_green_normalized[1:])
        plt.plot(timestamps, vps_green_filtered_normalized, 'g', linewidth=4)
        #plt.plot(timestamps, vps_green[1:], 'b')

        plt.xlabel('t')
        plt.grid(True)

        #plt.show()
        
        
        
        return (vps_green_filtered_normalized, vps_red_filtered_normalized, duration, timestamps, fps)
    
    
    def spo2_estimation(self, video_frames, timestamps):
        vps_940_green_filtered, vps_600_red_filtered = self.ppg_filtered(video_frames)
        return utils.spo2_estimation(vps_940_green_filtered, vps_600_red_filtered, timestamps)
    
    
    def spo2_estimation(self, video_file):
        vps_green_filtered_normalized, vps_red_filtered_normalized, duration, timestamps, fps = self.ppg_filtered_from_video_file(video_file)
        
        print("Estimation")
        #print("time", timestamps)
        
        return utils.spo2_estimation(vps_green_filtered_normalized, vps_red_filtered_normalized, timestamps, fps)
        
    
    
