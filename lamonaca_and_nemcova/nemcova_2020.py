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
        #print("means", blue_channel_mean, green_channel_mean, red_channel_mean)
        return (blue_channel_mean, green_channel_mean, red_channel_mean)

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
                if frame_count >= 150:
                    break
                
                timestamps_list.append(frame_count / fps)
                
                frame_count = frame_count + 1
            
                blue_channel_mean, green_channel_mean, red_channel_mean = self.ppg_nemcova(frame)
                ppg_full_green.append(green_channel_mean)
                ppg_full_red.append(red_channel_mean)
                ppg_full_blue.append(blue_channel_mean)
            else:
                break
        
        cap.release()
        
        print("Video length", frame_count / fps)
        print("timestamps", timestamps_list)
        
        #ppg_full_green.reverse()
        #ppg_full_red.reverse()
        #ppg_full_blue.reverse()
        #timestamps_list.reverse()
        
        
        #Slice ten second in the middle
        
        five_sec = 5 * fps
        print("five sec", five_sec)
        
        ppg_full_green = np.asarray(ppg_full_green)
        ppg_full_red = np.asarray(ppg_full_red)
        ppg_full_blue = np.asarray(ppg_full_blue)
        timestamps_list = np.asarray(timestamps_list)
        
        l = int(len(ppg_full_green)/2)
        down = l - five_sec
        up = l + five_sec
        if down < 0:
            down = 0
        if up >= len(ppg_full_green):
            up = len(ppg_full_green) - 1
        print("Midddle", l, " from ", down , " to ", up)
        
        ppg_full_green = ppg_full_green[down : up]
        ppg_full_red = ppg_full_red[down : up]
        ppg_full_blue = ppg_full_blue[down : up]
        timestamps_list = timestamps_list[down : up]
        
        assert(len(ppg_full_green) == len(ppg_full_red) == len(ppg_full_blue))
        
        
        #vps_green_normalized = [(float(i) - min(vps_green))/(max(vps_green) - min(vps_green)) for i in vps_green]
        #vps_red_normalized = [(float(i) - min(vps_red))/(max(vps_red) - min(vps_red)) for i in vps_red]
        green_absolute_max = max([abs(i) for i in ppg_full_green])
        red_absolute_max = max([abs(i) for i in ppg_full_red])
        
        assert(green_absolute_max >= 0)
        assert(red_absolute_max >= 0)
        
        ppg_green_normalized = [float(i)/green_absolute_max for i in ppg_full_green]
        ppg_red_normalized = [float(i)/red_absolute_max for i in ppg_full_red]
        
        timestamps = np.array( timestamps_list )
        
        duration = frame_count/fps
        
        # First, design the Buterworth filter
        ppg_green_filtered_normalized = self.cutoff_fir(ppg_green_normalized, 3.2, fps)
        ppg_red_filtered_normalized = self.cutoff_fir(ppg_red_normalized, 2.2, fps)
        
        print(len(ppg_green_filtered_normalized))
        ppg_green_filtered_normalized = np.delete(ppg_green_filtered_normalized, [0])
        print(len(ppg_green_filtered_normalized))
        
        ppg_red_filtered_normalized = np.delete(ppg_red_filtered_normalized, [0])
        print(len(ppg_red_filtered_normalized))
        
        timestamps = np.delete(timestamps, [0])
        print(len(timestamps))
        
        
        # The phase delay of the filtered signal.
        #delay = 0.5 * (N-1) / sample_rate


        #print("time", timestamps)
        #print("signal", vps_red_normalized)
        plt.figure(0)
        plt.plot(timestamps, ppg_red_normalized[1:])
        plt.plot(timestamps, ppg_red_filtered_normalized, 'g')
        #plt.plot(timestamps, vps_red[1:], 'b')

        plt.xlabel('t')
        plt.grid(True)
        
        plt.figure(1)
        plt.plot(timestamps, ppg_green_normalized[1:])
        plt.plot(timestamps, ppg_green_filtered_normalized, 'g')
        #plt.plot(timestamps, vps_green[1:], 'b')

        plt.xlabel('t')
        plt.grid(True)

        #plt.show()
        
        
        
        return (ppg_green_filtered_normalized, ppg_full_green, ppg_red_filtered_normalized, ppg_full_red, ppg_full_blue, duration, timestamps, fps)
    
    
    #def spo2_estimation(self, video_frames, timestamps):
        #vps_940_green_filtered, vps_600_red_filtered = self.ppg_filtered(video_frames)
        #return utils.spo2_estimation(vps_940_green_filtered, vps_600_red_filtered, timestamps)
    
    
    def spo2_estimation(self, video_file, optimize = True):
        ppg_green_filtered_normalized, ppg_full_green, ppg_red_filtered_normalized, ppg_full_red, ppg_full_blue, duration, timestamps, fps = self.ppg_filtered_from_video_file(video_file)
        
        print("Estimation")
        spo2, hr = utils.spo2_estimation(ppg_green_filtered_normalized, ppg_red_filtered_normalized, timestamps, fps)
        
        
        if optimize == True:
            print("Before optimization", spo2, hr)
            pr = np.amax(ppg_full_red) - np.amin(ppg_full_red)
            pg = np.amax(ppg_full_green) - np.amin(ppg_full_green)
            pb = np.amax(ppg_full_blue) - np.amin(ppg_full_blue)
            
            print(pr, " " , pg, " ", pb)
            
            spo2 = (spo2 + (pr * 0.25) - (pg * 0.05) - (pb * 0.55)) - 25.9152
        
        return spo2, hr
            
            
            #(SpO 2 × 100 + (P R × 0.25) − (P G × 0.05) − (P B × 0.55)) − 25.9152
            
            
            
        
    
    
