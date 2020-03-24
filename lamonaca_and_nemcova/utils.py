import cv2
import scipy.signal as signal
import numpy as np
import math


import matplotlib.pyplot as plt 

def getColorComponentsMean(frame, blue = 0, green = 1, red = 2, normalize = False):
    
    blue_channel = frame[:,:,blue]
    green_channel = frame[:,:,green]
    red_channel = frame[:,:,red]
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

#def cutoff(list_input, cutoff_freq = 4):
    #list_input_array = np.array(list_input)
    #list_input_array = list_input_array.astype(np.float)
    #N  = 2    # Filter order
    ##cutoff_freq = 4 # Cutoff frequency
    #B, A = signal.butter(N, cutoff_freq, output='ba')
    #return signal.filtfilt(B, A, list_input_array)


def get_peak_height_slope(ppg_signal, timestamps, fps):
    
    
    ## distance is 1.5 because it correspond to the time in second for a heart beat at 40 bpm which seemed low for me
    ## prominence is half the maximum possible amplitude of the signal
    
    print("finding the peaks", fps * 1.5)
    max_v = np.amax(ppg_signal)
    min_v =np.amin(ppg_signal) 
    peaks, peaks_properties = signal.find_peaks(ppg_signal, prominence= (max_v - min_v) * 0.3, distance = 1.5)
    
    #Calculate heart beat
    times = timestamps[peaks]
    print(times)
    times_heart_beat = np.diff(times)
    print(times_heart_beat)
    heart_beat_s = times_heart_beat.mean()
    
    print("heart beat in s ", heart_beat_s, " and in minutes ", heart_beat_s * 60 , ' in fps ', heart_beat_s * fps)
    
    ## We calculate the base of the peak by only looking for the base between a given peak and the last peak on its left.
    left_base_list = list()
    count = 0
    for peak in peaks:
    #Calculate the local base of the peak in the 40bpm window
        if count == 0:
            window = None
        else:
            window = 2 * times_heart_beat[count -1] * fps 
            print(window)
        count = count + 1
            
        peak_prom = signal.peak_prominences(ppg_signal, [peak], wlen = window)
        left_base_list.append(peak_prom[1][0])
    
    left_bases = np.array( left_base_list )
    
    print("peaks", ppg_signal[peaks])
    print("bottom", ppg_signal[left_bases])
    print("timestamps", timestamps[peaks])
    
    print("Found", len(peaks), "in the signal")
    #left_bases = peak_prom[1]
    peak_values = ppg_signal[peaks]
    bottom_values = ppg_signal[left_bases]
    peak_heights = peak_values - bottom_values
    
    print("m ", peak_heights)
    
    assert(len(bottom_values) == len(peak_values))
    
    #Dirty approximation of the slope could be refined by using points in the middle of the measurements
    duration = timestamps[peaks] - timestamps[left_bases]
    
    
    slope_rad = np.arctan2(peak_heights, duration)
    assert((slope_rad >= 0).all())
    #slope_rad = math.atan2( , duration);
    # Convert to degre (?)
    slope = slope_rad * (180/math.pi)
    
    
    plt.figure(3)
    plt.plot(timestamps[peaks], ppg_signal[peaks], "ob")
    plt.plot(timestamps[2-1:], ppg_signal[2-1:], 'g')
    plt.plot(timestamps[left_bases], ppg_signal[left_bases], "or")
    plt.legend(['prominence'])
    #plt.subplot(2, 2, 3)
    #plt.show()
    
    return (peak_heights, slope, heart_beat_s * 60)


def get_peaks_heights_slopes_average(ppg_signal, timestamps, fps):
    
    #print("time slope", timestamps)
    
    peak_heights, slopes = get_peak_height_slope(ppg_signal, timestamps, fps)
    
    peak_heights_mean = peak_heights.mean()
    peak_slope_mean = slopes.mean()
    
    print("ph", peak_heights)
    print("Avg ", peak_heights_mean, " ", peak_slope_mean)
    
    return (peak_heights_mean, peak_slope_mean)


def spo2_estimation(ppg_green_940, ppg_red_600, timestamps, fps):
    '''
    * Ehb_gamma is absorption coefficients of deoxyhemoglobinat wavelength gamma
    * Ehb_0_gamma is absorption coefficients of oxyhemoglobin at wavelength gamma
    * m_600 is the slope of the systolic peak for the red color
    * m_940 is the slope of the systolic peak for the green color
    * vp_600 is the heigh systolic peak for the red color
    * vp_940 is the heigh systolic peak for the green color
    '''
    
    #print("time", timestamps)
    
    Ehb_600 = 0.81
    Ehb_940 = 0.18
    Ehb0_600 = 0.08
    Ehb0_940 = 0.29
    
    print("Getting the peaks")
    
    vp_940, m_940, hr_940 = get_peak_height_slope(ppg_green_940 ,timestamps, fps)
    vp_600, m_600, hr_600 = get_peak_height_slope(ppg_red_600, timestamps, fps)
    
    
    
    vp_940_imaginary = np.array(vp_940, dtype=complex)
    m_940_imaginary = np.array(m_940, dtype=complex)
    vp_600_imaginary = np.array(vp_600, dtype=complex)
    m_600_imaginary = np.array(m_600, dtype=complex)

    #print( m_940, " * ", math.log10(vp_940), " since ", vp_940)  
    
    #assert(vp_940 >= 0)
    #assert(m_940 >= 0)
    #assert(vp_600 >= 0)
    #assert(m_940 * math.log10(vp_940) >= 0)
    
    print("Formulation")    
    log = np.log10(vp_940_imaginary)
    #print("l", log)
    #print("m", m_940)
    mult = np.multiply(m_940_imaginary, log)
    sqrt_g = np.sqrt(mult)
    
    numerator_1 = (Ehb_600 * sqrt_g)
    numerator_2 =( Ehb_940 * np.sqrt( np.multiply(m_600_imaginary, np.log10(vp_600_imaginary)))) 
    numerator = numerator_1 - numerator_2
    denominator_1 = (np.sqrt( np.multiply(m_940_imaginary, np.log10(vp_940_imaginary))) * (Ehb_600 - Ehb0_600))
    denominator_2 =(np.sqrt( np.multiply(m_600_imaginary, np.log10(vp_600_imaginary))) * (Ehb_940 - Ehb0_940))
    denominator = denominator_1 - denominator_2
    
    spo2 = np.divide(numerator, denominator)
    
    assert(np.imag(spo2.all()) == 0)
    
    spo2 = np.real(spo2)
    
    spo2 = spo2 * 100
    
    hr_940_mean = hr_940.mean()
    hr_600_mean = hr_600.mean()
    
    hr = (hr_940_mean + hr_600_mean) / 2
    

    return spo2.mean(), hr

