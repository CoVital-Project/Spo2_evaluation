import cv2
import wang_2017


def main():
    cv2.getBuildInformation()
    #lamonaca = lamonaca_2015.Lamonaca2015()
    
    #lamonaca.lacomana("Video/S98T89.avi")
    
    wang = wang_2017.Wang2017()
    
    ppg_green_interpolated, ppg_red_interpolated, video_length, timestamps_spline, fps_spline = wang.ppg_from_video_file("../Video/S87T78.mp4")
    
    hr = wang.heart_rate(ppg_red_interpolated, fps_spline, timestamps_spline)
    
    print(hr)
    
    #o2, hr = nemcova.spo2_estimation("Video/S87T78.mp4", optimize = True)
    #print("Done:" , o2, " and hr ", hr)

if __name__== "__main__":
  main()

