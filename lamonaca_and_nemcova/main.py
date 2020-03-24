import cv2
import lamonaca_2015
import nemcova_2020


def main():
    cv2.getBuildInformation()
    #lamonaca = lamonaca_2015.Lamonaca2015()
    
    #lamonaca.lacomana("Video/S98T89.avi")
    
    nemcova = nemcova_2020.Nemcova2020()
    o2, hr = nemcova.spo2_estimation("Video/S98T89.avi")
    print("Done:" , o2, " and hr ", hr)

if __name__== "__main__":
  main()

