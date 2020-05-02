import sys
sys.path.append('..')

import cv2
import spo2evaluation
from spo2evaluation.modelling.lamonaca_and_nemcova import lamonaca_2015
from spo2evaluation.modelling.lamonaca_and_nemcova import nemcova_2020


def main():
    cv2.getBuildInformation()
    # lamonaca = lamonaca_2015.Lamonaca2015()

    # lamonaca.lacomana("Video/S98T89.avi")

    nemcova = nemcova_2020.Nemcova2020()
    o2, hr = nemcova.spo2_estimation("Video/S87T78.mp4", optimize=True)
    print("Done:", o2, " and hr ", hr)


if __name__ == "__main__":
    main()
