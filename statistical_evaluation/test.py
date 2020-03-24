'''
Documentation, License etc.

@package Spo2_evaluation
'''
import numpy as np

import blant_altman
#import the Python plotting tool
from matplotlib import pyplot as plt


def main():
    print("Hello World!")
    #Data group 1
    #Create an array with the first dataset of group 1
    Data1 = np.array([10.3, 5.1, 3.2, 19.1, 8.1, 11.7, 7.1, 13.9, 4.0, 20.1, 27.5, 6.4, 30.1, 13.0, 10.0,
                            16.8, 17.3, 3.0, 25.6, 19.3, 15.0, 27.3, 14.2, 13.0,14.4, 22.1, 19.0, 18.0, 13.0, 25.6,
                            18.4, 12.6, 25.5, 15.7, 20.2, 16.5, 19.3, 10.0, 18.8, 24.0, 22.8])


    #Create an array with the second data set of group 1
    Data2 = np.array([8.9, 4.0, 8.1, 21.2, 8.1, 12.0, 4.5, 13.9, 4.0, 20.1, 27.5, 6.4, 40.3, 13.0, 10.0, 32.2,
                            17.1, 9.4, 25.2, 18.8, 15.0, 27.3, 21.3, 13.0, 14.4,22.1, 17.9, 3.0, 13.0, 19.0, 18.4,
                            12.6, 25.5, 15.7, 21.2, 16.5, 19.3, 10.0, 30.8, 9.0, 22.8])

    #This is a different group of data, which is the second group.
    #Create an array of data set one for the second group
    Dataset1 = np.array([10.3, 15.8, 6.4, 20.2, 17.1, 17.1, 21.2, 15.8, 15.0, 10.4, 16.4,
                            21.0, 19.0, 11.2, 10.0, 17.9, 10.2, 21.2, 23.0, 17.5, 8.9, 23.8, 26.1])


    #Create an array of data for set two for the second group
    Dataset2 = np.array([13.9, 16.8, 3.0, 25.6, 19.3, 15.0, 27.3, 13.0, 22.1, 19.0, 18.0, 13.0,
                            25.6, 18.4, 12.6, 25.5, 15.7, 20.2, 16.5, 19.3, 10.0, 18.8, 22.8])
    
    
    try:
        
        plt.figure(0)
        plt.title("correlation")
        least_square_slope_1, least_square_y_intercept_1, R_and_P_1 = blant_altman.Correlation(Data1, Data2)
        plt.figure(1)
        plt.title("BlandAltman")
        mean_difference_1, std_difference_1,  upper_limit_1, lower_limit_1 = blant_altman.BlandAltmanPlot(Data1, Data2)
    except Exception as error:
        print("Could not complete: ", error)
        plt.close(0)
        plt.close(1)
    plt.show()
    
    try:
        plt.figure(2)
        plt.title("correlation d1")
        least_square_slope_d1, least_square_y_intercept_d1, R_and_P_d1 = blant_altman.Correlation(Dataset1, Dataset2, 2)
        
        plt.figure(3)
        plt.title("BlandAltman bad1")
        mean_difference_d1, std_difference_d1,  upper_limit_d1, lower_limit_d1 = blant_altman.BlandAltmanPlot(Dataset1, Dataset2, 3)
        print("upper limit :", upper_limit_d1, " and lower limit:", lower_limit_d1, "with median ", mean_difference_d1, " and standard deviation ", std_difference_d1 )
    except Exception as error:
        print("Could not complete: ", error)
        plt.close('all')
 
    plt.show()
  
if __name__== "__main__":
  main()

