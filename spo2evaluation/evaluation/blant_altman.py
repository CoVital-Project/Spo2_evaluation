# Import numeric python library
import numpy as np

# import the Python plotting tool
from matplotlib import pyplot as plt

# Import the stats library from the scipy module
from scipy import stats


def is_gaussian(p_stat):
    alpha = 0.05
    if p_stat > alpha:
        print("Sample looks Gaussian (fail to reject H0)")
        return True
    else:
        print("Sample does not look Gaussian (reject H0)")
        return False


# Function to create a correlation plot comparing two data sets.
def Correlation(Xdata, Ydata, fig_num=None):
    # Create a matrix for the horizontal axis data for least squares fitting.
    # The T takes the transpose of a matrix and turns horizontal data into
    # columns instead of rows.
    A = np.vstack([Xdata, np.ones(len(Xdata))]).T
    # Calculate the least squares fit of the data. m refers to the slope
    # of the line and b refers to the y-intercept.
    least_square_slope, least_square_y_intercept = np.linalg.lstsq(A, Xdata)[0]
    # Calculate the Pearson correlation coefficient and the 2-tailed p-value).
    R_and_P = np.round(stats.pearsonr(Xdata, Ydata), 2)

    if fig_num != None:

        # Set the axis limits for the plot.  X-range is given followed by Y-range.
        plt.axis(
            [0, (np.max(Xdata) + 5), 0, (np.max(Ydata) + 5)], fontsize="16", lw="2"
        )
        # Plot the data on the correlation plot.
        # Use open circles with no marker face color, no lines, blue edge color, marker edge width of 2, marker size 8
        plt.plot(
            Xdata,
            Ydata,
            marker="o",
            mfc="none",
            ls="None",
            mec="blue",
            mew="2",
            ms="8",
            lw="2",
        )
        # Plot the least squares fit line
        plt.plot(
            Xdata, least_square_slope * Xdata + least_square_y_intercept, "k", lw="2"
        )
        ##Horizontal axis label.
        plt.xlabel("Dataset 2", fontsize="16")
        ##Vertical axis label
        plt.ylabel("Dataset 1", fontsize="16")
        # Change the font size of the tick mark labels to be bigger
        # on both axes.
        # plt.tick_params(axis='both', which='major', labelsize= '16', width = '2')
        # This if statement is used to determine the text box on the plot
        # to show the P value beside the r value.  If P is less than 0.01
        # then set the string to show just P < 0.01.  If not, just show the
        # P-value rounded to two decimal places.
        if R_and_P[1] < 0.01:
            P = "(P<0.01)"
        else:
            P = "(P=" + str(R_and_P[1]) + ")"
        # Add a text box with correlation coefficient and p-value
        # on the plot.
        plt.text(2, 28, "r=" + str(R_and_P[0]) + P, fontsize="16")
    return (least_square_slope, least_square_y_intercept, R_and_P)


# Function to create a Bland-Altman plot comparing agreement between two data sets.
def BlandAltmanPlot(Xdata, Ydata, fig_num=None):
    # Calculate the difference between the CACT and actual angles.  Will create a difference array comparing each angle
    Difference = Ydata - Xdata
    # Calculate the average value of the two datasets for each data point
    Average = (Xdata + Ydata) / 2

    # make sure the difference are a normal distribution
    w, p_stat = stats.shapiro(Difference)

    fig_normal = plt.figure()
    plt.hist(Difference, figure=fig_normal)
    plt.title("Distribution of the diff", figure=fig_normal)
    # fig_normal.show()

    print("\nw:", w, "\np_stat:", p_stat)
    if not is_gaussian(p_stat):
        print("The data is not normal so we try a logartihmic distribution")

        print(Difference)

        min_x = Xdata.min()
        min_y = Ydata.min()

        # Removing the zeros and negative numbers for the calculation of log.
        if min_x <= 0 or min_y <= 0:
            min_val = min(min_x, min_y)
            Xdata = Xdata - min_val + (min_val + 0.0000000001)
            Ydata = Ydata - min_val + (min_val + 0.0000000001)
        Xdata_log = np.log10(Xdata)
        Ydata_log = np.log10(Ydata)
        Difference = Ydata_log - Xdata_log
        # Calculate the average value of the two datasets for each data point
        Average_log = (Xdata_log + Ydata_log) / 2
        # make sure the difference are a normal distribution
        w_log, p_stat_log = stats.shapiro(Difference)
        print("\nw:", w_log, "\np_stat:", p_stat_log)

        fig_normal_log = plt.figure()
        plt.hist(Difference, figure=fig_normal_log)
        plt.title("Distribution of the diff of log(data)", figure=fig_normal_log)

        if not is_gaussian(p_stat_log):
            print("The data is still not normal")
            raise Exception("The distribution of difference is not normal")

    # Calculate the mean of the differences.
    mean_difference = np.mean(Difference)
    # Calculate the sample standard deviation of the difference.
    std_difference = np.std(Difference)
    # Calculate the upper and lower limits of the agreement (95% confidence).
    upper_limit = mean_difference + 1.96 * std_difference
    lower_limit = mean_difference - 1.96 * std_difference

    if fig_num != None:

        plt.figure(fig_num)

        ##Set axis limits, this will account for the maximum average difference and the upper and lower
        ##limits of agreement, and all data points.
        plt.axis(
            [0, np.max(Average) + 5, np.min(Difference) - 5, np.max(Difference) + 5],
            fontsize="16",
            lw="2",
        )
        ##Do the Bland-Altman plot.
        plt.plot(
            Average,
            Difference,
            marker="o",
            mfc="none",
            ls="None",
            mec="blue",
            mew="2",
            ms="8",
            lw="2",
        )
        # Add the mean, upper and lower levels of agreement to the Bland-Altman plot.
        plt.axhline(y=mean_difference, lw="2", color="k", ls="dashed")
        plt.axhline(y=upper_limit, lw="2", color="k", ls="dashed")
        plt.axhline(y=lower_limit, lw="2", color="k", ls="dashed")
        # Horizontal axis label
        plt.xlabel("Average difference", fontsize="16")
        # Vertical axis label
        plt.ylabel("Difference", fontsize="16")
    return (mean_difference, std_difference, upper_limit, lower_limit)
