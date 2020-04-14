import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


def power_analysis(df: pd.DataFrame, field_prefix='mean_', show=False):
    # Adapted from: https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python

    fs = 30.0       # sample rate, Hz

    # signal.welch
    plt.figure()
    for colour in ['red', 'green', 'blue']:
        x = df[f"{field_prefix}{colour}"]
        f, Pxx_spec = welch(x, fs, 'flattop', 1024, scaling='spectrum')
        plt.semilogy(f, np.sqrt(Pxx_spec), color=colour)

    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectra (welch)')

    if show:
        plt.show()


def rgb_to_ppg(df: pd.DataFrame) -> pd.DataFrame:

    _df = df.copy()

    red = _df['mean_red'].values
    green = _df['mean_green'].values
    blue = _df['mean_blue'].values

    '''4Hz low-pass filter
        Source: https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    '''

    # T = 5.0         # Sample Period
    fs = 30.0       # sample rate, Hz
    cutoff = 4.5    # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = df.shape[0]  # total number of samples

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    for new_field, data in {'tx_red': red, 'tx_green': green, 'tx_blue': blue}.items():
        _df[new_field] = filtfilt(b, a, data)

    # Normalisation
    tx_fields = ['tx_red', 'tx_green', 'tx_blue']
    _df[tx_fields] = normalize(_df[tx_fields])

    return _df


def inspect_ppg(df: pd.DataFrame, *args, **kwargs):

    fig, ax = plt.subplots(3, 1)

    for i, colour in enumerate(['red', 'green', 'blue']):
        field = f"tx_{colour}"
        ax[i].plot(df['frame'], df[field], color=colour, *args, **kwargs)
    ax[1].set_ylabel('Value (after norm.)')
    ax[2].set_xlabel('Frame')

    return fig