import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def power_analysis(df: pd.DataFrame, field_prefix='mean_', show=False):
    """Generate Welch Power Spectrum plot on RGB channels.

    NOTE: Heavily adapted from https://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python


    Parameters
    ----------
    df : pd.DataFrame
        Sample data with RGB channels.
    field_prefix : str, optional
        The prefix for fields to consider when, by default 'mean_'
    show : bool, optional
        Whether to call plt.show() automatically, by default False
    """

    fs = 30.0       # sample rate, Hz

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


def inspect_ppg(df: pd.DataFrame, xlim=None, *args, **kwargs):
    """Renders subplots of PPG data for each colour.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'tx_red', 'tx_green', 'tx_blue' fields.
    xlim : Tuple
        the range of frames to show, optional, by default None

    Returns
    -------
    Figure
    """

    fig, ax = plt.subplots(3, 1)

    for i, colour in enumerate(['red', 'green', 'blue']):
        field = f"tx_{colour}"
        ax[i].plot(df['frame'], df[field], color=colour, *args, **kwargs)
        if xlim is not None:
            ax[i].set_xlim(xlim)
    ax[1].set_ylabel('Value (after norm.)')
    ax[2].set_xlabel('Frame')

    return fig
