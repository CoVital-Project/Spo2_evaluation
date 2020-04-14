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


def rgb_to_ppg(df: pd.DataFrame, filter='band') -> pd.DataFrame:
    """Turn RGB means into PPG curves as per Lamonaca.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the colour data per frame (`mean_red`, etc.)
    filter : str, optional
        The kind of signal filter to apply, by default 'band'.
        `band` = bandpass Butterworth filter
        `low` = lowpass Butterworth filter
        `firwin` = lowpass Firwin filter

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe, but with 'tx_red', 'tx_blue', 'tx_green'.

    """

    if filter not in ['band', 'low', 'firwin']:
        raise ValueError("filter must be 'band', 'low', or 'firwin'")

    _df = df.copy()

    for colour in ['red', 'green', 'blue']:
        _df[f'tx_{colour}'] = _df[f'mean_{colour}']

    tx_fields = ['tx_red', 'tx_green', 'tx_blue']

    '''Filtering'''
    fs = 30.0       # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2

    if filter == 'band':
        low = 0.3 / nyq
        high = 4.2 / nyq
        b, a = butter(order, [low, high], btype='band', analog=False)
    else:
        cutoff = 4.0 / nyq    # desired cutoff frequency of the filter, Hz
        if filter == 'low':
            b, a = butter(order, cutoff, btype='low', analog=False)
        if filter == 'firwin':
            b = firwin(order+1, cutoff, )
            a = 1.0

    # Apply the filter
    for field in tx_fields:
        _df[field] = filtfilt(b, a, _df[field])

    '''Normalization /  Standardization'''
    # _df[tx_fields] = normalize(_df[tx_fields], axis=1)
    _df[tx_fields] = StandardScaler().fit_transform(_df[tx_fields])

    return _df


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
