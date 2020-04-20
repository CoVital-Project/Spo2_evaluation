import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif


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


def create_path_field(df: pd.DataFrame) -> pd.DataFrame:
    """Create `path` field from `sample_source`, then remove `sample_source`.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries dataset from Nemcova or sample data.

    Returns
    -------
    pd.DataFrame
        A copy of the input data, but with `path` field instead of `sample_source`.
    """
    _df = df.copy()
    _df['path'] = _df['sample_source'].apply(lambda x: str(Path(x).parent))
    _df.drop('sample_source', axis=1, inplace=True)

    return _df


def engineer_features(df: pd.DataFrame, labels: pd.DataFrame, target='SpO2'):

    _df = df.copy()
    _df = create_path_field(_df)
    ids = _df['path'].unique()

    _labels = labels[labels['path'].isin(ids)]
    y = _labels.set_index('path')[target].astype(np.float)

    extracted_features = extract_features(
        _df,
        # y,
        column_id="path",
        column_sort="frame",
    )

    impute(extracted_features)
    # features_filtered = select_features(
    #     extracted_features, y, ml_task='regression',)

    out_df = extracted_features.join(y, how='left')

    return out_df


def select_features(df: pd.DataFrame, n_features=20, target='SpO2') -> pd.DataFrame:
    """Perform feature selection using k-best strategy and return dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with all features present.
    n_features : int, optional
        k in k-best algorithm (the number of features), by default 20
    target : str, optional
        the target feature name, by default 'SpO2'

    Returns
    -------
    pd.DataFrame
        A copy of the original dataframe, with only the k-best features.
    """

    kbest = SelectKBest(f_classif, k=n_features)
    _X = df.drop(target, axis=1).values
    _y = df[target].values
    kbest.fit(_X, _y)

    indices = kbest.get_support()
    best_cols = df.drop(target, axis=1).columns[indices]

    _df = df[[*best_cols, target]]

    return _df


def augment_dataset(df: pd.DataFrame, n_frames=200, overlap=False) -> pd.DataFrame:

    def gen_sample_id(sid, frame):
        return f"{sid}_{frame // n_frames}"

    def gen_frame(frame):
        return frame % n_frames

    _df = df.copy()

    if overlap:
        raise NotImplementedError("Overalapping not yet supported")

    _df['original_sample_id'] = _df['sample_id']

    _df['sample_id'] = _df.apply(lambda x: gen_sample_id(
        x['original_sample_id'], x['frame']), axis=1)

    _df['frame'] = _df['frame'].apply(gen_frame)

    return _df
