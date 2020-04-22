import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, firwin
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
import pywt
from ..lamonaca_and_nemcova.utils import spo2_estimation, get_peak_height_slope


def calc_spo2(df: pd.DataFrame):

    """Not working"""
    raise NotImplementedError()

    e_hb_600 = 15.0
    e_hb_940 = 0.8
    e_hbo_600 = 3.5
    e_hbo_940 = 1.1

    timestamps = np.linspace(0, int(float(df.shape[0])/30.0), num=df.shape[0])

    vp_940, m_940, hr_940 = get_peak_height_slope(df['tx_green'].values, timestamps, 30.0)
    vp_600, m_600, hr_600 = get_peak_height_slope(df['tx_red'].values, timestamps, 30.0)

    ln_vp_600 = np.log(vp_600)
    ln_vp_940 = np.log(vp_940)

    print(ln_vp_600, ln_vp_940)
    print(m_600, m_940)
    print(hr_600, hr_940)

    spo2 = (e_hb_600 * np.sqrt(m_940 * ln_vp_940) - e_hb_940 * np.sqrt(m_600 * ln_vp_600)) / \
        (np.sqrt(m_940 * ln_vp_940)*(e_hb_600 - e_hbo_600) -
         np.sqrt(m_600 * ln_vp_600)*(e_hb_940 - e_hbo_940))

    return spo2.mean(), spo2.std()


def _wavelet_filter_signal(s, wave='db4', *args, **kwargs):
    cA, cD = pywt.dwt(s, wave)
    cD_mod = pywt.threshold(cD, *args, **kwargs)
    s_mod = pywt.idwt(cA, cD_mod, wave)
    return s_mod


def rgb_to_ppg(df: pd.DataFrame, filter='band', qe_constants=(0.004989, 0.001193, 0.001), scale=False, block_id='sample_id') -> pd.DataFrame:
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
    qe_constants : (float, float, float)
        The Quantum Efficiency constants (Lamonaca et al.) to use for scaling the
        means of the (red, green, blue) channels respectively.
    block_id : str, optional
        The field to use for chunking the data before applying transforms,
        by default 'sample_id'.

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe, but with 'tx_red', 'tx_blue', 'tx_green'.

    """

    if filter not in ['band', 'low', 'firwin', None]:
        raise ValueError("filter must be 'band', 'low', 'firwin', or None")

    _df = df.copy()

    for i, colour in enumerate(['red', 'green', 'blue']):
        qe_const = qe_constants[i]
        _df[f'tx_{colour}'] = _df[f'mean_{colour}'] * qe_const

    tx_fields = ['tx_red', 'tx_green', 'tx_blue']

    '''Filtering'''
    fs = 30.0       # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2

    if filter is not None:
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

        # Apply the filter (and normalise)
        for bid in _df[block_id].unique():
            for field in tx_fields:
                _df.loc[_df[block_id] == bid, field] = filtfilt(
                    b, a, _df[_df[block_id] == bid][field])

                # Scaling
                if scale:
                    _df.loc[_df[block_id] == bid, field] = MinMaxScaler(
                    ).fit_transform(_df[_df[block_id] == bid][field].values.reshape(-1, 1))

    return _df


def _create_path_field(df: pd.DataFrame) -> pd.DataFrame:
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


def _attach_sample_id_to_ground_truth(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of `labels` with added `sample_id` field.

    Parameters
    ----------
    df : pd.DataFrame
        [description]
    labels : pd.DataFrame
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    _df = df[['sample_id', 'sample_source']]
    _labels = labels.copy()

    _df = _create_path_field(_df)
    _df = _df.drop_duplicates(subset=['sample_id', 'path'], keep='first')

    out = _df.merge(_labels, on='path', how='inner')

    return out


def engineer_features(df: pd.DataFrame, labels: pd.DataFrame, target='SpO2', select=True):
    """Automatic feature engineering (with optional selection) for timeseries dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of rgb and/or ppg timeseries for all subjects.
    labels : pd.DataFrame
        A dataframe matching sample IDs to ground truth (e.g. SpO2)
    target : str, optional
        The name of the column you're trying to predict, by default 'SpO2'
    select : bool, optional
        Whether to automatically filter down to statistically significant features, by default True

    Returns
    -------
    pd.DataFrame
        A dataframe with new features added, including the target feature.
    """

    _df = df.copy()
    _labels = labels.copy()

    if 'sample_id' not in labels.columns:
        _labels = _attach_sample_id_to_ground_truth(_df, _labels)

    ids = _df['sample_id'].unique()

    _labels = _labels[_labels['sample_id'].isin(ids)]
    y = _labels.set_index('sample_id')[target].astype(np.float)

    _df.drop('sample_source', axis=1, inplace=True)

    extracted_features = extract_features(
        _df,
        column_id="sample_id",
        column_sort="frame",
    )

    impute(extracted_features)
    features = extracted_features
    if select:
        features_filtered = select_features(
            extracted_features, y, ml_task='regression',)
        print(extracted_features.shape, features_filtered.shape)
        features = features_filtered

    out_df = features.join(y, how='left')

    return out_df


def select_best_features(df: pd.DataFrame, n_features=20, target='SpO2') -> pd.DataFrame:
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
    """Simple data augmentation by chopping timeseries into blocks.

    This doesn't modify the data. It just overwrites the `sample_id`
    field with new values. i.e. a single sample of `N` frames is now
    seen as `M` different samples of `N/M` frames.

    Parameters
    ----------
    df : pd.DataFrame
        Your timeries data for all samples.
    n_frames : int, optional
        The lenth (in frames) of each new chunk/sample, by default 200
    overlap : bool, optional
        Whether blocks should overlap (not yet implemented), by default False

    Returns
    -------
    pd.DataFrame
        The augmented timeseries data.show()
    """

    def gen_sample_id(sid, frame):
        return f"{sid}_{frame // n_frames}"

    def gen_frame(frame):
        return frame % n_frames

    _df = df.copy()

    if overlap:
        raise NotImplementedError("Overalapping not yet supported")

    # Make a copy of the sample_id for later reference
    _df['original_sample_id'] = _df['sample_id']

    # Apply the gen_sample_id function to create an augmented sample_id
    _df['sample_id'] = _df.apply(lambda x: gen_sample_id(
        x['original_sample_id'], x['frame']), axis=1)

    # Apply the gen_frame function to augment the frame values
    _df['frame'] = _df['frame'].apply(gen_frame)

    # Trim the excess (samples which have fewer than `n_frames` frames)
    drop_sids = []
    for sid in _df['sample_id'].unique():
        frame_count = _df[_df['sample_id'] == sid]['frame'].tail(1).values
        if (frame_count + 1) < n_frames:
            drop_sids.append(sid)
    _df = _df[~_df['sample_id'].isin(drop_sids)]

    return _df
