from data_loader.data_loader import DataLoader, Spo2Dataset
import pandas as pd


def dataset_to_dataframe(dataset: Spo2Dataset):
    """Transforms a Spo2Dataset object into a single pandas DataFrame.

    Parameters
    ----------
    dataset : Spo2Dataset
        A dataset of the mean and std. dev. of each colour channel
        for each frame of each video. This comes from `data_loader`.


    Returns
    -------
    pd.DataFrame
        A pandas dataframe with two columns (mean, std) for each colour
        channel, as well as columns for the frame number, sample_id,
        and sample_source.
    """

    example_dict = {i: dataset[i][0].numpy() for i in range(len(dataset))}
    meta_dict = {i: dataset[i][1] for i in range(len(dataset))}
    dset_dicts = [
        {
            'mean_red': frames[:, 2, 0],
            'std_red': frames[:, 2, 1],
            'mean_green': frames[:, 1, 0],
            'std_green': frames[:, 1, 1],
            'mean_blue': frames[:, 0, 0],
            'std_blue': frames[:, 0, 1],
        } for frames in example_dict.values()]

    dfs = []
    for i, dset in enumerate(dset_dicts):
        _df = pd.DataFrame.from_dict(dset_dicts[i])
        _df['sample_id'] = i
        _df.reset_index(drop=False, inplace=True)
        _df['frame'] = _df['index']
        _df.drop('index', axis=1, inplace=True)
        dfs.append(_df)
    df = pd.concat(dfs)

    df.rename({'index': 'sample_id'}, axis=1, inplace=True)
    df['sample_source'] = df['sample_id'].apply(
        lambda x: meta_dict[x]['video_source'])

    ordered_cols = list(df.columns)
    ordered_cols.remove('sample_id')
    ordered_cols.remove('sample_source')
    df = df[[*ordered_cols, 'sample_id', 'sample_source']]

    return df
