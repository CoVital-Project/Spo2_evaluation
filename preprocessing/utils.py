from preprocessing.data_loader import DataLoader, Spo2Dataset
import pandas as pd
from pathlib import Path
import json


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
            "mean_red": frames[:, 2, 0],
            "std_red": frames[:, 2, 1],
            "mean_green": frames[:, 1, 0],
            "std_green": frames[:, 1, 1],
            "mean_blue": frames[:, 0, 0],
            "std_blue": frames[:, 0, 1],
        }
        for frames in example_dict.values()
    ]

    dfs = []
    for i, dset in enumerate(dset_dicts):
        _df = pd.DataFrame.from_dict(dset_dicts[i])
        _df["sample_id"] = i
        _df.reset_index(drop=False, inplace=True)
        _df["frame"] = _df["index"]
        _df.drop("index", axis=1, inplace=True)
        dfs.append(_df)
    df = pd.concat(dfs)

    df.rename({"index": "sample_id"}, axis=1, inplace=True)

    ordered_cols = list(df.columns)
    ordered_cols.remove("sample_id")
    df = df[[*ordered_cols, "sample_id"]]

    df["sample_source"] = df["sample_id"].apply(lambda x: meta_dict[x]["video_source"])

    return df


def ground_truth_dataframe(data_path="sample_data/"):
    """Retrieves SpO2 and HR values from all `gt.json` files in a path as a DataFrame.

    Parameters
    ----------
    data_path : str, optional
        The path to the data you want to extract values for`, by default 'sample_data/'

    Returns
    -------
    pd.DataFrame
        A dataframe mapping paths to HR and SpO2 values.
    """

    path = Path(data_path)
    gt_dict = {"path": [], "HR": [], "SpO2": []}
    for file in path.glob("**/gt.json"):
        with open(file, "r") as json_file:
            json_vals = json.load(json_file)
            gt_dict["path"].append(str(file.parent))
            for key in ["HR", "SpO2"]:
                gt_dict[key].append(json_vals[key])

    df = pd.DataFrame.from_dict(gt_dict)

    return df
