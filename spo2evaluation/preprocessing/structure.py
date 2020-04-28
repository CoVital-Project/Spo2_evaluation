""" Simple script for structuring the sample data. """
import pandas as pd
import numpy as np
import sys
import os
import datetime
import json
from pathlib import Path
from shutil import copy

VIDSPATH = Path("/Users/gianlucatruda/Downloads/CoVital clips/dataset")
DATASET = "all_label_data.csv"


if __name__ == "__main__":
    print("Importing dataset...")
    df = pd.read_csv(DATASET)
    df["RecordTimestamp"] = pd.to_datetime(df["RecordTimestamp"])
    print(df.info())

    # Generate the directory (folder) name from the timestamp
    df["FolderName"] = df["RecordTimestamp"].dt.strftime("%Y%m%d%H%M%S%f")

    # Hacky trick to split the DataFrame into several dictionaries (for later JSON files)
    json_dict = {}
    json_dict["gt.json"] = df[["SpO2", "HR", "VideoFilename", "Notes"]].to_dict(
        orient="index"
    )
    json_dict["phone.json"] = df[["PhoneMake", "PhoneModel"]].to_dict(orient="index")
    json_dict["device.json"] = df[["DeviceMake", "DeviceModel"]].to_dict(orient="index")
    json_dict["user.json"] = df[["PatientID"]].to_dict(orient="index")

    # Loop through each potential folder, populating with corresponding JSON and video files.
    for i, folder in enumerate(df["FolderName"]):
        if os.path.exists(folder):
            print(f"Folder {folder} already exists. Skipping.")
            continue

        print(f"Making and populating folder: {folder}")
        os.makedirs(folder)
        path = Path(folder)

        for key, vals in json_dict.items():
            with open(path / key, "w") as file:
                json.dump(vals[i], file)

        vid_filename = json_dict["gt.json"][i]["VideoFilename"]
        vidpath = VIDSPATH / vid_filename
        print(f"Collecting video: {vidpath}")
        copy(vidpath, path)
