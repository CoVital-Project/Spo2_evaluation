""" Script to pull and extract .zip archives of video data from dropbox.
"""

import requests
import os
from pathlib import Path
import zipfile
import io
import shutil


DATADIR = Path("data")
SAMPLEDIR = DATADIR / "sample_data"
NEMCOVADIR = DATADIR / "nemcova_data"
SAMPLEURL = "https://www.dropbox.com/s/dl8o9htbfsmzl50/sample_data.zip?dl=1"
NEMCOVAURL = "https://www.dropbox.com/s/v77ldm3goakx2rj/nemcova_data.zip?dl=1"

dir_url_dict = {SAMPLEDIR: SAMPLEURL, NEMCOVADIR: NEMCOVAURL}


def check_datadir():
    """Checks if data directory is complete. If not, return list
        of directories to update. Makes missing directories automatically.

    Returns
    -------
    List
        A list of the directories that are incomplete.
    """

    dirs_to_fetch = []
    for directory in list(dir_url_dict.keys()):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if len(os.listdir(directory)) == 0:
            dirs_to_fetch.append(directory)
    return dirs_to_fetch


def fetch_data(dirs_to_fetch: list):
    """Fetch video data from Dropbox links, unzip, and clean up.

    Retrieval and unzipping code comes from https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url#14260592

    Parameters
    ----------
    dirs_to_fetch : list
        A list of directories that need updating.
    """

    fetch_dict = {k: v for k, v in dir_url_dict.items() if k in dirs_to_fetch}

    for path, url in fetch_dict.items():
        print(f"Retrieving from {url}")
        try:
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path)

        except Exception as e:
            print(f"Error retrieving {url}:\n{e}")

        # Do some cleanup of junk dirs
        junk_path = path / "__MACOSX"
        try:
            shutil.rmtree(junk_path)
        except Exception as e:
            print(f"Error cleaning up '{junk_path}'")


if __name__ == "__main__":

    to_fetch = check_datadir()
    if len(to_fetch) > 0:
        print(f"Fetching data for: {to_fetch}")
        fetch_data(to_fetch)
    else:
        print("It doesn't seem like you need to fetch anything.")

    print("Done.")
