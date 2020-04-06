""" Script to pull and extract .zip archives of video data from dropbox.
"""

import requests
import os
from pathlib import Path
import zipfile
import io
import shutil


DATADIR = Path('data')
NEMCOVADIR = DATADIR / 'nemcova_data'
SAMPLEDIR = DATADIR / 'sample_data'

NEMCOVAURL = "https://www.dropbox.com/s/v77ldm3goakx2rj/nemcova_data.zip?dl=1"
SAMPLEURL = "https://www.dropbox.com/s/dl8o9htbfsmzl50/sample_data.zip?dl=1"


def check_datadir():
    # Check that datadir exists and has videos
    if os.path.exists(NEMCOVADIR) and os.path.exists(SAMPLEDIR):
        return True

    os.makedirs(NEMCOVADIR)
    os.makedirs(SAMPLEDIR)
    return False


def fetch_data():
    # Fetch video data from Dropbox links
    # See https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url#14260592
    for url, path in {SAMPLEURL: SAMPLEDIR, NEMCOVAURL: NEMCOVADIR}.items():
        print(f"Retrieving from {url}")
        try:
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(path)

            # Do some cleanup of junk dirs
            shutil.rmtree(path/'__MACOSX')

        except Exception as e:
            print(f"Error retrieving {url}:\n{e}")
    print("Done")


if __name__ == '__main__':

    if check_datadir():
        print(f"Data in {DATADIR} appears intact!")
    else:
        fetch_data()
