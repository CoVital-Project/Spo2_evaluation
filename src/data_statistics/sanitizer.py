import sys
sys.path.append('../..')
from spo2evaluation.preprocessing import data_sanitization
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-directory",
        dest="data_folder",
        help="select folder holding the sanitize",
        metavar="FOLDER",
    )
    args = parser.parse_args()
    data_folder = args.data_folder
    while data_folder is None:
        data_folder = input("Folder to sanitize> ")

    print("Launching the sanitization for", data_folder)
    data_sanitization.recursive_sanitizing_of_json(data_folder)


if __name__ == "__main__":
    main()
