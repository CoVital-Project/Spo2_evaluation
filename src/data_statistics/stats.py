"""
Documentation, License etc.

@package Spo2_evaluation
"""
import sys
sys.path.append('../..')
from argparse import ArgumentParser
from spo2evaluation.preprocessing import data_stats


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dir-data",
        dest="dir_data",
        help="select folder holding the data",
        metavar="FILE",
    )
    args = parser.parse_args()
    dir_data = args.dir_data
    while dir_data is None:
        dir_data = input("Folder to evaluate> ")

    stats = data_stats.DatasetStatistics()
    stats.get_data(dir_data)
    stats.calculate()
    stats.print()

if __name__ == "__main__":
    main()

