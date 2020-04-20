# Preprocessed data

## Structure

```
preprocessed_data/
├── README.md
├── nemcova_data
│   ├── ground_truths_nemcova.csv
│   └── nemcova_data.csv
└── sample_data
    ├── ground_truths_sample.csv
    └── sample_data.csv
```

* For both the `nemcova_data` and `sample_data` datasets, we have preprocessed the videos and ground truth (HR, SpO2) using the tools in `data_loader/` and `preprocessing/`. 
* The results are stored as .csv files in `preprocessed_data/`.
* Each subdirectory contains one or more .csv files with the timeseries data and a .csv file with the ground truth values. 
* To re-link the ground truth with the timeseries, you can join the path of `sample_source` onto `path`. 

## Examples

`preprocessed_data/sample_data/ground_truths_sample.csv`

| path                             | HR | SpO2 |
|----------------------------------|----|------|
| sample_data/20200327153700000000 | 67 | 98   |
| sample_data/20200327151800000000 | 66 | 98   |
| sample_data/20200329132000000000 | 60 | 97   |

---

`preprocessed_data/sample_data/sample_data.csv`


| mean_red   | std_red    | mean_green | std_green | mean_blue | std_blue  | frame | sample_id | sample_source                                 |
|------------|------------|------------|-----------|-----------|-----------|-------|-----------|-----------------------------------------------|
| 231.990429 | 0.14259326 | 0.0        | 0.0       | 43.844105 | 3.5604148 | 0     | 0         | sample_data/20200327153700000000/PC000004.mp4 |
| 231.98946  | 0.1450865  | 0.0        | 0.0       | 43.8068   | 3.567825  | 1     | 0         | sample_data/20200327153700000000/PC000004.mp4 |
| 231.98949  | 0.14572552 | 0.0        | 0.0       | 43.737774 | 3.5790896 | 2     | 0         | sample_data/20200327153700000000/PC000004.mp4 |
| 231.988039 | 0.15766032 | 0.0        | 0.0       | 43.66933  | 3.5964715 | 3     | 0         | sample_data/20200327153700000000/PC000004.mp4 |
| 231.98349  | 0.18321012 | 0.0        | 0.0       | 43.592094 | 3.62277   | 4     | 0         | sample_data/20200327153700000000/PC000004.mp4 |
