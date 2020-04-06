# CoVital sample data

(If github LFS gives you problems. You can directly download the data [here](https://www.dropbox.com/s/dl8o9htbfsmzl50/sample_data.zip?dl=1))

This directory contains sample data compiled as a preliminary dataset. It was gathered under lockdown conditions on asymptomatic individuals. It should not be considered a comprehensive sample. Rather, it is intended to indicate the structure of future data and provide sanity checks for early model implementations.

## Overview
* [Version control for large files](#version-control-for-large-files)
* [Format and structure](#format-and-structure)
* [Examples](#examples)

## Version control for large files
Each mp4 video file is approximately 1.3 MB, which could result in bloating of the repository. However, these files are almost certainly not going to have their contents changed. This allows us to make use of [Git Large File Storage (LFS)](https://git-lfs.github.com/) for tracking the mp4 files in the `sample_data/` subdirectories.

You need to [install LFS](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage) on your system to utilise these files. If you don't install LFS, your system will only fetch [the pointers to the large file objects](https://help.github.com/en/github/managing-large-files/collaboration-with-git-large-file-storage). This is probably fine for anyone who's using this repo but not the sample data.

### Modifying what's tracked by LFS
Git LFS uses the `.gitattributes` file to specify what must be tracked. Currently, this is configured to track all files with a `.mp4` extension (recursively). 

```  
sample_data/**/*.mp4 filter=lfs diff=lfs merge=lfs -text
```

### Checking what's tracked by LFS

```
git ls-files ':(attr:filter=lfs)'
```

You can append entries to the file, or use the CLI to track more files. Read more about how to track and add files using LFS [here](https://help.github.com/en/github/managing-large-files/configuring-git-large-file-storage).

### Oh shit, I think I broke it
You may stuff this up if you add new files or change the structure. I stuffed it up a few times. LFS lets you [migrate files](https://github.com/git-lfs/git-lfs/wiki/Tutorial#migrating-existing-repository-data-to-lfs) that you accidentally forgot to track before committing. 

## Format and structure

The data format and structure are an amalgamation of those found in [this outline](https://www.overleaf.com/read/kwfmchzmmgtm) and the methodology of [Nemcova et al.](https://doi.org/10.1016/j.bspc.2020.101928)

```
sample_data/
├── 20200327151500000000
│   ├── PC000001.mp4
│   ├── device.json
│   ├── gt.json
│   ├── phone.json
│   └── user.json
├── 20200327151800000000
│   ├── PC000002.mp4
│   ├── device.json
│   ├── gt.json
│   ├── phone.json
│   └── user.json
.
.
.
├── README.md
└── all_label_data.csv
```

* Each directory is named by the datetime stamp from collection in the format `"%Y%m%d%H%M%S%f`.
* Each directory contains:
	* A ~1 minute long video file of the subject's finger with flash on
		* mp4 format
		* 30 frames per second
		* No audio
	* `device.json`, which contains information about the pulse oximeter device used to measure the ground truth values for SpO2 and heart rate.
	* `gt.json`, which contains the ground truth values for SpO2 and heart rate (HR).
	* `phone.json`, which contains the make and model of the phone used to capture the video.
	* `user.json`, which contains the subject identifier for the patient (but which is likely to hold information about the age, sex, weight, etc. of the patient in future).


Additionally, the `sample_data/` directory contains:

* `all_label_data.csv`, which contains all the information from all the JSON files in a single CSV file. 
* This readme.

	
## Examples

### device.json
```json
{
    "DeviceMake": "Beurer GmbH",
    "DeviceModel": "PO 30"
}
```

### gt.json
```json
{
    "SpO2": 98,
    "HR": 63,
    "VideoFilename": "PC000001.mp4",
    "Notes": "Phone on first right finger. Phone cover on. Device on first left finger. "
}
```

### phone.json
```json
{
    "PhoneMake": "Apple",
    "PhoneModel": "iPhone 8"
}
```

### user.json
```json
{
    "PatientID": "SA001"
}
```


### all\_label\_data.csv

| RecordTimestamp  | PatientID | SpO2 | HR | PhoneMake | PhoneModel | VideoFilename | DeviceMake  | DeviceModel | Notes  |
|------------------|-----------|------|----|-----------|------------|---------------|-------------|-------------|--------|
| 2020-03-27 15:15 | SA001     | 98   | 63 | Apple     | iPhone 8   | PC000001.mp4  | Beurer GmbH | PO 30       | <text> |
| 2020-03-27 15:18 | SA001     | 98   | 66 | Apple     | iPhone 8   | PC000002.mp4  | Beurer GmbH | PO 30       | <text> |
| 2020-03-27 15:20 | SA001     | 98   | 69 | Apple     | iPhone 8   | PC000003.mp4  | Beurer GmbH | PO 30       | <text> |

