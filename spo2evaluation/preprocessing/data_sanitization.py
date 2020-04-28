## sadly the data has two different json format and this tool is here to convert from the old on to the new one
import json
import typing
import pytest
import copy
import os
from pathlib import Path


def is_old_format(file: str) -> bool:
    with open(file) as json_file:
        data = json.load(json_file)
    try:
        data['survey']
    except:
        return True
    return False


def run_fast_scandir_json(dir):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        print("file", f)
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if f.name == "data.json" and is_old_format(f):
                files.append(f.path)


    for dir in list(subfolders):
        sf, f = run_fast_scandir_json(dir)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def recursive_sanitizing_of_json(folder: str) -> None:
    '''
    Recursively convert old json data file to the new format. The old format is saved as data_old_formal.json while the data.json file is converted.
    '''
    # Get all json files 
    _, files = run_fast_scandir_json(folder)
    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            new_json = convert_old_json_format_to_new(data, False)
            
            path = Path(file).parent
            os.rename(file, os.path.join(path, "data_old_formal.json"))
        with open(file, 'w+') as outfile:
            json.dump(new_json, outfile)
    


def convert_old_json_format_to_new(old_json: dict, licensed_medical_professional: bool = False) -> dict:
    '''
    Return a dict representing old_json had the new data format.
    '''
    old_json_tmp =  copy.deepcopy(old_json)
    
    new_json = dict()
        
    spo2Device = copy.deepcopy(old_json_tmp['spo2Device'])
    old_json_tmp.pop('spo2Device', None)
    
    new_json['spo2Device'] = spo2Device
    new_json['userProfile'] = {'licensed_medical_professional': licensed_medical_professional}
    
    new_json['survey'] = old_json
    
    #print(json.dumps(new_json, indent=4, sort_keys=True))
    
    return new_json
    
    



