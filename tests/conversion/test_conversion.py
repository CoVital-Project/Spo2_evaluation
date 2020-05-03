import sys
import os
script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../..')
sys.path.append(myutils_path)
#sys.path.append('../..')
import json
import pytest
from spo2evaluation.preprocessing import data_sanitization


def test_detect_format():
    pathfile = os.path.dirname(os.path.abspath(__file__))
    assert(data_sanitization.format(
            os.path.join(pathfile, 'data_for_test_old_format.json')
        ) == data_sanitization.DataFormatCovital.old)


def test_conversion():
    pathfile = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pathfile, 'data_for_test_old_format.json')) as json_file:
        data = json.load(json_file)
        # print("Old dat:\n", data['spo2Device'])

    new_json = data_sanitization.convert_old_json_format_to_new(data, True)

    assert(new_json['spo2Device'] == data['spo2Device'])
    
    assert(new_json['survey']['accelerometerValues'] == data['accelerometerValues'])
    assert(new_json['survey']['gyroscopeValues'] == data['gyroscopeValues'])
    assert(new_json['survey']['userAccelerometerValues'] == data['userAccelerometerValues'])
    
    assert(new_json['survey']['accelerometerTimestamps'] == data['accelerometerTimestamps'])
    assert(new_json['survey']['gyroscopeTimestamps'] == data['gyroscopeTimestamps'])
    assert(new_json['survey']['userAccelerometerTimestamps'] == data['userAccelerometerTimestamps'])
    assert(new_json['survey']['spo2'] == data['spo2'])
    assert(new_json['survey']['hr'] == data['hr'])
    assert(new_json['survey']['age'] == data['age'])
    assert(new_json['survey']['weight'] == data['weight'])
    assert(new_json['survey']['height'] == data['height'])
    assert(new_json['survey']['skinColor'] == data['skinColor'])
    assert(new_json['survey']['respiratorySymptoms'] == data['respiratorySymptoms'])
    assert(new_json['survey']['phoneBrand'] == data['phoneBrand'])
    assert(new_json['survey']['phoneModel'] == data['phoneModel'])
    assert(new_json['survey']['date'] == data['date'])
    assert(new_json['survey']['id'] == data['id'])
    assert(new_json['survey']['startTimeOfRecording'] == data['startTimeOfRecording'])
    
    
    
    assert(len(new_json['survey']) == 20)
    assert(new_json['userProfile']['licensed_medical_professional'] == True)
    
    new_json = data_sanitization.convert_old_json_format_to_new(data, False)
    
    assert(new_json['userProfile']['licensed_medical_professional'] == False)
    
    
def test_sanitizing():
    
    folder = "."
    data_sanitization.recursive_sanitizing_of_json(folder)
    
    
