import sys
sys.path.append('..')


import spo2evaluation
from spo2evaluation.modelling.healthwatcher import healthwatcher
from spo2evaluation.preprocessing import data_loader_pandas


if __name__== "__main__":
    dataset = data_loader_pandas.Spo2DatasetPandas('test_data')
    
    for i in range(dataset.number_of_videos):
        df = dataset.get_video(i)
        
        df = df.T
        
        blue = df['blue'].to_numpy()
        red = df['red'].to_numpy()
        green = df['green'].to_numpy()
        blue_std = df['blue std'].to_numpy()
        red_std = df['red std'].to_numpy()
        green_std = df['green std'].to_numpy()
        fps = df['fps'][0]
        
        spo2 = healthwatcher.health_watcher(blue, blue_std, red, red_std, fps, smooth=False)
        print(spo2)
    
