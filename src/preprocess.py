import numpy as np
import pandas as pd
import h5py
import os
import re

# Raw data paths
BASE_DIR = "D:/internship/denoise-fnirs-ufabc/data"
ORIGIN = BASE_DIR + "/raw_data/"
OUTPUT_DIR = BASE_DIR + "/csv_data/"
pattern_data_a = re.compile(r'.*\.snirf$')

matching_folders = []

# Gettinhg the subjects of each experiment
def get_subject(ORIGIN):
    for item in os.listdir(ORIGIN):
        full_path = os.path.join(ORIGIN, item)
        matching_folders.append(full_path)
    return matching_folders

# Converting the .snitf data into .csv
def snirf_to_csv(matching_folders):
    for subject in matching_folders:
        for item in os.listdir(subject + '/nirs'):
            print(item)
            if pattern_data_a.match(item):
                filename = subject + '/nirs/' + item
                print(filename)
                with h5py.File(filename, 'r') as f:
                    data = f['nirs']['data1']['dataTimeSeries'][:]
                    time = f['nirs']['data1']['time'][:]
                    
                df = pd.DataFrame(data, columns=[f'CH{i+1}' for i in range(data.shape[1])])
                df.insert(0, 'Time (s)', time)
                print(OUTPUT_DIR)
                df.to_csv(OUTPUT_DIR + item + ".csv", index=False)


matching_folders = get_subject(ORIGIN)

snirf_to_csv(matching_folders)
