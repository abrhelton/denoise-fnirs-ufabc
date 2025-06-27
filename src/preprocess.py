import numpy as np
import pandas as pd
import h5py
import os
import re
from sklearn.preprocessing import MinMaxScaler

# Raw data paths
BASE_DIR = "/home/hsouza/internship/denoise-fnirs-ufabc/"
BASE_DIR_A = BASE_DIR + "raw_data/a/bids_raw/"
OUTPUT_DIR = BASE_DIR + "csv_data/"
pattern_subject_a = re.compile(r'.*-\d{2}$')
pattern_data_a = re.compile(r'.*\.snirf$')

matching_folders = []

def get_subject(BASE_DIR_A):
    for item in os.listdir(BASE_DIR_A):
        full_path = os.path.join(BASE_DIR_A, item)
        if os.path.isdir(full_path) and pattern_subject_a.match(item):
            matching_folders.append(full_path)
    return matching_folders

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

matching_folders = get_subject(BASE_DIR_A)

snirf_to_csv(matching_folders)
# # Assuming 'fnirs_data' is a NumPy array of shape (time_points, channels)
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(fnirs_data)

# # Segment the data into windows
# window_size = 128  # Example window size
# X = []
# for i in range(len(normalized_data) - window_size):
#     X.append(normalized_data[i:i+window_size])

# X = np.array(X)