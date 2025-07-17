import numpy as np
import pandas as pd
import os
import mne

# Raw data paths
BASE_DIR = "D:/internship/denoise-fnirs-ufabc/data"
ORIGIN = BASE_DIR + "/raw_data/"
OUTPUT_DIR = BASE_DIR + "/csv_data/"

for file in os.listdir(ORIGIN):
    if file.endswith(".snirf"):
        snirf_path = os.path.join(ORIGIN, file)
        print(f"Processando: {file}")

        try:
            raw = mne.io.read_raw_snirf(snirf_path, preload=True)
            df = raw.to_data_frame()

            csv_filename = file.replace(".snirf", ".csv")
            csv_path = os.path.join(OUTPUT_DIR, csv_filename)
            df.to_csv(csv_path, index=False)

            print(f"Salvo: {csv_path}")

        except Exception as e:
            print(f"Erro em {file}: {e}")