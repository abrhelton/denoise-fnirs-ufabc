import numpy as np
import pandas as pd
import os
import mne

# Diretórios
BASE_DIR = "D:/internship/denoise-fnirs-ufabc/data"
ORIGIN = os.path.join(BASE_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "csv_data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CHANNELS = 48  # Número fixo de canais para treinamento

def csv_data(df, target_channels=TARGET_CHANNELS):
    # Remove colunas de tempo
    for time_col in ["time", "Time", "Time (s)"]:
        if time_col in df.columns:
            df = df.drop(columns=[time_col])
            
    # Garante que os dados são numéricos
    df = df.select_dtypes(include=[np.number])

    # Remove canais com std zero (sem variação)
    # df = df.loc[:, df.std() > 1e-10]

    # Normalização canal a canal
    # df = (df - df.min()) / (df.max() - df.min())

    # # Padding ou truncamento para target_channels
    # current = df.shape[1]
    # if current < target_channels:
    #     missing = target_channels - current
    #     padding = pd.DataFrame(np.zeros((df.shape[0], missing)), columns=[f"pad_{i}" for i in range(missing)])
    #     df = pd.concat([df, padding], axis=1)
    # elif current > target_channels:
    #     df = df.iloc[:, :target_channels]

    # Renomear colunas para ch_0, ch_1, ...
    # df.columns = [f"ch_{i}" for i in range(df.shape[1])]

    return df

for file in os.listdir(ORIGIN):
    if file.endswith(".snirf"):
        snirf_path = os.path.join(ORIGIN, file)
        print(f"Processando: {file}")

        try:
            raw = mne.io.read_raw_snirf(snirf_path, preload=True)
            df = raw.to_data_frame()

            current = df.shape[1]
            df = csv_data(df)

            csv_filename = file.replace(".snirf", ".csv")
            csv_path = os.path.join(OUTPUT_DIR, csv_filename)
            df.to_csv(csv_path, index=False)

            print(f"✅ Salvo: {csv_path}")

        except Exception as e:
            print(f"Erro ao processar {file}: {e}")
