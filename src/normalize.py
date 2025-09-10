import os
import pandas as pd
import numpy as np
import re

# Coeficientes de extinção (exemplo, em mmol^-1·mm^-1, valores reais devem vir da literatura)
# λ1 = 757 nm, λ2 = 843 nm
ext_coeffs = {
    756: {"HbO": 570.0, "HbR": 1650.0},
    757: {"HbO": 576.0, "HbR": 1660.0},
    843: {"HbO": 1040.0, "HbR": 840.0},
    853: {"HbO": 1030.0, "HbR": 830.0},
}

# Distância fonte-detector (em mm, precisa ajustar para seu setup real)
d = 30  
DPF = 6  # fator médio
L = d * DPF

EPS = 1e-8  # evita log(0)

num_channels = 48

def mbll(intensity_df):
    """
    Converte intensidade óptica em HbO/HbR via Beer-Lambert Law modificada.
    Suporta pares (756-853) e (757-843).
    """
    hbo_data, hbr_data = {}, {}

    for col in intensity_df.columns:
        # Captura base do canal e comprimento de onda (ex.: S1_D2 757)
        match = re.match(r"(S\d+_D\d+)\s*(\d+)", col)
        if not match:
            continue

        base, wl = match.groups()
        wl = int(wl)

        # Determina par válido
        if wl in [756, 757]:
            other_wl = 853 if wl == 756 else 843
        elif wl in [843, 853]:
            other_wl = 756 if wl == 853 else 757
        else:
            continue

        col_other = f"{base} {other_wl}"
        if col_other not in intensity_df.columns:
            continue

        # Intensidades com correção para evitar log(0/negativo)
        I1 = np.clip(intensity_df[col].to_numpy(), EPS, None)
        I2 = np.clip(intensity_df[col_other].to_numpy(), EPS, None)

        A1 = -np.log(I1 / I1[0])
        A2 = -np.log(I2 / I2[0])

        # Matriz de coeficientes
        E = np.array([
            [ext_coeffs[wl]["HbO"], ext_coeffs[wl]["HbR"]],
            [ext_coeffs[other_wl]["HbO"], ext_coeffs[other_wl]["HbR"]],
        ]) * L

        A = np.vstack([A1, A2]).T

        try:
            inv = np.linalg.inv(E)
            C = A @ inv.T
            hbo_data[f"{base}_HbO"] = C[:, 0]
            hbr_data[f"{base}_HbR"] = C[:, 1]
        except np.linalg.LinAlgError:
            print(f"[X] Matriz não invertível em {base} ({wl}/{other_wl})")

    result = pd.DataFrame({**hbo_data, **hbr_data})
    # limpa inf e NaN
    result = result.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return result


def get_datasets():
    BASE_DIR = "D:/internship/denoise-fnirs-ufabc/data/csv_data"
    return [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.csv')]

def zscore_normalize(df):
    return df.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)

def process_and_save():
    input_files = get_datasets()
    output_dir = "D:/internship/denoise-fnirs-ufabc/data/norm_data"
    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_files:
        try:

            df = pd.read_csv(file_path)
            current = df.shape[1]
            if current <= num_channels:

                # Remove colunas de tempo
                for time_col in ["time", "Time", "Time (s)"]:
                    if time_col in df.columns:
                        df = df.drop(columns=[time_col])

                # Aplica MBLL -> HbO/HbR
                df_conc = mbll(df)

                # Normaliza
                df_normalized = zscore_normalize(df_conc)

                # Salva
                filename = os.path.basename(file_path)
                output_path = os.path.join(output_dir, filename)
                df_normalized.to_csv(output_path, index=False)

                print(f"[✓] Processado: {filename}")

        except Exception as e:
            print(f"[X] Erro em {file_path}: {e}")

    print("✔️ Todos os arquivos processados e salvos em:", output_dir)


if __name__ == "__main__":
    process_and_save()
