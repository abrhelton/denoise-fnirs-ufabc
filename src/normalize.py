import os
import pandas as pd
import numpy as np

# Coeficientes de extinção (exemplo, em mmol^-1·mm^-1, valores reais devem vir da literatura)
# λ1 = 757 nm, λ2 = 843 nm
ext_coeffs = {
    757: {"HbO": 0.6, "HbR": 1.5},
    843: {"HbO": 1.4, "HbR": 0.8}
}

# Distância fonte-detector (em mm, precisa ajustar para seu setup real)
d = 30  
DPF = 6  # fator médio
L = d * DPF

num_channels = 48

def mbll(intensity_df):
    """
    Converte intensidade óptica em HbO/HbR via Beer-Lambert Law modificada.
    """
    # Identifica pares de canais (mesmo Sx_Dy com λ 757 e 843)
    hbo_data, hbr_data = {}, {}


    for col in intensity_df.columns:
        if col.endswith("757"):
            base = col.replace("757", "")
            col_843 = base + "843"
            if col_843 in intensity_df.columns:
                I757 = intensity_df[col]
                I843 = intensity_df[col_843]

                # Normaliza em relação ao baseline (primeiro valor como I0)
                A757 = -np.log(I757 / I757.iloc[0])
                A843 = -np.log(I843 / I843.iloc[0])

                # Monta matriz de coeficientes
                E = np.array([
                    [ext_coeffs[757]["HbO"], ext_coeffs[757]["HbR"]],
                    [ext_coeffs[843]["HbO"], ext_coeffs[843]["HbR"]]
                ])

                A = np.vstack([A757, A843]).T

                # Resolve para HbO e HbR (ΔC = (E·L)^-1 · ΔA)
                try:
                    inv = np.linalg.inv(E * L)
                    C = A @ inv.T
                    hbo_data[base[:-1] + "HbO"] = C[:, 0]
                    hbr_data[base[:-1] + "HbR"] = C[:, 1]
                except np.linalg.LinAlgError:
                    print(f"[X] Problema invertendo matriz para {base}")
    
    return pd.DataFrame({**hbo_data, **hbr_data})

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
