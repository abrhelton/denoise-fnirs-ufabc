import os
import pandas as pd

# Caminhos
BASE_DIR = "D:/internship/denoise-fnirs-ufabc/data"
NORM_DIR = os.path.join(BASE_DIR, "csv_data")  # Agora lendo os arquivos .csv normalizados
REPORT_PATH = os.path.join(BASE_DIR, "canal_por_arquivo.csv")

# Lista de resultados
report = []

for file in os.listdir(NORM_DIR):
    if file.endswith(".csv"):
        csv_path = os.path.join(NORM_DIR, file)
        print(f"Verificando: {file}")
        try:
            df = pd.read_csv(csv_path)

            num_channels = df.shape[1]
            num_samples = df.shape[0]

            report.append({
                "arquivo": file,
                "num_canais": num_channels,
                "num_amostras": num_samples
            })

        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
            report.append({
                "arquivo": file,
                "num_canais": "erro",
                "num_amostras": "erro"
            })

# Salva o relatório em CSV
report_df = pd.DataFrame(report)
report_df.to_csv(REPORT_PATH, index=False)

print(f"\n✅ Relatório salvo em: {REPORT_PATH}")
