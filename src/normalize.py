import os
import pandas as pd

def get_datasets():
    BASE_DIR = '/content/denoise-fnirs-ufabc/data/csv_data'
    datasets = []
    for dataset in os.listdir(BASE_DIR):
        if dataset.endswith('.csv'):
            datasets.append(os.path.join(BASE_DIR, dataset))
    return datasets

def zscore_normalize(df):
    """
    Normaliza o DataFrame canal a canal (coluna a coluna) usando z-score.
    """
    return df.apply(lambda col: (col - col.mean()) / col.std(ddof=0), axis=0)

def normalize_and_save_datasets_zscore():
    input_files = get_datasets()
    output_dir = '/content/denoise-fnirs-ufabc/data/norm_data'
    os.makedirs(output_dir, exist_ok=True)

    for file_path in input_files:
        try:
            df = pd.read_csv(file_path)

            # Verifica e remove a coluna de tempo se existir
            time_col = None
            if 'Time (s)' in df.columns:
                time_col = df['Time (s)']
                df = df.drop(columns=['Time (s)'])

            # Normaliza apenas colunas numéricas (HbO, HbR etc.)
            df_numeric = df.select_dtypes(include='number')
            df_normalized = zscore_normalize(df_numeric)

            # Reinsere a coluna de tempo como a primeira coluna, se aplicável
            if time_col is not None:
                df_normalized.insert(0, 'Time', time_col)

            # Salva o arquivo no diretório de saída
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            df_normalized.to_csv(output_path, index=False)

            print(f"[✓] Normalizado (tempo preservado): {filename}")

        except Exception as e:
            print(f"[X] Erro ao processar {file_path}: {e}")

    print("✔️  Todos os arquivos foram normalizados e salvos em:", output_dir)


normalize_and_save_datasets_zscore()
