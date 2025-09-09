import os
import torch
import pandas as pd
import mlflow.pytorch

# Caminhos
BASE_DIR = "/content/denoise-fnirs-ufabc/data"
INPUT_DIR = os.path.join(BASE_DIR, "norm_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carregar último modelo treinado do MLflow
MLFLOW_DIR = "file:///content/drive/MyDrive/internship/mlruns"
mlflow.set_tracking_uri(MLFLOW_DIR)
experiment = mlflow.get_experiment_by_name("fNIRS-Denoising")
runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time desc"])
last_run_id = runs.iloc[0].run_id
model = mlflow.pytorch.load_model(f"runs:/{last_run_id}/model")

print(f"✅ Modelo carregado do run_id: {last_run_id}")

# Função para denoising
def denoise_file(file_path, output_dir, model):
    df = pd.read_csv(file_path)
    X = torch.Tensor(df.values)

    # Forward pass
    with torch.no_grad():
        denoised = model(X).numpy()

    # Salvar CSV reconstruído
    denoised_df = pd.DataFrame(denoised, columns=df.columns)
    filename = os.path.basename(file_path).replace(".csv", "_denoised.csv")
    save_path = os.path.join(output_dir, filename)
    denoised_df.to_csv(save_path, index=False)

    print(f"Arquivo denoised salvo: {save_path}")

# Processar todos os arquivos normalizados
for file in os.listdir(INPUT_DIR):
    if file.endswith(".csv"):
        file_path = os.path.join(INPUT_DIR, file)
        denoise_file(file_path, OUTPUT_DIR, model)
