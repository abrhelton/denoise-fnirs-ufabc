import os
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch

# --- INÍCIO DA CORREÇÃO ---
import sys
# Adiciona o diretório que contém o seu módulo 'model.py' ao path do Python
# Altere este caminho se a sua estrutura de pastas for diferente
SRC_DIR = "/content/denoise-fnirs-ufabc/src"
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
# --- FIM DA CORREÇÃO ---

from model import VanillaAutoencoder

# Diretório onde estão os .csv já normalizados
DATA_DIR = "/content/denoise-fnirs-ufabc/data/norm_data"

# Diretório onde serão salvos os logs do MLflow (no seu Google Drive)
MLFLOW_DIR = "file:///content/drive/MyDrive/internship/mlruns"
mlflow.set_tracking_uri(MLFLOW_DIR)
mlflow.set_experiment("fNIRS-Denoising")


def load_all_preprocessed_data(path, train_size=0.8):
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    dfs = [pd.read_csv(f) for f in all_files]
    all_data = pd.concat(dfs, axis=0, ignore_index=True)

    if 'time' in all_data.columns:
        all_data = all_data.drop(columns=['time'])

    X_train, X_test = train_test_split(all_data.values, train_size=train_size, shuffle=True, random_state=42)
    return X_train, X_test


def evaluate_model(model, X_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.Tensor(X_test)
        outputs = model(inputs)
        loss_fn = torch.nn.MSELoss()
        test_loss = loss_fn(outputs, inputs).item()
    return test_loss


def train_model(X_train, X_test, input_dim, latent_dim, epochs, learning_rate, batch_size):
    with mlflow.start_run() as run:
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("latent_dim", latent_dim)

        model = VanillaAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_dataset = TensorDataset(torch.Tensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for data in train_loader:
                inputs, = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Avaliação no conjunto de teste
        test_loss = evaluate_model(model, X_test)
        mlflow.log_metric("test_loss", test_loss)
        print(f"Final Test Loss: {test_loss:.4f}")

        # Salvar o modelo no MLflow
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    X_train, X_test = load_all_preprocessed_data(DATA_DIR, train_size=0.8)

    INPUT_DIM = X_train.shape[1]
    LATENT_DIM = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64

    train_model(X_train, X_test, INPUT_DIM, LATENT_DIM, EPOCHS, LEARNING_RATE, BATCH_SIZE)

