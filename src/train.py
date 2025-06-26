import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from model import VanillaAutoencoder
from preprocess import load_and_preprocess_data # Assuming you have this function

# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000") # Or your remote server
mlflow.set_experiment("fNIRS-Denoising")

def train_model(X_train, input_dim, latent_dim, epochs, learning_rate, batch_size):
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("latent_dim", latent_dim)

        # Initialize model, loss, and optimizer
        model = VanillaAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create DataLoader
        train_dataset = TensorDataset(torch.Tensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Enable autologging
        mlflow.pytorch.autolog()

        # Training loop
        for epoch in range(epochs):
            for data in train_loader:
                inputs, = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Load and preprocess data
    X_train, n_features = load_and_preprocess_data('data/your_fnirs_data.csv')

    # Define hyperparameters
    INPUT_DIM = X_train.shape[1] * X_train.shape[2] # Flattened window
    LATENT_DIM = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64

    # Reshape data for the model
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    train_model(X_train_flat, INPUT_DIM, LATENT_DIM, EPOCHS, LEARNING_RATE, BATCH_SIZE)