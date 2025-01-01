import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from exact_engine import generate_pong_states
import numpy as np
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch as mlflow_pytorch

from models import PongDataset, ModelConfiguration
from runtime_configuration import Model, model_path

config = ModelConfiguration()

mlflow_server_url = "http://localhost:8080"
try:
    response = requests.get(mlflow_server_url)
    if response.status_code == 200:
        mlflow.set_tracking_uri(mlflow_server_url)
        print(f"MLflow Tracking URL set to: {mlflow_server_url}")
    else:
        print(f"MLflow server at {mlflow_server_url} is not available. Status code: {response.status_code}")
        print("Logging locally. To view results, run `python -m mlflow server --port 5000`")
except requests.exceptions.RequestException as e:
    print(f"Failed to connect to MLflow server at {mlflow_server_url}. Error: {e}")
    print("Logging locally. To view results, run `python -m mlflow server --port 5000`")

batch_size = 1000
train_data_set_steps = 3200000
validate_dataset_steps = 10000
num_workers = int(os.cpu_count() / 16)
learning_rate = 0.001
gamma = 0.95
epochs = 5

def train():
    train_dataset = PongDataset(generate_pong_states, train_data_set_steps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
    validate_dataset = PongDataset(generate_pong_states, validate_dataset_steps)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    model = Model().to(device=config.device)
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam([
    #     {'params': model.classification_head.parameters(), 'lr': .001},  # Classification head
    #     {'params': list(model.fc_feature_expansion.parameters()) + list(model.lstm.parameters()) + list(
    #         model.regression_head.parameters()), 'lr': learning_rate}
    #
    # ])RandomPaddleFactory

    scaler = torch.amp.GradScaler()
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    train_loss = []
    train_mse = []
    validation_loss = []
    validation_mse = []
    experiment = mlflow.set_experiment("Pong model")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_params(config.__dict__ | {
            # training hyper parameters
            "optimizer": type(optimizer).__name__,
            "optimizer_detailed": str(optimizer),
            "lr_scheduler": type(lr_scheduler).__name__,
            "loss_function": f"{type(regression_loss_fn).__name__} + {type(classification_loss_fn).__name__}",
            "batch_size": batch_size,
            "train_data_set_steps": train_data_set_steps,
            "validate_data_set_steps": validate_dataset_steps,
            "num_workers": num_workers,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epochs": epochs,
            # "window_size": train_arguments.window_size,
        })

        mlflow.set_tags({
                            "model": model.__class__.__name__
                        })
        for epoch in range(epochs):
            total_loss = 0
            avg_loss = 0
            total_mse = 0
            avg_mse = 0
            with tqdm(train_dataloader, unit=" batch", desc=f"Training (epoch {epoch + 1} of {epochs})") as loader:
                model.train()

                for idx, batch in enumerate(loader):
                    batch_states, batch_next_states = batch

                    batch_states = torch.tensor(batch_states).float().to(device=config.device)
                    batch_next_states = torch.tensor(batch_next_states).float().to(device=config.device)
                    target_continuous_states = batch_next_states[:, :config.output_size]
                    target_discrete_states = batch_next_states[:, config.output_size:]
                    # Forward pass
                    with torch.autocast(device_type=config.device, dtype=torch.float16):
                        continuous_states, discrete_states = model(batch_states)
                        classification_loss = classification_loss_fn(discrete_states, target_discrete_states)
                        regression_loss = regression_loss_fn(continuous_states, target_continuous_states)
                        combined_loss = classification_loss + regression_loss

                    # Backward pass
                    optimizer.zero_grad()
                    scaler.scale(combined_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += combined_loss.item()
                    avg_loss = total_loss / (idx + 1)

                    mse = np.mean(
                        np.square(target_continuous_states.cpu().detach().numpy() - continuous_states.cpu().detach().numpy()))
                    total_mse += mse.item()
                    avg_mse += total_mse / (idx + 1)
                    loader.set_postfix({"loss": avg_loss, "mse": avg_mse, "classification_loss": classification_loss.item(),
                                        "regression_loss": regression_loss.item(), "combined_loss": combined_loss.item()})
                train_loss.append(avg_loss)
                train_mse.append(avg_mse)

                mlflow.log_metrics({
                    f"train_loss": avg_loss,
                    f"train_mse": avg_mse,
                    f"learning_rate": scheduler.get_lr()[0],
                }, step=epoch)
            total_loss = 0
            avg_loss = 0
            total_mse = 0
            avg_mse = 0
            with tqdm(validate_dataloader, unit=" batch", desc=f"Validation (epoch {epoch + 1} of {epochs})") as loader:
                model.eval()
                for idx, batch in enumerate(loader):
                    batch_states, batch_next_states = batch
                    batch_states = torch.tensor(batch_states).float().to(device=config.device)
                    batch_next_states = torch.tensor(batch_next_states).float().to(device=config.device)
                    target_continuous_states = batch_next_states[:, :config.output_size]
                    target_discrete_states = batch_next_states[:, config.output_size:]
                    # Forward pass
                    continuous_states, discrete_states = model(batch_states)
                    classification_loss = classification_loss_fn(discrete_states, target_discrete_states)
                    combined_loss = classification_loss + regression_loss_fn(continuous_states, target_continuous_states)

                    total_loss += combined_loss.item()
                    avg_loss = total_loss / (idx + 1)

                    mse = np.mean(
                        np.square(target_continuous_states.cpu().detach().numpy() - continuous_states.cpu().detach().numpy()))
                    total_mse += mse.item()
                    avg_mse += total_mse / (idx + 1)
                    loader.set_postfix({"loss": avg_loss, "mse": mse})

                validation_loss.append(avg_loss)
                validation_mse.append(avg_mse)
                mlflow.log_metrics({
                    f"validate_loss": avg_loss,
                    f"validate_mse": avg_mse,
                }, step=epoch)
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                mlflow_pytorch.log_model(model, f"model_e{epoch}")

    torch.save(model.state_dict(), model_path)

    x = np.arange(epochs) + 1
    fix, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.plot(x, train_loss, label="train loss")
    ax1.plot(x, validation_loss, label="validation loss")
    ax1.legend(loc="upper right")
    ax2.plot(x, train_mse, label="train mse")
    ax2.plot(x, validation_mse, label="validation mse")
    ax2.legend(loc="upper right")
    plt.savefig(f"{os.path.basename(model_path)}.train_results.png")
    plt.show()


if __name__ == "__main__":
    train()

