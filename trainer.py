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

from model import PongDataset
from model_configuration import device, discrete_output_size, output_size
from runtime_configuration import Model, model_path

batch_size = 1000
train_data_set_steps = 4000000
validate_dataset_steps = 10000
num_workers = int(os.cpu_count() / 16)
learning_rate = 0.001
gamma = 0.90
epochs = 150

def train():
    train_dataset = PongDataset(generate_pong_states, train_data_set_steps)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
    validate_dataset = PongDataset(generate_pong_states, validate_dataset_steps)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

    model = Model().to(device=device)
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam([
    #     {'params': model.classification_head.parameters(), 'lr': .001},  # Classification head
    #     {'params': list(model.fc_feature_expansion.parameters()) + list(model.lstm.parameters()) + list(
    #         model.regression_head.parameters()), 'lr': learning_rate}
    #
    # ])RandomPaddleFactory

    scaler = torch.cuda.amp.GradScaler()
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    train_loss = []
    train_mse = []
    validation_loss = []
    validation_mse = []

    for epoch in range(epochs):
        total_loss = 0
        avg_loss = 0
        total_mse = 0
        avg_mse = 0
        count = 1
        with tqdm(train_dataloader, unit=" batch", desc=f"Training (epoch {epoch + 1} of {epochs})") as loader:
            model.train()

            for idx, batch in enumerate(loader):
                batch_states, batch_next_states = batch

                batch_states = torch.tensor(batch_states).float().to(device=device)
                batch_next_states = torch.tensor(batch_next_states).float().to(device=device)
                target_continuous_states = batch_next_states[:, :output_size]
                target_discrete_states = batch_next_states[:, output_size:]
                # Forward pass
                with torch.autocast(device_type=device, dtype=torch.float16):
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

        total_loss = 0
        avg_loss = 0
        total_mse = 0
        avg_mse = 0
        with tqdm(validate_dataloader, unit=" batch", desc=f"Validation (epoch {epoch + 1} of {epochs})") as loader:
            model.eval()
            for idx, batch in enumerate(loader):
                batch_states, batch_next_states = batch
                batch_states = torch.tensor(batch_states).float().to(device=device)
                batch_next_states = torch.tensor(batch_next_states).float().to(device=device)
                target_continuous_states = batch_next_states[:, :output_size]
                target_discrete_states = batch_next_states[:, output_size:]
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
        scheduler.step()

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

