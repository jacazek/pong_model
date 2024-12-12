import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import generate_pong_states
import numpy as np
import os
import matplotlib.pyplot as plt


from fuzzy_engine import PongDataset, RNNModel
from model_configuration import device, model_path

batch_size = 100

train_data_set_steps = 100000
train_dataset = PongDataset(generate_pong_states, train_data_set_steps)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=int(os.cpu_count() / 4), pin_memory=True)

validate_dataset_steps = 10000
validate_dataset = PongDataset(generate_pong_states, validate_dataset_steps)
validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

model = RNNModel().to(device=device)
criterion = nn.MSELoss()

learning_rate = 0.01
gamma=0.5

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
train_loss = []
train_mse = []
validation_loss = []
validation_mse = []


epochs = 20
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
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_next_states)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (idx+1)

            mse = np.mean(np.square(batch_next_states.cpu().detach().numpy() - predictions.cpu().detach().numpy()))
            total_mse += mse.item()
            avg_mse += total_mse / (idx+1)
            loader.set_postfix({"loss": avg_loss, "mse": mse})
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
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_next_states)
            total_loss += loss.item()
            avg_loss = total_loss / (idx + 1)

            mse = np.mean(np.square(batch_next_states.cpu().detach().numpy() - predictions.cpu().detach().numpy()))
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
plt.show()
