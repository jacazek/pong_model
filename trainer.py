import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import finite_pong_state
import numpy as np

from fuzzy_engine import PongDataset, RNNModel

train_dataset = PongDataset(finite_pong_state, 100000)
train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=False, num_workers=40, pin_memory=True)

validate_dataset = PongDataset(finite_pong_state, 500)
validate_dataloader = DataLoader(validate_dataset, batch_size=10, shuffle=False)

device="cuda"

model = RNNModel(8, 8, 4, 2).to(device=device)
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

epochs = 20
for epoch in range(epochs):

    total_loss = 0
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
            loader.set_postfix({"loss": avg_loss, "mse": mse})

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
            loader.set_postfix({"loss": avg_loss, "mse": mse})

    scheduler.step()

torch.save(model.state_dict(), "pong_rnn_model.pth")





        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
