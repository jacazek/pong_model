import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from engine import finite_pong_state

from fuzzy_engine import PongDataset, RNNModel

train_dataset = PongDataset(finite_pong_state, 10000)
train_dataloader = DataLoader(train_dataset, batch_size=40, shuffle=False)

validate_dataset = PongDataset(finite_pong_state, 500)
validate_dataloader = DataLoader(validate_dataset, batch_size=10, shuffle=False)

model = RNNModel(8, 8, 4, 1)
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 100
for epoch in range(epochs):

    total_loss = 0
    count = 1
    with tqdm(train_dataloader, unit=" batch", desc="Training...") as loader:
        model.train()
        for idx, batch in enumerate(loader):
            batch_states, batch_next_states = batch
            batch_states = torch.tensor(batch_states).float()
            batch_next_states = torch.tensor(batch_next_states).float()
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_next_states)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / (idx+1)
            loader.set_postfix({"loss": avg_loss})

    with tqdm(validate_dataloader, unit=" batch", desc="Validation...") as loader:
        model.eval()
        for idx, batch in enumerate(loader):
            batch_states, batch_next_states = batch
            batch_states = torch.tensor(batch_states).float()
            batch_next_states = torch.tensor(batch_next_states).float()
            # Forward pass
            predictions = model(batch_states)
            loss = criterion(predictions, batch_next_states)

            total_loss += loss.item()

            avg_loss = total_loss / (idx + 1)
            loader.set_postfix({"loss": avg_loss})

torch.save(model.state_dict(), "pong_rnn_model.pth")





        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
