from engine import finite_pong_state
from fuzzy_engine import PongDataset, RNNModel
import torch

model = RNNModel(8, 10, 4, 1)
model.load_state_dict(torch.load("pong_rnn_model.pth", weights_only=True))
model.eval()

dataset = PongDataset(finite_pong_state, 10)
for input, target in dataset:
    output = model(torch.tensor([input]).float())
    print(target, output.tolist()[0])


