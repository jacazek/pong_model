from engine import generate_pong_states
from fuzzy_engine import PongDataset, RNNModel
import torch

model = RNNModel(8, 8, 4, 1)
model.load_state_dict(torch.load("pong_rnn_model.pth", weights_only=True))
model.eval()

dataset = PongDataset(generate_pong_states, 10)
for input, target in dataset:
    output = model(torch.tensor([input]).float())
    print(target, output.tolist()[0])


