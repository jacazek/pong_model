# general parameters
device = "cpu"
model_path = "pong_rnn_model.pth"

# model parameters
input_size = 16
hidden_size = 64
output_size = 4
discrete_output_size = 6
classification_threshold = 0.5
num_layers = 2
number_heads = 4

# training parameters
input_sequence_length=10
