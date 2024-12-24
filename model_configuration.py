# general parameters
device = "cuda"
model_path = "pong_rnn_model.pth"

# model parameters
input_size = 16
hidden_size = 128
output_size = 4
discrete_output_size = 6
classification_threshold = 0.99
num_layers = 2
number_heads = 8

# training parameters
input_sequence_length=10
