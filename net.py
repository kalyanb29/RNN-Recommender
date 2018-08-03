import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(num_layers * hidden_size, output_size)

    def forward(self, input, hx):
        output, hx = self.lstm(input, hx)
        output = output[-1]
        output = self.linear(output)
        return output

    def reset_parameters(self):
        self.lstm.reset_parameters()

