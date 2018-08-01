import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        self.linear = nn.Linear(num_layers * hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hx):
        output, hx = self.lstm(input, hx)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output

    def reset_parameters(self):
        self.lstm.reset_parameters()

