import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, hidden_dim, rnn_type, bidir, checkpoint_name,
                 num_layers=1, dropout=0):
        super(RNNModel, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.hidden_dim = hidden_dim
        self.bidir = bidir
        self.bidir_mult = 2 if bidir else 1
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout

        if rnn_type == "RNN_TANH":
            self.rnn = nn.RNN(1, hidden_dim, nonlinearity='tanh', bidirectional=bidir,
                              num_layers=num_layers, dropout=dropout, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(1, hidden_dim, bidirectional=bidir,
                              num_layers=num_layers, dropout=dropout, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(1, hidden_dim, bidirectional=bidir,
                               num_layers=num_layers, dropout=dropout, batch_first=True)
        else:
            raise ValueError("Please choose either RNN_TANH, GRU, or LSTM as RNN type")

        if bidir:
            self.out = nn.Linear(hidden_dim * 2, 1)
        else:
            self.out = nn.Linear(hidden_dim, 1)

    def forward(self, input, hidden):
        # [batch, career_len] -> [batch, career_len, 1]
        input = input.unsqueeze(2)
        # [batch, career_len, 1] -> [batch, career_len, hidden_dim]
        if self.rnn_type == "LSTM":
            rnn_out, hidden = self.rnn(input, hidden)
        else:
            rnn_out, hidden = self.rnn(input, hidden)
        # [batch, career_len, hidden_dim] -> [batch, career_len, 1]
        output = self.out(rnn_out).permute(1, 0, 2)
        return torch.sigmoid(output.squeeze(2)), hidden
    
    def initHidden(self, batch_size):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.num_layers * self.bidir_mult, batch_size, self.hidden_dim),
                    torch.zeros(self.num_layers * self.bidir_mult, batch_size, self.hidden_dim))
        else:
            return torch.zeros(self.num_layers * self.bidir_mult, batch_size, self.hidden_dim)
    