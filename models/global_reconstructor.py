import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalReconstructor(nn.Module):
    def __init__(self, n_layers, decoder_hidden_size, hidden_size, dropout):
        super(GlobalReconstructor, self).__init__()
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.combine = nn.Linear(self.decoder_hidden_size * 2, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, decoder_hiddens):
        mean_pooled = decoder_hiddens.transpose(0, 2)
        mean_pooled = mean_pooled.transpose(1, 3)
        mean_pooled = mean_pooled.mean(2)
        mean_pooled = mean_pooled.mean(2)

        output = torch.cat((input[0], mean_pooled), 1)
        output = self.combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = self.out(output[0])
        return output, hidden

