import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalReconstructor(nn.Module):
    def __init__(self, model_name, n_layers, decoder_hidden_size, hidden_size, dropout, decoder_dropout, caption_max_len):
        super(GlobalReconstructor, self).__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.decoder_dropout_p = decoder_dropout
        self.caption_max_len = caption_max_len

        self.decoder_dropout = nn.Dropout(self.decoder_dropout_p)
        if self.model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.decoder_hidden_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, decoder_hiddens):
        batch_specific_len = decoder_hiddens.size()[0]

        mean_pooled = decoder_hiddens.transpose(0, 2)
        mean_pooled = mean_pooled.transpose(1, 3)
        mean_pooled = mean_pooled.mean(2)
        mean_pooled = mean_pooled.mean(2)
        mean_pooled = mean_pooled / batch_specific_len * self.caption_max_len
        mean_pooled = self.decoder_dropout(mean_pooled)

        input_combined = torch.cat((input[0], mean_pooled), 1)
        input_combined = input_combined.unsqueeze(0)

        output, hidden = self.rnn(input_combined, hidden)

        output = self.out(output[0])
        return output, hidden

