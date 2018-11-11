import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, model_name, n_layers, encoder_size, embedding_size, embedding_scale, hidden_size,
                 output_size, embedding_dropout, dropout, out_dropout):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.encoder_size = encoder_size
        self.embedding_size = embedding_size
        self.embedding_scale = embedding_scale
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dropout_p = embedding_dropout
        self.dropout_p = dropout
        self.out_dropout_p = out_dropout

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_p)

        if model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.embedding_size + self.encoder_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.out_dropout = nn.Dropout(self.out_dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = embedded * self.embedding_scale
        embedded = self.embedding_dropout(embedded)

        global_encoder_feature = encoder_outputs.mean(dim=1, keepdim=True)
        global_encoder_feature = global_encoder_feature.transpose(0, 1)

        input_combined = torch.cat((embedded, global_encoder_feature), dim=2)

        output, hidden = self.rnn(input_combined, hidden)

        output = self.out(output[0])
        output = self.out_dropout(output)
        return output, hidden

