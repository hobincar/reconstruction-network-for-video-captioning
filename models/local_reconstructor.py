import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalReconstructor(nn.Module):
    def __init__(self, model_name, n_layers, decoder_hidden_size, hidden_size, dropout, decoder_dropout, attn_size):
        super(LocalReconstructor, self).__init__()
        self.model_name = model_name
        self.n_layers = n_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout
        self.decoder_dropout_p = decoder_dropout
        self.attn_size = attn_size

        self.attn_W = nn.Linear(self.hidden_size, self.attn_size, bias=False)
        self.attn_U = nn.Linear(self.decoder_hidden_size, self.attn_size, bias=False)
        self.attn_b = nn.Parameter(torch.ones(self.attn_size), requires_grad=True)
        self.attn_tanh = nn.Tanh()
        self.attn_w = nn.Linear(self.attn_size, 1, bias=False)
        self.attn_softmax = nn.Softmax()

        self.decoder_dropout = nn.Dropout(self.decoder_dropout_p)
        if self.model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.decoder_hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, decoder_hiddens):
        if self.model_name == "LSTM":
            Wh = self.attn_W(hidden[0][-1])
        else:
            Wh = self.attn_W(hidden[-1])
        Uv = self.attn_U(decoder_hiddens)
        Wh = Wh.unsqueeze(0).unsqueeze(0).expand_as(Uv)
        betas = Wh + Uv + self.attn_b
        betas = self.attn_tanh(betas)
        betas = self.attn_w(betas)
        betas = betas.expand_as(decoder_hiddens)
        weighted_decoder_hiddens = betas * decoder_hiddens
        input = weighted_decoder_hiddens.mean(dim=0)
        input = self.decoder_dropout(input)

        output, hidden = self.rnn(input, hidden)

        output = self.out(output[0])
        return output, hidden

