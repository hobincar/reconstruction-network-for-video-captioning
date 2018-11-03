import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, model_name, n_layers, encoder_size, embedding_size, embedding_scale, hidden_size,
                 output_size, embedding_dropout, dropout, max_length):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.encoder_size = encoder_size
        self.embedding_size = embedding_size
        self.embedding_scale = embedding_scale
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dropout_p = embedding_dropout
        self.dropout_p = dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_p)
        self.attn = nn.Linear(self.embedding_size + self.hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.encoder_size, self.hidden_size)

        self.embed_att_W = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.embed_att_U = nn.Linear(self.encoder_size, self.hidden_size, bias=False)
        self.embed_att_b = nn.Parameter(torch.zeros(self.hidden_size))
        self.embed_att_w = nn.Linear(self.hidden_size, 1, bias=False)

        if model_name == "LSTM":
            rnn_unit = nn.LSTM
        else:
            rnn_unit = nn.GRU
        self.rnn = rnn_unit(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input.squeeze(0))
        embedded = embedded * self.embedding_scale
        embedded = self.embedding_dropout(embedded)

        alphas = self.embed_att_w(
            self.embed_att_W(hidden) + self.embed_att_U(encoder_output) + self.embed_att_b
        )

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden.squeeze(0)), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attn_applied = attn_applied.squeeze(1)

        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

