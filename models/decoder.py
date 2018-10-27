import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attn import Attn


class Decoder(nn.Module):
    def __init__(self, attn_model, encoder_output_size, embedding, embedding_dropout, input_size, hidden_size,
                 output_size, n_layers=1, dropout=0.1):
        super(Decoder, self).__init__()

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(encoder_output_size + hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size=encoder_output_size, attn_size=hidden_size)


    def forward(self, last_word, last_hidden, encoder_outputs):
        # NOTE: we run this one step (word) at a time

        embedded = self.embedding(last_word)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)

        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

