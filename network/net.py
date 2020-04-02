import torch
import torch.nn as nn


class CharacterRNN(nn.Module):

    def __init__(self, in_dim, hidden_dim, layer_dim, out_dim):
        super(CharacterRNN, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_size=in_dim,
                          hidden_size=hidden_dim,
                          num_layers=layer_dim,
                          batch_first=True,
                          nonlinearity='relu')
        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        '''

        :param x: x.shape                   batch_size,seq_dim,in_dim
        :param:h0 ho.shape                  layer_dim, batch_size,hidden_dim
        :return: RNN return shape           batch_size,seq_dim,hidden_dim
        '''
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        out, hn = self.rnn(x, h0)

        out = out.contiguous().view(-1, self.hidden_dim)

        out = self.fc(out)

        return out
