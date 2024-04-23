import torch.nn as nn
import torch
from torch.nn  import TransformerEncoder, TransformerEncoderLayer

class Bert_like(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(Bert_like, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads) for _ in range(self.num_layers)])
        self.encoder = nn.TransformerEncoder(self.layers, num_layers=self.num_layers)
        self.fc = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = self.softmax(x)
        return x
