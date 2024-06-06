import torch.nn as nn
import torch
from torch.nn  import TransformerEncoder, TransformerEncoderLayer

# 当前对局信息的序列的维度为55
input_size = 55

d_model = 32 # 嵌入后向量维度
max_len = 512 # 序列最大长度
nhead = 4 # 多头注意力的头数
num_encoder_layers = 6 # 编码器的层数
dim_feedforward = 4*d_model # 前馈网络的维度

# 全连接层
class Token_embed(nn.Module):
    def __init__(self, input_size=input_size, output_size=d_model):
        super(Token_embed, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size*seq_len, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.view(batch_size, seq_len, -1)
        return x
    

class RoPE(nn.Module):
    def __init__(self, d_model=d_model, max_len=max_len):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        print(position.shape, div_term.shape)
        sinusoid_inp = torch.einsum('i,j->ij', position, div_term)
        
        self.sin = torch.sin(sinusoid_inp).unsqueeze(0)
        self.cos = torch.cos(sinusoid_inp).unsqueeze(0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim)
        seq_len = x.size(1)
        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]

        sin = self.sin[:, :seq_len, :]
        cos = self.cos[:, :seq_len, :]
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        x_rot = torch.stack((x1_rot, x2_rot), dim=-1).reshape_as(x)
        return x_rot
    
class Bert_like(nn.Module):
    def __init__(self, d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, 
                 dim_feedforward=dim_feedforward, max_len=max_len):
        super(Bert_like, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        
        self.token_embed = Token_embed()
        self.rope = RoPE(d_model, max_len)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
    def forward(self, x):
        x = self.token_embed(x)
        x = self.rope(x)
        x = self.transformer_encoder(x)
        return x
    