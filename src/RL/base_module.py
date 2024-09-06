import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import GPT2Model, GPT2Config


# 参考suphx的方法，输入的特征形状为(batch, channel, height, width)
# 其中height=4, width=34表示所有的牌
# 不同的channel表示不同的特征，具体如下
# ------------------player0 自己------------------
# 0~3: 自己的手牌 player0
# 4~7: 自己的副露
# 8: 自己是否持有红宝牌
# ------------------player1 下家------------------
# 9~12: player1的副露
# 13: player1是否持有红宝牌
# ------------------player2 对家------------------
# 14~17: player2的副露
# 18: player2是否持有红宝牌
# ------------------player3 上家------------------
# 19~22: player3的副露
# 23: player3是否持有红宝牌
# ------------------dora 指示牌------------------
# 24~27: dora牌
# 28: 是否有赤宝牌
# ------------------剩余牌 视野中未出现的牌------------------
# 29~32: 剩余牌
# 33: 是否有赤宝牌


class Rsidual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Rsidual_Block, self).__init__()
        # 3x3 convolution
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.squential = nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2)

    def forward(self, x):
        residual = x
        out = self.squential(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Tiles_CNN(nn.Module):
    def __init__(self):
        super(Tiles_CNN, self).__init__()
        # 3x3 convolution
        self.conv1 = nn.Conv2d(34, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.resnet_list = nn.ModuleList([Rsidual_Block(128, 128) for i in range(18)])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*4*34, 512)
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        for resnet in self.resnet_list:
            x = resnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

class pub_info_embedding(nn.Module):
    def __init__(self):
        super(pub_info_embedding, self).__init__()

        self.concat_vector_size = 32 + 16 + 16
        self.scores_embedding_layer_output_size = 32
        self.oya_embedding_layer_output_size = 16
        self.honba_riichi_sticks_embedding_layer_output_size = 16
        self.scores_embedding_layer = nn.Linear(4, self.scores_embedding_layer_output_size, dtype=torch.float32)    # 4 * 8 = 32
        self.scores_layernorm = nn.LayerNorm(4, dtype=torch.float32)
        self.oya_embedding_layer = nn.Embedding(4, self.oya_embedding_layer_output_size, dtype=torch.float32)      # 4 * 4 = 16
        self.honba_riichi_sticks_embedding_layer = nn.Linear(2, self.honba_riichi_sticks_embedding_layer_output_size, dtype=torch.float32) # 2 * 8 = 16 

    def forward(self, scores, oya, honba, riichi_sticks):
        scores = self.scores_layernorm(scores)
        scores = self.scores_embedding_layer(scores)
        oya = self.oya_embedding_layer(oya)
        honba_riichi_sticks = self.honba_riichi_sticks_embedding_layer(honba_riichi_sticks)
        x = torch.cat((scores, oya, honba_riichi_sticks), dim=1)

        return x
    
# 对动作进行编码，编码形式如下
action_encoding = {
    # 0~48同时也是动作空间
    'player0': {
        'discard': {
            **{
                f'{i}{suit}': idx + suit_idx * 10
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30
                for idx, i in enumerate(range(1, 8))
            },
        },
        'chi_left': 37,
        'chi_middle': 38,
        'chi_right': 39,
        'pon': 40,
        'minkan': 41,
        'ankan': 42,
        'kakan': 43,
        'reach': 44,
        'agari': 45,
        'pass_naki': 46,
        'pass_reach': 47,
        'kyushukyuhai': 48,
        },
    'player1': {
        'discard': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 49
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 49
                for idx, i in enumerate(range(1, 8))
            },
        },
        'chi_left': 86,
        'chi_middle': 87,
        'chi_right': 88,
        'pon': 89,
        'minkan': 90,
        'ankan': 91,
        'kakan': 92,
        'reach': 93,
        'agari': 94,
        'kyushukyuhai': 95,
        },
    'player2': {
        'discard': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 96
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 96
                for idx, i in enumerate(range(1, 8))
            },
        },
        'chi_left': 133,
        'chi_middle': 134,
        'chi_right': 135,
        'pon': 136,
        'minkan': 137,
        'ankan': 138,
        'kakan': 139,
        'reach': 140,
        'agari': 141,
        'kyushukyuhai': 142,
        },
    'player3': {
        'discard': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 143
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 143
                for idx, i in enumerate(range(1, 8))
            },
        },
        'chi_left': 180,
        'chi_middle': 181,
        'chi_right': 182,
        'pon': 183,
        'minkan': 184,
        'ankan': 185,
        'kakan': 186,
        'reach': 187,
        'agari': 188,
        'kyushukyuhai': 189,
        }
    }

class action_GPT2(nn.Module):
    def __init__(self):
        super(action_GPT2, self).__init__()
        self.action_embedding_layer = nn.Embedding(225, self.d_model, dtype=torch.float32)

        
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        return x
    
class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()

        
    def forward(self, x):

        return x