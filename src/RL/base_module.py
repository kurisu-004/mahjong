import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import GPT2Model, GPT2Config


# 参考suphx的方法，输入的特征形状为(batch, channel, height, width)
# 其中height=4, width=9表示所有的牌
# 不同的channel表示不同的特征，具体如下
# ------------------player0 自己------------------
# 0~4: 自己的手牌 player0
# 5~9: 自己的副露
# ------------------player1 下家------------------
# 10~14: player1的副露
# ------------------player2 对家------------------
# 15~19: player2的副露
# ------------------player3 上家------------------
# 20~24: player3的副露
# ------------------dora 指示牌------------------
# 25~29: dora牌
# ------------------剩余牌 视野中未出现的牌------------------
# 30~34: 剩余牌



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
    def __init__(self, hidden_channel=128):
        super(Tiles_CNN, self).__init__()
        # 3x3 convolution
        self.conv1 = nn.Conv2d(34, hidden_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.resnet_list = nn.ModuleList([Rsidual_Block(hidden_channel, hidden_channel) for i in range(18)])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channel*4*34, 512)
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = x.view(-1, 34, 4, 34)
        x = self.conv1(x)
        x = self.bn1(x)
        for resnet in self.resnet_list:
            x = resnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

# TODO 环境仅提供了单局的模拟，没有提供多局的模拟
# 因此有用的信息只有oya，riichi_sticks
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

    def forward(self, scores, oya, honba_riichi_sticks):
        # 调整输入的形状
        scores = scores.view(-1, 4)
        oya = oya.view(-1)
        honba_riichi_sticks = honba_riichi_sticks.view(-1, 2)
        scores = self.scores_layernorm(scores)
        scores = self.scores_embedding_layer(scores)
        oya = self.oya_embedding_layer(oya.to(torch.long))
        honba_riichi_sticks = self.honba_riichi_sticks_embedding_layer(honba_riichi_sticks)
        # print(scores.shape, oya.shape, honba_riichi_sticks.shape)
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

class action_GPT2(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.embeded = nn.Embedding(190, config.n_embd)
        self.config = config
        
    def forward(self, action_list, attention_mask=None):
        action_list = self.embeded(action_list)
        outputs = super().forward(inputs_embeds=action_list, attention_mask=attention_mask)
        return outputs.last_hidden_state

    
class my_model(nn.Module):
    def __init__(self, config):
        super(my_model, self).__init__()
        self.tiles_cnn = Tiles_CNN()
        self.pub_info_embedding = pub_info_embedding()
        self.action_GPT2 = action_GPT2(config=config)

        

        
    def forward(self, action_list, info, tile_features, mask):
        tile_features = self.tiles_cnn(tile_features.to(torch.float32)) # (batch, 128)
        info = self.pub_info_embedding(**info) # (batch, 64)
        action_list = self.action_GPT2(action_list, attention_mask = mask) # (batch, seq_len, n_embd)
        # state_representation = torch.cat((tile_features, info, action_list), dim=1)

        return tile_features, info, action_list
    

class myCollator(nn.Module):

    def __init__(self, batch_size=32, max_len=128, 
                 tile_features_channel=34, 
                 tile_features_height=4, 
                 tile_features_width=34,
                 device='cuda'):
        self.max_len = max_len
        self.tile_features_channel = tile_features_channel
        self.tile_features_height = tile_features_height
        self.tile_features_width = tile_features_width
        self.device = device

        pass

    def compute_Q(self, num_actions, reward, gamma=0.99):
        Q = torch.zeros(num_actions)
        Q[num_actions-1] = reward
        for i in reversed(range(num_actions-1)):
            Q[i] = gamma * Q[i+1]
        return Q


    def __call__(self, features):

        action_list, info, tile_features, reward= zip(*[(item['action_list'], 
                                                         item['info'],
                                                         item['tile_features'],
                                                         item['reward']) for item in features])
        
        self_action_mask, legal_action_mask = zip(*[(item['self_action_mask'], 
                                                     item['legal_action_mask']) for item in features])
        batch_size = len(features)

        # 生成mask
        mask = torch.zeros(batch_size, self.max_len, dtype=torch.float32)
        for i in range(batch_size):
            mask[i, :len(action_list[i])] = 1

        # 计算Q值
        num_actions = [item.sum().item() for item in self_action_mask]
        Q = []
        for num_action, r in zip(num_actions, reward):
            Q.append(self.compute_Q(num_action, r))
        Q = torch.cat(Q, dim=0)
        # 填充action_list到max_len
        action_list_tensor = [torch.tensor(item + [0] * (self.max_len - len(item))) for item in action_list]
        action_list_tensor = torch.stack(action_list_tensor, dim=0)

        # 填充self_action_mask,到max_len
        # self_action_mask是一个list，每个元素是一个np.array
        # self_action_mask[i]是一个np.array，形状为(seq_len,)
        self_action_mask = [torch.tensor(item, dtype=torch.bool) for item in self_action_mask]
        self_action_mask = torch.stack([torch.cat([item, torch.zeros(self.max_len - len(item), dtype=torch.bool)]) for item in self_action_mask], dim=0)

        # 将legal_action_mask转换为tensor
        # legal_action_mask是一个list，每个元素是一个np.array，形状为(len(self_action_mask[i]), 49)
        legal_action_mask = [torch.tensor(item, dtype=torch.bool) for item in legal_action_mask]
        legal_action_mask = torch.cat([epsisode for epsisode in legal_action_mask], dim=0)




        # 这里的info是一个list，每个元素是一个dict
        # info[i]是一个dict，包含'scores', 'oya', 'honba_riichi_sticks'三个key
        # info[i]['scores']是一个np.array，形状为(len(self_action_mask[i]), 4)
        # info[i]['oya']是一个np.array，形状为(len(self_action_mask[i]),)
        # info[i]['honba_riichi_sticks']是一个np.array，形状为(len(self_action_mask[i]), 2)

        oya = torch.cat([torch.tensor(item['oya'], dtype=torch.long) for item in info], dim=0)
        scores = torch.cat([torch.tensor(item['scores'], dtype=torch.float32) for item in info], dim=0)
        honba_riichi_sticks = torch.cat([torch.tensor(item['honba_riichi_sticks'], dtype=torch.float32) for item in info], dim=0)
        info_tensor = {
            'scores': scores.to(self.device),
            'oya': oya.to(self.device),
            'honba_riichi_sticks': honba_riichi_sticks.to(self.device)
        }

        # tile_features是一个list，每个元素是一个(len(self_action_mask[i]), 34, 4, 34)的np.array
        # 调整tile_features的形状为(-1, channel, height, width)
        tile_features_tensor = torch.cat([torch.tensor(item, dtype=torch.float32) for item in tile_features], dim=0)


        return {
            'action_list': action_list_tensor.to(self.device),
            'self_action_mask': self_action_mask.to(self.device),
            'legal_action_mask': legal_action_mask.to(self.device),
            'info': info_tensor,
            'tile_features': tile_features_tensor.to(self.device),
            'mask': mask.to(self.device),
            'Q': Q.to(self.device)
        }



class Policy_Network(nn.Module):
    def __init__(self, config, action_space=49):
        super(Policy_Network, self).__init__()
        self.my_model = my_model(config)
        self.fc1 = nn.Linear(128+64+config.n_embd, 128)
        self.fcList = nn.ModuleList([nn.Linear(128, 128) for i in range(4)])
        self.fc2 = nn.Linear(128, action_space)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, action_list, info, tile_features, mask,
                self_action_mask, legal_action_mask, Q):

        tile_features, info, action_list = self.my_model(action_list, info, tile_features, mask)
        # self_action_mask形状为ie(batch, max_len)
        # action_list形状为(batch, max_len, n_embd)
        # 从action_list中取出self_action_mask为True的部分
        action_list = action_list[self_action_mask]

        state_representation = torch.cat((tile_features, info, action_list), dim=1)
        action_logits = self.fc1(state_representation)
        action_logits = F.relu(action_logits)
        for fc in self.fcList:
            action_logits = fc(action_logits)
            action_logits = F.relu(action_logits)
        action_logits = self.fc2(action_logits)
        # print(action_logits.shape)
    
        # 将legal_action_mask中为0的位置的logits设置为负无穷
        action_logits[~legal_action_mask] = float('-inf')
        action_probs = self.softmax(action_logits)
        loss = self.compute_loss(Q, action_probs)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'loss': loss
        }
    
    def compute_loss(self, Q, action_probs):
        max_action_indices = torch.argmax(action_probs, dim=1)
        log_probs = torch.log(action_probs[torch.arange(action_probs.shape[0]), max_action_indices])
        loss = -torch.sum(log_probs * Q)
        return loss