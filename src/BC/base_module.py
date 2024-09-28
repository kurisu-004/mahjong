import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import GPT2Model, GPT2Config
import numpy as np


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
    def __init__(self, input_channel=35, input_height=4, input_width=9, hidden_channel=128):
        super(Tiles_CNN, self).__init__()
        self.input_channel = input_channel
        self.input_height = input_height
        self.input_width = input_width
        # 3x3 convolution
        self.conv1 = nn.Conv2d(input_channel, hidden_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channel)
        self.resnet_list = nn.ModuleList([Rsidual_Block(hidden_channel, hidden_channel) for i in range(18)])
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_channel*input_height*input_width, 512)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_height, self.input_width)
        x = self.conv1(x)
        x = self.bn1(x)
        for resnet in self.resnet_list:
            x = resnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

# 环境仅提供了单局的模拟，没有提供多局的模拟
# 因此有用的信息只有oya，riichi_sticks
# 输入形状为(batch, seq_len, 2)
class pub_info_embedding(nn.Module):
    def __init__(self):
        super(pub_info_embedding, self).__init__()
        self.oya_sticks_embedding = nn.Linear(2, 16, dtype=torch.float32)
        self.layernorm = nn.LayerNorm(2, dtype=torch.float32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, oya, riichi_sticks):
        oya_riichi_sticks = torch.concat((oya, riichi_sticks), dim=1)
        oya_riichi_sticks = self.layernorm(oya_riichi_sticks)
        x = self.oya_sticks_embedding(oya_riichi_sticks)
        x = self.relu(x)
        return x
    
# 对动作进行编码，编码形式如下
    action_encoding = {
        # 0~48同时也是动作空间
        'player0': {
            'discard': {
                **{
                    f'{i}{suit}': idx + suit_idx * 9
                    for suit_idx, suit in enumerate(['m', 'p', 's'])
                    for idx, i in enumerate(range(1, 10))
                },
                **{
                    f'{i}z': idx + 27
                    for idx, i in enumerate(range(1, 8))
                },
            },
            'chi_left': 34,
            'chi_middle': 35,
            'chi_right': 36,
            'pon': 37,
            'ankan': 38,
            'kan': 39,
            'kakan': 40,
            'riichi': 41,
            'ron': 42,
            'tsumo': 43,
            'kyushukyuhai': 44,
            'pass_naki': 45,
            'pass_riichi': 46,
            },
        'player1': {
            'discard': {
                **{
                    f'{i}{suit}': idx + suit_idx * 9 + 47
                    for suit_idx, suit in enumerate(['m', 'p', 's'])
                    for idx, i in enumerate(range(1, 10))
                },
                **{
                    f'{i}z': idx + 27 + 47
                    for idx, i in enumerate(range(1, 8))
                },
            },
            'chi_left': 81,
            'chi_middle': 82,
            'chi_right': 83,
            'pon': 84,
            'ankan': 85,
            'kan': 86,
            'kakan': 87,
            'riichi': 88,
            'ron': 89,
            'tsumo': 90,
            'kyushukyuhai': 91,
            'pass_naki': 92,
            'pass_riichi': 93,
        },
        'player2': {
            'discard': {
                **{
                    f'{i}{suit}': idx + suit_idx * 9 + 94
                    for suit_idx, suit in enumerate(['m', 'p', 's'])
                    for idx, i in enumerate(range(1, 10))
                },
                **{
                    f'{i}z': idx + 27 + 94
                    for idx, i in enumerate(range(1, 8))
                },
            },
            'chi_left': 128,
            'chi_middle': 129,
            'chi_right': 130,
            'pon': 131,
            'ankan': 132,
            'kan': 133,
            'kakan': 134,
            'riichi': 135,
            'ron': 136,
            'tsumo': 137,
            'kyushukyuhai': 138,
            'pass_naki': 139,
            'pass_riichi': 140,
        },
        'player3': {
            'discard': {
                **{
                    f'{i}{suit}': idx + suit_idx * 9 + 141
                    for suit_idx, suit in enumerate(['m', 'p', 's'])
                    for idx, i in enumerate(range(1, 10))
                },
                **{
                    f'{i}z': idx + 27 + 141
                    for idx, i in enumerate(range(1, 8))
                },
            },
            'chi_left': 175,
            'chi_middle': 176,
            'chi_right': 177,
            'pon': 178,
            'ankan': 179,
            'kan': 180,
            'kakan': 181,
            'riichi': 182,
            'ron': 183,
            'tsumo': 184,
            'kyushukyuhai': 185,
            'pass_naki': 186,
            'pass_riichi': 187,
        },
    }

class RoPE(nn.Module):
    def __init__(self, d_model, max_len):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # print(position.shape, div_term.shape)
        sinusoid_inp = torch.einsum('i,j->ij', position, div_term)
        
        self.sin = torch.sin(sinusoid_inp).unsqueeze(0)
        self.cos = torch.cos(sinusoid_inp).unsqueeze(0)

    def forward(self, x):
        device = x.device

        # x shape: (batch_size, seq_len, dim)
        seq_len = x.size(1)
        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]

        sin = self.sin[:, :seq_len, :].to(device)
        cos = self.cos[:, :seq_len, :].to(device)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        x_rot = torch.stack((x1_rot, x2_rot), dim=-1).reshape_as(x)
        return x_rot
    
class action_GPT2(GPT2Model):
    def __init__(self, config, num_actions=189):
        super().__init__(config)
        self.embeded = nn.Embedding(num_actions, config.n_embd)
        self.rope = RoPE(config.n_embd, config.n_positions)
        self.config = config
        
    def forward(self, action_list, attention_mask):
        action_list = self.embeded(action_list)
        action_list = self.rope(action_list)
        outputs = super().forward(inputs_embeds=action_list, attention_mask=attention_mask)
        return outputs.last_hidden_state

    
class my_model(nn.Module):
    def __init__(self, config):
        super(my_model, self).__init__()
        self.tiles_cnn = Tiles_CNN()
        self.pub_info_embedding = pub_info_embedding()
        self.config = config
        self.action_GPT2 = action_GPT2(config=config)


    def forward(self, action_list, self_action_mask, attention_mask, info, tiles_features):
        # input shape: (batch, input_channel, 4, 9)
        # output shape: (batch, 128)
        tiles_features = self.tiles_cnn(tiles_features)

        info = self.pub_info_embedding(**info) # (batch, 16)
        # (batch, seq_len, n_embd)
        action_list = self.action_GPT2(action_list, attention_mask = attention_mask)
        self_action_logits = action_list[self_action_mask].view(-1, self.config.n_embd)

        state_representation = torch.cat((tiles_features, info, self_action_logits), dim=1)

        return state_representation
    
    def inference(self, action_list, attention_mask, oya, riichi_sticks, tiles_features):
        tiles_features = self.tiles_cnn(tiles_features)
        info = self.pub_info_embedding(oya, riichi_sticks)
        action_list = self.action_GPT2(action_list, attention_mask = attention_mask)
        # 找到attention_mask中最后一个为True的位置
        indice = torch.where(attention_mask == 1)[1][-1].item()

        self_action_logits = action_list[:, indice, :].view(-1, self.config.n_embd)
        state_representation = torch.cat((tiles_features, info, self_action_logits), dim=1)
        return state_representation
    

class myCollator(nn.Module):

    def __init__(self, batch_size=32, max_len=128, 
                 tile_features_channel=35, 
                 tile_features_height=4, 
                 tile_features_width=9,
                 device='cuda'):
        self.max_len = max_len
        self.tile_features_channel = tile_features_channel
        self.tile_features_height = tile_features_height
        self.tile_features_width = tile_features_width
        self.device = device

        pass


    def __call__(self, features):
        tiles_features, oya, riichi_sticks, action_lists, attention_masks, self_action_masks, legal_action_mask, Q_values = self.unpack(features)
        augmented_oya, augmented_riichi_sticks, augmented_attention_masks, augmented_self_action_masks, augmented_Q_values = self.aug_unchange_features(oya, riichi_sticks, attention_masks, self_action_masks, Q_values)
        augmented_tiles_features = self.aug_tiles_features(tiles_features)
        augmented_action_lists = self.aug_action_list(action_lists, attention_masks)
        augmented_legal_action_mask = self.aug_legal_action_mask(legal_action_mask)

        return_data = {
            'tiles_features': augmented_tiles_features.to(self.device),
            'info': {
                'oya': augmented_oya.to(self.device).view(-1, 1).to(torch.float32),
                'riichi_sticks': augmented_riichi_sticks.to(self.device).view(-1, 1).to(torch.float32),
            },
            'action_list': augmented_action_lists.to(self.device),
            'attention_mask': augmented_attention_masks.to(self.device),
            'self_action_mask': augmented_self_action_masks.to(self.device),
            'legal_action_mask': augmented_legal_action_mask.to(self.device),
            'Q_values': augmented_Q_values.to(self.device).to(torch.float32),
        }

        return return_data
    def unpack(self, features):
        # features是一个列表，列表中的每个元素是一个字典
        tiles_features = [torch.tensor(feature['tiles_features'], dtype=torch.float32) for feature in features]
        tiles_features = torch.cat(tiles_features, dim=0).view(-1, self.tile_features_channel, self.tile_features_height, self.tile_features_width)
        infos = [feature['info'] for feature in features]
        oya = [info['oya'] for info in infos]
        riichi_sticks = [info['riichi_sticks']for info in infos]
        action_lists = [feature['action_list'] for feature in features]
        attention_masks = [feature['attention_mask'] for feature in features]
        self_action_masks = [feature['self_action_mask'] for feature in features]
        legal_action_masks = [feature['legal_action_mask'] for feature in features]
        Q_values = [feature['Q_values'] / 1000 for feature in features]
        return tiles_features, oya, riichi_sticks, action_lists, attention_masks, self_action_masks, legal_action_masks, Q_values
    
    def _expand_list(self, list_):
        return [item for item in list_ for _ in range(3)]

    def aug_unchange_features(self, oya, riichi_sticks, attention_masks, self_action_masks, Q_values):
        augmented_oya = [torch.tensor(array) for array in self._expand_list(oya)]
        augmented_oya = torch.cat(augmented_oya, dim=0)
        augmented_riichi_sticks = [torch.tensor(array) for array in self._expand_list(riichi_sticks)]
        augmented_riichi_sticks = torch.cat(augmented_riichi_sticks, dim=0)

        augmented_attention_masks = [torch.tensor(array, dtype=bool) for array in self._expand_list(attention_masks)]
        augmented_attention_masks = torch.stack(augmented_attention_masks, dim=0)

        augmented_self_action_masks = [torch.tensor(array, dtype=bool) for array in self._expand_list(self_action_masks)]
        augmented_self_action_masks = torch.stack(augmented_self_action_masks, dim=0)

        augmented_Q_values = [torch.tensor(array) for array in self._expand_list(Q_values)]
        augmented_Q_values = torch.cat(augmented_Q_values, dim=0)
        return augmented_oya, augmented_riichi_sticks, augmented_attention_masks, augmented_self_action_masks, augmented_Q_values

    def aug_tiles_features(self, tiles_features):
        # tiles_features shape: (batch, channel, height, width)
        # 将手牌三种花色进行轮换
        augmented_tiles_features = torch.tensor(np.zeros((tiles_features.shape[0]*3, tiles_features.shape[1], tiles_features.shape[2], tiles_features.shape[3])), dtype=torch.float32)
        augmented_tiles_features[0: tiles_features.shape[0]] = tiles_features
        augmented_tiles_features[tiles_features.shape[0]: tiles_features.shape[0]*2] = tiles_features.index_select(2, torch.tensor([2, 0, 1, 3]))
        augmented_tiles_features[tiles_features.shape[0]*2: tiles_features.shape[0]*3] = tiles_features.index_select(2, torch.tensor([1, 2, 0, 3]))
        return augmented_tiles_features

    def aug_action_list(self, action_lists, attention_masks):
        augmented_action_lists = []
        for action_list, attention_mask in zip(action_lists, attention_masks):
            augmented_action_lists.append(torch.tensor(np.array(action_list)))
            action_list_copy= action_list[attention_mask==1].copy()
            player_id = action_list_copy//47
            action = action_list_copy%47

            trans1_action = []
            trans2_action = []
            for a, p in zip(action[1:], player_id[1:]):
                if a < 27:
                    trans1_action.append((a + 9) % 27 + p*47)
                    trans2_action.append((a + 18) % 27 + p*47)
                else:
                    trans1_action.append(a + p*47)
                    trans2_action.append(a + p*47)
            augmented_action_lists.append(torch.tensor(np.array([188] + trans1_action + [0]*(128-1-len(trans1_action)))))
            augmented_action_lists.append(torch.tensor(np.array([188] + trans2_action + [0]*(128-1-len(trans2_action)))))

        return torch.stack(augmented_action_lists, dim=0)

    def aug_legal_action_mask(self, legal_action_mask):
        augmented_legal_action_mask = []
        for mask in legal_action_mask:
            mask = torch.tensor(mask)
            # 这里mask有可能是空的，即玩家根本没有行动
            if mask.shape[0] == 0:
                augmented_legal_action_mask.append(torch.cat((mask, mask, mask), dim=0))
            else:
                augmented_legal_action_mask1 = mask.index_select(1, 
                                                torch.tensor([18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]))
                augmented_legal_action_mask2 = mask.index_select(1,
                                                torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]))
                augmented_legal_action_mask.append(torch.cat((mask, augmented_legal_action_mask1, augmented_legal_action_mask2), dim=0))
        return torch.cat(augmented_legal_action_mask, dim=0)


class inference_Collator(myCollator):
    def __init__(self, batch_size=32, max_len=128, 
                 tile_features_channel=35, 
                 tile_features_height=4, 
                 tile_features_width=9,
                 device='cuda'):
        super().__init__(batch_size, max_len, tile_features_channel, tile_features_height, tile_features_width, device)

    def __call__(self, obs):
        tiles_features = torch.tensor(obs['tiles_features'], dtype=torch.float32).to(self.device).view(-1, self.tile_features_channel, self.tile_features_height, self.tile_features_width)
        oya = torch.tensor(obs['info']['oya'], dtype=torch.float32).unsqueeze(0).to(self.device).view(-1, 1)
        riichi_sticks = torch.tensor(obs['info']['riichi_sticks'],dtype=torch.float32).unsqueeze(0).to(self.device).view(-1, 1)
        action_list = torch.tensor(obs['action_list'],dtype=torch.long).unsqueeze(0).to(self.device).view(-1, self.max_len)
        attention_mask = torch.tensor(obs['attention_mask'],dtype=torch.long).unsqueeze(0).to(self.device).view(-1, self.max_len)
        legal_action_mask = torch.tensor(obs['legal_action_mask'], dtype=bool).unsqueeze(0).to(self.device).view(-1, 47)

        input = {
            "tiles_features": tiles_features,
            "oya": oya,
            "riichi_sticks": riichi_sticks,
            "action_list": action_list,
            "attention_mask": attention_mask,
            "legal_action_mask": legal_action_mask,
            }
        return input


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, output_size)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(self.fc1, self.relu, self.fc2, self.relu, self.fc3, self.relu, self.fc4, self.relu, self.fc5, self.relu, self.fc6)
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class Policy_Network(nn.Module):
    def __init__(self, config, action_space=47):
        super(Policy_Network, self).__init__()
        self.my_model = my_model(config)
        self.layernorm = nn.LayerNorm(128 + 16 + config.n_embd, dtype=torch.float32)
        self.mlp = MLP(128 + 16 + config.n_embd, action_space)
        # self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, action_list, self_action_mask, attention_mask, info, tiles_features, legal_action_mask, Q_values):

        state_representation = self.my_model(action_list, self_action_mask, attention_mask, info, tiles_features)
        state_representation = self.layernorm(state_representation)
        action_logits = self.mlp(state_representation)
    
        # 将legal_action_mask中为0的位置的logits设置为负无穷
        action_logits[~legal_action_mask] = float('-inf')
        action_probs = self.softmax(action_logits)
        loss = self.compute_loss(Q_values, action_probs)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'loss': loss,
        }
    
    def inference(self, action_list, attention_mask, oya, riichi_sticks, tiles_features, legal_action_mask):

        state_representation = self.my_model.inference(action_list, attention_mask, oya, riichi_sticks, tiles_features)
        state_representation = self.layernorm(state_representation)


        action_logits = self.mlp(state_representation)
        action_logits[~legal_action_mask] = float('-inf')
        action_probs = self.softmax(action_logits)

        return {
            'action_probs': action_probs,
            'action': torch.argmax(action_probs, dim=1),
        }

    def compute_loss(self, Q, action_probs):
        max_action_indices = torch.argmax(action_probs, dim=1)
        log_probs = torch.log(action_probs[torch.arange(action_probs.shape[0]), max_action_indices])
        loss = -torch.sum(log_probs * Q)
        return loss