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
        self.fc2 = nn.Linear(512, 128)
        
    def forward(self, x):
        x = x.view(-1, self.input_channel, self.input_height, self.input_width)
        x = self.conv1(x)
        x = self.bn1(x)
        for resnet in self.resnet_list:
            x = resnet(x)
        x = self.flatten(x)
        x = self.fc1(x)
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
        self.relu = nn.ReLU()

    def forward(self, oya, riichi_sticks):
        oya_riichi_sticks = torch.stack((oya, riichi_sticks), dim=1)
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

class action_GPT2(GPT2Model):
    def __init__(self, config, num_actions=189):
        super().__init__(config)
        self.embeded = nn.Embedding(num_actions, config.n_embd)
        self.config = config
        
    def forward(self, action_list, attention_mask):
        action_list = self.embeded(action_list)
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


    def __call__(self, features, is_train=True):
        batch_size = len(features)
        # features是一个列表，列表中的每个元素是一个字典
        tiles_features = [torch.tensor(feature['tiles_features'], dtype=torch.float32).to(self.device) for feature in features]
        infos = [feature['info'] for feature in features]
        oya = [torch.tensor(info['oya'], dtype=torch.float32).to(self.device) for info in infos]
        riichi_sticks = [torch.tensor(info['riichi_sticks'], dtype=torch.float32).to(self.device) for info in infos]
        action_list = [torch.tensor(feature['action_list'], dtype=torch.long).to(self.device) for feature in features]
        attention_mask = [torch.tensor(feature['attention_mask'], dtype=torch.float32).to(self.device) for feature in features]
        self_action_mask = [torch.tensor(feature['self_action_mask'], dtype=bool).to(self.device) for feature in features]
        legal_action_mask = [torch.tensor(feature['legal_action_mask'], dtype=bool).to(self.device) for feature in features]
        return_data = {
            'tiles_features': torch.cat(tiles_features, dim=0).to(self.device),
            'info': {
                'oya': torch.cat(oya, dim=0).to(self.device),
                'riichi_sticks': torch.cat(riichi_sticks, dim=0).to(self.device),
            },
            'action_list': torch.stack(action_list, dim=0).to(self.device),
            'attention_mask': torch.stack(attention_mask, dim=0).to(self.device),
            'self_action_mask': torch.stack(self_action_mask, dim=0).to(self.device),
            'legal_action_mask': torch.cat(legal_action_mask, dim=0).to(self.device),
        }
        if is_train:
            Q_values = [torch.tensor(feature['Q_values'], dtype=torch.float32).to(self.device) for feature in features]
            return_data['Q_values'] = torch.cat(Q_values, dim=0).to(self.device)
        return return_data


class Policy_Network(nn.Module):
    def __init__(self, config, action_space=47):
        super(Policy_Network, self).__init__()
        self.my_model = my_model(config)
        self.fc1 = nn.Linear(128+16+config.n_embd, 128)
        self.fcList = nn.ModuleList([nn.Linear(128, 128) for i in range(4)])
        self.fc2 = nn.Linear(128, action_space)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, action_list, self_action_mask, attention_mask, info, tiles_features, legal_action_mask, Q_values):

        state_representation = self.my_model(action_list, self_action_mask, attention_mask, info, tiles_features)
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
        loss = self.compute_loss(Q_values, action_probs)

        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'loss': loss,
        }
    
    def inference(self, action_list, attention_mask, oya, riichi_sticks, tiles_features, legal_action_mask):
        state_representation = self.my_model.inference(action_list, attention_mask, oya, riichi_sticks, tiles_features)
        
        action_logits = self.fc1(state_representation)
        action_logits = F.relu(action_logits)

        # print("action_logits")
        # print(action_logits.shape)
        # print(action_logits)
        for fc in self.fcList:
            action_logits = fc(action_logits)
            action_logits = F.relu(action_logits)
        action_logits = self.fc2(action_logits)
        # print(action_logits.shape)
        # print(action_logits)
        # print("legal_action_mask:")
        # print(legal_action_mask.shape)
        # # return action_logits
        legal_action_mask = legal_action_mask.unsqueeze(0)
        action_logits[~legal_action_mask] = float('-inf')
        action_probs = self.softmax(action_logits)
        # print(action_logits)
        # # print(action_probs)
        # # print(action_logits)

        return {
            'action_probs': action_probs,
            'action': torch.argmax(action_probs, dim=1),
        }

    def compute_loss(self, Q, action_probs):
        max_action_indices = torch.argmax(action_probs, dim=1)
        log_probs = torch.log(action_probs[torch.arange(action_probs.shape[0]), max_action_indices])
        loss = -torch.sum(log_probs * Q)
        return loss