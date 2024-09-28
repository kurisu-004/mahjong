from transformers import GPT2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, gzip, pickle
import copy

pai_dict = {
    # 0~35: 万
    0: '1m', 1: '1m', 2: '1m', 3: '1m', 4: '2m', 5: '2m', 6: '2m', 7: '2m', 8: '3m', 9: '3m', 10: '3m', 11: '3m',
    12: '4m', 13: '4m', 14: '4m', 15: '4m', 16: '0m', 17: '5m', 18: '5m', 19: '5m', 20: '6m', 21: '6m', 22: '6m',
    23: '6m', 24: '7m', 25: '7m', 26: '7m', 27: '7m', 28: '8m', 29: '8m', 30: '8m', 31: '8m', 32: '9m', 33: '9m',
    34: '9m', 35: '9m',
    # 36~71: 筒
    36: '1p', 37: '1p', 38: '1p', 39: '1p', 40: '2p', 41: '2p', 42: '2p', 43: '2p', 44: '3p', 45: '3p', 46: '3p',
    47: '3p', 48: '4p', 49: '4p', 50: '4p', 51: '4p', 52: '0p', 53: '5p', 54: '5p', 55: '5p', 56: '6p', 57: '6p',
    58: '6p', 59: '6p', 60: '7p', 61: '7p', 62: '7p', 63: '7p', 64: '8p', 65: '8p', 66: '8p', 67: '8p', 68: '9p',
    69: '9p', 70: '9p', 71: '9p',
    # 72~107: 条
    72: '1s', 73: '1s', 74: '1s', 75: '1s', 76: '2s', 77: '2s', 78: '2s', 79: '2s', 80: '3s', 81: '3s', 82: '3s',
    83: '3s', 84: '4s', 85: '4s', 86: '4s', 87: '4s', 88: '0s', 89: '5s', 90: '5s', 91: '5s', 92: '6s', 93: '6s',
    94: '6s', 95: '6s', 96: '7s', 97: '7s', 98: '7s', 99: '7s', 100: '8s', 101: '8s', 102: '8s', 103: '8s', 104: '9s',
    105: '9s', 106: '9s', 107: '9s',
    # 108~135: 字
    108: '1z', 109: '1z', 110: '1z', 111: '1z', 112: '2z', 113: '2z', 114: '2z', 115: '2z', 116: '3z', 117: '3z',
    118: '3z', 119: '3z', 120: '4z', 121: '4z', 122: '4z', 123: '4z', 124: '5z', 125: '5z', 126: '5z', 127: '5z',
    128: '6z', 129: '6z', 130: '6z', 131: '6z', 132: '7z', 133: '7z', 134: '7z', 135: '7z'
}

# 定义牌的映射表
tile_map = {
    '0m': 0, '1m': 1, '2m': 2, '3m': 3, '4m': 4, '5m': 5, '6m': 6, '7m': 7, '8m': 8, '9m': 9,
    '0p': 10, '1p': 11, '2p': 12, '3p': 13, '4p': 14, '5p': 15, '6p': 16, '7p': 17, '8p': 18, '9p': 19,
    '0s': 20, '1s': 21, '2s': 22, '3s': 23, '4s': 24, '5s': 25, '6s': 26, '7s': 27, '8s': 28, '9s': 29,
    '1z': 30, '2z': 31, '3z': 32, '4z': 33, '5z': 34, '6z': 35, '7z': 36
}

class MahjongDataset(Dataset):
    def __init__(self, file_path, file_name_list):
        self.file_path = file_path
        self.file_name_list = file_name_list
        self.num_files = len(file_name_list)

        self.inputs = []
        self.targets = []
        self.mask = []

        self.max_len = 256

        for file_name in self.file_name_list:

            data = self.open_gz_file(self.file_path, file_name + '.gz')

            if len(data['action']) > 0 and len(data['action'][0]) > 0 and len(data['action'][0][0]) > 0:
                if 'scores' in data['action'][0][0][0] and data['action'][0][0][0]['scores'] is not None:
                    # 确保数据结构有效后再执行逻辑
                    # print(file_name)

                    num_kyoku = len(data['action'])
                    # print('num_kyoku:', num_kyoku)

                    # 把不同对局的数据放在一起
                    for i in range(num_kyoku):
                        self.inputs.append(data['action'][i])

                    tar = np.load(self.file_path + '/' + file_name + '_tehai.npz')
                    # print(tar['tehai'].shape)

                    # 不同对局的手牌信息np.array的形状不同
                    # (num_kyoku, 4, seq_len, 4, 37)
                    self.targets.append(tar['tehai'])
                else:
                    # 处理数据不完整的情况
                    print("Invalid data structure or missing 'scores' field.")
                    print(file_name)
            else:
                print("Data structure is empty or incomplete.")
                print(file_name)



 

        # 调整targets的形状
        for i in range(len(self.targets)):
            temp = np.zeros((self.targets[i].shape[0], 4, self.max_len, 4, 37))
            temp[:, :, :self.targets[i].shape[2], :, :] = self.targets[i]
            self.targets[i] = temp

        # 将target在第一个维度上拼接
        self.targets = np.concatenate(self.targets, axis=0)
        print('targets_shape:', self.targets.shape)
        print('inputs_len:', len(self.inputs))

    # 定义打开gz文件的函数
    def open_gz_file(self, path, file_name):
        with gzip.open(path + '/' + file_name, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # print('调用getitem')
        return {
            'input': self.inputs[index],
            'target': self.targets[index],
            # 'mask': self.mask[index]
        }



class MahjongCollator():
    def __init__(self, batch_size=32, max_len=256):
        self.max_len = max_len
        # self.device = device

        pass


    def __call__(self, features):

        input, target = zip(*[(item['input'], item['target']) for item in features])
        # print(len(input))
        # target的每一个元素的形状为(4, 256, 4, 37)，将其转化为(batch_size, 4, 256, 4, 37)的tensor
        target = torch.tensor(np.array(target), dtype=torch.long)
        # print('target:', target.shape)

        # input的序列中可能包括info信息，遍历input，将info信息提取出来并记录在序列中的哪个位置
        info_list = []
        temp = []
        mask = []
        max_kyoku_len = 0
        # count = 0
        # for batch_idx, data in enumerate(input):
        for kyoku_idx, kyoku in enumerate(input):
            kyoku = copy.deepcopy(kyoku)
            target_kyoku = copy.deepcopy(target[kyoku_idx])
            for i in range(4):
                for token_idx, token in enumerate(kyoku[i]):
                    if isinstance(token, dict):
                        # 提取info信息
                        info_list.append(token)
                        # 暂时将info信息替换为224
                        kyoku[i][token_idx] = 224

                temp.append(kyoku[i])
                max_kyoku_len = max(max_kyoku_len, len(kyoku[i]))
                if max_kyoku_len > self.max_len:
                    raise ValueError('max_len is too small', max_kyoku_len)
                    
        # 将temp的长度补齐为self.max_len
        for kyoku in temp:
            mask_temp = [1] * len(kyoku) + [0] * (self.max_len - len(kyoku))
            kyoku.extend([0] * (self.max_len - len(kyoku)))
            # 为填充的位置添加mask
            mask.append(mask_temp)
        
        # 此时temp的形状为(kyoku_num, self.max_len)
        input = torch.tensor(temp, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.bool)

        # 预处理info_list
        scores_list = []
        oya_list = []
        dora_list = []
        honba_riichi_sticks_list = []

        for info in info_list:
            scores_list.append(info['scores'])
            oya_list.append(info['oya_id'])

            # dora的处理
            doras = info['dora_tiles']
            # 将dora的编码转化为0-37
            dora = [tile_map[pai_dict[pai]] for pai in doras]

            dora.extend([37] * (5 - len(dora)))
            dora_list.append(dora)
            # print(dora_list)
            honba_riichi_sticks_list.append([info['honba'], info['riichi_sticks']])

        # 将info_list中的信息转化为张量
        scores_list = torch.tensor(scores_list, dtype=torch.float32)
        oya_list = torch.tensor(oya_list, dtype=torch.long)
        dora_list = torch.tensor(dora_list, dtype=torch.long)
        honba_riichi_sticks_list = torch.tensor(honba_riichi_sticks_list, dtype=torch.float32)

        if scores_list.shape[-1] != 4:
            # 跳过这个batch
            return {
                'info': None,
                'action': None,
                'mask': None,
                'target': None
            }

        # return input, target, mask, info_list
        

            
        return {
            'info': {
                'scores': scores_list,
                'oya': oya_list,
                'dora': dora_list,
                'honba_riichi_sticks': honba_riichi_sticks_list
            },
            'action': input,
            'mask': mask,
            'target': target
        }

class MahjongEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.concat_vector_size = 32 + 16 + 320 + 16
        self.scores_embedding_layer_output_size = 32
        self.oya_embedding_layer_output_size = 16
        self.dora_embedding_layer_output_size = 64
        self.honba_riichi_sticks_embedding_layer_output_size = 16

        self.action_embedding_layer = nn.Embedding(225, self.d_model, dtype=torch.float32)
        self.info_embedding_layer = nn.Linear(self.concat_vector_size, self.d_model, dtype=torch.float32)

        self.scores_embedding_layer = nn.Linear(4, self.scores_embedding_layer_output_size, dtype=torch.float32)    # 4 * 8 = 32
        self.oya_embedding_layer = nn.Embedding(4, self.oya_embedding_layer_output_size, dtype=torch.float32)      # 4 * 4 = 16
        self.dora_embedding_layer = nn.Embedding(38, self.dora_embedding_layer_output_size, dtype=torch.float32)   # 64 * 5 = 320
        self.honba_riichi_sticks_embedding_layer = nn.Linear(2, self.honba_riichi_sticks_embedding_layer_output_size, dtype=torch.float32) # 2 * 8 = 16 


        self.scores_layer_norm = nn.LayerNorm(4, dtype=torch.float32)
        # self.concat_layer_norm = nn.LayerNorm(self.concat_vector_size)

        
        pass

    # scores的形状为(batch_size, 4)
    def scores_embedding(self, scores):
        # 以第二个维度归一化
        # print(scores)
        if scores.shape[-1] != 4:
            raise ValueError('scores的第二个维度不等于4')
        scores = self.scores_layer_norm(scores)
        return self.scores_embedding_layer(scores)

    # oya的形状为(batch_size, 1)
    def oya_embedding(self, oya):
        return self.oya_embedding_layer(oya)

    # dora的形状为(batch_size, 5)
    def dora_embedding(self, dora):
        output = self.dora_embedding_layer(dora)
        # output的形状为(batch_size, 5, self.dora_embedding_layer_output_size)
        # 将output的形状变为(batch_size, self.dora_embedding_layer_output_size * 5)
        return output.view(output.shape[0], -1)

    # honba_riichi_sticks的形状为(batch_size, 2)
    def honba_riichi_sticks_embedding(self, honba_riichi_sticks):
        return self.honba_riichi_sticks_embedding_layer(honba_riichi_sticks)

    def action_embedding(self, action):
        return self.action_embedding_layer(action)
        # 有224种行动
    
    def concat_info(self, info):
        concat_vector = torch.cat([self.scores_embedding(info['scores']), self.oya_embedding(info['oya']), self.dora_embedding(info['dora']), self.honba_riichi_sticks_embedding(info['honba_riichi_sticks'])], dim=1)
        # 归一化
        # concat_vector = self.concat_layer_norm(concat_vector)
        # 将concat_vector嵌入为(batch_size, self.d_model)
        return self.info_embedding_layer(concat_vector)
    
    
    def forward(self, info, action, mask):
        action, info, mask = action, info, mask
        # # 将数据转移到GPU上
        # action = action.cuda()
        # info = {key: value.cuda() for key, value in info.items()}
        # mask = mask.cuda()

        # action的形状为(kyoku_num, max_len=256)
        embeded_action = self.action_embedding(action)
        # scores的形状为(batch_size, 4)
        embeded_info = self.concat_info(info)

        # print(embeded_action.shape)
        # print(embeded_info.shape)

        # 查找action中初始为224的位置，然后将info替换到embeded_action中对应的位置
        indices = torch.where(action == 224)
        
        # print(len(indices[0]))
        if len(indices[0]) == embeded_info.shape[0]:
            for i in range(len(indices[0])):
                embeded_action[indices[0][i], indices[1][i], :] = embeded_info[i, :]
        else:
            raise ValueError('info 的数量和action中的224的数量不一致')
        
        # print('embeded_action:', embeded_action.shape)

        return embeded_action
    

class GPTmodel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # self.embeded = MahjongEmbedding(dmodel=512)
        self.config = config
        
    def forward(self, inputs_embeds=None, attention_mask=None):
        logits = super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return logits
    

class myModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = GPTmodel(config)
        self.embedding_layer = MahjongEmbedding()

        # 定义37个MLP层每一个用于预测一种牌
        # 每种牌的数量为0～4，使用交叉熵损失函数
        self.mlp_layers_player0 = nn.ModuleList([nn.Linear(512, 5) for _ in range(37)])
        self.mlp_layers_player1 = nn.ModuleList([nn.Linear(512, 5) for _ in range(37)])
        self.mlp_layers_player2 = nn.ModuleList([nn.Linear(512, 5) for _ in range(37)])
        self.mlp_layers_player3 = nn.ModuleList([nn.Linear(512, 5) for _ in range(37)])
        self.mlp_layers = [self.mlp_layers_player0, self.mlp_layers_player1, self.mlp_layers_player2, self.mlp_layers_player3]
        self.softmax = nn.Softmax(dim=1)    # 沿着第二个维度进行softmax

    def forward(self, info, action, mask, target):
        if info is None:
            return None
        # 将batch中的数据转化为embeded
        embeded = self.embedding_layer(info, action, mask)
        # 将embeded传入GPT2模型
        gpt_outputs = self.gpt(inputs_embeds=embeded, attention_mask=mask)
        logits = gpt_outputs.last_hidden_state

        # 将logits传入MLP层并调整形状
        pred_logits = []
        for player_layer in self.mlp_layers:
            player_pred_logits = []
            for i, mlp_layer in enumerate(player_layer):
                player_pred_logits.append(mlp_layer(logits))

            player_pred_logits = torch.stack(player_pred_logits, dim=2)
            pred_logits.append(player_pred_logits)
            # print('player_pred_logits:', player_pred_logits.shape)

        pred_logits = torch.stack(pred_logits, dim=2)
        # print('pred_logits',pred_logits.shape)

        mask = mask.unsqueeze(-1).unsqueeze(-1)

        # 计算损失
        pred_logits = pred_logits.view(-1, 5)
        target = target.view(-1)
        mask = mask.expand(-1, -1, 4, 37)

        loss = F.cross_entropy(pred_logits, target, reduction='none')

        loss = loss * mask

        return {
            'loss': loss.sum() / mask.sum().float(),
            'sum_loss': loss.sum(),
            'sum_mask': mask.sum(),
            'logits': pred_logits.view(-1, 256, 4, 37, 5)

        }
