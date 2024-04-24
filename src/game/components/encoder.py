# 将天凤的牌谱文件转换为神经网络的输入
from components.dict import *
import numpy as np
from typing import List, Dict



# 定义编码器类
# 接受一个Taikyoku_loader进行初始化
class Encoder():
    def __init__(self, exported_info: Dict[str, object]) -> None:
        # 参数:
        # exported_info: 一个字典，包含 'taikyoku_info' 和 'player0~3' 5个键。
        # 由Taikyoku_loader的export方法导出的信息

        self.exported_info = exported_info
        pass

    def _encode_tiles(self, tiles: List[int]) -> np.ndarray:

        # 前34行代表牌，最后一行代表是否是赤宝牌
        encoded_tile = np.zeros((35, len(tiles)))

        for index, tile in enumerate(tiles):

            if tile == None:
                continue

            tile_type = tile // 4
            encoded_tile[tile_type, index] = 1
            if tile == 16 or tile == 52 or tile == 88:
                encoded_tile[-1, index] = 1

        return encoded_tile
    
    def _encode_actions(self, actions: List[Dict[str, object]]) -> np.ndarray:

        # 参数:
        # actions: 一个列表，其中每个元素是一个字典，包含 'player' 和 'action' 两个键。
        #          'player' 键对应整数类型，'action' 键对应 Action 枚举类型


        # 每一行代表一种动作，最后一行表示执行动作的玩家
        encoded_action = np.zeros((len(Action) + 1, len(actions)))

        for index, action in enumerate(actions):
            encoded_action[action['action'].value, index] = 1
            encoded_action[-1, index] = action['player']

        return encoded_action
    
    def _encode_public_info(self) -> np.ndarray:
        # 第一行表示场风：0-东，1-南，2-西，3-北
        # 第二行表示局数
        # 第三行表示本场
        # 第四行庄家
        # 第五行表示立直棒数
        # 第六行表示剩余摸牌数
        # 第7～10行表示立直玩家：0-不是立直玩家，1-是立直玩家
        # 第11～14行表示各个玩家的点数

        public_info = np.zeros((14, 1))
        public_info[0, 0] = self.exported_info['taikyoku_info']['bakaze'].value
        public_info[1, 0] = self.exported_info['taikyoku_info']['kyoku']
        public_info[2, 0] = self.exported_info['taikyoku_info']['honba']
        public_info[3, 0] = self.exported_info['taikyoku_info']['oya']
        public_info[4, 0] = self.exported_info['taikyoku_info']['reach_stick']
        public_info[5, 0] = self.exported_info['taikyoku_info']['remain_draw']
        public_info[6:10, 0] = np.array([
            self.exported_info['player0']['isReach'], 
            self.exported_info['player1']['isReach'], 
            self.exported_info['player2']['isReach'], 
            self.exported_info['player3']['isReach']
        ])
        public_info[10:14, 0] = np.array([
            self.exported_info['player0']['point'], 
            self.exported_info['player1']['point'], 
            self.exported_info['player2']['point'], 
            self.exported_info['player3']['point']
        ])

        return public_info

    def encode(self, mask_type=0) -> np.ndarray:
        # 返回值:
        # 一个numpy数组，包含了所有的信息

        # 公共信息public_info
        public_info = self._encode_public_info()

        # dora表示牌
        dora_indicators = self._encode_tiles(self.exported_info['taikyoku_info']['dora'])

        # 四个玩家的手牌和副露信息
        tehai = []
        naki = []
        masked_position = []
        for i in range(4):
            # 添加手牌信息
            tehai.append(self._encode_tiles(self.exported_info[f'player{i}']['tehai']))

            # 添加副露信息
            temp_naki = []
            for naki_info in self.exported_info[f'player{i}']['naki']:
                temp_naki.extend(naki_info['result'])
            naki.append(self._encode_tiles(temp_naki))

            # mask手牌
            if mask_type == 0:
            # 每个玩家随机mask一半的牌           
                masked_position.append(self._mask_tehai(self.exported_info[f'player{i}']['tehai'], type=0))
            elif mask_type == 1:
            # mask除了自己的所有牌
                if i == 0:
                    masked_position.append(self._mask_tehai(self.exported_info[f'player{i}']['tehai'], type=2))
                else:
                    masked_position.append(self._mask_tehai(self.exported_info[f'player{i}']['tehai'], type=1))
        
        # 对局信息
        actions = []
        tiles = []
        for record in self.exported_info['taikyoku_info']['record']:
            actions.append({'action': record['action'], 'player': record['player']})
            tiles.append(record['tile'])
            # print({'action': record['action'], 'player': record['player']}, record['tile'])
        encoded_record = np.concatenate((self._encode_tiles(tiles), self._encode_actions(actions)), axis=0)
        # print(encoded_record[:, 0:10])

        # 补齐
        max_row = encoded_record.shape[0]
        public_info = self._pad_matrix(public_info, max_row)
        dora_indicators = self._pad_matrix(dora_indicators, max_row)
        for i in range(4):
            tehai[i] = self._pad_matrix(tehai[i], max_row)
            naki[i] = self._pad_matrix(naki[i], max_row)

        
        # 定义CLS
        cls = np.ones((max_row, 1))
        
        # 拼接
        encoded_info = np.concatenate(( cls, public_info, dora_indicators, 
                                        tehai[0], naki[0],
                                        tehai[1], naki[1],
                                        tehai[2], naki[2],
                                        tehai[3], naki[3],
                                        encoded_record), axis=1)
        masked_info = np.concatenate((  np.zeros((1, cls.shape[1]), dtype=np.int8), 
                                        np.zeros((1, public_info.shape[1]), dtype=np.int8),
                                        np.zeros((1, dora_indicators.shape[1]), dtype=np.int8),
                                        masked_position[0], np.zeros((1, naki[0].shape[1]), dtype=np.int8),
                                        masked_position[1], np.zeros((1, naki[1].shape[1]), dtype=np.int8),
                                        masked_position[2], np.zeros((1, naki[2].shape[1]), dtype=np.int8),
                                        masked_position[3], np.zeros((1, naki[3].shape[1]), dtype=np.int8),
                                        np.zeros((1, encoded_record.shape[1]), dtype=np.int8)), axis=1)

        return encoded_info, masked_info

    # 随机对手牌进行mask
    def _mask_tehai(self, tehai: List[int], type=0) -> np.ndarray:
        masked_position = np.zeros((1,len(tehai)), dtype=np.int8)

        if type == 0:
        # 随机mask一半的牌
            mask_num = len(tehai) // 2
            mask_index = np.random.choice(len(tehai), mask_num, replace=False)
            masked_position[0, mask_index] = 1

        elif type == 1:
            # mask所有的牌
            masked_position[0, :] = 1
        
        elif type == 2:
            # 不mask任何牌
            pass

        return masked_position

    def _pad_matrix(self, matrix: np.ndarray, max_row: int) -> np.ndarray:
        # 参数:
        # matrix: 一个numpy数组
        # max_row: 最大行数

        # 返回值:
        # 一个numpy数组，将matrix填充到max_row行

        if matrix.shape[0] >= max_row:
            return matrix
        else:
            return np.concatenate((matrix, np.zeros((max_row - matrix.shape[0], matrix.shape[1]))), axis=0)