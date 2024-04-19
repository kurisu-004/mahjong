# 将天凤的牌谱文件转换为神经网络的输入
from components.dict import *
import numpy as np
from components.Taikyoku_loader import Taikyoku_loader
from typing import List, Dict



# 定义编码器类
# 接受一个Taikyoku_loader进行初始化
class Encoder():

    def _encode_tiles(tiles: List[int]) -> np.ndarray:

        # 前34行代表牌，最后一行代表是否是赤宝牌
        encoded_tile = np.zeros((35, len(tiles)))

        for index, tile in enumerate(tiles):

            tile_type = tile // 4
            encoded_tile[tile_type, index] = 1
            if tile == 16 or tile == 52 or tile == 88:
                encoded_tile[-1, index] = 1

        return encoded_tile
    
    def _encode_actions(actions: List[Dict[str, object]]) -> np.ndarray:

        # 参数:
        # actions: 一个列表，其中每个元素是一个字典，包含 'player' 和 'Action' 两个键。
        #          'player' 键对应整数类型，'Action' 键对应 Action 枚举类型


        # 每一行代表一种动作，最后一行表示执行动作的玩家
        encoded_action = np.zeros((len(Action) + 1, len(actions)))

        for index, action in enumerate(actions):
            encoded_action[action['Action'].value, index] = 1
            encoded_action[-1, index] = action['player']

        return encoded_action
    
    def _encode_public_info() -> np.ndarray:
        pass

    def encode(exportedInfo: Dict[str, object]) -> np.ndarray:
        pass