import gymnasium as gym
import numpy as np
import warnings
from gym.spaces import Discrete, Box, Dict, Sequence
from . import MahjongPyWrapper as pm
from env_pymahjong import MahjongEnv

class myMahjongEnv(MahjongEnv):
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
    
    SELF_ACTION_DIM = 49
    RECORDER_ACTION_DIM = 190
    ACTIONS_MAX_LEN = 128

    TILES_FEATURES_CHANNEL = 34
    TILES_FEATURES_HEIGHT = 4
    TILES_FEATURES_WIDTH = 34



    def __init__(self):
        super().__init__()
        self.action_space = Discrete(self.SELF_ACTION_DIM)
        self.tiles_features_space = Box(low=0, high=1, shape=(self.TILES_FEATURES_CHANNEL, 
                                                              self.TILES_FEATURES_HEIGHT, 
                                                              self.TILES_FEATURES_WIDTH), dtype=np.bool)
        self.scores_space = Box(low=-500000, high=500000, shape=(4,), dtype=np.int32)
        self.oya_space = Discrete(4)
        self.honba_riichi_sticks_space = Box(low=0, high=100, shape=(2,), dtype=np.int32)
        self.info_space = Dict(
            {
                'scores': self.scores_space,
                'oya': self.oya_space,
                'honba_riichi_sticks': self.honba_riichi_sticks_space,
            }
        )
        self.recorder_action_space = Discrete(self.RECORDER_ACTION_DIM)
        self.action_list_space = Sequence(self.action_space)
        self.self_action_mask_space = Sequence(Discrete(2))
        self.observation_space = Dict(
            {
                'tiles_features': self.tiles_features_space,
                'info': self.info_space,
                'action_list': self.action_list_space,
                'self_action_mask': self.self_action_mask_space
            }
        )
        self.reset()

    def get_observation(self):
        
        tiles_features = self.tiles_features_space.sample()
        info = {
            'scores': self.scores_space.sample(),
            'oya': self.oya_space.sample(),
            'honba_riichi_sticks': self.honba_riichi_sticks_space.sample(),
        }
        action_list = self.action_list_space.sample()
        self_action_mask = self.self_action_mask_space.sample()
        return {
            'tiles_features': tiles_features,
            'info': info,
            'action_list': action_list,
            'self_action_mask': self_action_mask
        }


