import gymnasium as gym
import numpy as np
import warnings
from gymnasium.spaces import Discrete, Box, Dict, Sequence
from . import MahjongPyWrapper as pm
from .env_pymahjong import MahjongEnv
import re
from collections import Counter

class myMahjongEnv(MahjongEnv):
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
    
    # 动作映射
    action_mapping = {
        0: 0
    }

    suit_mapping = {
        'm': 0,
        'p': 1,
        's': 2,
        'z': 3
    }

    SELF_ACTION_DIM = 47
    RECORDER_ACTION_DIM = 188 + 1
    ACTIONS_MAX_LEN = 128

    TILES_FEATURES_CHANNEL = 35
    TILES_FEATURES_HEIGHT = 4
    TILES_FEATURES_WIDTH = 9



    def __init__(self):
        super().__init__()
        self.action_space = Discrete(self.SELF_ACTION_DIM)
        self.tiles_features_space = Box(low=0, high=1, shape=(self.TILES_FEATURES_CHANNEL, 
                                                              self.TILES_FEATURES_HEIGHT, 
                                                              self.TILES_FEATURES_WIDTH), dtype=bool)
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
        self.action_record = [] # 以绝对位置记录动作
        self.tiles_features_record = [[], [], [], []]
        self.oya_record = [[], [], [], []]
        self.riichi_sticks_record = [[], [], [], []]
        self.legal_actions_mask_record = [[], [], [], []]
        self.reset()

    def _proceed(self):
        while not self.is_over():  # continue until game over or one player has choices
            phase = self.t.get_phase()
            if phase < 4:
                aval_actions = self.t.get_self_actions()
            elif phase < 16:
                aval_actions = self.t.get_response_actions()
            else:
                raise ValueError(f"Invalid phase {phase}")
            if len(aval_actions) > 1:
                return phase, aval_actions
            else:
                self.t.make_selection(0)

    # 这里的player_id是需要获取观察的玩家的id
    # 返回一个list，list中的元素是一个字符串，表示玩家的手牌
    def _get_hand_tiles(self, player_id):
        hand = []
        hand_list = self.t.players[player_id].hand
        for tile in hand_list:
            hand.append(tile.to_string())
        return hand

    def _get_fuuros(self, player_id):
        fuuros = []
        fuuro_list = self.t.players[player_id].get_fuuros()
        for fuuro in fuuro_list:
            tiles = fuuro.tiles
            for tile in tiles:
                fuuros.append(tile.to_string())
        return fuuros
    
    def _get_dora_indicators(self):
        dora_indicators = []
        dora_list = self.t.dora_indicator
        n_active_dora = self.t.n_active_dora
        for dora in dora_list[:n_active_dora]:
            dora_indicators.append(dora.to_string())
        return dora_indicators
    
    def _get_points(self, player_id):
        return self.t.players[player_id].score
    
    def _get_oya(self, player_id):
        return (self.t.oya-player_id)%4
    
    def _get_riichi_sticks(self):
        return self.t.riichibo

    def _get_action_record(self):
        return self.action_record
    
    # 这里指的是当前player看不见的牌
    # 包括yama, 另外三家的手牌，再扣去dora指示牌
    def _get_remaining_tiles(self, player_id):
        remaining_tiles = [tile.to_string() for tile in self.t.yama]
        # print("yama", len(remaining_tiles))
        dora_indicators = self._get_dora_indicators()
        # print("dora", len(dora_indicators))
        for dora in dora_indicators:
            if dora in remaining_tiles:
                remaining_tiles.remove(dora)
            else:
                raise ValueError(f"Invalid dora indicator {dora}")
        for i in range(4):
            if i != player_id:
                hand = self._get_hand_tiles(i)
                # print(f"hand{i}", len(hand))
                remaining_tiles.extend(hand)
        return remaining_tiles

    
    def _change_action_record_aspects(self, player_id):
        action_list = [188]
        self_action_mask = [0]
        for action in self.action_record:
            player = action['player_id']
            action_type = action['action']
            action_list.append(((player-player_id)%4) * 47 + action_type)
            if player == player_id:
                self_action_mask.append(1)
            else:
                self_action_mask.append(0)
        return action_list, self_action_mask
    def _padding_action_record(self, action_list, self_action_mask):
        padding_mask = [1]*len(action_list)+[0]*(self.ACTIONS_MAX_LEN-len(action_list))
        padding_action = action_list + [0]*(self.ACTIONS_MAX_LEN-len(action_list))
        padding_self_action_mask = self_action_mask + [0]*(self.ACTIONS_MAX_LEN-len(action_list))
        padding_action = np.array(padding_action)
        padding_self_action_mask = np.array(padding_self_action_mask)
        padding_mask = np.array(padding_mask)
        return padding_action, padding_self_action_mask, padding_mask


    def _transfer_to_2D(self, tiles_list):
        # 统计list中每种元素出现的次数
        counter = Counter(tiles_list)
        matrix = np.zeros((5, self.TILES_FEATURES_HEIGHT, self.TILES_FEATURES_WIDTH), dtype=bool)
        # 检查是否有0m, 0p, 0s
        if '0m' in counter.keys():
            matrix[4, 0, 4] = 1
            if '5m' in counter.keys():
                counter['5m'] += 1
            else:
                counter['5m'] = 1
        if '0p' in counter.keys():
            matrix[4, 1, 4] = 1
            if '5p' in counter.keys():
                counter['5p'] += 1
            else:
                counter['5p'] = 1
        if '0s' in counter.keys():
            matrix[4, 2, 4] = 1
            if '5s' in counter.keys():
                counter['5s'] += 1
            else:
                counter['5s'] = 1

        for tile in counter.keys():
            suit = tile[-1]
            number = int(tile[0])
            if number == 0:
                continue
            elif suit in ['m', 'p', 's', 'z'] and number in range(1, 10):
                matrix[0:counter[tile], self.suit_mapping[suit], number-1] = 1
            else:
                raise ValueError(f"Invalid tile {tile}")
        return matrix


    # 这里的返回值应该以gymnasium的格式返回
    def _get_tiles_features(self, player_id):

        for i in range(4):
            if i == player_id:
                player0_hand = self._transfer_to_2D(self._get_hand_tiles(i))
                player0_fuuros = self._transfer_to_2D(self._get_fuuros(i))
            elif (i-player_id)%4==1:
                player1_fuuros = self._transfer_to_2D(self._get_fuuros(i))
            elif (i-player_id)%4==2:
                player2_fuuros = self._transfer_to_2D(self._get_fuuros(i))
            elif (i-player_id)%4==3:
                player3_fuuros = self._transfer_to_2D(self._get_fuuros(i))
        dora_indicators = self._transfer_to_2D(self._get_dora_indicators())
        remaining_tiles = self._transfer_to_2D(self._get_remaining_tiles(player_id))

        tiles_features = np.concatenate([player0_hand, player0_fuuros, player1_fuuros, player2_fuuros, player3_fuuros, dora_indicators, remaining_tiles], axis=0)
        oya = self._get_oya(player_id)

        return tiles_features, oya
    
    def _get_Q_values(self, player_id, self_action_len, gamma=0.99):
        if not self.is_over():
            raise ValueError("The game is not over yet.")
        payoff = self.get_payoffs()[player_id]
        Q = np.zeros(self_action_len, dtype=np.float32)
        Q[-1] = payoff
        for i in reversed(range(self_action_len-1)):
            Q[i] = Q[i+1] * gamma
        return Q

    def _get_legal_actions_mask(self):
        legal_actions_idx = self.get_valid_actions()
        legal_actions = np.zeros(self.SELF_ACTION_DIM, dtype=bool)
        # legal_actions_idx是一个np.array，里面存放的是合法动作的索引
        legal_actions[legal_actions_idx] = 1
        return legal_actions


    def reset(self, oya=None, game_wind=None, seed=None):
        if oya is None:
            oya = np.random.randint(4)
        else:
            assert oya in [0, 1, 2, 3]

        if game_wind is None:
            game_wind = "east"
        else:
            assert game_wind in ["east", "south", "west", "north"]

        self.t = pm.Table()
        if seed is not None:
            self.t.seed = seed

        self.t.game_init_with_metadata({"oya": str(oya), "wind": game_wind})
        self.riichi_stage2 = False
        self.may_riichi_tile_id = None

        self.game_count += 1
        self.action_record = []
        self.tiles_features_record = [[], [], [], []]
        self.oya_record = [[], [], [], []]
        self.riichi_sticks_record = [[], [], [], []]
        self.legal_actions_mask_record = [[], [], [], []]


        self._proceed()

    def step(self, player_id: int, action: int):
        tiles_features, oya = self._get_tiles_features(player_id)


        legal_actions = self._get_legal_actions_mask()
        super().step(player_id, action)

        self.legal_actions_mask_record[player_id].append(legal_actions)
        self.tiles_features_record[player_id].append(tiles_features)
        self.oya_record[player_id].append(oya)
        self.riichi_sticks_record[player_id].append(self._get_riichi_sticks())
        self.action_record.append({
            'player_id': player_id,
            'action': action
        })

    def get_observation_with_return(self, player_id):
        if not self.is_over():
            raise ValueError("The game is not over yet.")
        tils_features = np.array(self.tiles_features_record[player_id])
        oya = np.array(self.oya_record[player_id])
        riichi_sticks = np.array(self.riichi_sticks_record[player_id])
        action_list, self_action_mask = self._change_action_record_aspects(player_id)
        action_list, self_action_mask, padding_mask = self._padding_action_record(action_list, self_action_mask)
        Q = self._get_Q_values(player_id, self_action_mask.sum())
        return {
            'tiles_features': tils_features,
            'info': {
                'oya': oya,
                'riichi_sticks': riichi_sticks
            },
            'action_list': action_list,
            'self_action_mask': self_action_mask,
            'attention_mask': padding_mask,
            'Q_values': Q,
            'legal_action_mask': np.array(self.legal_actions_mask_record[player_id])
        }

    def get_observation(self, player_id):
        tils_features, oya = self._get_tiles_features(player_id)
        riichi_sticks = self._get_riichi_sticks()
        action_list, self_action_mask = self._change_action_record_aspects(player_id)
        action_list, self_action_mask, padding_mask = self._padding_action_record(action_list, self_action_mask)
        legal_actions_mask = self._get_legal_actions_mask()
        return {
            'tiles_features': np.array(tils_features),
            'info': {
                'oya': oya,
                'riichi_sticks': riichi_sticks
            },
            'action_list': action_list,
            'attention_mask': padding_mask,
            'legal_action_mask': legal_actions_mask
        }