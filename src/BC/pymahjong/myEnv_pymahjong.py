import gymnasium as gym
import numpy as np
import warnings
from gymnasium.spaces import Discrete, Box, Dict, Sequence, MultiBinary
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
    action_id2name = {
        **{
            suit_idx*9+idx: f'Discard {i}{suit}'
            for suit_idx, suit in enumerate(['m', 'p', 's'])
            for idx, i in enumerate(range(1, 10))
        },
        **{
            idx+27: f'Discard {i}z'
            for idx, i in enumerate(range(1, 8))
        },
        34: 'Chi_left',
        35: 'Chi_middle',
        36: 'Chi_right',
        37: 'Pon',
        38: 'AnKan',
        39: 'Kan',
        40: 'KaKan',
        41: 'Riichi',
        42: 'Ron',
        43: 'Tsumo',
        44: 'Kyushukyuhai',
        45: 'Pass Naki',
        46: 'Pass Riichi',
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
        # self.scores_space = Box(low=-500000, high=500000, shape=(4,), dtype=np.int32)
        self.oya_space = Discrete(4)
        self.riichi_sticks_space = Discrete(100)
        # self.honba_riichi_sticks_space = Box(low=0, high=100, shape=(2,), dtype=np.int32)
        
        self.info_space = Dict(
            {
                # 'scores': self.scores_space,
                'oya': self.oya_space,
                'riichi_sticks': self.riichi_sticks_space,
            }
        )
        self.action_list_space = Box(low=0, high=self.RECORDER_ACTION_DIM, shape=(self.ACTIONS_MAX_LEN,), dtype=np.int32)
        self.attention_mask_space = MultiBinary(self.ACTIONS_MAX_LEN)
        self.legal_actions_mask_space = MultiBinary(self.SELF_ACTION_DIM)
        self.observation_space = Dict(
            {
                'tiles_features': self.tiles_features_space,
                'info': self.info_space,
                'action_list': self.action_list_space,
                'attention_mask': self.attention_mask_space,
                'legal_action_mask': self.legal_actions_mask_space
            }
        )
        self.action_record = [] # 以绝对位置记录动作
        self.tiles_features_record = [[], [], [], []]
        self.oya_record = [[], [], [], []]
        self.riichi_sticks_record = [[], [], [], []]
        self.legal_actions_mask_record = [[], [], [], []]
        self.last_discard_tile = None   
        self.riichi_stage2 = False # 是否进入了立直第二阶段即
        self.pass_riichi = False # 是否pass了立直
        # self.reset()


    def _proceed(self):
        if self.is_over():
            # print("The game is over.")
            pass
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
        if self_action_len == 0:
            return Q
        elif self_action_len == 1:
            Q[0] = payoff
            return Q
        elif self_action_len > 1:
            Q[-1] = payoff
            for i in reversed(range(self_action_len-1)):
                Q[i] = Q[i+1] * gamma
            return Q

    def _get_legal_actions_mask(self):
        legal_actions_idx = np.array(self.get_valid_actions())
        legal_actions = np.zeros(self.SELF_ACTION_DIM, dtype=bool)
        # legal_actions_idx是一个np.array，里面存放的是合法动作的索引
        legal_actions[legal_actions_idx] = 1
        return legal_actions


    def reset(self, oya=None, game_wind=None, seed=None, table=None):
        if oya is None:
            oya = np.random.randint(4)
        else:
            assert oya in [0, 1, 2, 3]

        if game_wind is None:
            game_wind = "east"
        else:
            assert game_wind in ["east", "south", "west", "north"]

        if table is None:
            self.t = pm.Table()
            self.t.use_seed = True
            self.t.seed = np.random.randint(1000000000)

            self.t.game_init_with_metadata({"oya": str(oya), "wind": game_wind})
        else:
            self.t = table

        
        self.riichi_stage2 = False
        self.may_riichi_tile_id = None

        self.game_count += 1
        self.action_record = []
        self.tiles_features_record = [[], [], [], []]
        self.oya_record = [[], [], [], []]
        self.riichi_sticks_record = [[], [], [], []]
        self.legal_actions_mask_record = [[], [], [], []]


        self._proceed()

        obs = self.get_observation(self.get_curr_player_id())
        termination = self.is_over()
        if termination:
            r = np.array(self.get_payoffs())
        else:
            r = np.array([0, 0, 0, 0])

        return obs, {'curr_player': self.get_curr_player_id(), 'payoffs': r}

    def step(self, action_idx: int):
        player_id = self.get_curr_player_id()
        tiles_features, oya = self._get_tiles_features(player_id)
        legal_actions = self._get_legal_actions_mask()
        self.legal_actions_mask_record[player_id].append(legal_actions)
        self.tiles_features_record[player_id].append(tiles_features)
        self.oya_record[player_id].append(oya)
        self.riichi_sticks_record[player_id].append(self._get_riichi_sticks())
        self.action_record.append({
            'player_id': player_id,
            'action': action_idx
        })
        # super().step(player_id, action)

        # 判断是否是正确的玩家
        if not player_id == self.get_curr_player_id():
            raise ValueError("current acting player ID is {}, but you are trying to ask player {} to act !!".format(
                self.get_curr_player_id(), player_id))
        
# --------------------------获取当前阶段的可选动作--------------------------------
        phase = self.t.get_phase()
        if phase < 4:
            aval_actions = self.t.get_self_actions()
        elif phase < 16:
            aval_actions = self.t.get_response_actions()
        else:
            raise ValueError(f"Invalid phase {phase}")
# --------------------------处理立直相关的动作--------------------------------
        if self.action_id2name[action_idx] == 'Riichi':
            self.riichi_stage2 = True

        elif self.action_id2name[action_idx] == 'Pass Riichi':
            self.pass_riichi = True
# --------------------------处理其他动作--------------------------------
        # 舍牌动作
        elif action_idx < 34:
            self.last_discard_tile = action_idx
            # 从aval_actions中找到对应的动作
            for action in aval_actions:
                base_action = action.action
                if (base_action.name == 'Riichi' and self.riichi_stage2 == True) or (base_action.name == 'Discard' and self.riichi_stage2 == False):
                    corr_tiles = action.correspond_tiles
                    # print(corr_tiles[0].id//4, action_idx)
                    assert len(corr_tiles) == 1
                    if corr_tiles[0].id//4 == action_idx:
                        self.t.make_selection_from_action_basetile(base_action, [corr_tiles[0].tile], False)
                        # 将标志位还原
                        self.riichi_stage2 = False
                        self.pass_riichi = False
                        # print("player{} discard {}".format(player_id, corr_tiles[0].tile))
                        break

        # 吃碰杠动作   
        elif self.action_id2name[action_idx] == 'Chi_left':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Chi':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 2
                    if corr_tiles[0].id//4 > self.last_discard_tile and corr_tiles[1].id//4 > self.last_discard_tile:
                        self.t.make_selection_from_action_tile(base_action, corr_tiles)
                        # print("chi_left")
                        break

        elif self.action_id2name[action_idx] == 'Chi_middle':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Chi':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 2
                    if corr_tiles[0].id//4 < self.last_discard_tile and corr_tiles[1].id//4 > self.last_discard_tile:
                        self.t.make_selection_from_action_tile(base_action, corr_tiles)
                        # print("chi_middle")
                        break

        elif self.action_id2name[action_idx] == 'Chi_right':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Chi':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 2
                    if corr_tiles[0].id//4 < self.last_discard_tile and corr_tiles[1].id//4 < self.last_discard_tile:
                        self.t.make_selection_from_action_tile(base_action, corr_tiles)
                        # print("chi_right")
                        break

        elif self.action_id2name[action_idx] == 'Pon':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Pon':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 2
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'Kan':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Kan':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 3
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'AnKan':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'AnKan':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 4
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'KaKan':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'KaKan':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 1
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'Ron':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Ron':
                    corr_tiles = action.correspond_tiles
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'Tsumo':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Tsumo':
                    corr_tiles = action.correspond_tiles
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break
        
        elif self.action_id2name[action_idx] == 'Kyushukyuhai':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Kyushukyuhai':
                    corr_tiles = action.correspond_tiles
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break

        elif self.action_id2name[action_idx] == 'Pass Naki':
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Pass':
                    corr_tiles = action.correspond_tiles
                    self.t.make_selection_from_action_tile(base_action, corr_tiles)
                    break
        
        self._proceed()

        termination = self.is_over()

        if termination:
            curr_player = -1
            r = np.array(self.get_payoffs())
            obs = self.observation_space.sample()
            obs_with_return = []
            for i in range(4):
                obs_with_return.append(self.get_observation_with_return(i))
        else:
            curr_player = self.get_curr_player_id()
            r = np.array([0, 0, 0, 0])
            obs = self.get_observation(self.get_curr_player_id())
            obs_with_return = None


        return obs, 0, termination, False, {'curr_player': curr_player, 'payoffs': r,
                                            'obs_with_return': obs_with_return}


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
    

    # 重写get_valid_actions
    # 返回一个list，里面存放的是合法动作的索引
    # 似乎可以自定义动作空间？
    def get_valid_actions(self):
        phase = self.t.get_phase()
        valid_actions = []
        if phase < 4:
            aval_actions = self.t.get_self_actions()
        elif phase < 16:
            aval_actions = self.t.get_response_actions()
        else:
            raise ValueError(f"Invalid phase {phase}")
        
        aval_actions_str = [action.action.name for action in aval_actions]
        selfAction_set = set(['Discard', 'AnKan', 'KaKan', 'Riichi', 'Tsumo', 'Kyushukyuhai'])
        responseAction_set = set(['Chi', 'Pon', 'Kan', 'Ron', 'Pass', 'ChanAnKan', 'ChanKan'])
# --------------------------处理立直相关的动作--------------------------------
        # 如果没有pass立直
        if self.pass_riichi == False and self.riichi_stage2 == False and 'Riichi' in aval_actions_str:
                return [self.action_encoding['player0']['riichi'], 
                        self.action_encoding['player0']['pass_riichi']]

        # 如果进入了立直第二阶段
        elif self.riichi_stage2 == True:
            assert 'Riichi' in aval_actions_str
            for action in aval_actions:
                base_action = action.action
                # 找到所有的立直动作
                if base_action.name == 'Riichi':
                    # 找到对应的舍牌
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 1
                    valid_actions.append(corr_tiles[0].id//4)
                    # 消除重复的动作
            return list(set(valid_actions))
        elif self.pass_riichi == True:
            assert 'Riichi' in aval_actions_str
            for action in aval_actions:
                base_action = action.action
                if base_action.name == 'Discard':
                    corr_tiles = action.correspond_tiles
                    assert len(corr_tiles) == 1
                    valid_actions.append(corr_tiles[0].id//4)
            return list(set(valid_actions))

# --------------------------处理其他动作--------------------------------
        if phase < 4:
            for action in aval_actions:
                base_action = action.action
                corr_tiles = action.correspond_tiles
                
                assert base_action.name in selfAction_set, f"Invalid action {base_action.name}, should not be in selfAction_set"
                
                if base_action.name == 'Discard':
                    assert len(corr_tiles) == 1, f"Invalid corr_tiles {corr_tiles}, should be 1 at {base_action.name}"
                    valid_actions.append(corr_tiles[0].id//4)
                if base_action.name == 'Ankan':
                    valid_actions.append(self.action_encoding['player0']['ankan'])
                if base_action.name == 'Kakan':
                    valid_actions.append(self.action_encoding['player0']['kakan'])
                if base_action.name == 'Tsumo':
                    valid_actions.append(self.action_encoding['player0']['tsumo'])
                if base_action.name == 'Kyushukyuhai':
                    valid_actions.append(self.action_encoding['player0']['kyushukyuhai'])

        elif phase < 16:
            for action in aval_actions:
                base_action = action.action
                corr_tiles = action.correspond_tiles
                assert base_action.name in responseAction_set, f"Invalid action {base_action.name}, should not be in responseAction_set"

                if base_action.name == 'Chi':
                    assert len(corr_tiles) == 2
                    # 比较corr_tiles中的两个tile和last_discard_tile的大小
                    if corr_tiles[0].id//4 < self.last_discard_tile and corr_tiles[1].id//4 < self.last_discard_tile:
                        valid_actions.append(self.action_encoding['player0']['chi_right']) # chi_right
                    elif corr_tiles[0].id//4 > self.last_discard_tile and corr_tiles[1].id//4 > self.last_discard_tile:
                        valid_actions.append(self.action_encoding['player0']['chi_left']) # chi_left
                    else:
                        valid_actions.append(self.action_encoding['player0']['chi_middle']) # chi_middle

                elif base_action.name == 'Pon':
                    valid_actions.append(self.action_encoding['player0']['pon'])

                elif base_action.name == 'Kan':
                    valid_actions.append(self.action_encoding['player0']['kan'])

                elif base_action.name == 'Ron' or base_action.name == 'ChanKan' or base_action.name == 'ChanAnKan':
                    valid_actions.append(self.action_encoding['player0']['ron'])    

                elif base_action.name == 'Pass':
                    valid_actions.append(self.action_encoding['player0']['pass_naki'])

        return list(set(valid_actions))