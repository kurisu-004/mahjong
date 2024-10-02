from .myEnv_pymahjong import myMahjongEnv
import gzip
import xml.etree.ElementTree as ET
from pymahjong import MahjongPyWrapper as mp
import numpy as np
from collections import deque
# from pymahjong.utils import *

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

reversed_encoding = {}

for player, actions in action_encoding.items():
    for action, value in actions.items():
        if action == 'discard':
            for card, code in value.items():
                reversed_encoding[code] = (player, action, card)
        else:
            reversed_encoding[value] = (player, action)
# 把副露的5位数解码成具体的东西
def decodem(naru_tiles_int, naru_player_id):
    # 54279 : 4s0s chi 6s
    # 35849 : 6s pon
    # 51275 : chu pon
    # ---------------------------------
    binaries = bin(naru_tiles_int)[2:]

    opened = True

    if len(binaries) < 16:
        binaries = "0" * (16 - len(binaries)) + binaries

    bit2 = int(binaries[-3], 2)
    bit3 = int(binaries[-4], 2)
    bit4 = int(binaries[-5], 2)

    if bit2:
        naru_type = "Chi"

        bit0_1 = int(binaries[-2:], 2)

        if bit0_1 == 3:  # temporally not used
            source = "kamicha"
        elif bit0_1 == 2:
            source = "opposite"
        elif bit0_1 == 1:
            source = "shimocha"
        elif bit0_1 == 0:
            source = "self"

        bit10_15 = int(binaries[:6], 2)
        bit3_4 = int(binaries[-5:-3], 2)
        bit5_6 = int(binaries[-7:-5], 2)
        bit7_8 = int(binaries[-9:-7], 2)

        which_naru = bit10_15 % 3

        source_player_id = (naru_player_id + bit0_1) % 4  # TODO: 包牌

        start_tile_id = int(int(bit10_15 / 3) / 7) * 9 + int(bit10_15 / 3) % 7

        side_tiles_added = [[start_tile_id * 4 + bit3_4, 0], [start_tile_id * 4 + 4 + bit5_6, 0],
                            [start_tile_id * 4 + 8 + bit7_8, 0]]
        # TODO: check aka!
        side_tiles_added[which_naru][1] = 1

        hand_tiles_removed = []
        for kk, ss in enumerate(side_tiles_added):
            if kk != which_naru:
                hand_tiles_removed.append(ss[0])


    else:
        naru_type = "Pon"

        if bit3:

            bit9_15 = int(binaries[:7], 2)

            which_naru = bit9_15 % 3
            pon_tile_id = int(int(bit9_15 / 3))

            side_tiles_added = [[pon_tile_id * 4, 0], [pon_tile_id * 4 + 1, 0], [pon_tile_id * 4 + 2, 0],
                                [pon_tile_id * 4 + 3, 0]]

            bit5_6 = int(binaries[-7:-5], 2)
            which_not_poned = bit5_6

            del side_tiles_added[which_not_poned]

            side_tiles_added[which_naru][1] = 1

            hand_tiles_removed = []
            for kk, ss in enumerate(side_tiles_added):
                if kk != which_naru:
                    hand_tiles_removed.append(ss[0])

        else:  # An-Kan, Min-Kan, Ka-Kan

            bit5_6 = int(binaries[-7:-5], 2)
            which_kan = bit5_6

            if bit4:
                naru_type = "Ka-Kan"
                bit9_15 = int(binaries[:7], 2)

                kan_tile_id = int(bit9_15 / 3)

                side_tiles_added = [[kan_tile_id * 4 + which_kan, 1]]

                hand_tiles_removed = [kan_tile_id * 4 + which_kan]

            else:  # An-Kan or # Min-Kan

                which_naru = naru_tiles_int % 4

                bit8_15 = int(binaries[:8], 2)

                kan_tile = bit8_15
                kan_tile_id = int(kan_tile / 4)
                which_kan = kan_tile % 4

                side_tiles_added = [[kan_tile_id * 4, 0], [kan_tile_id * 4 + 1, 0], [kan_tile_id * 4 + 2, 0],
                                    [kan_tile_id * 4 + 3, 0]]
                if which_naru == 0:
                    naru_type = "An-Kan"
                    hand_tiles_removed = []
                    for kk, ss in enumerate(side_tiles_added):
                        hand_tiles_removed.append(ss[0])
                    opened = False

                else:
                    naru_type = "Min-Kan"
                    side_tiles_added[which_kan][1] = 1

                    hand_tiles_removed = []
                    for kk, ss in enumerate(side_tiles_added):
                        if kk != which_kan:
                            hand_tiles_removed.append(ss[0])

    return side_tiles_added, hand_tiles_removed, naru_type, opened



def open_file(path: str, file_name: str):
    if file_name.endswith('.gz'):
        try:
            with gzip.open(path +'/'+ file_name, 'rt') as f:
                tree = ET.parse(f)
        except Exception as e:
            raise RuntimeError(e.__str__(), f"Cannot read paipu {path +'/'+ file_name}")
    elif file_name.endswith('.log'):
        try:
            with open(path +'/'+ file_name, 'r') as f:
                tree = ET.parse(f)
        except Exception as e:
            raise RuntimeError(e.__str__(), f"Cannot read paipu {path +'/'+ file_name}")
    else:
        raise ValueError('Unknown file type')
    
    return tree

# 返回每一局初始化信息和动作信息
def get_inst_and_action_list(tree):
    root = tree.getroot()
    return_data = {
        'replayer': [],
        'action_list': [],
    }

    for child_no, child in enumerate(root):
        if child.tag == "SHUFFLE":
            seed_str = child.get("seed")
            prefix = 'mt19937ar-sha512-n288-base64,'
            if not seed_str.startswith(prefix):
                self.log('Bad seed string')
                continue
            seed = seed_str[len(prefix):]
            inst = mp.TenhouShuffle.instance()
            inst.init(seed)

        elif child.tag == "GO" or child.tag == "UN" or child.tag == "TAIKYOKU" or child.tag == "DORA":
            pass

        elif child.tag == "INIT":

            action_que = deque() # 用于存储每一局的动作
            # 开局时候的各家分数
            scores_str = child.get("ten").split(',')
            init_scores = [int(tmp) * 100 for tmp in scores_str]

            # Oya ID
            oya_id = int(child.get("oya"))

            # 什么局
            game_order = int(child.get("seed").split(",")[0])

            # 本场和立直棒
            honba = int(child.get("seed").split(",")[1])
            riichi_sticks = int(child.get("seed").split(",")[2])

            # 骰子数字
            dice_numbers = [int(child.get("seed").split(",")[3]) + 1, int(child.get("seed").split(",")[4]) + 1]

            # 牌山
            yama = inst.generate_yama()

            # 利用PaiPuReplayer进行重放
            replayer = mp.PaipuReplayer()
            replayer.init(yama, init_scores, riichi_sticks, honba, game_order // 4, oya_id)
            return_data['replayer'].append(replayer)

        elif child.tag == "REACH":
            player_id = int(child.get("who"))
            if int(child.get("step")) == 1:
                action = player_id * 47 + 41
                action_que.append(action)

        # discard
        elif child.tag[0] in ['D', 'E', 'F', 'G']:
            player_id = "DEFG".find(child.tag[0])
            try:
                discarded_tile = int(child.tag[1:])
            except:
                print(child.tag)

            # 从action_encode中找到对应的动作
            if player_id == -1:
                raise ValueError('Unknown player')
            action = player_id * 47 + discarded_tile // 4
            action_que.append(action)

        elif child.tag == "N":
            player_id = int(child.get("who"))
            naru_tiles_int = int(child.get("m"))
            side_tiles_added_by_naru, hand_tiles_removed_by_naru, naru_type, opened = decodem(
                naru_tiles_int, player_id)

            naru_type_set = {"Chi", "Pon", "Ka-Kan", "An-Kan", "Min-Kan"}
            if naru_type not in naru_type_set:
                raise ValueError('Unknown naru type')
            if naru_type == "Chi":
                for idx, tile in enumerate(side_tiles_added_by_naru):
                    if tile[1] == 1:
                        action = player_id * 47 + 34 + idx
                    
            elif naru_type == "Pon":
                action = player_id * 47 + 37
            elif naru_type == "Ka-Kan":
                action = player_id * 47 + 40
            elif naru_type == "An-Kan":
                action = player_id * 47 + 38
            elif naru_type == "Min-Kan":
                action = player_id * 47 + 39

            action_que.append(action)
        
        elif  child.tag == "BYE":
            # print("掉线，跳过本局")
            return None
        
        elif child.tag == "AGARI":
            if child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI":
                # print("一炮多响，跳过本局")
                return None
            else:
                player_id = int(child.get("who"))
                from_who = int(child.get("fromWho"))
                if player_id == from_who:
                    action = player_id * 47 + 43
                else:
                    action = player_id * 47 + 42
                action_que.append(action)

                return_data['action_list'].append(action_que)
                action_que = deque()


        
        elif child.tag == "RYUUKYOKU":
            if child.get("type") == "yao9":
                # print(child.tag, child.attrib)
                action_que.append(44)
            return_data['action_list'].append(action_que)
            action_que = deque()

    return return_data
    
def replay(path, file_name):
    # for file_name in file_name_list:
    label = [[], [], [], []]
    if True:
        tree = open_file(path=path, file_name=file_name)
        return_data = get_inst_and_action_list(tree)
        if return_data is None:
            return None


        # for kyoku in range(len(return_data['replayer'])):
        if True:
            kyoku = 0
            queue = return_data['action_list'][kyoku].copy()
            env = myMahjongEnv()
            obs, info = env.reset(table=return_data['replayer'][kyoku].table)

            done = False
            while not done:
                # if queue[0] == 5 + 47 * 2:
                #     return env, queue

                is_riichi = [player.riichi for player in env.t.players]
                if len(queue) > 0:

                    while is_riichi[queue[0]//47] and queue[0]%47 < 34:
                        # 已立直玩家的舍牌动作会进入到这个循环
                        # 如果这个舍牌动作的同时还有其他的可执行动作，则不能跳过这个舍牌动作
                        curr_player = env.get_curr_player_id()
                        legal_actions = env.get_valid_actions()
                        if queue[0] % 47 in legal_actions and len(legal_actions) > 1 and curr_player == queue[0]//47:
                            break
                        queue.popleft()
                        if len(queue) == 0:
                            return {'success': False}

                    action = queue[0]
                curr_player = env.get_curr_player_id()
                legal_actions = env.get_valid_actions()
                action_idx = action % 47
                player_idx = action // 47
                if action_idx == 44: # 九种九牌
                    player_idx = curr_player
                if curr_player == player_idx and action_idx in legal_actions:
                    obs, reward, done, _, info = env.step(action_idx=action_idx)
                    label[curr_player].append(action_idx)
                    # print(f'{reversed_encoding[action]}')
                    if len(queue) > 0:
                        queue.popleft()
                    else:
                        print(f'file_name {file_name} kyoku {kyoku} failed')
                        print(f'player {player_idx} action {reversed_encoding[action]} is invalid')
                        print(f'curr_player {curr_player} legal_actions {legal_actions}')
                        return {'success': False}
                elif 45 in legal_actions:
                    obs, reward, done, _, info = env.step(action_idx=45)
                    label[curr_player].append(45)
                    # print(f'player {curr_player} pass_naki')

                elif 46 in legal_actions:
                    obs, reward, done, _, info = env.step(action_idx=46)
                    label[curr_player].append(46)
                    # print(f'player {curr_player} pass_riichi')
                    
                else:
                    print(f'file_name {file_name} kyoku {kyoku} failed')
                    print(f'player {player_idx} action {reversed_encoding[action]} is invalid')
                    print(f'curr_player {curr_player} legal_actions {legal_actions}')
                    return {'success': False}
                    
            # print(f'file_name {file_name} kyoku {kyoku} success')
    return {'data' :info['obs_with_return'], 'success': True, 'label': label}

def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        # if result is not None:
        output.put(result)

def calculate(func, args):
    result = func(*args)
    return result