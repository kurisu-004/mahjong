from enum import Flag
import re
import os
import time
import numpy as np
import shutil
import sys, gzip, pickle
import xml.etree.ElementTree as ET
from pymahjong.tenhou_paipu_check import *
from tqdm import tqdm
from collections import Counter

# 定义牌的映射表
tile_map = {
    '0m': 0, '1m': 1, '2m': 2, '3m': 3, '4m': 4, '5m': 5, '6m': 6, '7m': 7, '8m': 8, '9m': 9,
    '0p': 10, '1p': 11, '2p': 12, '3p': 13, '4p': 14, '5p': 15, '6p': 16, '7p': 17, '8p': 18, '9p': 19,
    '0s': 20, '1s': 21, '2s': 22, '3s': 23, '4s': 24, '5s': 25, '6s': 26, '7s': 27, '8s': 28, '9s': 29,
    '1z': 30, '2z': 31, '3z': 32, '4z': 33, '5z': 34, '6z': 35, '7z': 36
}

action_encode = {
    'draw': {
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
    'discard': {
        'player0': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 37
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 37
                for idx, i in enumerate(range(1, 8))
            },
        },
        'player1': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 74
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 74
                for idx, i in enumerate(range(1, 8))
            },
        },
        'player2': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 111
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 111
                for idx, i in enumerate(range(1, 8))
            },
        },
        'player3': {
            **{
                f'{i}{suit}': idx + suit_idx * 10 + 148
                for suit_idx, suit in enumerate(['m', 'p', 's'])
                for idx, i in enumerate(range(10))
            },
            **{
                f'{i}z': idx + 30 + 148
                for idx, i in enumerate(range(1, 8))
            },
        }
        
    },
    'Chi':{
        'player0': 185,
        'player1': 186,
        'player2': 187,
        'player3': 188
    },
    'Pon':{
        'player0': 189,
        'player1': 190,
        'player2': 191,
        'player3': 192
    },
    'Min-Kan':{
        'player0': 193,
        'player1': 194,
        'player2': 195,
        'player3': 196
    },
    'An-Kan':{
        'player0': 197,
        'player1': 198,
        'player2': 199,
        'player3': 200
    },
    'Ka-Kan':{
        'player0': 201,
        'player1': 202,
        'player2': 203,
        'player3': 204
    },
    'riichi':{
        'player0': 205,
        'player1': 206,
        'player2': 207,
        'player3': 208
    },
    'ron':{
        'player0': 209,
        'player1': 210,
        'player2': 211,
        'player3': 212
    },
    'tsumo':{
        'player0': 213,
        'player1': 214,
        'player2': 215,
        'player3': 216
    },
    'kyushukyuhai':{
        'player0': 217,
        'player1': 218,
        'player2': 219,
        'player3': 220
    },
    'pass_for_naki': 221,
    'pass_for_riichi': 222,
    'ryuukyoku': 223
    }



class myPaipuReplay(PaipuReplay):
    def __init__(self, path, paipu_list, year):
        super().__init__()
        self.path = path
        self.paipu_list = paipu_list
        self.year = year

        # 暂存一局的信息
        self.action_buffer = [[], [], [], []] # 四个玩家的视角
        self.tehai_buffer = [[], [], [], []]

        # 输出整局的信息
        self.action_record = [] # 四个玩家的视角
        self.tehai_record = []
        self.scores_change = []
        self.new_dora_list = []

    def _compress_and_save_data(self, output_path, filename):
        # 将手牌信息编码
        max_length = 0
        tehai_vector = []
        for kyoku in self.tehai_record:
            temp = [[], [], [], []]
            for i in range(4):
                for tile in kyoku[i]:
                    # print(tile)
                    tiles = self._parse_hand_and_calls(tile)
                    vector = self._encode_tiles(tiles)
                    temp[i].append(vector)
                if len(temp[i]) > max_length:
                    max_length = len(temp[i])
            tehai_vector.append(temp)
        # np.array的形状是(kyoku_num, 4, max_length, 4, 37)
        # 补齐

        mask = np.zeros((len(tehai_vector), 4, max_length))
        for idx, kyoku in enumerate(tehai_vector):
            for i in range(4):
                while len(kyoku[i]) < max_length:
                    kyoku[i].append(np.zeros((4, 37)))
                    mask[idx][i][len(kyoku[i]) - 1] = 1

        # 压缩np.array
        tehai_vector = np.array(tehai_vector)
        np.savez_compressed(output_path + '/' + str(self.year) + '_' + filename[:-3] + '_tehai.npz', tehai = tehai_vector, mask = mask)
        # print('tehai_vector shape:', tehai_vector.shape)
        # print(tehai_vector)
        # print('mask shape:', mask.shape)
        # print(mask)


        data = {
            'action': self.action_record,
            # 'tehai': self.tehai_record,
            'scores_change': self.scores_change
        }
        with gzip.open(output_path + '/' + str(self.year) + '_' + filename, 'wb') as f:
            pickle.dump(data, f)

        


    def _paipu_replay(self, path, paipu):
        if not paipu.endswith('gz'): 
            raise RuntimeError(f"Cannot read paipu {paipu}")
        filename = path + '/' + paipu
        # log(filename)
        try:
            with gzip.open(filename, 'rt') as f:
                tree = ET.parse(f)
        except Exception as e:
            raise RuntimeError(e.__str__(), f"Cannot read paipu {filename}")
        root = tree.getroot()        
        self.log("解析牌谱为ElementTree成功！")
        replayer = None
        riichi_status = False
        after_kan = False

        # 依次解析每个标签
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

            elif child.tag == "GO":  # 牌桌规则和等级等信息.
                # self.log(child.attrib)
                try:
                    type_num = int(child.get("type"))
                    tmp = str(bin(type_num))

                    game_info = dict()
                    game_info["is_pvp"] = int(tmp[-1])
                    if not game_info["is_pvp"]:
                        break

                    game_info["no_aka"] = int(tmp[-2])
                    if game_info["no_aka"]:
                        break

                    game_info["no_kuutan"] = int(tmp[-3])
                    if game_info["no_kuutan"]:
                        break

                    game_info["is_hansou"] = int(tmp[-4])
                    # no requirement

                    game_info["is_3ma"] = int(tmp[-5])
                    if game_info["is_3ma"]:
                        break

                    game_info["is_pro"] = int(tmp[-6])
                    if not game_info["is_pro"]:
                        break

                    game_info["is_fast"] = int(tmp[-7])
                    if game_info["is_fast"]:
                        break

                    game_info["is_joukyu"] = int(tmp[-8])

                except Exception as e:
                    self.log(e)
                    continue

                for key in game_info:
                    self.log(key, game_info[key])

                # 0x01	如果是PVP对战则为1
                # 0x02	如果没有赤宝牌则为1
                # 0x04	如果无食断则为1
                # 0x08	如果是半庄则为1
                # 0x10	如果是三人麻将则为1
                # 0x20	如果是特上卓或凤凰卓则为1
                # 0x40	如果是速卓则为1
                # 0x80	如果是上级卓则为1

            elif child.tag == "TAIKYOKU":
                self.log("=========== 此场开始 =========")
                is_new_round = False
                round_no = 0

                # 清空buffer
                self.action_buffer = [[], [], [], []]
                self.tehai_buffer = [[], [], [], []]

                self.action_record = []
                self.tehai_record = []
                self.scores_change = []

            elif child.tag == "UN":
                # 段位信息
                pass

            elif child.tag == "INIT":
                self.log("-----   此局开始  -------")

                # 清空action_buffer和tehai_buffer
                self.action_buffer = [[], [], [], []]
                self.tehai_buffer = [[], [], [], []]
                self.new_dora_list = []

                riichi_status = False
                # 开局时候的各家分数
                scores_str = child.get("ten").split(',')
                init_scores = [int(tmp) * 100 for tmp in scores_str]
                # print("开局各玩家分数：", init_scores)
                # self.log("开局各玩家分数：", scores)

                # Oya ID
                oya_id = int(child.get("oya"))
                self.log("庄家是玩家{}".format(oya_id))

                # 什么局
                
                game_order = int(child.get("seed").split(",")[0])
                #self.log("此局是{}{}局".format(winds[game_order // 4], chinese_numbers[game_order % 4]))

                # 本场和立直棒
                honba = int(child.get("seed").split(",")[1])
                riichi_sticks = int(child.get("seed").split(",")[2])
                #self.log("此局是{}本场, 有{}根立直棒".format(honba, riichi_sticks))

                # 骰子数字
                dice_numbers = [int(child.get("seed").split(",")[3]) + 1, int(child.get("seed").split(",")[4]) + 1]
                self.log("骰子的数字是", dice_numbers)

                # 牌山
                inst = mp.TenhouShuffle.instance()
                yama = inst.generate_yama()
                self.log("牌山(count={}): ".format(len(yama)), yama)

                # 利用PaiPuReplayer进行重放
                replayer = mp.PaipuReplayer()
                if self.write_log:
                    replayer.set_write_log(True)
                
                self.log(f'Replayer.init: {yama} {init_scores} {riichi_sticks} {honba} {game_order // 4} {oya_id}')
                replayer.init(yama, init_scores, riichi_sticks, honba, game_order // 4, oya_id)
                #self.log('Init over.')

                # 开局的dora
                dora_tiles = [int(child.get("seed").split(",")[5])]
                self.log("开局DORA：{}".format(dora_tiles[-1]))

                # 开局的手牌信息:
                hand_tiles = []
                for pid in range(4):
                    tiles_str = child.get("hai{}".format(pid)).split(",")
                    hand_tiles_player = [int(tmp) for tmp in tiles_str]
                    hand_tiles_player.sort()
                    hand_tiles.append(hand_tiles_player)
                    self.log("玩家{}的开局手牌是{}".format(pid, get_tiles_from_id(hand_tiles_player)))

                game_has_init = True  # 表示这一局开始了
                remaining_tiles = 70

                # 将开局信息存入buffer
                info = {
                    "scores": init_scores,
                    "oya_id": oya_id,
                    "honba": honba,
                    "riichi_sticks": riichi_sticks,
                    "dora_tiles": dora_tiles,
                }
                for i in range(4):
                    self.action_buffer[i].append(info)
                    self.tehai_buffer[i].append({
                        "player0": replayer.table.players[0].hand_to_string(),
                        "player1": replayer.table.players[1].hand_to_string(),
                        "player2": replayer.table.players[2].hand_to_string(),
                        "player3": replayer.table.players[3].hand_to_string()
                        })
                # print("kyoku_buffer", self.kyoku_buffer)
                # break

            # ------------------------- 对局过程中信息 ---------------------------
            elif child.tag == "DORA":
                new_dora = int(child.get("hai"))
                self.new_dora_list.append(new_dora)
                # self.log("翻DORA：{}".format(dora_tiles[-1]))

                # 将信息存入buffer
                for i in range(4):
                    self.action_buffer[i].append({
                        "scores": init_scores,
                        "oya_id": oya_id,
                        "honba": honba,
                        "riichi_sticks": riichi_sticks,
                        "dora_tiles": dora_tiles + self.new_dora_list,
                    })
                    self.tehai_buffer[i].append({
                        "player0": replayer.table.players[0].hand_to_string(),
                        "player1": replayer.table.players[1].hand_to_string(),
                        "player2": replayer.table.players[2].hand_to_string(),
                        "player3": replayer.table.players[3].hand_to_string()
                        })


            elif child.tag == "REACH":
                # 立直
                player_id = int(child.get("who"))
                if int(child.get("step")) == 1:
                    riichi_status = True
                    self.log("玩家{}宣布立直".format(player_id))

                if int(child.get("step")) == 2:
                    riichi_status = False
                    # self.log("玩家{}立直成功".format(player_id))
                    # scores[player_id] -= 1000

            elif child.tag[0] in ["T", "U", "V", "W"] and child.attrib == {}:  # 摸牌
                # 进行4次放弃动作
                if after_kan:
                    after_kan = False
                else:
                    if not (child_no - 1 < 0 or 
                        root[child_no - 1].tag == "INIT"):

                        phase = replayer.get_phase()
                        while phase > int(mp.PhaseEnum.P4_ACTION): #回复阶段
                            # 检查是否有鸣牌动作
                            response_actions = replayer.get_response_actions()
                            if len(response_actions) > 1:
                                player_id = replayer.table.who_make_selection()
                                # 将pass动作存入buffer
                                self.action_buffer[player_id].append(action_encode['pass_for_naki'])
                                self.tehai_buffer[player_id].append({
                                    "player0": replayer.table.players[0].hand_to_string(),
                                    "player1": replayer.table.players[1].hand_to_string(),
                                    "player2": replayer.table.players[2].hand_to_string(),
                                    "player3": replayer.table.players[3].hand_to_string()
                                    })

                            replayer.make_selection(0)
                            phase = replayer.get_phase()
                        response_actions = replayer.get_response_actions()
                        # replayer.make_selection(0)
                        # replayer.make_selection(0)
                        # replayer.make_selection(0)
                        # replayer.make_selection(0)

                player_id = "TUVW".find(child.tag[0])
                remaining_tiles -= 1
                obtained_tile = int(child.tag[1:])

                # 向buffer中存入信息
                self.action_buffer[player_id].append(action_encode['draw'][get_tile_from_id(obtained_tile)])
                self.tehai_buffer[player_id].append({
                    "player0": replayer.table.players[0].hand_to_string(),
                    "player1": replayer.table.players[1].hand_to_string(),
                    "player2": replayer.table.players[2].hand_to_string(),
                    "player3": replayer.table.players[3].hand_to_string()
                    })
                        
            elif child.tag[0] in ["D", "E", "F", "G"] and child.attrib == {}:  # 打牌
                player_id = "DEFG".find(child.tag[0])
                discarded_tile = int(child.tag[1:])
                phase = int(replayer.table.get_phase())
                self_actions = replayer.get_self_actions()
                player_id = replayer.table.who_make_selection()
                if riichi_status:
                    selection = replayer.get_selection_from_action(mp.BaseAction.Riichi, [discarded_tile])
                    ret = replayer.make_selection(selection)
                    for i in range(4):
                        self.action_buffer[i].append(action_encode['riichi'][f'player{i}']) # 立直
                        self.action_buffer[i].append(action_encode['discard'][f'player{player_id}'][get_tile_from_id(discarded_tile)]) # 打牌
                        self.tehai_buffer[i].append({
                            "player0": replayer.table.players[0].hand_to_string(),
                            "player1": replayer.table.players[1].hand_to_string(),
                            "player2": replayer.table.players[2].hand_to_string(),
                            "player3": replayer.table.players[3].hand_to_string()
                            })
                        
                    if not ret:
                        self.log('phase', int(phase))
                        self.log(f'要打 {get_tile_from_id(discarded_tile)}, Fail.\n')
                            
                        raise ActionException('立直打牌', paipu, game_order, honba)
                else:                    
                    selection = replayer.get_selection_from_action(mp.BaseAction.Discard, [discarded_tile])
                    ret = replayer.make_selection(selection)
                    for i in range(4):
                        self.action_buffer[i].append(action_encode['discard'][f'player{player_id}'][get_tile_from_id(discarded_tile)])
                        self.tehai_buffer[i].append({
                            "player0": replayer.table.players[0].hand_to_string(),
                            "player1": replayer.table.players[1].hand_to_string(),
                            "player2": replayer.table.players[2].hand_to_string(),
                            "player3": replayer.table.players[3].hand_to_string()
                            })

                    if not ret:
                        self.log('phase', int(phase))
                        self.log(f'要打 {get_tile_from_id(discarded_tile)}, Fail.')
                            
                        raise ActionException('打牌', paipu, game_order, honba)
            

            elif child.tag == "N":  # 鸣牌 （包括暗杠）
                naru_player_id = int(child.get("who"))
                player_id = naru_player_id
                naru_tiles_int = int(child.get("m"))

                # self.log("==========  Naru =================")
                side_tiles_added_by_naru, hand_tiles_removed_by_naru, naru_type, opened = decodem(
                    naru_tiles_int, naru_player_id)

                # opened = True表示是副露，否则是暗杠

                self.log("玩家{}用手上的{}进行了一个{}，形成了{}".format(
                    naru_player_id, hand_tiles_removed_by_naru, naru_type, side_tiles_added_by_naru))
                for i in range(4):
                    if replayer.get_phase() > int(mp.PhaseEnum.P4_ACTION): #回复阶段，除该人之外
                        if i != naru_player_id:
                            self.log(f'Select: {0}')
                            replayer.make_selection(0)
                    # else: #自己暗杠或者鸣牌
                    if i == naru_player_id:
                        response_types = ['Chi', 'Pon', 'Min-Kan']
                        action_types = {'Chi': mp.BaseAction.Chi, 'Pon': mp.BaseAction.Pon, 'Min-Kan': mp.BaseAction.Kan,
                                        'An-Kan': mp.BaseAction.AnKan, 'Ka-Kan': mp.BaseAction.KaKan}
                        
                        if naru_type == 'Min-Kan':
                            after_kan = True

                        if naru_type in response_types:
                            self.log('Response Options')
                            response_actions = replayer.get_response_actions()
                            for ra in response_actions:
                                self.log(ra.to_string() + "|")
                            self.log('Response Options End')
                        else:
                            self.log('SelfAction Options')
                            self_actions = replayer.get_self_actions()
                            for sa in self_actions:
                                self.log(sa.to_string()+"|")
                            self.log('SelfAction Options End')

                        selection = replayer.get_selection_from_action(action_types[naru_type], 
                            hand_tiles_removed_by_naru)
                        self.log(f'Select: {selection}')
                        ret = replayer.make_selection(selection)

                        # 向buffer中存入信息
                        for i in range(4):
                            self.action_buffer[i].append(action_encode[naru_type][f'player{i}'])
                            self.tehai_buffer[i].append({
                                "player0": replayer.table.players[0].hand_to_string(),
                                "player1": replayer.table.players[1].hand_to_string(),
                                "player2": replayer.table.players[2].hand_to_string(),
                                "player3": replayer.table.players[3].hand_to_string()
                                })

                        if not ret:                            
                            self.log(f'要{naru_type} {get_tiles_from_id(hand_tiles_removed_by_naru)}, Fail.\n'
                                    f'{replayer.table.players[naru_player_id].to_string()}')
                            raise ActionException(f'{naru_type}', paipu, game_order, honba)
                        if naru_type not in response_types:
                            break

            elif child.tag == "BYE":  # 掉线
                bye_player_id = int(child.get("who"))
                self.log("### 玩家{}掉线！！ ".format(bye_player_id))

                # 筛选数据时剔除掉线的局
                break

            elif child.tag == "RYUUKYOKU" or child.tag == "AGARI":
                self.log('~Game Over~')
                score_info_str = child.get("sc").split(",")
                score_info = [int(tmp) for tmp in score_info_str]
                score_changes = [score_info[1] * 100, score_info[3] * 100, score_info[5] * 100, score_info[7] * 100]
                self.scores_change.append(score_changes)
                # print("本局各玩家的分数变化是", score_changes)
                
                # score_changes1 = [_ for _ in score_changes]
                # score_changes2 = None
                double_ron = False
                if child.tag == "RYUUKYOKU":
                    if child.get('type') == 'yao9':
                        self.log("九种九牌！")
                        player_id = replayer.table.who_make_selection()
                        replayer.make_selection(14)
                        for i in range(4):
                            self.action_buffer[i].append(action_encode['kyushukyuhai'][f'player{player_id}'])
                            self.tehai_buffer[i].append({
                                "player0": replayer.table.players[0].hand_to_string(),
                                "player1": replayer.table.players[1].hand_to_string(),
                                "player2": replayer.table.players[2].hand_to_string(),
                                "player3": replayer.table.players[3].hand_to_string()
                                })
                            
                            self.action_record.append(self.action_buffer)
                            self.tehai_record.append(self.tehai_buffer)


                    elif child.get('type') == 'ron3':
                        self.log('三家和牌！')   
                        continue                
                    else:
                        phase = replayer.get_phase()
                        while phase > int(mp.PhaseEnum.P4_ACTION): #回复阶段
                            # 检查是否有鸣牌动作
                            response_actions = replayer.get_response_actions()
                            if len(response_actions) > 1:
                                player_id = replayer.table.who_make_selection()
                                # 将pass动作存入buffer
                                self.action_buffer[player_id].append(action_encode['pass_for_naki'])
                                self.tehai_buffer[player_id].append({
                                    "player0": replayer.table.players[0].hand_to_string(),
                                    "player1": replayer.table.players[1].hand_to_string(),
                                    "player2": replayer.table.players[2].hand_to_string(),
                                    "player3": replayer.table.players[3].hand_to_string()
                                    })

                            replayer.make_selection(0)
                            phase = replayer.get_phase()
                            if phase == 16: # GAME_OVER
                                break
                        for i in range(4):
                            self.action_buffer[i].append(action_encode['ryuukyoku'])
                            self.tehai_buffer[i].append({
                                "player0": replayer.table.players[0].hand_to_string(),
                                "player1": replayer.table.players[1].hand_to_string(),
                                "player2": replayer.table.players[2].hand_to_string(),
                                "player3": replayer.table.players[3].hand_to_string()
                                })
                            
                        # 把和牌信息存入record
                        self.action_record.append(self.action_buffer)
                        self.tehai_record.append(self.tehai_buffer)
    
                    if replayer.get_phase() != int(mp.PhaseEnum.GAME_OVER):
                        raise ActionException('牌局未结束', paipu, game_order, honba)

                    # result = replayer.get_result()
                    # print(result.score)
                    # result_score = result.score
                    # target_score = [score_changes[i] + scores[i] for i in range(4)]                    
                    # result_score_change = [result_score[i] - scores[i] for i in range(4)]    
                    # self.log(score_changes1, score_changes2, scores, result_score)
                    # self.log(result.to_string())
                    # for i in range(4):
                    #     if score_changes[i] + scores[i] == result_score[i]:
                    #         continue
                    #     else:
                    #         raise ScoreException(f'Now: {result_score}({result_score_change}) Expect: {target_score}({score_changes}) Original: {scores}', paipu, game_order, honba)
                    
                    # self.log('OK!')
                                   
                elif child.tag == "AGARI":
                    who_agari = []
                    if child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI":
                        double_ron = True
                        self.log("这局是Double Ron!!!!!!!!!!!!")
                        # 忽略双响的对局
                        self.action_record = []
                        self.tehai_record = []
                        self.scores_change = []
                        break

                        who_agari.append(int(root[child_no + 1].get("who")))
                        
                        score_info_str2 = root[child_no + 1].get("sc").split(",")
                        score_info2 = [int(tmp) for tmp in score_info_str2]
                        score_changes2 = [score_info2[1] * 100, score_info2[3] * 100, score_info2[5] * 100, score_info2[7] * 100]
                        for i in range(4):
                            score_changes[i] += score_changes2[i]

                    agari_player_id = int(child.get("who"))  
                    who_agari.append(int(child.get("who")))

                    for i in range(4):
                        phase = replayer.get_phase()
                        self.log(f'phase {phase}')
                        ret = True
                        if phase <= int(mp.PhaseEnum.P4_ACTION):
                            self.log('SelfAction Options')
                            player_id = replayer.table.who_make_selection()
                            self_actions = replayer.get_self_actions()
                            for sa in self_actions:
                                self.log(sa.to_string()+"|")
                            self.log('SelfAction Options End')
                            selection = replayer.get_selection_from_action(mp.BaseAction.Tsumo, [])
                            for i in range(4):
                                self.action_buffer[i].append(action_encode['tsumo'][f'player{agari_player_id}'])
                                self.tehai_buffer[i].append({
                                    "player0": replayer.table.players[0].hand_to_string(),
                                    "player1": replayer.table.players[1].hand_to_string(),
                                    "player2": replayer.table.players[2].hand_to_string(),
                                    "player3": replayer.table.players[3].hand_to_string()
                                    })
                                
                            # 把和牌信息存入record
                            self.action_record.append(self.action_buffer)
                            self.tehai_record.append(self.tehai_buffer)
                            ret = replayer.make_selection(selection)
                            
                            if not ret:
                                self.log(replayer.table.players[int(phase)].to_string(),
                                    '听牌:' + replayer.table.players[int(phase)].tenpai_to_string()
                                )
                                
                                raise ActionException('自摸', paipu, game_order, honba)
                            break

                        else:
                            if i not in who_agari:
                                self.log('Select: 0')
                                replayer.make_selection(0)
                            else:
                                
                                self.log('Response Options')
                                response_actions = replayer.get_response_actions()
                                for ra in response_actions:
                                    self.log(ra.to_string()+"|")
                                self.log('Response Options End')
                                if phase <= int(mp.PhaseEnum.P4_RESPONSE):
                                    player_id = replayer.table.who_make_selection()
                                    selection = replayer.get_selection_from_action(mp.BaseAction.Ron, [])
                                    for i in range(4):
                                        self.action_buffer[i].append(action_encode['ron'][f'player{agari_player_id}'])
                                        self.tehai_buffer[i].append({
                                            "player0": replayer.table.players[0].hand_to_string(),
                                            "player1": replayer.table.players[1].hand_to_string(),
                                            "player2": replayer.table.players[2].hand_to_string(),
                                            "player3": replayer.table.players[3].hand_to_string()
                                            })
                                        
                                    # 把和牌信息存入record
                                    self.action_record.append(self.action_buffer)
                                    self.tehai_record.append(self.tehai_buffer)
                                    ret = replayer.make_selection(selection)  
                                elif phase <= int(mp.PhaseEnum.P4_chankan):
                                    player_id = replayer.table.who_make_selection()
                                    selection = replayer.get_selection_from_action(mp.BaseAction.ChanKan, [])
                                    for i in range(4):
                                        self.action_buffer[i].append(action_encode['ron'][f'player{agari_player_id}'])
                                        self.tehai_buffer[i].append({
                                            "player0": replayer.table.players[0].hand_to_string(),
                                            "player1": replayer.table.players[1].hand_to_string(),
                                            "player2": replayer.table.players[2].hand_to_string(),
                                            "player3": replayer.table.players[3].hand_to_string()
                                            })

                                    # 把和牌信息存入record
                                    self.action_record.append(self.action_buffer)
                                    self.tehai_record.append(self.tehai_buffer)
                                    ret = replayer.make_selection(selection)

                                elif phase <= int(mp.PhaseEnum.P4_chanankan):
                                    player_id = replayer.table.who_make_selection()
                                    selection = replayer.get_selection_from_action(mp.BaseAction.ChanAnKan, [])
                                    for i in range(4):
                                        self.action_buffer[i].append(action_encode['ron'][f'player{agari_player_id}'])
                                        self.tehai_buffer[i].append({
                                            "player0": replayer.table.players[0].hand_to_string(),
                                            "player1": replayer.table.players[1].hand_to_string(),
                                            "player2": replayer.table.players[2].hand_to_string(),
                                            "player3": replayer.table.players[3].hand_to_string()
                                            })
                                    # 把和牌信息存入record
                                    self.action_record.append(self.action_buffer)
                                    self.tehai_record.append(self.tehai_buffer)
                                    ret = replayer.make_selection(selection)
                                if not ret:                                    
                                    raise ActionException('荣和', paipu, game_order, honba)
                    if replayer.get_phase() != int(mp.PhaseEnum.GAME_OVER):
                        raise ActionException('牌局未结束', paipu, game_order, honba)

                    # result = replayer.get_result()
                    # print(result.score)
                    # result_score = result.score
                    # target_score = [score_changes[i] + scores[i] for i in range(4)]                    
                    # result_score_change = [result_score[i] - scores[i] for i in range(4)]    
                    # self.log(score_changes1, score_changes2, scores, result_score)
                    # # self.log(result.to_string())
                    # for i in range(4):
                    #     if score_changes[i] + scores[i] == result_score[i]:
                    #         continue
                    #     else:
                    #         raise ScoreException(f'Now: {result_score}({result_score_change}) Expect: {target_score}({score_changes}) Original: {scores}', paipu, game_order, honba)
                    
                    self.log('OK!')

                    # from_who = int(child.get("fromWho"))
                    # agari_tile = int(child.get("machi"))
                    # honba = int(child.get("ba").split(",")[0])
                    # riichi_sticks = int(child.get("ba").split(",")[1])

                    # if from_who == agari_player_id:
                    #     info = "自摸"
                    # else:
                    #     info = "玩家{}的点炮".format(from_who)

                    # self.log("玩家{} 通过{} 用{} 和了{}点 ({}本场,{}根立直棒)".format(
                    #     agari_player_id, info, agari_tile, score_changes[agari_player_id] , honba, riichi_sticks))
                    # self.log("和牌时候的手牌 (不包含副露):", [int(hai) for hai in child.get("hai").split(",")])
                    if double_ron:
                        break
                # self.log("本局各玩家的分数变化是", score_changes)

                if not (child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI"):
                    game_has_end = True  # 这一局结束了
                    # num_games += 1

                    # if num_games % 100 == 0:
                    #     self.log("****************** 已解析{}局 ***************".format(num_games))

                # if "owari" in child.attrib:
                #     owari_scores = child.get("owari").split(",")
                #     self.log("========== 此场终了，最终分数 ==========")
                #     self.log(int(owari_scores[0]) * 100, int(owari_scores[2]) * 100,
                #         int(owari_scores[4]) * 100, int(owari_scores[6]) * 100)
            else:
                raise ValueError(child.tag, child.attrib, "Unexpected Element!")

    def extract(self, output_path):
        for paipu in tqdm(self.paipu_list):
            self._paipu_replay(self.path, paipu)
            self._compress_and_save_data(output_path = output_path, filename=paipu)


    def _split_tiles(self, call_str):
        """将类似于'1p2p3p'的字符串拆分为['1p', '2p', '3p']"""
        return [call_str[i:i+2] for i in range(0, len(call_str), 2)]

    def _parse_hand_and_calls(self,  data):
        all_tiles = [[], [], [], []]
        for i in range(4):
            hand = data[f'player{i}'].split('Hand: ')[1].split(' \n')[0].split()
            all_tiles[i].extend(hand)
            calls = []
            if 'Calls: ' in data[f'player{i}']:
                raw_calls = data[f'player{i}'].split('Calls: ')[1].split(' \n')[0].replace('(', '').replace(')', '').split()
                for call in raw_calls:
                    calls.extend(self._split_tiles(call))  # 拆分每个call
                all_tiles[i].extend(calls)
            # print(hand, calls)
        return all_tiles

    def _encode_tiles(self, tiles):
        vector = np.zeros((4, 37), dtype=int)
        for i in range(4):

            counter = Counter(tiles[i])
        
            for tile, count in counter.items():
                if tile in tile_map:
                    idx = tile_map[tile]
                    vector[i, idx] = count
                    
        return vector

# # 示例输入数据
# data = {
#     'player0': 'Pt: 25000\nWind: East\nHand: 1m 2m 3m 4m 6m 7m 8m 9m 6z 6z \nCalls: (7z)7z7z \nRiver: 2s1h 3p5h- 7s9h 8s13h 5s17 1s21 2z25 6m29 1p33- \nRiichi: No\nMenzen: No',
#     'player1': 'Pt: 25000\nWind: South\nHand: 3p 4p 4p 5p 6p 8p 9p \nCalls: 1p2p(3p) (1p)2p3p \nRiver: 9m2h 3m6h 6s10 7z14 2s18h 1s22h 5z26h 3m30 6z34h \nRiichi: No\nMenzen: No',
#     'player2': 'Pt: 25000\nWind: West\nHand: 4m 5m 0p 7p 9p 9p 2s 3s 0s 5s 7s 8s 9s \nRiver: 3z3h 1z7 1z11 1p15h 1m19 2p23 1z27 4m31h \nRiichi: No\nMenzen: Yes',
#     'player3': 'Pt: 25000\nWind: North\nHand: 3m 4m 0m 6m 7m 8m 9m 4p 3s 4s 4s 6s 6s \nRiver: 5z4h 4z8 7z12h- 9s16h 8p20h 7s24h 9p28 4z32 \nRiichi: No\nMenzen: Yes'
# }

# # 解析数据并编码
# tiles = parse_hand_and_calls(data)
# vector = encode_tiles(tiles)

# print(vector)
        



def multi_process(path, paipu_list, year, ouput_path):
    myreplayer = myPaipuReplay(path=path, paipu_list=paipu_list, year=year)
    myreplayer.extract(output_path=ouput_path)
