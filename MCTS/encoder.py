# This file contains the encoder class which is used to encode the game state into a tensor

import re, random
from bs4 import BeautifulSoup
from dict import *
import numpy as np
from Taikyoku_loader import Taikyoku_loader


class Encoder(Taikyoku_loader):
    def __init__(self, log) -> None:
        self.log = log
        self.info = {}
        self.dora = []
        self.tehai = [[], [], [], []]
        self.naki = [[], [], [], []]
        self.record = []
        # print(self.log)
        pass

    def _encode_public(self)->np.ndarray:
        pass

    def _encode_dora(self)->np.ndarray:
        pass

    def _encode_tehai(self)->np.ndarray:
        pass

    def _encode_record(self)->np.ndarray:
        pass

    def _update(self, tag:str)->None:
        pattern_draw = re.compile(r'<([TUVW])(\d{1,3})/>')
        pattern_discard = re.compile(r'<([DEFG])(\d{1,3})/>')

        # 先找到init标签，获取手牌和场面信息
        soup = BeautifulSoup(tag, 'lxml')
        if soup.find('init'):
            print("init")
            self.tehai = list(self._get_hai(soup.find('init')))
            self.info['oya'] = soup.init.get('oya')
            self.info['ten'] = soup.init.get('ten')
            self.info['seed'] = soup.init.get('seed')
            self.dora.append(int(soup.init.get('seed').strip().split(',')[-1]))
            print(self.dora)
            print(self.tehai)
            print(self.info)

        # 抽牌标签
        elif pattern_draw.match(tag):
            match = pattern_draw.match(tag)
            # print('draw')
            if match.group(1) == 'T':
                self.tehai[0].append(int(match.group(2)))
            elif match.group(1) == 'U':
                self.tehai[1].append(int(match.group(2)))
            elif match.group(1) == 'V':
                self.tehai[2].append(int(match.group(2)))
            elif match.group(1) == 'W':
                self.tehai[3].append(int(match.group(2)))
        # 弃牌标签
        elif pattern_discard.match(tag):
            match = pattern_discard.match(tag)
            # print('discard')
            if match.group(1) == 'D':
                self.tehai[0].remove(int(match.group(2)))
            elif match.group(1) == 'E':
                self.tehai[1].remove(int(match.group(2)))
            elif match.group(1) == 'F':
                self.tehai[2].remove(int(match.group(2)))
            elif match.group(1) == 'G':
                self.tehai[3].remove(int(match.group(2)))
        # 副露标签
        elif soup.find('n'):
            m = soup.find('n').get('m')
            who = int(soup.find('n').get('who'))
            # result反映参与副露的牌
            result = self._get_naki(m, tehai=self.tehai)
            self.naki[who].extend(result)
            temp = [x for x in self.tehai[who] if x not in result]
            print("附录前的手牌：", self.tehai[who])
            self.tehai[who] = temp
            # print("玩家：", player_dict[str(who)])
            # print("副露后的手牌：", self.tehai[who])
            # print("副露:", self.naki[who])
            # print("副露")

        # 立直标签
        elif soup.find('reach'):
            step = soup.find('reach').get('step')
            who = soup.find('reach').get('who')
            if step == '2':
                ten = soup.find('reach').get('ten')
                self.info['ten'] = ten
                print(player_dict[who], "立直成功")
                print("立直后的点数：", ten)
            else:
                print(player_dict[who], "宣布立直")

        # dora标签
        elif soup.find('dora'):
            print("新宝牌为：", pai_dict[int(soup.dora.get('hai'))])
            self.dora.append(int(soup.dora.get('hai')))

        pass

    def encode(self, ratio = 10)->np.ndarray:
        random_num = random.sample(range(1, len(self.log)+1), len(self.log)//ratio)
        random_num.sort()
        print(random_num)

        for index, tag in enumerate(self.log):

            self._update(tag)

            # 如果是随机选中的局数，打印手牌
            if index+1 in random_num:
                for i in range(4):
                    temp = self.tehai[i]
                    temp.sort()
                    pai = [pai_dict[x] for x in temp]
                    print(player_dict[str(i)], pai)
                print("=====================================")
                continue
            


        #     elif soup.find('agari'):
        #         who = soup.agari.get('who')
        #         fromWho = soup.agari.get('fromwho')
        #         ten_temp = soup.agari.get('ten')
        #         ten = ten_temp.strip().split(',')
        #         sc_temp = soup.agari.get('sc')
        #         sc = sc_temp.strip().split(',')
        #         # 将sc中的点数转化为整数
        #         sc = [int(x) for x in sc]

        #         if who == fromWho:
        #             print(player_dict[who], "自摸")
        #         else:
        #             print(player_dict[who], "荣和", player_dict[fromWho])
        #         print(ten[0],"符",ten[1],"点")
        #         print("点数变化：")
        #         for i in range(4):
        #             print(player_dict[str(i)], sc[2*i]*100, "+", sc[2*i+1]*100)
        #     elif soup.find('ryuukyoku'):
        #         sc_temp = soup.ryuukyoku.get('sc')
        #         sc = sc_temp.strip().split(',')
        #         sc = [int(x) for x in sc]

        #         print("流局")
        #         if soup.ryuukyoku.get('type'):
        #             print("流局原因：", soup.ryuukyoku.get('type'))
        #             print("文件：", self.file, "第", num, "局")
        
        #         print("点数变化：")
        #         for i in range(4):
        #             print(player_dict[str(i)], sc[2*i]*100, "+", sc[2*i+1]*100)

        #     else:
        #         print("未知标签")
        #         # 打印出对局文件名称和局数
        #         print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])
        #         break

        # print("=====================================")

        