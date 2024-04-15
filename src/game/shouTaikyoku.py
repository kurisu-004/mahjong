# 将xml格式的牌谱数据转换为编码格式

import re
import os
import csv
import numpy as np
from itertools import islice
from bs4 import BeautifulSoup
from mahjong.MCTS.components.dict import *


class taikyoku(object):

    def __init__(self, file, doubleRon = False ) -> None:
        self.file = file
        self.data = self._load_file(file) # 存放牌谱数据

        # 存放当前对局的信息
        self.info = {
            'seed': '',
            'ten': '',
            'oya': '',
            'hai0': '',
            'hai1': '',
            'hai2': '',
            'hai3': '',
        }
        self.log = self._get_kyoku(doubleRon)
        pass

    def _load_file(self, file):
        with open(file, 'rb') as f:
            data_b = f.read()
            data = data_b.decode('utf-8')
        return data
    
    def _get_tag(self, data, tag):
        soup = BeautifulSoup(data, 'lxml')
        return soup.find_all(tag)
    
    # 用正则表达式匹配所有的对局数据，并转化为实际的对局信息并返回
    def _get_kyoku(self, doubleRon):
        # 存放每一局的数据
        kyoku_all = []
        # 将每一局的数据提取出来
        pattern_segment = re.compile(r'(<INIT.*?>)(.*?)(?=<INIT|$)', re.DOTALL)
        # 判断是否双响
        pattern_agari_2 = re.compile(r'(<AGARI.*?><AGARI.*?>)', re.DOTALL)
        # 分割标签
        pattern_tag = re.compile(r'<.*?/>', re.DOTALL)

        matches_kyoku = pattern_segment.finditer(self.data)
        index = 0
        for match_kyoku in matches_kyoku:
            index += 1
            log = [match_kyoku.group(1)]
            match_agari_2 = pattern_agari_2.findall(match_kyoku.group(2))
            matches_tag = pattern_tag.findall(match_kyoku.group(2))
            # print(matches_tag)
            if match_agari_2 and doubleRon:
                print("出现双响！")
                print("对局文件为：", self.file,"第", index, "局")
                print("=====================================")
            for tag in matches_tag:
                log.append(tag)

            
            kyoku_all.append(log)               

        return kyoku_all

    def _get_hai(self, tag):
        hai0 = tag.get('hai0')
        hai0 = hai0.split(',')
        integer_hai0 = [int(x) for x in hai0]
        integer_hai0.sort()
        # hand_hai0 = [pai_dict[x] for x in integer_hai0]
        hai1 = tag.get('hai1')
        hai1 = hai1.split(',')
        integer_hai1 = [int(x) for x in hai1]
        integer_hai1.sort()
        # hand_hai1 = [pai_dict[x] for x in integer_hai1]
        hai2 = tag.get('hai2')
        hai2 = hai2.split(',')
        integer_hai2 = [int(x) for x in hai2]
        integer_hai2.sort()
        # hand_hai2 = [pai_dict[x] for x in integer_hai2]
        hai3 = tag.get('hai3')
        hai3 = hai3.split(',')
        integer_hai3 = [int(x) for x in hai3]
        integer_hai3.sort()
        # hand_hai3 = [pai_dict[x] for x in integer_hai3]
        # return hand_hai0, hand_hai1, hand_hai2, hand_hai3
        return integer_hai0, integer_hai1, integer_hai2, integer_hai3
    
    def _get_seed(self):
        temp = self.info['seed'].strip().split(',')
        kyoku = kyoku_dict[temp[0]]
        honba = temp[1]
        kyotaku = temp[2]
        dora_indicator = pai_dict[int(temp[-1])]
        return kyoku, honba, kyotaku, dora_indicator
    
    # 获取副露信息
    def _get_naki(self, m):
        # 转换为16位的2进制数
        m = bin(int(m))[2:].zfill(16)
        print("m=", m)

        bit0_1 = m[-2:]
        bit2 = m[-3]
        bit3 = m[-4]
        bit4 = m[-5]
        bit4_5 = m[-6:-4]
        bit6_7 = m[-8:-6]
        bit8_9 = m[-10:-8]
        bit10_15 = m[:6]
        bit9_15 = m[:7]
        bit8_15 = m[:8]

        bit5_6 = m[-5:-3]

        # bit0~bit1表示来源
        fromWho = int(bit0_1, 2)
        print("来源：", player_dict[str(fromWho)], fromWho)

        # bit2判断是否为吃
        if bit2 == '1':
            print("吃")
            # bit10~bit15表示吃的牌
            chi = int(bit10_15, 2)
            # 除以3取整和取余数
            temp, position = divmod(chi, 3)
            # 再除以7取整和取余数
            color, number = divmod(temp, 7)
            print("吃的牌为：", number_dict[number], color_dict[color], "第", position+1, "张")
            print("面子编号：", number)

            # bit3_
            if (number == 2 and bit8_9 == '00') or (number == 3 and bit6_7 == '00') or (number == 4 and bit4_5 == '00'):
                print("含有红宝牌")
                
                

        # bit3判断是否为碰
        elif bit3 == '1':
            print("碰")
            pong = int(bit9_15, 2)
            pong_type, position = divmod(pong, 3)
            pong_type = pong_type * 4
            # print("pong_type:", pong_type)
            # print("position:" ,position)
            print("碰的牌为：", pai_dict[pong_type])

            # 碰的牌为5m, 5p, 5s时打印出对局文件名称和局数
            if pong_type == 16 or pong_type == 52 or pong_type == 88:
                print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])

            # bit5_6确定没有使用到的那一张牌
            if (pong_type == 16 and bit5_6 != '00') or (pong_type == 52 and bit5_6 != '00') or (pong_type == 88 and bit5_6 != '00'):
                print("含有红宝牌")

        # bit3_5判断是否为杠
        elif bit3 == '0' and bit4_5 == '00':
            type = int(bit8_15, 2)
            if fromWho == 0:
                print("暗杠")
                print("杠的牌为：", pai_dict[type])
            else:
                print("明杠，来自", player_dict[str(fromWho)])
                print("杠的牌为：", pai_dict[type])

        elif bit4 == '1':
            print("加杠")
            kang = int(bit9_15, 2)
            kang_type, position = divmod(kang, 3)
            kang_type = kang_type * 4
            print("杠的牌为：", pai_dict[kang_type])

        else:
            print("未知副露")
            # 打印出对局文件名称和局数
            print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])


    def print_tehai(self):
        kyoku, honba, kyotaku, dora_indicator = self._get_seed()
        print("局数：", kyoku, "本场：", honba, "供托：", kyotaku, "宝牌指示牌：", dora_indicator)
        hai0 = [pai_dict[x] for x in self.info['hai0']]
        hai1 = [pai_dict[x] for x in self.info['hai1']]
        hai2 = [pai_dict[x] for x in self.info['hai2']]
        hai3 = [pai_dict[x] for x in self.info['hai3']]
        print("自家：", hai0)
        print("下家：", hai1)
        print("对家：", hai2)
        print("上家：", hai3)

    def _print_kyoku(self, num):
        pattern_draw = re.compile(r'<[TUVW](\d{1,3})/>')
        pattern_discard = re.compile(r'<[DEFG](\d{1,3})/>')
        try:
            log = self.log[num-1]
        except:
            print("对局不存在！")
            return
        print("第", num, "局")
        for tag in log:
            soup = BeautifulSoup(tag, 'lxml')
            if soup.find('init'):
                tehai = self._get_hai(soup.find('init'))
                self.info['hai0'] = tehai[0]
                self.info['hai1'] = tehai[1]
                self.info['hai2'] = tehai[2]
                self.info['hai3'] = tehai[3]
                self.info['oya'] = soup.init.get('oya')
                self.info['ten'] = soup.init.get('ten')
                self.info['seed'] = soup.init.get('seed')
                self.print_tehai()

            elif pattern_draw.match(tag):
                match = pattern_draw.match(tag)
                print(draw_dict[tag[1]], pai_dict[int(match.group(1))])
            elif pattern_discard.match(tag):
                match = pattern_discard.match(tag)
                print(discard_dict[tag[1]], pai_dict[int(match.group(1))])
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
            elif soup.find('dora'):
                print("新宝牌为：", pai_dict[int(soup.dora.get('hai'))])

            elif soup.find('n'):
                m = soup.find('n').get('m')
                result = self._get_naki(m)
                # print("副露")
            elif soup.find('agari'):
                who = soup.agari.get('who')
                fromWho = soup.agari.get('fromwho')
                ten_temp = soup.agari.get('ten')
                ten = ten_temp.strip().split(',')
                sc_temp = soup.agari.get('sc')
                sc = sc_temp.strip().split(',')
                # 将sc中的点数转化为整数
                sc = [int(x) for x in sc]

                if who == fromWho:
                    print(player_dict[who], "自摸")
                else:
                    print(player_dict[who], "荣和", player_dict[fromWho])
                print(ten[0],"符",ten[1],"点")
                print("点数变化：")
                for i in range(4):
                    print(player_dict[str(i)], sc[2*i]*100, "+", sc[2*i+1]*100)
            elif soup.find('ryuukyoku'):
                sc_temp = soup.ryuukyoku.get('sc')
                sc = sc_temp.strip().split(',')
                sc = [int(x) for x in sc]

                print("流局")
                if soup.ryuukyoku.get('type'):
                    print("流局原因：", soup.ryuukyoku.get('type'))
                    print("文件：", self.file, "第", num, "局")
        
                print("点数变化：")
                for i in range(4):
                    print(player_dict[str(i)], sc[2*i]*100, "+", sc[2*i+1]*100)

            else:
                print("未知标签")
                # 打印出对局文件名称和局数
                print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])
                break

        print("=====================================")

    def print_taikyoku(self, num=0):
        
        if num == 0: # 打印所有对局
            for index, log in zip(range(len(self.log)), self.log):
                self._print_kyoku(index+1)
        else: # 打印指定对局
            self._print_kyoku(num)
        pass


    # 检查文件中的DORA标签，并打印出对局文件名称和局数
    def check_dora(self):

        # 将含有dora标签的文件名记录在csv文件中

        # .cs文件路径
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset/Tenhou')
        
        # 遍历文件夹下的所有文件
        pattern_dora = re.compile(r'<DORA.*?/>')
        for log in self.log:
            for tag in log:
                soup = BeautifulSoup(tag, 'lxml')
                if soup.find('init'):
                    # tehai = self._get_hai(soup.find('init'))
                    # self.info['hai0'] = tehai[0]
                    # self.info['hai1'] = tehai[1]
                    # self.info['hai2'] = tehai[2]
                    # self.info['hai3'] = tehai[3]
                    # self.info['oya'] = soup.init.get('oya')
                    # self.info['ten'] = soup.init.get('ten')
                    self.info['seed'] = soup.init.get('seed')
                    # self.print_tehai()
                
                if pattern_dora.match(tag):
                    # print("对局文件：", self.file, "局数：", kyoku_dict[self.info['seed'].strip().split(',')[0]], self.info['seed'].strip().split(',')[1], "本场")
                    with open(os.path.join(path, 'dora.csv'), 'a') as f:
                        f.write(self.file + ',' + kyoku_dict[self.info['seed'].strip().split(',')[0]] + ',' + self.info['seed'].strip().split(',')[1] + '\n')

    # 输入天凤中牌的编码，返回自定义的编码
    def _transfer_hai(self, hai)->list:
        new_hai = []
        for x in hai:
            if x == 16:
                temp = 34

            elif x == 52:
                temp = 35

            elif x == 88:
                temp = 36

            else:
                temp = x//4

            new_hai.append(temp)

        new_hai.sort()
        return new_hai

    def encode_kyouku(self, log) -> np.array:

        # 对所有的牌进行one-hot编码, 37种牌1-9m 0m 1-9p 0p 1-9s 0s 1-7z
        # 编码字典
        pattern_draw = re.compile(r'<[TUVW](\d{1,3})/>')
        pattern_discard = re.compile(r'<[DEFG](\d{1,3})/>')

        # 返回的矩阵包括几个部分
        # 1. [CLS]标识符
        # 2. 公共信息：局顺 本场 供托
        # 3. [SEP]分隔符
        # 4. DORA表示牌
        # 5. [SEP]分隔符
        # 6. 自家手牌、下家手牌、对家手牌、上家手牌
        # 7. [SEP]分隔符
        # 8. 动作序列

        # 矩阵的维度
        row = 109

        cls = np.zeros((row, 1))
        cls[-1] = 1

        sep = np.zeros((row, 1))
        sep[-2] = 1



        for tag in log:
            soup = BeautifulSoup(tag, 'lxml')
            if soup.find('init'):
                tehai = self._get_hai(soup.find('init'))
                self.info['hai0'] = tehai[0]
                self.info['hai1'] = tehai[1]
                self.info['hai2'] = tehai[2]
                self.info['hai3'] = tehai[3]
                self.info['oya'] = soup.init.get('oya')
                self.info['ten'] = soup.init.get('ten')
                self.info['seed'] = soup.init.get('seed')
                self.print_tehai() 
                # print(self.info)            
                self._transfer_hai(self.info['hai2'])   

if __name__ == '__main__':

    # 验证编码
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/Tenhou')
    game = taikyoku(os.path.join(path, 'test.log'))
    array = game.encode_kyouku(game.log[0])
    


    # # 读取文件
    # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/Tenhou')
    # files = os.listdir(path)
    # pattern = re.compile(r'test.log')
    # # # pattern = re.compile(r'15790.log') # 双响 第12局
    # # pattern = re.compile(r'52672.log') # 杠

    # for file in files:
    #     if pattern.match(file):
    #         file_path = os.path.join(path, file)
    #         game = taikyoku(file_path)
    #         # game.print_tehai(num=1)
    #         game.print_taikyoku()
    #         # game.check_dora()