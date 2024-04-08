# 将xml格式的牌谱数据转换为编码格式

import re
import os
import csv
import numpy as np
from itertools import islice
from bs4 import BeautifulSoup

# 0~135表示136种牌
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
player_dict = {
    '0': '自家',
    '1': '下家',
    '2': '对家',
    '3': '上家'
}
kyoku_dict = {
    '0': '东一局',
    '1': '东二局',
    '2': '东三局',
    '3': '东四局',
    '4': '南一局',
    '5': '南二局',
    '6': '南三局',
    '7': '南四局',
    '8': '西一局',
    '9': '西二局',
    '10': '西三局',
    '11': '西四局',
    '12': '北一局',
    '13': '北二局',
    '14': '北三局',
    '15': '北四局'
}
draw_dict = {
    'T': '自家摸牌',
    'U': '下家摸牌',
    'V': '对家摸牌',
    'W': '上家摸牌'
}
discard_dict = {
    'D': '自家弃牌',
    'E': '下家弃牌',
    'F': '对家弃牌',
    'G': '上家弃牌'
}
encode_dict = {
    'action': {

    },

    'hai': {
        '1m': 0, '2m': 1, '3m': 2, '4m': 3, '5m': 4, '6m': 5, '7m': 6, '8m': 7, '9m': 8, '0m': 9,
        '1p': 10, '2p': 11, '3p': 12, '4p': 13, '5p': 14, '6p': 15, '7p': 16, '8p': 17, '9p': 18, '0p': 19,
        '1s': 20, '2s': 21, '3s': 22, '4s': 23, '5s': 24, '6s': 25, '7s': 26, '8s': 27, '9s': 28, '0s': 29,
        '1z': 30, '2z': 31, '3z': 32, '4z': 33
    }
}


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
        color_dict = {
            0: 'm',
            1: 'p',
            2: 's',
        }
        number_dict = {
            0: '123',
            1: '234',
            2: '345',
            3: '456',
            4: '567',
            5: '678',
            6: '789'
        }
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
            
    def encode_kyouku(self, log) -> np.array:

        # 对所有的牌进行one-hot编码, 37种牌1-9m 0m 1-9p 0p 1-9s 0s 1-7z
        # 编码字典
        pattern_draw = re.compile(r'<[TUVW](\d{1,3})/>')
        pattern_discard = re.compile(r'<[DEFG](\d{1,3})/>')

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

            # elif pattern_draw.match(tag):
            #     match = pattern_draw.match(tag)
            #     print(draw_dict[tag[1]], pai_dict[int(match.group(1))])
            # elif pattern_discard.match(tag):
            #     match = pattern_discard.match(tag)
            #     print(discard_dict[tag[1]], pai_dict[int(match.group(1))])
            # elif soup.find('reach'):
            #     step = soup.find('reach').get('step')
            #     who = soup.find('reach').get('who')
            #     if step == '2':
            #         ten = soup.find('reach').get('ten')
            #         self.info['ten'] = ten
            #         print(player_dict[who], "立直成功")
            #         print("立直后的点数：", ten)
            #     else:
            #         print(player_dict[who], "宣布立直")
            # elif soup.find('dora'):
            #     print("新宝牌为：", pai_dict[int(soup.dora.get('hai'))])

            # elif soup.find('n'):
            #     m = soup.find('n').get('m')
            #     result = self._get_naki(m)
            #     # print("副露")
            # elif soup.find('agari'):
            #     who = soup.agari.get('who')
            #     fromWho = soup.agari.get('fromwho')
            #     ten_temp = soup.agari.get('ten')
            #     ten = ten_temp.strip().split(',')
            #     sc_temp = soup.agari.get('sc')
            #     sc = sc_temp.strip().split(',')
            #     # 将sc中的点数转化为整数
            #     sc = [int(x) for x in sc]

            #     if who == fromWho:
            #         print(player_dict[who], "自摸")
            #     else:
            #         print(player_dict[who], "荣和", player_dict[fromWho])

        

        return 0
    

if __name__ == '__main__':

    # 验证编码
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
    game = taikyoku(os.path.join(path, 'test.log'))
    array = game.encode_kyouku(game.log[0])
    


    # # 读取文件
    # path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/Tenhou/2011_nan_4_kuitan_aka')
    # files = os.listdir(path)
    # pattern = re.compile(r'\d+.log')
    # # # pattern = re.compile(r'15790.log') # 双响 第12局
    # # pattern = re.compile(r'52672.log') # 杠

    # for file in files:
    #     if pattern.match(file):
    #         file_path = os.path.join(path, file)
    #         game = taikyoku(file_path)
    #         # game.print_tehai(num=1)
    #         game.print_taikyoku()
    #         # game.check_dora()
