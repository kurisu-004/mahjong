# 将xml格式的牌谱数据转换为编码格式
import re
from bs4 import BeautifulSoup
from components.dict import *
from typing import List
from enum import Enum, auto
from components.player import Player

# 定义场风类
class Bakaze(Enum):
    EAST = auto()
    SOUTH = auto()
    WEST = auto()
    NORTH = auto()

# 定义对局结束原因类
class EndKyoku(Enum):
    TSUMO = auto()
    RON = auto()
    RYUUKYOKU = auto()

# 定义对局信息类
class Taikyoku_info(object):
    def __init__(self) -> None:
        # dora指示牌
        self.dora: List[str] = []
        # 牌山
        self.yama: List[str] = []
        # 剩余摸牌数
        self.remain_draw: int = 70 # 四人局
        # 立直玩家数
        self.reach_num: int = 0
        # 场风
        self.bakaze: Bakaze = Bakaze.EAST
        # 局数
        self.kyoku: int = 1
        # 本场
        self.honba: int = 0
        # 场上立直棒的数目
        self.reach_stick: int = 0
        # 拱托数
        self.kyotaku: int = 0

    def set_dora(self, dora: List[str]):
        self.dora = dora
        pass

    # 从所有牌中移除dora指示牌和玩家手牌
    def set_yama(self, dora: List[str], tehai: List[List[str]]):
        # TODO: 从所有牌中移除dora指示牌和玩家手牌
        pass

    def set_remain_draw(self, remain_draw: int):
        self.remain_draw = remain_draw
        pass

    def reach(self):
        self.reach_num += 1
        self.reach_stick += 1
        pass

    def _calc_kyotaku(self):
        # 拱托数 = 立直棒数*1000 + 本场数*300
        self.kyotaku = self.reach_stick * 1000 + self.honba * 300
        pass

    def kyoku_start(self, info: dict):

        # TODO: 传入的info为dict类型，包含局数，本场，供托，宝牌指示牌等信息
        # 由函数kyoku_end返回一个dict包含每个玩家的点数变化以及下一局的初始化信息
        pass

    # 对局结束后返回一个dict包含每个玩家的点数变化以及下一局的初始化信息
    def kyoku_end(self, endType: EndKyoku)->dict:
        self.kyoku += 1
        self.honba = 0
        self.reach_stick = 0
        self._calc_kyotaku()
        
        # TODO: 根据不同的结束原因计算点数变化以及下一局的初始化信息
        return {

        }


class Taikyoku_loader(object):

    def __init__(self, file, doubleRon = False ) -> None:
        self.file = file
        self.data = self._load_file(file) # 存放牌谱数据

        # TODO: 引入Taikyoku_info类，存放对局信息
        # 引入Player类，存放玩家信息
        self.taikyoku_info = Taikyoku_info()
        self.players = {
            '0': Player('0'),
            '1': Player('1'),
            '2': Player('2'),
            '3': Player('3')
        } # 玩家id为键，Player类为值


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
    
    # 获取副露信息,返回参与副露的牌的编号
    def _get_naki(self, m, tehai=[[]])->list:
        # 转换为16位的2进制数
        m = bin(int(m))[2:].zfill(16)
        print("m=", m)

        bit0_1 = m[-2:]
        bit2 = m[-3]
        bit3 = m[-4]
        bit4 = m[-5]
        bit4_5 = m[-6:-4]
        bit5_6 = m[-7:-5]
        bit6_7 = m[-8:-6]
        bit8_9 = m[-10:-8]
        bit10_15 = m[:6]
        bit9_15 = m[:7]
        bit8_15 = m[:8]



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

            # bit4_9两两表示每张牌在同种牌中是第几张
            # 分别转换为10进制数
            small = int(bit4_5, 2)
            middle = int(bit6_7, 2)
            big = int(bit8_9, 2)
            
            # 根据面子花色和编号以及吃的牌的大小确定参与吃的牌的编号
            temp = color * 36 + number * 4 # 该面子的第一张牌的第一枚的编号
            m1 = temp + small
            m2 = temp + 4 + middle
            m3 = temp + 8 + big
            return [m1, m2, m3]
                
                

        # bit3判断是否为碰
        elif bit3 == '1':
            print("碰")
            pong = int(bit9_15, 2)
            pong_type, position = divmod(pong, 3)
            pong_type = pong_type * 4
            # print("pong_type:", pong_type)
            # print("position:" ,position)
            print("碰的牌为：", pai_dict[pong_type])

            # # 碰的牌为5m, 5p, 5s时打印出对局文件名称和局数
            # if pong_type == 16 or pong_type == 52 or pong_type == 88:
            #     print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])

            # # bit5_6确定没有使用到的那一张牌
            # if (pong_type == 16 and bit5_6 != '00') or (pong_type == 52 and bit5_6 != '00') or (pong_type == 88 and bit5_6 != '00'):
            #     print("含有红宝牌")

            temp = [pong_type, pong_type + 1, pong_type + 2, pong_type + 3]

            print(int(bit5_6, 2))
            temp.remove(pong_type + int(bit5_6, 2))
            print(temp)
            return temp

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

