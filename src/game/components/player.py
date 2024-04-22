from typing import List, Set
from components.dict import *
import sys


# 定义玩家类
class Player:
    def __init__(self, id: int):
        self.id = id
        self.tehai: List[int] = []
        self.naki: List[int] = []
        self.sutehai: List[dict] = [] # 每一张弃牌都是一个字典，包含牌面和手摸切标志
        self.point: int = 25000
        self.isReach: bool = False
        self.isOya: bool = False
        self.ipatsu: bool = False
        self.tenpai: bool = False
        # self.machi: Set[str] = set()
        # self.syan_ten: int = 0
        self.last_action: Action = Action.NONE

    def reset_next_kyoku(self):
        self.tehai = []
        self.naki = []
        self.sutehai = []
        self.isReach = False
        self.isOya = False
        self.ipatsu = False
        self.tenpai = False
        self.last_action = Action.NONE

    def draw(self, hai: int)->Action:
        self.tehai.append(hai)
        self.last_action = Action.DRAW

        return self.last_action

    def discard(self, hai: int)->Action:

        # 如果手牌中没有这张牌，报错
        if hai not in self.tehai:
            print("手牌中没有这张牌")
            sys.exit()

        self.last_action = Action.DISCARD_tegiri
        # 判断上一个动作是否为摸牌
        if self.last_action == Action.DRAW:
            # 判断这张牌是否是手牌list中的最后一张牌
            if self.tehai[-1] == hai:
                tsumokiri = True
                self.last_action = Action.DISCARD_tsumokiri

        tsumokiri = False

        # 将这张牌从手牌中移除
        self.tehai.remove(hai)
        self.tehai.sort()

        # 将这张牌加入到弃牌list中
        self.sutehai.append({'hai': hai, 'tsumokiri': tsumokiri})

        return self.last_action

    def reach(self, step: int):
        if step == 1:
            action = Action.REACH_declear
            # print("player:", self.id, "action:", action)
        elif step == 2:
            action = Action.REACH_success

            self.isReach = True
            self.point -= 1000
            # print("player:", self.id, "action:", action)
        else:
            print("立直阶段错误")
            sys.exit()
        
        return action

    def handle_naki(self, m, print_info=False):
        # 转换为16位的2进制数
        m = bin(int(m))[2:].zfill(16)
        # print("m=", m)

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
        bit3_4 = m[-5:-3]
        bit7_8 = m[-9:-7]

        # bit0~bit1表示来源
        fromWho = int(bit0_1, 2)
        # print("来源：", player_dict[str(fromWho)], fromWho)

        # bit2判断是否为吃
        if bit2 == '1':
            
            # bit10~bit15表示吃的牌
            chi = int(bit10_15, 2)
            # 除以3取整和取余数
            temp, position = divmod(chi, 3)
            # 再除以7取整和取余数
            color, number = divmod(temp, 7)
            if print_info:
                print("吃")
                print("吃的牌为：", number_dict[number], color_dict[color], "第", position+1, "张")
                print("面子编号：", number)

            if position == 0:
                action_type = Action.CHI_bottom
            elif position == 1:
                action_type = Action.CHI_middle
            else:
                action_type = Action.CHI_top

            # 根据面子编号和吃的牌的大小确定吃的牌的编号
            nakihai = 4*position + 36*color + number*4

            # bit4_9两两表示每张牌在同种牌中是第几张
            # 分别转换为10进制数
            small = int(bit3_4, 2)
            middle = int(bit5_6, 2)
            big = int(bit7_8, 2)
            
            # 根据面子花色和编号以及吃的牌的大小确定参与吃的牌的编号
            temp = color * 36 + number * 4 # 该面子的第一张牌的第一枚的编号
            m1 = temp + small
            m2 = temp + 4 + middle
            m3 = temp + 8 + big

            self.naki.append({'type': Naki.CHI, 'fromWho': fromWho, 'result': [m1, m2, m3], 'nakihai': nakihai})
            # 从手牌中移除副露的牌
            temp = [x for x in self.tehai if x not in [m1, m2, m3]]
            self.tehai = temp

            return nakihai, [m1, m2, m3], action_type
                
                

        # bit3判断是否为碰
        elif bit3 == '1':
            
            pong = int(bit9_15, 2)
            pong_type, position = divmod(pong, 3)
            pong_type = pong_type * 4
            # print("pong_type:", pong_type)
            # print("position:" ,position)
            if print_info:
                print("碰")
                print("碰的牌为：", pai_dict[pong_type])

            temp = [pong_type, pong_type + 1, pong_type + 2, pong_type + 3]

            print(int(bit5_6, 2))
            temp.remove(pong_type + int(bit5_6, 2))
            
            self.naki.append({'type': Naki.PON, 'fromWho': fromWho, 'result': temp, 'nakihai': pong_type })
            # 从手牌中移除副露的牌
            temp_2 = [x for x in self.tehai if x not in temp]
            self.tehai = temp_2

            if fromWho == 1:
                return pong_type, temp, Action.PON_fromShimo
            elif fromWho == 2:
                return pong_type, temp, Action.PON_fromToimen
            elif fromWho == 3:
                return pong_type, temp, Action.PON_fromKami
            else:
                print("碰来源错误")
                sys.exit()


        # bit3_5判断是否为杠
        elif bit3 == '0' and bit4_5 == '00':
            type = int(bit8_15, 2)
            # print(type)
            if fromWho == 0:
                if print_info:
                    print("暗杠")
                    print("杠的牌为：", pai_dict[type])
                # type除以4取余数
                a = type % 4
                type = type - a

                self.naki.append({'type': Naki.ANKAN, 'fromWho': fromWho, 'result': [type, type+1, type+2, type+3], 'nakihai': type})
                # 从手牌中移除副露的牌
                temp = [x for x in self.tehai if x not in [type, type+1, type+2, type+3]]
                self.tehai = temp

                return type, [type, type+1, type+2, type+3], Action.ANKAN
            else:
                if print_info:
                    print("明杠，来自", player_dict[str(fromWho)])
                    print("杠的牌为：", pai_dict[type])
                # type除以4取余数
                a = type % 4
                type = type - a

                self.naki.append({'type': Naki.MINKAN, 'fromWho': fromWho, 'result': [type, type+1, type+2, type+3], 'nakihai': type})
                # 从手牌中移除副露的牌
                temp = [x for x in self.tehai if x not in [type, type+1, type+2, type+3]]
                self.tehai = temp

                if fromWho == 1:
                    return type, [type, type+1, type+2, type+3], Action.MINKAN_fromShimo
                elif fromWho == 2:
                    return type, [type, type+1, type+2, type+3], Action.MINKAN_fromToimen
                elif fromWho == 3:
                    return type, [type, type+1, type+2, type+3], Action.MINKAN_fromKami
                else:
                    print("明杠来源错误")
                    sys.exit()

        elif bit4 == '1':

            kakan_position = int(bit5_6, 2)
            kang = int(bit9_15, 2)
            kang_type, pong_position = divmod(kang, 3)
            kang_type = kang_type * 4
            if print_info:
                print("加杠")
                print("杠的牌为：", pai_dict[kang_type])
            return kang_type, kang_type + kakan_position, Action.KAKAN

        else:
            print("未知副露")
            # 打印出对局文件名称和局数
            print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])
            sys.exit()

    def print_player(self):
        print("player", self.id)
        print("====================================")
        print("手牌：", [pai_dict[x] for x in self.tehai])
        print("副露：", self.naki)
        print("弃牌：", [(pai_dict[x['hai']], x['tsumokiri']) for x in self.sutehai])
        print("点数：", self.point)
        print("立直：", self.isReach)
        print("一发：", self.ipatsu)
        print("听牌：", self.tenpai)
        print("====================================")
        print("\n")
        
