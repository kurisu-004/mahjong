# 将xml格式的牌谱数据转换为编码格式
import re
from bs4 import BeautifulSoup
from components.dict import *
from components.player import Player
from components.taikyoku_info import *


class Taikyoku_loader(object):

    def __init__(self, file, doubleRon = False ) -> None:

        # 初始化牌谱数据
        self.file = file
        self.data = self._load_file(file) # 存放牌谱数据
        self.log = self._get_kyoku(doubleRon)

        # 引入Taikyoku_info类，存放对局信息
        # 引入Player类，存放玩家信息
        self.taikyoku_info = Taikyoku_info()
        self.players = {
            0: Player('0'),
            1: Player('1'),
            2: Player('2'),
            3: Player('3')
        } # 玩家id为键，Player类为值

        # 默认初始化第一局的玩家信息和对局信息
        self.current_kyoku_index = 0    # 当前局的索引
        self.current_tag_index = 0      # 当前标签的索引
        self._init_player_taikyoku_info()

        pass

    # 初始化第num局的玩家信息, num=0表示默认初始化第一局
    def _init_player_taikyoku_info(self, num=0):
        try:
            soup = BeautifulSoup(self.log[num][0], 'lxml')
        except:
            print("对局不存在！")
            sys.exit()

        if soup.find('init'):
            tehai = self._get_hai(soup.find('init'))
            oya = int(soup.init.get('oya'))
            ten = soup.init.get('ten')
            ten = ten.strip().split(',')
            seed = soup.init.get('seed')
            seed = seed.strip().split(',')
            for i in range(4):
                self.players[i].reset_next_kyoku()
                self.players[i].tehai = tehai[i]
                self.players[i].point = int(ten[i])*100

                if i == oya:
                    self.players[i].isOya = True
            self.taikyoku_info.oya = oya

            # 根据seed信息确定场风
            if int(seed[0]) < 4:
                self.taikyoku_info.bakaze = Bakaze.EAST
            elif int(seed[0]) < 8:
                self.taikyoku_info.bakaze = Bakaze.SOUTH
            elif int(seed[0]) < 12:
                self.taikyoku_info.bakaze = Bakaze.WEST
            elif int(seed[0]) < 16:
                self.taikyoku_info.bakaze = Bakaze.NORTH
            else:
                print("场风错误！")
                return

            # 更新本场数，立直棒数，拱托数，宝牌
            self.taikyoku_info.kyoku = int(seed[0])+1
            self.taikyoku_info.honba = int(seed[1])
            self.taikyoku_info.reach_stick = int(seed[2])
            self.taikyoku_info.calc_kyotaku()
            self.taikyoku_info.set_dora(int(seed[-1]))
            self.taikyoku_info.set_yama(tehai)
            self.taikyoku_info.set_remain_draw(70)

        
        else:
            print("初始化信息不存在！")
            return
        
        pass

    def _load_file(self, file):
        with open(file, 'rb') as f:
            data_b = f.read()
            data = data_b.decode('utf-8')
        return data
    
    # def _get_tag(self, data, tag):
    #     soup = BeautifulSoup(data, 'lxml')
    #     return soup.find_all(tag)
    
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
        hai = [[], [], [], []]
        integer_hai = [[], [], [], []]
        for i in range(4):
            hai[i] = tag.get(f'hai{i}')
            hai[i] = hai[i].split(',')
            integer_hai[i] = [int(x) for x in hai[i]]
            integer_hai[i].sort()
        return integer_hai
    
    # 获取副露信息,返回参与副露的牌的编号
    def _get_naki(self, m)->list:
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
            small = int(bit3_4, 2)
            middle = int(bit5_6, 2)
            big = int(bit7_8, 2)
            
            # 根据面子花色和编号以及吃的牌的大小确定参与吃的牌的编号
            temp = color * 36 + number * 4 # 该面子的第一张牌的第一枚的编号
            m1 = temp + small
            m2 = temp + 4 + middle
            m3 = temp + 8 + big

            return [m1, m2, m3], Naki.CHI
                
                

        # bit3判断是否为碰
        elif bit3 == '1':
            print("碰")
            pong = int(bit9_15, 2)
            pong_type, position = divmod(pong, 3)
            pong_type = pong_type * 4
            # print("pong_type:", pong_type)
            # print("position:" ,position)
            print("碰的牌为：", pai_dict[pong_type])

            temp = [pong_type, pong_type + 1, pong_type + 2, pong_type + 3]

            print(int(bit5_6, 2))
            temp.remove(pong_type + int(bit5_6, 2))
            # print(temp)
            return temp, Naki.PON

        # bit3_5判断是否为杠
        elif bit3 == '0' and bit4_5 == '00':
            type = int(bit8_15, 2)
            # print(type)
            if fromWho == 0:
                print("暗杠")
                print("杠的牌为：", pai_dict[type])
                # type除以4取余数
                a = type % 4
                type = type - a

                return [type, type+1, type+2, type+3], Naki.ANKAN
            else:
                print("明杠，来自", player_dict[str(fromWho)])
                print("杠的牌为：", pai_dict[type])
                # type除以4取余数
                a = type % 4
                type = type - a

                return [type, type+1, type+2, type+3], Naki.MINKAN

        elif bit4 == '1':
            print("加杠")
            kakan_position = int(bit5_6, 2)
            kang = int(bit9_15, 2)
            kang_type, pong_position = divmod(kang, 3)
            kang_type = kang_type * 4
            print("杠的牌为：", pai_dict[kang_type])
            return kang_type + kakan_position, Naki.KAKAN

        else:
            print("未知副露")
            # 打印出对局文件名称和局数
            print("对局文件：", self.file, "局数：", self.info['seed'].strip().split(',')[0])
            sys.exit()

    def _print_kyoku(self, num):
        pass

    def print_taikyoku(self, num=0): 
        if num == 0: # 打印所有对局
            for index, log in zip(range(len(self.log)), self.log):
                self._print_kyoku(index+1)
        else: # 打印指定对局
            self._print_kyoku(num)
        pass


    #_handdle_tag仅处理标签，具体的玩家信息和对局信息由自身的函数处理
    def _haddle_tag(self, tag, forward: bool):
        soup = BeautifulSoup(tag, 'lxml')
        pattern_draw = re.compile(r'<([TUVW])(\d{1,3})/>')
        pattern_discard = re.compile(r'<([DEFG])(\d{1,3})/>')
        if forward:
            if pattern_draw.match(tag):
                # 解析标签
                match = pattern_draw.match(tag)

                player = draw_discard_dict_int[match.group(1)]
                tile = int(match.group(2))

                # 更新玩家信息
                action = self.players[player].draw(tile)
                self.taikyoku_info.yama.remove(tile)
                self.taikyoku_info.remain_draw -= 1

                # 记录
                self.taikyoku_info.record.append({'player': player, 'action': action, 'tile': tile})

                print("player", player, "action", action, "tile", pai_dict[tile])

            elif pattern_discard.match(tag):
                match = pattern_discard.match(tag)

                player = draw_discard_dict_int[match.group(1)]
                tile = int(match.group(2))
                action = self.players[player].discard(tile)

                # 记录
                self.taikyoku_info.record.append({'player': player, 'action': action, 'tile': tile})

                print("player", player, "action", action, "tile", pai_dict[tile])

            elif soup.find('reach'):
                step = int(soup.find('reach').get('step'))
                who = int(soup.find('reach').get('who'))

                action = self.players[who].reach(step)

                # 记录
                self.taikyoku_info.record.append({'player': who, 'action': action, 'tile': None})


            elif soup.find('dora'):
                self.taikyoku_info.append_dora(int(soup.dora.get('hai')))
                print("新宝牌为：", pai_dict[int(soup.dora.get('hai'))])


            # TODO
            elif soup.find('n'):
                m = soup.find('n').get('m')
                who = int(soup.find('n').get('who'))
                print("player", who, "副露")
                result, naki_type= self._get_naki(m)

                print("result:", result)

                if naki_type == Naki.KAKAN:
                    # 从副露的牌中找到加杠对应的碰
                    for i in range(len(self.players[who].naki)):
                        if self.players[who].naki[i]['type'] == Naki.PON and self.players[who].naki[i]['result'][0]//4 == result//4:
                            self.players[who].naki[i]['type'] = Naki.KAKAN
                            self.players[who].naki[i]['result'].append(result)
                            break
                    # 从手牌中移除加杠的牌
                    temp = [x for x in self.players[who].tehai if x != result]
                    self.players[who].tehai = temp

                    pass

                else:
                    # 将副露的牌加入到玩家的副露list中
                    self.players[who].naki.append({'type': naki_type, 'result': result})
                    # 从手牌中移除副露的牌
                    temp = [x for x in self.players[who].tehai if x not in result]
                    self.players[who].tehai = temp

                pass

            elif soup.find('agari'):
                who = int(soup.agari.get('who'))
                fromWho = int(soup.agari.get('fromwho'))
                ten_temp = soup.agari.get('ten')
                ten = ten_temp.strip().split(',')
                sc_temp = soup.agari.get('sc')
                sc = sc_temp.strip().split(',')
                sc = [int(x) for x in sc]
                print("player", who, "agari", "from player", fromWho)
                print(sc)

                for i in range(4):
                    self.players[i].point += sc[i*2+1]*100
            
            elif soup.find('ryuukyoku'):
                print("流局")
                ryuukyoku_type = soup.ryuukyoku.get('type')
                if ryuukyoku_type:
                    print("流局类型：", ryuukyoku_type)
                pass
                sc_temp = soup.ryuukyoku.get('sc')
                sc = sc_temp.strip().split(',')
                sc = [int(x) for x in sc]
                print(sc)
                

        else:
            print("后退")
        pass

    # 每执行一次step函数，读取一个xml标签，更新玩家信息和对局信息
    def step_forward(self):
        self.current_tag_index += 1
        if self.current_tag_index >= len(self.log[self.current_kyoku_index]):
            print("对局结束！")
            self.reset(self.current_kyoku_index+2)# 重置到下一局
        else:
            tag = self.log[self.current_kyoku_index][self.current_tag_index]
            self._haddle_tag(tag, forward=True)
        pass

    def step_backward(self):
        if self.current_tag_index == 0:
            print("已经是第一步了！")
            return
        else:
            tag = self.log[self.current_kyoku_index][self.current_tag_index]
            self._haddle_tag(tag, forward=False)
            self.current_tag_index -= 1
        pass

    # 回到第num局初始化的状态
    def reset(self, num=1):
        self._init_player_taikyoku_info(num-1)
        self.current_kyoku_index = num-1
        self.current_tag_index = 0
        pass

    def print_loader(self):
        self.taikyoku_info.print_info()
        for i in range(4):
            self.players[i].print_player()

    # 导出当前状态的信息，用于Encoder的encode函数
    def export_info(self):
        info = {
            'taikyoku_info': {
                'bakaze': self.taikyoku_info.bakaze,
                'kyoku': self.taikyoku_info.kyoku,
                'oya': self.taikyoku_info.oya,
                'honba': self.taikyoku_info.honba,
                'reach_stick': self.taikyoku_info.reach_stick,
                'kyotaku': self.taikyoku_info.kyotaku,
                'dora': self.taikyoku_info.dora,
                'remain_draw': self.taikyoku_info.remain_draw
            },
            'players': self.players
        }
        return info