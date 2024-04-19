
from typing import List
from components.dict import *
import sys


# 定义对局信息类
class Taikyoku_info(object):
    def __init__(self) -> None:
        self.dora: List[int] = []                       # dora指示牌
        self.yama: List[int] = [x for x in range(136)]  # 牌山
        self.remain_draw: int = 70                      # 剩余摸牌数(四人场)
        self.reach_num: int = 0                         # 立直玩家数
        self.bakaze: Bakaze = Bakaze.EAST               # 场风
        self.kyoku: int = 1                             # 局数
        self.honba: int = 0                             # 本场
        self.reach_stick: int = 0                       # 场上立直棒的数目
        self.kyotaku: int = 0                           # 拱托数
        self.oya: int = 0                               # 庄家
        self.record: List[dict] = []                    # 记录

    def set_next_kyoku(self):
        pass

    def set_dora(self, new_dora: int):
        self.dora = []
        self.dora.append(new_dora)
        pass

    def append_dora(self, new_dora: int):
        self.dora.append(new_dora)
        pass

    # 从所有牌中移除玩家手牌
    def set_yama(self, tehai: List[List[int]]):
        # 初始化牌山
        self.yama = [x for x in range(136)]

        # 从所有牌中移除玩家手牌
        for i in range(4):
            for j in range(len(tehai[i])):
                try:
                    self.yama.remove(tehai[i][j])
                except ValueError:
                    print(tehai[i][j])
                    print("尝试从牌山中移除不存在的牌")
                    sys.exit()
        pass

    def set_remain_draw(self, remain_draw: int):
        self.remain_draw = remain_draw
        pass

    def reach(self):
        self.reach_num += 1
        self.reach_stick += 1
        pass

    def calc_kyotaku(self):
        # 拱托数 = 立直棒数*1000 + 本场数*300
        self.kyotaku = self.reach_stick * 1000 + self.honba * 300
        pass

    
    def print_info(self):
        print("=====================================")
        print("场风：", self.bakaze)
        print("局数：", self.kyoku)
        print("本场：", self.honba)
        print("宝牌指示牌：", self.dora)
        print("剩余摸牌数：", self.remain_draw)
        print("立直棒数：", self.reach_stick)
        print("拱托数：", self.kyotaku)
        print("庄家：", self.oya)
        print("牌山：", self.yama)
        print("=====================================")
        print("\n")
        pass
