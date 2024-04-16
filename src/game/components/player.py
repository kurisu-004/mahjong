from typing import List
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

    def draw(self, hai: int):
        self.tehai.append(hai)
        self.last_action = Action.DRAW

    def discard(self, hai: int):

        # 如果手牌中没有这张牌，报错
        if hai not in self.tehai:
            print("手牌中没有这张牌")
            sys.exit()

        # 判断上一个动作是否为摸牌
        if self.last_action == Action.DRAW:
            # 判断这张牌是否是手牌list中的最后一张牌
            if self.tehai[-1] == hai:
                tsumokiri = True

            else:
                tsumokiri = False

        else:
            tsumokiri = False

        # 将这张牌从手牌中移除
        self.tehai.remove(hai)
        self.tehai.sort()

        # 将这张牌加入到弃牌list中
        self.sutehai.append({'hai': hai, 'tsumokiri': tsumokiri})

        self.last_action = Action.DISCARD

    def handle_naki(self, hai: str):
        pass

    def ron(self, hai: str):
        pass

    
    def print_tehai(self):
        print("player", self.id, ":", self.tehai)

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
        
