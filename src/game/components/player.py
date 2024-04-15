from typing import List
from components.dict import *
from enum import Enum, auto

# 定义动作类
class Action(Enum):
    DRAW = auto()
    DISCARD = auto()
    CHI = auto()
    PON = auto()
    KAN = auto()
    RON = auto()
    RICHI = auto()
    PASS = auto()
    NONE = auto()


# 定义玩家类
class Player:
    def __init__(self, id: str):
        self.id = id
        self.tehai: List[str] = []
        self.naki: List[str] = []
        self.sutehai: List[dict] = [] # 每一张弃牌都是一个字典，包含牌面和手摸切标志
        self.point: int = 25000
        self.richi: bool = False
        self.ipatsu: bool = False
        self.tenpai: bool = False
        # self.syan_ten: int = 0
        self.last_action: Action = Action.NONE

    def draw(self, hai: str):
        self.tehai.append(hai)
        self.last_action = Action.DRAW

    def discard(self, hai: str):

        # 如果手牌中没有这张牌，抛出异常
        if hai not in self.tehai:
            raise Exception('Discard error: hai not in tehai')

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

        # 将这张牌加入到弃牌list中
        self.sutehai.append({'hai': hai, 'tsumokiri': tsumokiri})

        self.last_action = Action.DISCARD

    def naki(self, hai: str):
        pass

    def ron(self, hai: str):
        pass

    
    def print_tehai(self):
        print("player", self.id, ":", self.tehai)
        
