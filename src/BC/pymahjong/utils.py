# 为了方便从天凤replay牌谱，也为了方便后续的训练和demo的展示
# 在python端将对局的各类信息进行模块化处理
# TODO: 工作量较大，后面再做
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List


class TileType(Enum):
    _1m = 0
    _2m = 1
    _3m = 2
    _4m = 3
    _5m = 4
    _6m = 5
    _7m = 6
    _8m = 7
    _9m = 8
    _1p = 9
    _2p = 10
    _3p = 11
    _4p = 12
    _5p = 13
    _6p = 14
    _7p = 15
    _8p = 16
    _9p = 17
    _1s = 18
    _2s = 19
    _3s = 20
    _4s = 21
    _5s = 22
    _6s = 23
    _7s = 24
    _8s = 25
    _9s = 26
    _1z = 27
    _2z = 28
    _3z = 29
    _4z = 30
    _5z = 31
    _6z = 32
    _7z = 33

@dataclass
class Tile:
    type: TileType
    is_red: bool
    is_in_river: Optional[bool] = None # 如果被鸣走则为False，在手牌和牌山中为None
    from_player: Optional[int] = None
    
class ActionType(Enum):
    DRAW = auto()
    DISCARD = auto()
    CHI_LEFT = auto()
    CHI_CENTER = auto()
    CHI_RIGHT = auto()
    PON = auto()
    MINKAN = auto()
    ANKAN = auto()
    KAKAN = auto()
    RIICHI = auto()
    RON = auto()
    TSUMO = auto()
    KYUSHUKYUHAI = auto()
    PASS_NAKI = auto()
    PASS_RIICHI = auto()

@dataclass
class Action:
    type: ActionType
    tile: Tile
    player_id: int


@dataclass
class Player:
    id: int
    point: int
    hand: list[Tile]
    river: deque[Tile]
    naki: list[list[Tile]]
    is_riichi: bool
    is_oya: bool
    def draw(self, tile: Tile):
        self.hand.append(tile)

    def discard(self, tile: Tile):
        self.hand.remove(tile)
        self.river.append(tile)


# 定义场风类
class Wind(Enum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3

class EndReason(Enum):
    TSUMO = auto()
    RON = auto()
    RYUUKYOKU = auto()


class GameInfo:
    players: List[Player]
    wall: Optional[deque[Tile]] = None
    wind: Wind
    honba: int
    riichi_sticks: int
    dora_indicators: List[Tile]
    remaining_tiles: int
    current_player: int
    action_record: List[Action]
    is_end: bool
    end_reason: Optional[EndReason] = None
    payoffs: Optional[List[int]] = None
    replayer: Optional[object] = None
    def __init__(self, table) -> None:
        