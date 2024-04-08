# 将xml格式的牌谱数据转换为编码格式
import re
import os
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

oya_dict = {
    0: '自家',
    1: '下家',
    2: '对家',
    3: '上家',
    4: 'None'
}
bakaze_dict = {
    0: '东',
    1: '南',
    2: '西',
    3: '北'
}

class taikyoku(object):

    def __init__(self, file) -> None:
        self.file = file
        self.data = self._load_file(file)
        self.bakaze = 0
        self.last_oya = 4
        self.honba = 0
        pass

    def _load_file(self, file):
        with open(file, 'rb') as f:
            data_b = f.read()
            data = data_b.decode('utf-8')
        return data
    
    def _get_tag(self, tag):
        soup = BeautifulSoup(self.data, 'lxml')
        return soup.find_all(tag)
    
    def _get_hai(self, tag):
        hai0 = tag.get('hai0')
        hai0 = hai0.split(',')
        integer_hai0 = [int(x) for x in hai0]
        integer_hai0.sort()
        hand_hai0 = [pai_dict[x] for x in integer_hai0]
        hai1 = tag.get('hai1')
        hai1 = hai1.split(',')
        integer_hai1 = [int(x) for x in hai1]
        integer_hai1.sort()
        hand_hai1 = [pai_dict[x] for x in integer_hai1]
        hai2 = tag.get('hai2')
        hai2 = hai2.split(',')
        integer_hai2 = [int(x) for x in hai2]
        integer_hai2.sort()
        hand_hai2 = [pai_dict[x] for x in integer_hai2]
        hai3 = tag.get('hai3')
        hai3 = hai3.split(',')
        integer_hai3 = [int(x) for x in hai3]
        integer_hai3.sort()
        hand_hai3 = [pai_dict[x] for x in integer_hai3]
        return hand_hai0, hand_hai1, hand_hai2, hand_hai3
    
    def _get_oya(self, tag):
        oya = tag.get('oya')
        return int(oya)
    
    def print_tehai(self, num=0):
        if num == 0:
            init_tags = self._get_tag('init')
            for init_tag, index in zip(init_tags, range(len(init_tags))):
                hand_hai0, hand_hai1, hand_hai2, hand_hai3 = self._get_hai(init_tag)
                oya = self._get_oya(init_tag)

                # 判断是否连庄
                if oya == self.last_oya:
                    self.honba += 1
                else:
                    self.honba = 0
                
                #判断场风
                if oya == 0 and self.last_oya == 3:
                    self.bakaze += 1
                self.last_oya = oya


                print("第", index+1, "局", "庄家：", oya_dict[oya], " 场风：", bakaze_dict[self.bakaze], " 本场：", self.honba) 
                print("自家手牌：", hand_hai0)
                print("下家手牌：", hand_hai1)
                print("对家手牌：", hand_hai2)
                print("上家手牌：", hand_hai3)
                print("-----------------------------")
        else:
            try:
                init_tag = self._get_tag('init')[num-1]
            except:
                print("局数超出范围")
                return
            hand_hai0, hand_hai1, hand_hai2, hand_hai3 = self._get_hai(init_tag)
            oya = self._get_oya(init_tag)
            print("第", num, "局", "庄家：", oya_dict[oya], " 场风：", bakaze_dict[self.bakaze], " 本场：", self.honba) 
            print("自家手牌：", hand_hai0)
            print("下家手牌：", hand_hai1)
            print("对家手牌：", hand_hai2)
            print("上家手牌：", hand_hai3)
            print("-----------------------------")

if __name__ == '__main__':
    # 读取文件
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/Tenhou')
    files = os.listdir(path)
    # pattern = re.compile(r'(\d{4})_nan_4_kuitan_aka_(.*?).log')
    pattern_test = re.compile(r'test.log')
    # print(files)
    print(os.path.dirname(os.getcwd()))
    for file in files:
        if pattern_test.match(file):
            file_path = os.path.join(path, file)
            game = taikyoku(file_path)
            # game.print_tehai(num=2)
            game.print_taikyoku(2)

