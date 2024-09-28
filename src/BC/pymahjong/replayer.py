from .myEnv_pymahjong import myMahjongEnv
import gzip
import xml.etree.ElementTree as ET
from pymahjong import MahjongPyWrapper as mp

class myReplayer:
    def __init__(self, path: str, file_name: list):
        self.path = path
        self.file_name = file_name
        self.env = myMahjongEnv()

    def _open_file(self, path: str, file_name: str):
        if not file_name.endswith('.gz'):
            raise ValueError('File must be a .gz file')
        try:
            with gzip.open(path +'/'+ file_name, 'rt') as f:
                tree = ET.parse(f)
        except Exception as e:
            raise RuntimeError(e.__str__(), f"Cannot read paipu {path +'/'+ file_name}")
        return tree

    # 返回每一局初始化信息和动作信息
    def _replay(self, tree):
        root = tree.getroot()
        for child in root:
            if child.tag == "SHUFFLE":
                seed_str = child.get("seed")
                prefix = 'mt19937ar-sha512-n288-base64,'
                if not seed_str.startswith(prefix):
                    self.log('Bad seed string')
                    continue
                seed = seed_str[len(prefix):]
                inst = mp.TenhouShuffle.instance()
                inst.init(seed)