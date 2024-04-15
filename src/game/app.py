from components.Taikyoku_loader import Taikyoku_loader
from components.encoder import Encoder
import os, re, random


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../dataset/Tenhou')
    files = os.listdir(path)
    pattern = re.compile(r'test.log')
    for file in files:
        if pattern.search(file):
            loader = Taikyoku_loader(os.path.join(path, file))
            # loader.print_taikyoku()
            encoder = Encoder(loader.log[3])
            encoder.encode()

            break
        pass
