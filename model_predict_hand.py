import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MahjongEmbedding():
    def __init__(self, dmodel=512):
        self.d_model = dmodel
        self.concat_vector_size = 4

        self.record_embedding_layer = nn.Embedding(224, dmodel)
        self.info_embedding_layer = nn.Linear(self.concat_vector_size, dmodel)
        
        pass

    def scores_embedding(self, scores):
        # 归一化处理
        scores = np.array(scores)
        print(scores)
        pass

    def oya_embedding(self, oya):
        pass

    def dora_embedding(self, dora):
        pass

    def honba_riichi_sticks_embedding(self, honba, riichi_sticks):
        pass

    def record_embedding(self, record):
        # 有224种行动

        pass