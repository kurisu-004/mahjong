import os
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import zipfile



current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)

# 解压数据集
# 定义压缩文件路径
zip_file_path = parent_path + "/dataset/pymahjong-offline-data-20M.zip"

# 创建 ZipFile 对象
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # 获取压缩文件中的文件列表
    file_list = zip_ref.namelist()
    
    # 检查是否有文件存在
    if file_list:
        # 获取文件的名称
        first_file = file_list[0]
        second_file = file_list[1]
        
        # 解压文件到指定目录
        extract_path = parent_path + '/dataset/extract'
        zip_ref.extract(first_file, extract_path)
        zip_ref.extract(second_file, extract_path)

    else:
        print("压缩文件中没有文件。")


train_data_path = parent_path + '/dataset/extract/mahjong-offline-data-batch-0.mat'
test_data_path = parent_path + '/dataset/extract/mahjong-offline-data-batch-1.mat'

class MahjongDataset(Dataset):
    def __init__(self, data_path):
        data = loadmat(data_path)
        self.executor_obs = data["X"]
        self.oracle_obs = data["O"]

    def __len__(self):
        return self.executor_obs.shape[0]

    def __getitem__(self, idx):
        mask = np.full((18, 34), 2)
        train_data = np.concatenate([self.executor_obs[idx], mask], axis=0)
        label = np.concatenate([self.executor_obs[idx], self.oracle_obs[idx]], axis=0)
        return torch.from_numpy(train_data.T), torch.from_numpy(label.T)
    

class MahjongBERT(nn.Module):
    def __init__(self):
        super(MahjongBERT, self).__init__()
        self.embedding = nn.Linear(111, 256)  # 111个特征
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(256, 111)  # 最后的全连接层

    def forward(self, src):
        x = self.embedding(src)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x
    
# CPU
# if __name__ == '__main__':
#     train_dataset = MahjongDataset(train_data_path)
#     test_dataset = MahjongDataset(test_data_path)
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

#     model = MahjongBERT()
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(10):
#         for i, (train_data, label) in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             output = model(train_data.float())
#             loss = criterion(output, label.float())
#             loss.backward()
#             optimizer.step()
#             print("Epoch {}, batch {}, loss: {}".format(epoch, i, loss.item()))

#     print("Training finished!")
#     torch.save(model.state_dict(), parent_path + "/model/mahjong_BERT.pth")

#     # test
#     model.eval()
#     with torch.no_grad():
#         for i, (test_data, label) in enumerate(test_dataloader):
#             output = model(test_data.float())
#             loss = criterion(output, label.float())
#             print("Test batch {}, loss: {}".format(i, loss.item()))

#     print("Test finished!")

if __name__ == '__main__':
    # 检查 GPU 是否可用
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 创建数据加载器
    train_dataset = MahjongDataset(train_data_path)
    test_dataset = MahjongDataset(test_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 创建模型并将其移动到 GPU
    model = MahjongBERT().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        for i, (train_data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_data = train_data.float().to(device)  # 将输入数据移动到 GPU
            label = label.float().to(device)  # 将标签移动到 GPU
            output = model(train_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print("Epoch {}, batch {}, loss: {}".format(epoch, i, loss.item()))

    print("Training finished!")
    torch.save(model.state_dict(), parent_path + "/model/mahjong_BERT.pth")

    # 测试
    model.eval()
    with torch.no_grad():
        for i, (test_data, label) in enumerate(test_dataloader):
            test_data = test_data.float().to(device)  # 将测试数据移动到 GPU
            label = label.float().to(device)  # 将测试标签移动到 GPU
            output = model(test_data)
            loss = criterion(output, label)
            print("Test batch {}, loss: {}".format(i, loss.item()))

    print("Test finished!")
