import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        x = x.transpose(2, 1)  # [batch_size, 6, num_points]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.max(dim=2)[0]  # [batch_size, 1024]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        transform = self.fc3(x)
        transform = transform.view(-1, 3, 3)  # [batch_size, 3, 3]
        return transform

class PointNetClassifier(nn.Module):
    def __init__(self):
        super(PointNetClassifier, self).__init__()
        self.tnet = TNet()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # 二分类：完整或不完整

    def forward(self, x):
        transform = self.tnet(x)
        coords = x[:, :, :3]  # 提取坐标
        normals = x[:, :, 3:]  # 提取法向量

        # 对坐标进行变换
        transformed_coords = torch.bmm(coords, transform)  # [batch_size, num_points, 3]
        transformed_x = torch.cat([transformed_coords, normals], dim=2)  # 合并变换后的坐标和法向量

        # 特征提取
        x = self.feature_extractor(transformed_x.transpose(2, 1))  # [batch_size, 1024, num_points]
        x = x.max(dim=2)[0]  # [batch_size, 1024] 进行最大池化

        # 分类
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出未激活的 logits

        return x
