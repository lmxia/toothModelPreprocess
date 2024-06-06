import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_util as gu
from torch.utils.data import Dataset, DataLoader
import os


# 这个地方参考了PointNet 的转移矩阵的含义，只不过这里用到的是标准化的矩阵。
# 这里完全参考的PointNet，稍微改了下，增加了转移向量
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_rot = nn.Linear(256, k * k)
        self.fc3_trans = nn.Linear(256, k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        rot = self.fc3_rot(x)
        trans = self.fc3_trans(x)

        iden = torch.eye(self.k).view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        rot = rot + iden
        rot = rot.view(-1, self.k, self.k)
        return rot, trans


class TeethAlignmentModel(nn.Module):
    def __init__(self):
        super(TeethAlignmentModel, self).__init__()
        self.tnet = TNet(k=3)

    def forward(self, source, target):
        rot, trans = self.tnet(source)
        source_transformed = torch.bmm(source.transpose(2, 1), rot).transpose(2, 1)
        source_transformed = source_transformed + trans.unsqueeze(2).repeat(1, 1, source_transformed.size(2))
        loss = chamfer_distance(source_transformed, target)
        return loss, source_transformed


def chamfer_distance(pc1, pc2):
    batch_size, n_points, _ = pc1.size()
    pc1 = pc1.unsqueeze(1).repeat(1, n_points, 1, 1)
    pc2 = pc2.unsqueeze(2).repeat(1, 1, n_points, 1)
    dist = torch.sum((pc1 - pc2) ** 2, dim=-1)
    min_dist1, _ = torch.min(dist, dim=1)
    min_dist2, _ = torch.min(dist, dim=2)
    return torch.mean(min_dist1) + torch.mean(min_dist2)


def train(model, data_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for source, target in data_loader:
            source = source.permute(0, 2, 1).float()  # Transpose to match TNet input shape
            target = target.permute(0, 2, 1).float()
            optimizer.zero_grad()
            loss, _ = model(source, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")


def main():
    non_standard_path = "/data/non_standard"
    standard_path = "/data/xia.stl"
    train_loader = gu.get_generator_set(non_standard_path, standard_path)

    model = TeethAlignmentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_loader, optimizer, epochs=100)

    # 保存模型
    model_path = '/data/teeth_alignment_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
