import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from glob import glob

from Ushape.model import PointNetClassifier
from utils.gen_util import load_and_sample_mesh, logger


class UShapeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = {}
        self.stl_path_ls = []
        for dir_path in [x[0] for x in os.walk(data_dir)][1:]:
            self.stl_path_ls += glob(os.path.join(dir_path, "*Lower.stl"))
            self.stl_path_ls += glob(os.path.join(dir_path, "*Upper.stl"))

        # 读取标签文件
        with open(data_dir + "/labels.txt", 'r') as f:
            for line in f:
                file_name, label = line.strip().split(",")
                self.labels[file_name] = int(label)

    def __len__(self):
        return len(self.stl_path_ls)

    def __getitem__(self, idx):
        file_path = self.stl_path_ls[idx]
        # 加载点云数据
        pointcloud = load_and_sample_mesh(file_path)
        base_name = os.path.basename(file_path)
        if base_name in self.labels:
            label = self.labels[base_name]
            return torch.tensor(pointcloud), torch.tensor(label)
        else:
            logger.info("failed to find key in labels file...")
            return torch.tensor(pointcloud), torch.tensor(0)

# 训练循环
def train(model, train_loader, val_loader, criterion, optimizer, model_path, epochs=20):
    # 初始化最小损失为一个很大的数（可以是正无穷）
    best_val_loss = float('inf')
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for pointclouds, labels in train_loader:
            pointclouds, labels = pointclouds.float(), labels.long()
            optimizer.zero_grad()

            # 前向传播
            outputs = model(pointclouds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            logger.info(f"predict right {correct}/{total}")

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy*100:.2f}%")
        # 验证模式
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度
            for data in val_loader:
                source, labels = data  # 获取一个batch的验证数据
                pointclouds, labels = source.float(), labels.long()
                output = model(pointclouds)  # 进行前向传播
                loss = criterion(output, labels)  # 计算损失
                val_loss += loss.item()  # 累加验证损失

        # 计算验证集平均损失
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

        # 如果当前验证损失小于之前的最小损失，保存当前模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Validation loss decreased, saving model to {model_path}...")
            torch.save(model.state_dict(), model_path)  # 保存模型权重

        # 恢复训练模式
        model.train()


if __name__ == '__main__':
    # 使用数据加载器
    dataset = UShapeDataset("/data/teeth")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    val_dataset = UShapeDataset("/data/val_teeth")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    # 模型实例化
    model = PointNetClassifier()
    criterion = nn.CrossEntropyLoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 开始训练
    train(model, train_loader, val_loader, criterion, optimizer, model_path="/data/UModel.pth", epochs=20)
