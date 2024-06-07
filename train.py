import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_util as gu


class TeethAlignmentModel(nn.Module):
    def __init__(self):
        super(TeethAlignmentModel, self).__init__()

        # MLPs for source point cloud
        self.source_conv1 = nn.Conv1d(3, 64, 1)
        self.source_conv2 = nn.Conv1d(64, 64, 1)
        self.source_conv3 = nn.Conv1d(64, 64, 1)
        self.source_conv4 = nn.Conv1d(64, 128, 1)
        self.source_conv5 = nn.Conv1d(128, 1024, 1)

        # MLPs for template point cloud
        self.template_conv1 = nn.Conv1d(3, 64, 1)
        self.template_conv2 = nn.Conv1d(64, 64, 1)
        self.template_conv3 = nn.Conv1d(64, 64, 1)
        self.template_conv4 = nn.Conv1d(64, 128, 1)
        self.template_conv5 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers for pose estimation
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_rot = nn.Linear(256, 4)  # Quaternion for rotation
        self.fc_trans = nn.Linear(256, 3)  # Translation vector

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, source, template):
        # Source point cloud through MLPs
        src = F.relu(self.bn1(self.source_conv1(source)))
        src = F.relu(self.bn1(self.source_conv2(src)))
        src = F.relu(self.bn1(self.source_conv3(src)))
        src = F.relu(self.bn2(self.source_conv4(src)))
        src = F.relu(self.bn3(self.source_conv5(src)))
        src = torch.max(src, 2, keepdim=True)[0]
        src = src.view(-1, 1024)

        # Template point cloud through MLPs
        tmpl = F.relu(self.bn1(self.template_conv1(template)))
        tmpl = F.relu(self.bn1(self.template_conv2(tmpl)))
        tmpl = F.relu(self.bn1(self.template_conv3(tmpl)))
        tmpl = F.relu(self.bn2(self.template_conv4(tmpl)))
        tmpl = F.relu(self.bn3(self.template_conv5(tmpl)))
        tmpl = torch.max(tmpl, 2, keepdim=True)[0]
        tmpl = tmpl.view(-1, 1024)

        # Concatenate features from both point clouds
        combined_features = torch.cat((src, tmpl), 1)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(combined_features)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.relu(self.bn5(self.fc3(x)))

        # Output rotation (quaternion) and translation
        rot = self.fc_rot(x)
        trans = self.fc_trans(x)

        return rot, trans


def chamfer_distance(pc1, pc2):
    batch_size, num_points, _ = pc1.size()
    pc1 = pc1.unsqueeze(1).repeat(1, num_points, 1, 1)
    pc2 = pc2.unsqueeze(2).repeat(1, 1, num_points, 1)
    dist = torch.sum((pc1 - pc2) ** 2, dim=-1)
    min_dist1, _ = torch.min(dist, dim=1)
    min_dist2, _ = torch.min(dist, dim=2)
    return torch.mean(min_dist1) + torch.mean(min_dist2)


def train(model, data_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for source, target in data_loader:
            source = source.permute(0, 2, 1).float()
            target = target.permute(0, 2, 1).float()
            optimizer.zero_grad()
            rot, trans = model(source, target)
            source_transformed = gu.apply_transform(source, rot, trans)
            loss = chamfer_distance(source_transformed, target)
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
