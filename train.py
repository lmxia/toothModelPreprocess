import torch
import torch.nn as nn
import torch.nn.functional as F
import gen_util as gu
from chamferdist import ChamferDistance
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeethAlignmentModel(nn.Module):
    def __init__(self):
        super(TeethAlignmentModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        # Fully connected layers for pose estimation
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_rot = nn.Linear(256, 4)  # Quaternion for rotation
        self.fc_trans = nn.Linear(256, 3)  # Translation vector

        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, source, template):
        source_features = self.feature_extractor(source.transpose(2, 1).float()).max(dim=2)[0]
        target_features = self.feature_extractor(template.transpose(2, 1).float()).max(dim=2)[0]

        # Concatenate features from both point clouds
        combined_features = torch.cat((source_features, target_features), 1)

        # Fully connected layers
        x = F.relu(self.bn3(self.fc1(combined_features)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = F.relu(self.bn5(self.fc3(x)))

        # Output rotation (quaternion) and translation
        rot = self.fc_rot(x)
        trans = self.fc_trans(x)

        return rot, trans


def train(model, data_loader, optimizer, epochs=100):
    model.train()
    chamfer_dist = ChamferDistance()
    for epoch in range(epochs):
        epoch_loss = 0
        for source, target in data_loader:
            optimizer.zero_grad()
            rot, trans = model(source, target)
            source_transformed = gu.apply_transform(source, rot, trans)
            loss = compute_loss(chamfer_dist, source_transformed, rot, trans)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logger.info(f"handle a batch with loss {epoch_loss}")

        logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")


def compute_loss(chamfer_dist, source_transformed, pred_rot, target):
    loss_chamfer = chamfer_dist(source_transformed, target)
    # Regularize quaternion
    quat_norm = torch.norm(pred_rot, p=2, dim=1)
    loss_reg = torch.mean((quat_norm - 1) ** 2)

    # Symmetry penalty
    rot_matrix = gu.quat_to_rotmat(pred_rot)
    symmetry_loss = torch.mean((rot_matrix + rot_matrix.transpose(2, 1)) ** 2)

    loss = loss_chamfer + 0.1 * loss_reg + 0.01 * symmetry_loss
    return loss


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
    logger.info(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
