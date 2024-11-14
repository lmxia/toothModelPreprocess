import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import gen_util as gu
from chamferdist import ChamferDistance
from math import inf

class TeethAlignmentModel(nn.Module):
    def __init__(self):
        super(TeethAlignmentModel, self).__init__()

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


def train(model, data_loader, optimizer, model_path, standard_path, epochs=50):
    model.train()
    chamfer_dist = ChamferDistance()
    best_val_loss = inf

    target_points = gu.load_and_sample_mesh(standard_path)
    target_points_batch = np.expand_dims(target_points, axis=0)
    target_vector = gu.compute_centroid_direction_vector(target_points_batch[:, :, :3])[0]
    for epoch in range(epochs):
        epoch_loss = 0
        for source, target in data_loader:
            # 获取当前批次大小
            batch_size = source.size(0)
            # 重复 target_vector
            repeated_target_vector = np.tile(target_vector, (batch_size, 1))
            optimizer.zero_grad()
            rot, trans = model(source, target)
            source_transformed = gu.apply_transform(source, rot, trans)
            loss = gu.compute_loss(chamfer_dist, source_transformed, target, repeated_target_vector)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_total_loss = epoch_loss / len(data_loader)
        gu.logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")
        if best_val_loss > avg_total_loss:
            best_val_loss = avg_total_loss
            # 保存模型
            gu.save_checkpoint(model, optimizer, epoch, avg_total_loss, filename=model_path)
            # torch.save(model.state_dict(), model_path)
            gu.logger.info(f"Model saved to {model_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Teeth alignment train entrypoint')
    parser.add_argument('--target', type=str, default="upper", help='upper or lower')
    args = parser.parse_args()
    if args.target == "upper":
        standard_path = "/data/shang.stl"
        model_path = '/data/teeth_alignment_upper_model.pth'
    else:
        standard_path = "/data/xia.stl"
        model_path = '/data/teeth_alignment_lower_model.pth'
    non_standard_path = "/data/non_standard"
    train_loader = gu.get_generator_set(non_standard_path, standard_path, args.target)
    model = TeethAlignmentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 尝试加载上次保存的检查点
    try:
        start_epoch, state_dict, optimizer_state_dict, best_loss = gu.load_checkpoint(model_path)
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        print(f"Checkpoint loaded, starting from epoch {start_epoch + 1} with loss {best_loss}")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
    # 训练模型
    train(model, train_loader, optimizer, model_path, standard_path)


if __name__ == '__main__':
    main()
