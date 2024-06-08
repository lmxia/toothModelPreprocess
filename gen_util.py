import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
import trimesh
import random


class TeethDataset(Dataset):
    def __init__(self, non_standard_path, standard_path, num_points=48000):
        stl_path_ls = []
        for dir_path in [x[0] for x in os.walk(non_standard_path)][1:]:
            stl_path_ls += glob(os.path.join(dir_path, "*Lower-PreparationScan_transformed.stl"))

        self.mesh_paths = stl_path_ls
        self.standard_path = standard_path
        self.standard_cloud = load_and_sample_mesh(self.standard_path, num_points)
        self.num_points = num_points

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        source = load_and_sample_mesh(self.mesh_paths[idx], self.num_points)
        target = self.standard_cloud
        return torch.tensor(source, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_and_sample_mesh(path, num_points=48000):
    mesh = trimesh.load(path)

    points = mesh.vertices

    # Normalize the point cloud (optional, but often beneficial)
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    # Sample points
    if len(points) > num_points:
        points = downsample(points, num_points)
    elif len(points) < num_points:
        points = upsample(points, num_points)

    return points


# 特殊处理，face会欠缺。
def downsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    return points[indices]


def upsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=True)
    return points[indices]


def get_generator_set(non_standard_path, standard_path, batch_size=8, num_points=48000):
    point_loader = DataLoader(
        TeethDataset(
            non_standard_path,
            standard_path,
            num_points=num_points
        ),
        shuffle=True,
        batch_size=batch_size,
    )

    return point_loader


def apply_transform(points, rotation, translation):
    batch_size = points.size(0)
    rotation_matrix = quat_to_rotmat(rotation)
    points_transposed = points.transpose(2, 1)  # (B, 3, 48000)
    # Perform batch matrix multiplication
    points_rotated = torch.bmm(rotation_matrix, points_transposed)  # (B, 3, 48000)
    points_rotated = points_rotated.transpose(2, 1)  # (B, 48000, 3)

    points_transformed = points_rotated + translation.unsqueeze(1)  # (B, 48000, 3)
    return points_transformed


def quat_to_rotmat(quat):
    # Convert quaternion to rotation matrix
    batch_size = quat.size(0)
    norm_quat = quat / quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = batch_size

    rotmat = torch.zeros((B, 3, 3)).to(quat.device)
    rotmat[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotmat[:, 0, 1] = 2 * (x * y - z * w)
    rotmat[:, 0, 2] = 2 * (x * z + y * w)
    rotmat[:, 1, 0] = 2 * (x * y + z * w)
    rotmat[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotmat[:, 1, 2] = 2 * (y * z - x * w)
    rotmat[:, 2, 0] = 2 * (x * z - y * w)
    rotmat[:, 2, 1] = 2 * (y * z + x * w)
    rotmat[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return rotmat
