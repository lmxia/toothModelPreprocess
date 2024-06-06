import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
import trimesh
import random


class TeethDataset(Dataset):
    def __init__(self, non_standard_path, standard_path, num_points=24000):
        stl_path_ls = []
        for dir_path in [x[0] for x in os.walk(non_standard_path)][1:]:
            stl_path_ls += glob(os.path.join(dir_path, "*Lower-PreparationScan_transformed.stl"))

        self.mesh_paths = stl_path_ls
        self.standard_path = standard_path
        self.standard_cloud, _ = load_and_sample_mesh(self.standard_path, num_points)
        self.num_points = num_points

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        source, _ = load_and_sample_mesh(self.mesh_paths[idx], self.num_points)
        target = self.standard_cloud
        return torch.tensor(source, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_and_sample_mesh(path, num_points=24000):
    mesh = trimesh.load(path)
    faces = mesh.faces

    points = mesh.vertices

    # Normalize the point cloud (optional, but often beneficial)
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    # Sample points
    if len(points) > num_points:
        points, faces = downsample(points, faces, num_points)
    elif len(points) < num_points:
        points = upsample(points, num_points)

    return points, faces


# 特殊处理，face会欠缺。
def downsample(points, faces, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=False)

    faces = faces[np.all(np.isin(faces, indices), axis=1)]
    new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    faces = np.vectorize(new_indices.get)(faces)

    return points[indices], faces


def upsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=True)
    return points[indices]


def get_generator_set(non_standard_path, standard_path, batch_size=32, num_points=24000):
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