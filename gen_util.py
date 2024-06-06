import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
import trimesh
import random


class TeethDataset(Dataset):
    def __init__(self, non_standard_path, standard_path):
        stl_path_ls = []
        for dir_path in [x[0] for x in os.walk(non_standard_path)][1:]:
            stl_path_ls += glob(os.path.join(dir_path, "*Lower-PreparationScan_transformed.stl"))

        self.mesh_paths = stl_path_ls
        self.standard_path = standard_path
        self.standard_cloud = load_and_sample_mesh(self.standard_path)

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        source = load_and_sample_mesh(self.mesh_paths[idx])
        target = self.standard_cloud
        return torch.tensor(source, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_and_sample_mesh(path):
    mesh = trimesh.load(path)

    points = mesh.vertices

    # Normalize the point cloud (optional, but often beneficial)
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    return points


def get_generator_set(non_standard_path, standard_path, batch_size=32):
    point_loader = DataLoader(
        TeethDataset(
            non_standard_path,
            standard_path
        ),
        shuffle=True,
        batch_size=batch_size,
    )

    return point_loader