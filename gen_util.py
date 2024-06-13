import torch
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from glob import glob
import trimesh
import random
import logging
import open3d as o3d

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeethDataset(Dataset):
    def __init__(self, non_standard_path, standard_path, num_points=24000):
        stl_path_ls = []
        for dir_path in [x[0] for x in os.walk(non_standard_path)][1:]:
            stl_path_ls += glob(os.path.join(dir_path, "*Lower-PreparationScan_transformed.stl"))

        self.mesh_paths = stl_path_ls
        self.standard_path = standard_path
        vertices = load_and_sample_mesh(self.standard_path, num_points)
        self.standard_cloud = vertices
        self.num_points = num_points

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        vertices = load_and_sample_mesh(self.mesh_paths[idx], self.num_points)
        target = self.standard_cloud
        return torch.tensor(vertices, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def load_and_sample_mesh(path, num_points=24000):
    # 加载网格
    mesh = trimesh.load(path)
    points = mesh.vertices

    # 检查法向量是否存在，如果不存在则计算法向量
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()

    normals = mesh.vertex_normals

    # 归一化点云
    points = points - np.mean(points, axis=0)
    points = points / np.max(np.linalg.norm(points, axis=1))

    # 下采样或上采样点云
    if len(points) > num_points:
        sampled_indices = np.random.choice(len(points), num_points, replace=False)
        points = points[sampled_indices]
        normals = normals[sampled_indices]
    elif len(points) < num_points:
        sampled_indices = np.random.choice(len(points), num_points, replace=True)
        points = points[sampled_indices]
        normals = normals[sampled_indices]

    return np.hstack((points, normals))


# 特殊处理，face会欠缺。
def downsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    return points[indices]


def upsample(points, num_points):
    indices = np.random.choice(points.shape[0], num_points, replace=True)
    return points[indices]


def get_generator_set(non_standard_path, standard_path, batch_size=2, num_points=24000):
    point_loader = DataLoader(
        TeethDataset(
            non_standard_path,
            standard_path,
            num_points=num_points
        ),
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )

    return point_loader


def apply_transform(points, rotation, translation):
    """
    Apply rotation and translation to a batch of point clouds.
    Args:
        points (Tensor): Point cloud, shape (B, num_points, 6) (coordinates and normals)
        rotation (Tensor): Quaternion, shape (B, 4)
        translation (Tensor): Translation vector, shape (B, 3)
    Returns:
        Tensor: Transformed point cloud, shape (B, num_points, 6)
    """
    rotation_matrix = quat_to_rotmat(rotation)

    # Separate coordinates and normals
    coords = points[:, :, :3]  # (B, num_points, 3)
    normals = points[:, :, 3:]  # (B, num_points, 3)

    coords_transposed = coords.transpose(2, 1)  # (B, 3, num_points)
    normals_transposed = normals.transpose(2, 1)  # (B, 3, num_points)

    # Perform batch matrix multiplication
    coords_rotated = torch.bmm(rotation_matrix, coords_transposed)  # (B, 3, num_points)
    normals_rotated = torch.bmm(rotation_matrix, normals_transposed)  # (B, 3, num_points)

    coords_rotated = coords_rotated.transpose(2, 1)  # (B, num_points, 3)
    normals_rotated = normals_rotated.transpose(2, 1)  # (B, num_points, 3)

    coords_transformed = coords_rotated + translation.unsqueeze(1)  # (B, num_points, 3)

    # Concatenate transformed coordinates and rotated normals
    points_transformed = torch.cat((coords_transformed, normals_rotated), dim=2)  # (B, num_points, 6)

    return points_transformed


def normal_consistency_o3d(normals1, normals2, pc1, pc2, k=10):
    def compute_normal_consistency(pcd_tree, pcd_source, pcd_target):
        normal_consistency = []
        for i in range(len(pcd_source.points)):
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd_source.points[i], k)
            max_dot = max(np.dot(pcd_target.normals[j], pcd_source.normals[i]) for j in idx)
            normal_consistency.append(max_dot)
        return normal_consistency

    batch_size = pc1.shape[0]
    total_consistency = 0

    for b in range(batch_size):
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1[b])
        pcd1.normals = o3d.utility.Vector3dVector(normals1[b])

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2[b])
        pcd2.normals = o3d.utility.Vector3dVector(normals2[b])

        pcd_tree1 = o3d.geometry.KDTreeFlann(pcd1)
        pcd_tree2 = o3d.geometry.KDTreeFlann(pcd2)

        normal_consistency1 = compute_normal_consistency(pcd_tree2, pcd1, pcd2)
        normal_consistency2 = compute_normal_consistency(pcd_tree1, pcd2, pcd1)

        consistency = 1 - (np.mean(normal_consistency1) + np.mean(normal_consistency2)) / 2
        total_consistency += consistency

    return total_consistency / batch_size


def compute_loss(chamfer_dist, source_transformed, target):
    # Chamfer distance loss
    loss_chamfer = chamfer_dist(source_transformed[:, :, :3], target[:, :, :3])

    # Normal consistency loss
    normal_loss = normal_consistency_o3d(
        source_transformed[:, :, 3:].detach().cpu().numpy(),
        target[:, :, 3:].detach().cpu().numpy(),
        source_transformed[:, :, :3].detach().cpu().numpy(),
        target[:, :, :3].detach().cpu().numpy()
    )

    centroid_source = torch.mean(source_transformed[:, :, :3], dim=1)
    centroid_target = torch.mean(target[:, :, :3], dim=1)

    # 计算质心之间的欧几里得距离
    centroid_distance = torch.norm(centroid_source - centroid_target, dim=1)
    centroid_loss = centroid_distance.mean()

    logger.info(f'chamfer loss is {loss_chamfer} and normal loss is {normal_loss} '
                f'and centroid_loss is {centroid_loss}')
    # Combine losses
    total_loss = loss_chamfer + normal_loss * 500 + centroid_loss * 300
    return total_loss


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
