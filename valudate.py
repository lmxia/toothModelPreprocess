import torch
import numpy as np
import trimesh

from train import TeethAlignmentModel
from utils import gen_util as gu
from chamferdist import ChamferDistance
import os

def validate(model, source_path, target_path, output_path):
    chamfer_dist = ChamferDistance()
    # 加载源点云和目标点云
    source_points = gu.load_and_sample_mesh(source_path, 12000)  # 假设加载点云为 (N, 3) 形状
    target_points = gu.load_and_sample_mesh(target_path, 12000)  # 目标点云

    # 获取源点云的形状，目标点云的形状应当相同
    source_points_batch = np.expand_dims(source_points, axis=0)
    target_points_batch = np.expand_dims(target_points, axis=0)
    # 将点云转为 PyTorch 张量
    source_tensor = torch.tensor(source_points_batch, dtype=torch.float32)
    target_tensor = torch.tensor(target_points_batch, dtype=torch.float32)

    # 获取模型的预测（旋转、平移）
    model.eval()
    with torch.no_grad():
        rot, trans, source_features, target_features = model(source_tensor, target_tensor)
    # 将旋转与平移应用到源点云上
    source_transformed = gu.apply_transform(source_tensor, rot, trans)
    loss = gu.compute_loss(chamfer_dist, source_transformed, target_tensor,
                           target_tensor, source_features, target_features)

    print(f"loss is {loss}")

    mesh = trimesh.load(source_path)
    points = mesh.vertices
    # 检查法向量是否存在，如果不存在则计算法向量
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()

    normals = mesh.vertex_normals
    source_tensor = np.hstack((points, normals))
    source_points_batch = np.expand_dims(source_tensor, axis=0)
    source_tensor = torch.tensor(source_points_batch, dtype=torch.float32)

    # 将旋转与平移应用到源点云上
    source_transformed = gu.apply_transform(source_tensor, rot, trans)

    # 保存对齐后的点云
    aligned_source_points = source_transformed.squeeze().cpu().numpy()  # 转换为 (N, 6) 形状
    gu.save_mesh(output_path, aligned_source_points)  # 假设 save_mesh 函数可以保存为 .ply 或其他格式



    print(f"Aligned source point cloud saved to {output_path}")

def main():
    # 指定文件路径
    source_path = "/data/val_teeth/230401304/230401304-Upper.stl"  # 待对齐的源点云
    target_path = "/data/shang.stl"  # 目标点云，通常是标准模型
    output_path = "/data/aligned_pointcloud.ply"  # 保存对齐后点云的路径
    model_path = "/data/teeth_alignment_upper_model.pth"

    # 加载模型
    model = TeethAlignmentModel()
    try:
        _, state_dict, _, _ = gu.load_checkpoint(model_path)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"No model checkpoint found at {model_path}")
        return

    # 进行验证并保存对齐后的点云
    validate(model, source_path, target_path, output_path)

if __name__ == '__main__':
    main()
