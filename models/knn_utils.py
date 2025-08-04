import torch
import torch.nn.functional as F

def knn_point_efficient(k, xyz, new_xyz):
    """
    Input:
        k: number of nearest neighbors
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, k]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    # 使用批量矩阵乘法计算距离，更高效
    # 展开公式: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    xyz_norm = (xyz ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    new_xyz_norm = (new_xyz ** 2).sum(dim=-1, keepdim=True)  # [B, S, 1]
    
    # 计算点积
    dots = torch.bmm(new_xyz, xyz.transpose(1, 2))  # [B, S, N]
    
    # 计算距离
    dists = new_xyz_norm + xyz_norm.transpose(1, 2) - 2 * dots  # [B, S, N]
    
    # 找到k个最近邻
    _, group_idx = torch.topk(dists, k, dim=-1, largest=False, sorted=False)  # [B, S, k]
    
    return None, group_idx


def knn_point_heap(k, xyz, new_xyz):
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device
    
    # 如果点数太多，分块处理
    block_size = 2048
    if S > block_size:
        # 分块处理查询点
        all_idx = []
        for i in range(0, S, block_size):
            end_i = min(i + block_size, S)
            block_new_xyz = new_xyz[:, i:end_i, :]
            _, block_idx = knn_point_efficient(k, xyz, block_new_xyz)
            all_idx.append(block_idx)
        group_idx = torch.cat(all_idx, dim=1)
    else:
        _, group_idx = knn_point_efficient(k, xyz, new_xyz)
    
    return None, group_idx

knn_point = knn_point_efficient
