import torch
import torch.nn.functional as F


def block_partition_and_concat(tensor, block_size):
    B, C, H, W = tensor.shape
    new_C = C * (block_size ** 2)
    new_H, new_W = H // block_size, W // block_size

    # 先将 tensor 形状调整为 (B, C, new_H, block_size, new_W, block_size)
    tensor = tensor.view(B, C, new_H, block_size, new_W, block_size)

    # 调整维度顺序为 (B, C, block_size, block_size, new_H, new_W)
    tensor = tensor.permute(0, 1, 3, 5, 2, 4).contiguous()

    # 最后调整形状为 (B, new_C, new_H, new_W)
    tensor = tensor.view(B, new_C, new_H, new_W)

    return tensor


# 创建一个形状为 (16, 3, 128, 128) 的随机张量
tensor = torch.rand(16, 3, 128, 128)

# 按块大小 2 进行分块并拼接
output_tensor = block_partition_and_concat(tensor, 4)

print(output_tensor.shape)  # 应输出: (16, 12, 64, 64)
