import torch
import torch.nn as nn


class Rotate(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, angle):
        super(Rotate, self).__init__()
        angle_rad = torch.tensor(angle * (torch.pi / 180.0), dtype=torch.float32)
        angle_rad.expand_as(torch.zeros(16, 1))
        # 计算旋转矩阵
        self.rotation_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad), 0],
            [torch.sin(angle_rad), torch.cos(angle_rad), 0]
        ]).cuda()

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        rotated_images = torch.nn.functional.affine_grid(
            self.rotation_matrix.unsqueeze(0).repeat(noised_image.size(0), 1, 1),
            noised_image.size(), align_corners=False)

        # 双线性插值
        rotated_images = torch.nn.functional.grid_sample(noised_image, rotated_images, align_corners=False)
        noised_and_cover[0] = rotated_images

        return noised_and_cover


if __name__ == '__main__':
    a = torch.randn(16, 3, 128, 128).cuda()
    model = Rotate(45.).cuda()
    model([a, a])