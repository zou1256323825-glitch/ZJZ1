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
        noised_image = noised_and_cover
        rotated_images = torch.nn.functional.affine_grid(
            self.rotation_matrix.unsqueeze(0).repeat(noised_image.size(0), 1, 1),
            noised_image.size(), align_corners=False)

        # 双线性插值
        rotated_images = torch.nn.functional.grid_sample(noised_image, rotated_images, align_corners=False)
        return rotated_images


if __name__ == '__main__':
    import torch
    import cv2
    from torchvision import transforms
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # image to tensor
    image = cv2.imread(r'D:\PaperWorks\Paperwork4\TextureFind\000000073812.jpg')  # (H, W, C)
    image = cv2.resize(image, (128, 128))
    image_tensor = transforms.ToTensor()(image)  # (C, H, W)
    image_tensorBatch = image_tensor.unsqueeze(0)  # (B, C, H, W)
    model = Rotate(45.).cuda()
    print(image_tensorBatch.shape)
    image_tensorBatch = model([image_tensorBatch.cuda(), image_tensorBatch.cuda()])[0].cpu()
    # tensor to image
    import numpy as np

    image_numpy = np.transpose(image_tensorBatch.squeeze(0).numpy(), (1, 2, 0))  # (H, W, C)

    # show image
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_numpy)
    plt.show()

