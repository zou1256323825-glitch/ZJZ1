# @Time: 2023/8/12 14:48
import torch
import cv2
from torchvision import transforms
from PIL import Image

# image = Image.open(r'D:\python_project\sucess\python\data\train\009179.jpg')
# images = transforms.ToTensor()(image)
# images = images.unsqueeze(0)


def SVD(image_batch):

    # 将图像 reshape 成 (b, c, h * w) 的形状
    b, c, h, w = image_batch.shape
    images_reshaped = image_batch.view(b, c, -1)

    # 对每个图像执行奇异值分解
    u, s, v = torch.svd(images_reshaped)

    # u, s, v 分别是奇异值分解的结果，形状为 (b, c, h * w)，(b, c, h * w)，(b, c, h * w)
    return u, s, v


def iSVD(u, s, v):
    b, c, square = u.shape
    h, w = torch.sqrt(square)
    # 将奇异值转换为对角矩阵
    s_diag = torch.diag(s)

    # 将奇异值转换为对角矩阵
    s_diag = torch.diag_embed(s)

    # 重建图像数据
    reconstructed_images = torch.matmul(u, torch.matmul(s_diag, v.permute(0, 2, 1)))

    # 将图像数据 reshape 成原始形状
    reconstructed_images = reconstructed_images.view(b, c, h, w)

    # 将图像数据转换为范围在 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO] 的数据
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)

    return reconstructed_images
    # 显示重建后的图像（假设b=Wnet(SEP2Embedwater)+SEDecoder_COCO）
    # reconstructed_image = reconstructed_images[0].permute(Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0).cpu().numpy()
    # reconstructed_image_pil = Image.fromarray((reconstructed_image * 255).astype('uint8'))
    # image.show()
    # reconstructed_image_pil.show()



















