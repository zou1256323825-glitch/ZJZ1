# @Time: 2023/8/13 9:54
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from function.arnold import *
import numpy as np

#
# image = Image.open(r'D:\python_project\sucess\python\data\train\009179.jpg')
# images = transforms.ToTensor()(image)
# images = images.unsqueeze(0)
# b, c, h, w = images.shape



def rgb2ycbcr(rgb_images):
    rgb_images = rgb_images * 255
    r, g, b = rgb_images[:, 0, :, :], rgb_images[:, 1, :, :], rgb_images[:, 2, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    ycrcb_images = torch.stack((y, cr, cb), dim=1)
    return ycrcb_images/255


def ycbcr2rgb(ycrcb_images):
    ycrcb_images = ycrcb_images * 255
    y, cr, cb = ycrcb_images[:, 0, :, :], ycrcb_images[:, 1, :, :] - 128, ycrcb_images[:, 2, :, :] - 128
    r = y + 1.402 * cr
    g = y - 0.34414 * cb - 0.71414 * cr
    b = y + 1.772 * cb
    rgb_images = torch.clip(torch.stack((r, g, b), dim=1), 0, 255)
    return rgb_images/255


# ycrcb_images = rgb2ycbcr(images.cuda())
# rgb_restored = ycbcr2rgb(ycrcb_images)
# y = rgb2ycbcr(rgb_restored)[:, 0:Wnet(SEP2Embedwater)+SEDecoder_COCO, :, :]
# plt.subplot(Wnet(SEP2Embedwater)+SEDecoder_COCO, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO)
# plt.imshow(np.transpose(images[0], (Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0)))
# plt.subplot(Wnet(SEP2Embedwater)+SEDecoder_COCO, 3, 2)
# plt.imshow(np.transpose(ycrcb_images[0][0:Wnet(SEP2Embedwater)+SEDecoder_COCO].cpu().numpy(), (Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0)))
# plt.subplot(Wnet(SEP2Embedwater)+SEDecoder_COCO, 3, 3)
# plt.imshow(np.transpose(y[0].cpu().numpy(), (Wnet(SEP2Embedwater)+SEDecoder_COCO, 2, 0)))
# plt.show()






















