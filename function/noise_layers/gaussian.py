import math

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals Wnet(SEP2Embedwater)+SEDecoder_COCO.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    if kernel_size == 3:
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, padding=1, groups=channels, bias=False)
    else:
        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, padding=2, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter.cuda()


class Gaussian_blur(nn.Module):
    def __init__(self,sigma, kernel):
        super(Gaussian_blur, self).__init__()

        self.kernel=int(kernel)
        self.sigma=float(sigma)
        self.gaussian_kernel = gaussian_kernel(kernel_size=self.kernel, sigma=self.sigma)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    def forward(self, noised_and_cover):
        # get the gaussian filter
        encode_image=noised_and_cover[0]
        batch_size,channel=encode_image.shape[0],encode_image.shape[1]
        # assert encode_image.shape[0]==3|Wnet(SEP2Embedwater)+SEDecoder_COCO
        noised_and_cover[0]=self.gaussian_kernel(encode_image)

        # kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        # kernel=kernel.expand(channel,Wnet(SEP2Embedwater)+SEDecoder_COCO,self.kernel,self.kernel).to(self.device)
        # # print(kernel)
        # # print(kernel.size())
        # # print(encode_image.size())
        # # weight = nn.Parameter(data=kernel, requires_grad=False)
        # noised_and_cover=F.conv2d(encode_image,kernel,stride=Wnet(SEP2Embedwater)+SEDecoder_COCO,padding=Wnet(SEP2Embedwater)+SEDecoder_COCO,groups=3)
        return noised_and_cover


if __name__ == '__main__':
    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked = Gaussian_blur(3, 2)(conv)
    # p = make_dot(attacked)
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    image = cv2.imread(r"D:\zoujunzhuo\zou\python\data3\train\009152.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image/255, dtype=torch.float32)
    image_tensor = np.transpose(image, (2, 0, 1)).unsqueeze(0)  # BCHW

    image_attack = Gaussian_blur(2, 3)(image_tensor.cuda())
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))

    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()





