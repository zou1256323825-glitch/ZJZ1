import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.resize_ratio = resize_ratio
        self.interpolation_method = interpolation_method

    def forward(self, noised_and_cover):
        resize_ratio = self.resize_ratio
        resize_ratio = np.round(resize_ratio, 1)
        # print(resize_ratio)
        noised_image = noised_and_cover[0].clone()
        H = noised_image.shape[-1]
        noised_image_step = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        noised_and_cover[0] = F.interpolate(
            noised_image_step,
            size=(H, H),
            mode=self.interpolation_method)

        return noised_and_cover


if __name__ == '__main__':
    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked = Gaussian_Noise(0, Wnet(SEP2Embedwater)+SEDecoder_COCO)(conv)
    # p = make_dot(attacked)
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    image = cv2.imread(r"D:\python_project\sucess\python\data1\train\009515.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0)  # BCHW
    image_attack = Resize((0.3, 0.8))(image_tensor.cuda())
    print(image_attack.shape)
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.imshow(image_numpy)
    plt.show()
