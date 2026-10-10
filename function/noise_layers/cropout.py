import torch
import torch.nn as nn
import numpy as np
from function.noise_layers.crop import get_random_rectangle_inside


class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, ratio):
        super(Cropout, self).__init__()
        self.height_ratio_range = ratio
        self.width_ratio_range = ratio

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[1]
        cover_image = noised_and_cover[0]
        assert noised_image.shape == cover_image.shape

        cropout_mask = torch.zeros_like(noised_image)
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
                                                                     height_ratio_range=self.height_ratio_range,
                                                                     width_ratio_range=self.width_ratio_range)
        cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1

        noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
        return  noised_and_cover


if __name__ == '__main__':
    import os
    import torchvision.utils as vutils

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked, _ = Cropout(0.8)([conv, input])
    # p = make_dot(attacked)
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np

    image = cv2.imread(r"D:\Python_Dataset\COCO\train\000000072422.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image / 255, dtype=torch.float32)
    image_tensor = np.transpose(image, (2, 0, 1)).unsqueeze(0)  # BCHW
    zeros_tensor = torch.zeros_like(image_tensor).cuda()
    print(image_tensor.shape)
    image_attack = Cropout(0.6)([image_tensor.cuda(), zeros_tensor])

    # vutils.save_image(image_attack[0], r'D:\\Python_code\\zjz_work2\\Mycode\\model_testSavesImages\\uuu.png', nrow=Wnet(SEP2Embedwater)+SEDecoder_COCO,
    #                   padding=Wnet(SEP2Embedwater)+SEDecoder_COCO, normalize=False)
    image_numpy = np.transpose(image_attack[0].squeeze(0).cpu(), (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()