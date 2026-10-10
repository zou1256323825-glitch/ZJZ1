import torch
import torch.nn as nn
import numpy as np


class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        self.ratio = keep_ratio_range

    def forward(self, noised_and_cover):

        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        mask_percent = self.ratio  # random number in [min,max]
        mask = np.random.choice([0.0, 1.0], noised_image.shape[1:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=noised_image.device, dtype=torch.float)
        # mask_tensor.unsqueeze_(0)
        # mask_tensor.unsqueeze_(0)
        mask_tensor = mask_tensor.expand_as(noised_image)
        noised_image = noised_image * mask_tensor + cover_image * (1 - mask_tensor)
        noised_and_cover[0] = noised_image
        return noised_and_cover


if __name__ == '__main__':
    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked = Dropout((0.Wnet(SEP2Embedwater)+SEDecoder_COCO, 0.5))(conv)
    # p = make_dot(attacked[0])
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    image = cv2.imread(r"D:\python_project\sucess\python\data1\train\009515.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0)  # BCHW
    image1 = cv2.imread(r"D:\python_project\sucess\python\data1\train\021374.jpg")
    image1 = cv2.resize(image1, (256, 256))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = image1 / 255
    image1_tensor = torch.tensor(np.transpose(image1, (2, 0, 1))).unsqueeze(0)  # BCHW
    ratio = np.random.uniform(0.3, 0.7)
    image_attack = Dropout(ratio)((image_tensor.cuda(), image1_tensor.cuda()))
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.subplot(1, 3, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.subplot(1, 3, 3)
    plt.imshow(image1)
    plt.show()
