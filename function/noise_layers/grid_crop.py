#coding=utf-8
import numpy as np
import torch.nn as nn
import torch


class grid_crop(nn.Module):
    def __init__(self,rate):
        super(grid_crop, self).__init__()
        self.rate=float(rate)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noised_and_cover):
        encoded_image=noised_and_cover[0].clone()
        block_size = 8
        block_switch = np.random.uniform(0.0, high=1.0, size=[encoded_image.shape[2] // block_size, encoded_image.shape[3] // block_size])
        block_switch = block_switch < self.rate
        # new_img = img.copy()
        for i in range(block_switch.shape[0]):
            for j in range(block_switch.shape[1]):
                if block_switch[i, j] == 0:
                    continue
                encoded_image[:,:,block_size * i:block_size + block_size * i, block_size * j:block_size + block_size * j] = -1
        noised_and_cover[0]=encoded_image
        return noised_and_cover


if __name__ == '__main__':
    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked = JpegCompression('cuda')(conv)
    # p = make_dot(attacked)
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    image = cv2.imread(r"D:\python_project\sucess\python\data1\train\009515.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.tensor(image/255, dtype=torch.float32)
    image_tensor = np.transpose(image, (2, 0, 1)).unsqueeze(0)  # BCHW
    image_attack = grid_crop(0.7)(image_tensor.cuda())
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()