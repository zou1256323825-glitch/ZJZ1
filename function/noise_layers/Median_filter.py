import PIL
import torch
import torch.nn as nn
import kornia.filters as filters


class Median_filter(nn.Module):
    def __init__(self,kernel_size):
        super(Median_filter, self).__init__()
        self.kernel_size=int(kernel_size)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encoded=noise_and_cover[0].clone()
        noise_and_cover[0]=filters.MedianBlur(kernel_size=(self.kernel_size,self.kernel_size))(encoded)
        # noise_and_cover[0]=filters.motion.MotionBlur(ksize=7,angle=35.,direction=-0.5,border_type='reflect')(encoded)
        return noise_and_cover


if __name__ == '__main__':
    """测试一：计算图,且使用能够适应梯度计算"""
    # from torch.autograd import Variable
    # import torch.nn as nn
    # from torchviz import make_dot
    #
    # input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    # conv = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO).cuda()(input)
    # attacked = Median_filter(3)(conv)
    # p = make_dot(attacked)
    # p.view()
    """测试二：观察图像输出质量"""
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    image = cv2.imread(r"D:\Python_Dataset\COCO\train\000000072425.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0)  # BCHW
    image_attack = Median_filter(3)([image_tensor.cuda(), ])
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()
