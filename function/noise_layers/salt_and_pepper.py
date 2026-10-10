import torch
import torch.nn as nn
import numpy as np
import random


class Salt_and_Pepper(nn.Module):
    def __init__(self, ratio):
        super(Salt_and_Pepper, self).__init__()
        ratio = ratio
        self.ratio=float(ratio)
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encoded_image=noise_and_cover[0].clone()
        B,C,H,W=encoded_image.size()
        mask = np.random.choice((0, 1, 2), size=(B,1,H,W), p=[self.ratio, self.ratio, 1 - 2 * self.ratio])
        mask=torch.tensor(np.repeat(mask,C, axis=1),device=self.device)
        encoded_image[mask==0]=-1
        encoded_image[mask==1]=1
        noise_and_cover[0]=encoded_image
        return noise_and_cover


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
    image = cv2.imread(r"D:\python_project\sucess\python\data1\train\009515.jpg")
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    image_tensor = torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0)  # BCHW
    image_attack = Salt_and_Pepper(0.05)(image_tensor.cuda())
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.imshow(image_numpy)
    plt.show()


