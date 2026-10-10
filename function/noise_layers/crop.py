import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(height_ratio_range * image_height)
    remaining_width = int(width_ratio_range * image_width)

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, ratio):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = ratio
        self.width_ratio_range = ratio


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)
        H = noised_image.shape[-1]
        noised_and_cover_step = noised_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()
        noised_and_cover[0] = F.interpolate(
            noised_and_cover_step,
            size=(H, H),
            mode='nearest')

        return noised_and_cover[0]


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
    image = torch.tensor(image / 255, dtype=torch.float32)
    image_tensor = np.transpose(image, (2, 0, 1)).unsqueeze(0)  # BCHW
    image_attack = Crop(0.5, 0.5)(image_tensor.cuda())
    image_numpy = np.transpose(image_attack.squeeze(0).cpu(), (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image_numpy)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()

