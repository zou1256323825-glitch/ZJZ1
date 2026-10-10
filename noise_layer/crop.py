import torch.nn as nn
import numpy as np
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
    #np.rint是四舍五入取整，height_ratio_range[0]是为长变化的下限，[Wnet(SEP2Embedwater)+SEDecoder_COCO]是为上限，
    # 所以是得到了一个缩小了的图像的长宽，在原图像的范围内
    # remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[Wnet(SEP2Embedwater)+SEDecoder_COCO]) * image_height))
    # remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))
    remaining_height = int(np.rint(height_ratio_range * image_height))
    remaining_width = int(np.rint(width_ratio_range * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:#np.random.randint()返回一个随机整数，包括低范围，不包括高范围
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
    def __init__(self, ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = ratio_range
        self.width_ratio_range = ratio_range


    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)
        H = noised_image.shape[-1]
        h_start, h_end, w_start, w_end = get_random_rectangle_inside(noised_image, self.height_ratio_range, self.width_ratio_range)

        noised_image = noised_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()
        left = w_start
        right = H-w_end
        hight = H-h_end
        down = h_start
        noised_and_cover[0] = F.pad(noised_image, (left, right, hight, down), mode='constant')

        return noised_and_cover


if __name__ == '__main__':
    import torch
    import cv2
    from torchvision import transforms
    import os
    import torchviz

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # image to tensor
    image = cv2.imread(r"D:\Python_Dataset\COCO\train\000000072711.jpg")  # (H, W, C)
    image = cv2.resize(image, (640, 640))
    image_tensor = transforms.ToTensor()(image)  # (C, H, W)
    image_tensorBatch = image_tensor.unsqueeze(0)  # (B, C, H, W)
    # image_tensorBatch = nn.Conv2d(3, 3, 3, Wnet(SEP2Embedwater)+SEDecoder_COCO, Wnet(SEP2Embedwater)+SEDecoder_COCO)(image_tensorBatch)
    attack = Crop(0.4)

    attackimg= attack([image_tensorBatch, 0])
    import numpy as np
    image_numpy = np.transpose(attackimg[0].squeeze(0).numpy(), (1, 2, 0))  # (H, W, C)

    # show image
    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(image_numpy)
    plt.show()




