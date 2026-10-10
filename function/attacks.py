# @Time: 2023/7/16 15:25
import numpy as np
# import cv2
import torch
import torchvision.transforms.functional as TF
import random
# import pywt
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from function.noise_layers import Gaussian_noise, salt_and_pepper, gaussian, jpeg, identity, crop1, resize1, dropout, \
    grid_crop, Median_filter, cropout, roration
from function import Combine_attacks


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def attack(img_batch, hide_img, cover_img, ugpu=True):
    """
    对输入的图像进行攻击处理,对一个batch的图像随机进行攻击
    攻击方式： 高斯噪声、椒盐噪声、原图、缩放、剪切
    :param image_batch: 输入图像批次，形状为 (batch_size, channels, height, width)，范围是 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO.0]
    :return: 变换后的图像批次
    """
    sigma_ratio = np.round(random_float(0.5, 2.5), 1)  # gaussian blur
    ratio = np.round(random_float(0.3, 0.8), 1)  # random crop
    cropout_ratio = np.round(random_float(0.2, 0.8), 1)  # cropout
    dropout_ratio = np.round(random_float(0.3, 1), 1)  # dropout
    mf_ratio = random.choice([3, 5, 7])
    grid_crop_ratio = np.round(random_float(0.3, 0.8), 1)  # grid_crop
    resize_range = np.round(random_float(0.3, 1.8), 1)  # resize
    sigma = np.round(random_float(0.01, 0.25), 2)  # gaussian noise
    ratio_sp = np.round(random_float(0.01, 0.2), 2)  # salt and pepper
    Q_ratio = np.round(random_float(30, 70), -1)  # JPEG quantum
    Rotate_Angle = np.round(random_float(0, 360), -1)  # Rotation
    batch = img_batch.shape[0]
    attack_func = [
        gaussian.Gaussian_blur(sigma_ratio, 3),
        crop1.Crop(ratio),
        # cropout.Cropout(cropout_ratio),
        dropout.Dropout(dropout_ratio),
        Median_filter.Median_filter(mf_ratio),
        grid_crop.grid_crop(grid_crop_ratio),
        resize1.Resize(resize_range),
        Gaussian_noise.Gaussian_Noise(0, sigma),
        salt_and_pepper.Salt_and_Pepper(ratio_sp),
        jpeg.Jpeg(Q_ratio),
        # roration.Rotate(Rotate_Angle),
        identity.Identity()
    ]
    # attack_func = [jpeg_compression.JpegCompression('cuda')]
    # kernel = random.choice([3, 5])  # gaussian blur
    # ratio = np.round(random_float(0.3, 0.71), 2)  # random crop
    # dropout_ratio = np.round(random_float(0.5, Wnet(SEP2Embedwater)+SEDecoder_COCO), 2)  # dropout
    # resize_range = np.round(random_float(0.3, Wnet(SEP2Embedwater)+SEDecoder_COCO), 2)  # resize
    # sigma = np.round(random_float(0.01, 0.25), 3)  # gaussian noise
    # ratio_sp = np.round(random_float(0.01, 0.2), 3)  # salt and pepper
    # Q_ratio = np.round(random_float(0, 50), 0)  # JPEG quantum
    # batch = img_batch.shape[0]
    # attack_func = [
    #     gaussian.Gaussian_blur(2, kernel),
    #     crop1.Crop(ratio),
    #     dropout.Dropout(dropout_ratio),
    #     resize1.Resize(resize_range),
    #     Gaussian_noise.Gaussian_Noise(0, sigma),
    #     salt_and_pepper.Salt_and_Pepper(ratio_sp),
    #     # jpeg.Jpeg(Q_ratio),
    #     # jpeg_compression.JpegCompression('cuda'),
    #     identity.Identity()
    # ]
    attack_imgs = []
    attack_Hide_imgs = []
    choices = random.choices(range(len(attack_func)), k=batch)
    # random_noise_layer = np.random.choice(attack_func, Wnet(SEP2Embedwater)+SEDecoder_COCO)[0]
    for idx, i in enumerate(choices):
        attack_img = attack_func[i]([img_batch[idx:idx + 1], hide_img[idx:idx + 1], cover_img[idx:idx + 1]])
        attack_imgs.append(attack_img[0])
        attack_Hide_imgs.append(attack_img[1])
    #
    final_image = torch.cat(attack_imgs, dim=0)
    final_hide_image = torch.cat(attack_Hide_imgs, dim=0)
    # final_image = random_noise_layer([img_batch, cover_img])
    return final_image, final_hide_image



def attack_for_test(img_batch, cover_img, attack_funcs, ugpu=True):
    """
    对输入的图像进行攻击处理,对一个batch的图像随机进行攻击
    攻击方式： 高斯噪声、椒盐噪声、原图、缩放、剪切
    :param image_batch: 输入图像批次，形状为 (batch_size, channels, height, width)，范围是 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO.0]
    :return: 变换后的图像批次
    """

    attack_img = attack_funcs([img_batch, cover_img])
    return attack_img
    # """
    # 对输入的图像进行攻击处理,对一个batch的图像随机进行攻击
    # 攻击方式： 高斯噪声、椒盐噪声、原图、缩放、剪切
    # :param image_batch: 输入图像批次，形状为 (batch_size, channels, height, width)，范围是 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO.0]
    # :return: 变换后的图像批次
    # """
    # batch = img_batch.shape[0]
    # if rat > 0:
    #     attack_func = [
    #         # [],
    #         jpeg.Jpeg(rat),
    #         gaussian.Gaussian_blur(rat, 3),
    #         crop1.Crop(rat),
    #         dropout.Dropout(rat),
    #         Median_filter.Median_filter(rat),
    #         grid_crop.grid_crop(rat),
    #         resize1.Resize(rat),
    #         Gaussian_noise.Gaussian_Noise(0, rat),
    #         salt_and_pepper.Salt_and_Pepper(rat),
    #     ]
    # elif rat < 0:
    #     attack_func = Combine_attacks.Noiser(rat)
    # else:
    #     attack_func = [
    #         # [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         [],
    #         identity.Identity()
    #     ]
    # # attack_func = [jpeg_compression.JpegCompression('cuda')]
    # attack_imgs = []
    #
    # for idx in range(batch):
    #     # if number == 2:
    #     #     attack_img = attack_func[number]([img_batch[idx:idx+Wnet(SEP2Embedwater)+SEDecoder_COCO], cover_img[idx:idx+Wnet(SEP2Embedwater)+SEDecoder_COCO]])
    #     # else:
    #     if rat>=0:
    #         attack_img = attack_func[number]([img_batch[idx:idx + Wnet(SEP2Embedwater)+SEDecoder_COCO], cover_img[idx:idx + Wnet(SEP2Embedwater)+SEDecoder_COCO]])
    #     else:
    #         attack_img = attack_func([img_batch[idx:idx + Wnet(SEP2Embedwater)+SEDecoder_COCO], cover_img[idx:idx + Wnet(SEP2Embedwater)+SEDecoder_COCO]])
    #     attack_imgs.append(attack_img[0])
    #
    # final_image = torch.cat(attack_imgs, dim=0)
    # return final_image


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch.nn as nn
    from torchviz import make_dot

    input = torch.randn((8, 3, 128, 128), requires_grad=True).cuda()
    conv = nn.Conv2d(3, 3, 3, 1, 1).cuda()(input)
    attacked = attack(conv)
    p = make_dot(attacked)
    p.view()
