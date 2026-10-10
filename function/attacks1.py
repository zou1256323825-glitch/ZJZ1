# @Time: 2023/7/16 15:25
import numpy as np
import torch
import random
from noise_layers import Gaussian_noise, salt_and_pepper, gaussian, identity, crop1, resize1, jpeg, dropout


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def attack(img_batch, cover_img, ugpu=True):
    """
    对输入的图像进行攻击处理,对一个batch的图像随机进行攻击
    攻击方式： 高斯噪声、椒盐噪声、原图、缩放、剪切
    :param image_batch: 输入图像批次，形状为 (batch_size, channels, height, width)，范围是 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO.0]
    :return: 变换后的图像批次
    """
    kernel = random.choice([3, 5])  # gaussian blur
    ratio = np.round(random_float(0.3, 0.71), 2)  # random crop
    dropout_ratio = np.round(random_float(0.5, 1), 2)  # dropout
    resize_range = np.round(random_float(0.3, 1), 2)  # resize
    sigma = np.round(random_float(0.01, 0.25), 3)  # gaussian noise
    ratio_sp = np.round(random_float(0.01, 0.2), 3)  # salt and pepper
    Q_ratio = np.round(random_float(20, 70), -1)  # JPEG quantum
    batch = img_batch.shape[0]
    attack_func = [
        gaussian.Gaussian_blur(2, kernel),
        crop1.Crop(ratio),
        dropout.Dropout(dropout_ratio),
        resize1.Resize(resize_range),
        Gaussian_noise.Gaussian_Noise(0, sigma),
        salt_and_pepper.Salt_and_Pepper(ratio_sp),
        jpeg.Jpeg(Q_ratio),
        # jpeg_compression.JpegCompression('cuda'),
        identity.Identity()
    ]
    # attack_func = [jpeg_compression.JpegCompression('cuda')]
    attack_imgs = []
    choices = random.choices(range(len(attack_func)), k=batch)

    for idx, i in enumerate(choices):

        attack_img = attack_func[i]([img_batch[idx:idx+1], cover_img[idx:idx+1]])
        attack_imgs.append(attack_img)

    final_image = torch.cat(attack_imgs, dim=0)
    return final_image


def attack_for_test(img_batch, cover_img, number, rat, ugpu=True):
    """
    对输入的图像进行攻击处理,对一个batch的图像随机进行攻击
    攻击方式： 高斯噪声、椒盐噪声、原图、缩放、剪切
    :param image_batch: 输入图像批次，形状为 (batch_size, channels, height, width)，范围是 [0, Wnet(SEP2Embedwater)+SEDecoder_COCO.0]
    :return: 变换后的图像批次
    """
    batch = img_batch.shape[0]
    if rat !=0:
        attack_func = [
            gaussian.Gaussian_blur(2, rat),
            crop1.Crop(rat),
            dropout.Dropout(rat),
            resize1.Resize(rat),
            Gaussian_noise.Gaussian_Noise(0, rat),
            salt_and_pepper.Salt_and_Pepper(rat),
            jpeg.Jpeg(rat),
            identity.Identity()
        ]
    else:
        attack_func = [
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            identity.Identity()
        ]
    # attack_func = [jpeg_compression.JpegCompression('cuda')]
    attack_imgs = []

    for idx in range(batch):
        if number == 2:
            attack_img = attack_func[number]([img_batch[idx:idx+1], cover_img[idx:idx+1]])
        else:
            attack_img = attack_func[number](img_batch[idx:idx+1])
        attack_imgs.append(attack_img)

    final_image = torch.cat(attack_imgs, dim=0)
    return final_image


if __name__ == '__main__':
    pass
    # from torchviz import make_dot


























