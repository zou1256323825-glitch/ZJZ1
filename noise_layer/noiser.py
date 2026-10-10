import numpy as np
import torch.nn as nn

from function.noise_layers import cropout, gaussian, Gaussian_noise, salt_and_pepper\
    , Median_filter, jpeg1, grid_crop, dropout, identity, resize1, roration, Adjust
import numpy as np
import random
import torch


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, noise_layers: list):
        super(Noiser, self).__init__()

        self.GBRatio = np.round(random_float(0.5, 2.5), 1)  # gaussian blur
        self.crop_ratio = np.round(random_float(0.3, 0.8), 1)  # random crop
        self.cropout_ratio = np.round(random_float(0.1, 0.6), 1)  # cropout
        self.dropout_ratio = np.round(random_float(0.3, 1), 1)  # dropout
        self.mf_ratio = random.choice([3, 5, 7])
        self.grid_crop_ratio = np.round(random_float(0.3, 0.8), 1)  # grid_crop
        self.resize_range = np.round(random_float(0.3, 1.8), 1)  # resize
        self.GNRatio = np.round(random_float(0.01, 0.25), 2)  # gaussian noise
        self.ratio_sp = np.round(random_float(0.01, 0.2), 2)  # salt and pepper
        self.Q_ratio = np.round(random_float(30, 70), -1)  # JPEG quantum
        self.Rotate_ratio = np.round(random_float(0, 180), -1)

        self.noise_layers = []

        for layer in noise_layers:
            if type(layer) is str:
                if layer == 'Cropout':
                    self.noise_layers.append(cropout.Cropout(self.cropout_ratio))
                elif layer == 'Resize':
                    self.noise_layers.append(resize1.Resize(self.resize_range))
                # elif layer == 'Crop':
                #     self.noise_layers.append(gaussian.Gaussian_blur(self.GBRatio, 3))
                elif layer == 'GaussianBlur':
                    self.noise_layers.append(gaussian.Gaussian_blur(self.GBRatio, 3))
                elif layer == 'Dropout':
                    self.noise_layers.append(dropout.Dropout(self.dropout_ratio))
                elif layer == 'MedianBlur':
                    self.noise_layers.append(Median_filter.Median_filter(self.mf_ratio))
                elif layer == 'GridCrop':
                    self.noise_layers.append(grid_crop.grid_crop(self.grid_crop_ratio))
                elif layer == 'GaussianNoise':
                    self.noise_layers.append(Gaussian_noise.Gaussian_Noise(0, self.GNRatio))
                elif layer == 'SaltAndPepper':
                    self.noise_layers.append(salt_and_pepper.Salt_and_Pepper(self.ratio_sp))
                elif layer == 'JPEG':
                    self.noise_layers.append(jpeg1.Jpeg(self.Q_ratio))
                elif layer == 'Rotate':
                    self.noise_layers.append(roration.Rotate(self.Rotate_ratio))
                elif layer == 'Identity':
                    self.noise_layers.append(identity.Identity())
                elif layer == 'MoireCreat':
                    self.noise_layers.append(modelMoire)
                else:
                    raise ValueError(f'Wrong layer placeholder string in Noiser.__init__().'
                                     f' Expected "Cropout","GaussianBlur","Dropout","MedianBlur","GridCrop",'
                                     f'"GaussianNoise", "SaltandPepper","JPEG"'
                                     f' but got {layer} instead')
            else:
                self.noise_layers.append(layer)
        # self.noise_layers = nn.Sequential(*noise_layers)

    def forward(self, img_batch, cover_img, mode='val'):
        total = len(self.noise_layers)
        if mode == 'train':
            noise_idxs = [random.randint(0, total-1) for _ in range(img_batch.shape[0])]
            noise_images = []
            cover_imgs = []
            for idx, noise_idx in enumerate(noise_idxs):
                # print(idx)
                noise, cover = self.noise_layers[noise_idx]([img_batch[idx:idx+1, :, :, :], cover_img[idx:idx+1, :, :, :]])
                noise_images.append(noise)
                cover_imgs.append(cover)
            noise_images = torch.cat(noise_images, dim=0)
            cover_imgs = torch.cat(cover_imgs, dim=0)
            return [noise_images, cover_imgs], 0
        # total = len(self.noise_layers)
        else:
            noise_idx = random.randint(0, total-1)
            random_noise_layer = self.noise_layers[noise_idx]
            return random_noise_layer([img_batch, cover_img]), noise_idx


def attack(img_batch, cover_img, ugpu=True):
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
    # sigma_ratio = random_float(0.5, 3.0)  # gaussian blur
    # ratio = random_float(0.3, Wnet(SEP2Embedwater)+SEDecoder_COCO)  # random crop
    # cropout_ratio = random_float(0.Wnet(SEP2Embedwater)+SEDecoder_COCO, 0.9)  # cropout
    # dropout_ratio = random_float(0.5, Wnet(SEP2Embedwater)+SEDecoder_COCO)  # dropout
    # grid_crop_ratio = random_float(0.4, 0.8)  # grid_crop
    # resize_range = random_float(0.3, Wnet(SEP2Embedwater)+SEDecoder_COCO.8)  # resize
    # sigma = random_float(0.01, 0.25)  # gaussian noise
    # ratio_sp = random_float(0.01, 0.2)  # salt and pepper
    # Q_ratio = np.round(random_float(5, 70), 0) # JPEG quantum
    batch = img_batch.shape[0]
    attack_func = [
        gaussian.Gaussian_blur(sigma_ratio, 3),
        cropout.Cropout(cropout_ratio),
        dropout.Dropout(dropout_ratio),
        Median_filter.Median_filter(mf_ratio),
        grid_crop.grid_crop(grid_crop_ratio),
        Gaussian_noise.Gaussian_Noise(0, sigma),
        salt_and_pepper.Salt_and_Pepper(ratio_sp),
        jpeg.Jpeg(Q_ratio),
    ]

    attack_imgs = []
    choices = random.choices(range(len(attack_func)), k=batch)
    # random_noise_layer = np.random.choice(attack_func, Wnet(SEP2Embedwater)+SEDecoder_COCO)[0]
    for idx, i in enumerate(choices):
        attack_img = attack_func[i]([img_batch[idx:idx + 1], cover_img[idx:idx + 1]])
        attack_imgs.append(attack_img[0])
    #
    final_image = torch.cat(attack_imgs, dim=0)
    # final_image = random_noise_layer([img_batch, cover_img])
    return [final_image, cover_img]
