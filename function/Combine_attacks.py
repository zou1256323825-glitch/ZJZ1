import torch.nn as nn
from function.noise_layers import Gaussian_noise,salt_and_pepper,crop1,grid_crop,dropout,gaussian, resize1
from function.noise_layers import Median_filter


class Noiser(nn.Module):
    """
    This module allows to combine different noise layers into a sequential noise module. The
    configuration and the sequence of the noise layers is controlled by the noise_config parameter.
    """
    def __init__(self, rat):
        super(Noiser, self).__init__()
        self.noise_layers = []
        if rat == -1:
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                salt_and_pepper.Salt_and_Pepper(0.1)
            )
        elif rat == -2:
            self.noise_layers = nn.Sequential(
                crop1.Crop(0.3),
                grid_crop.grid_crop(0.5),
            )
        elif rat == -3:
            self.noise_layers = nn.Sequential(
                Median_filter.Median_filter(3),
                resize1.Resize(0.5),
            )
        elif rat == -4:
            self.noise_layers = nn.Sequential(
                dropout.Dropout(0.7),
                gaussian.Gaussian_blur(2.0, 3),
                grid_crop.grid_crop(0.5)
            )
        elif rat == -5:
            self.noise_layers = nn.Sequential(
                crop1.Crop(0.3),
                Median_filter.Median_filter(3),
                grid_crop.grid_crop(0.5),
            )
        elif rat == -6:  # -6 -7 证明不同的滤波是否造成模型鲁棒性降低
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                Median_filter.Median_filter(3),
                grid_crop.grid_crop(0.5),
            )
        elif rat == -7:  # -7 -8 -9 证明复合攻击顺序是否造成鲁棒性降低
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                gaussian.Gaussian_blur(2.0, 3),
                grid_crop.grid_crop(0.5)
            )
        elif rat == -8:
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                grid_crop.grid_crop(0.5),
                gaussian.Gaussian_blur(2.0, 3),
            )
        elif rat == -9:
            self.noise_layers = nn.Sequential(
                grid_crop.grid_crop(0.5),
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                gaussian.Gaussian_blur(2.0, 3),
            )
        elif rat == -10:  # N N F
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                salt_and_pepper.Salt_and_Pepper(0.1),
                Median_filter.Median_filter(3),
            )
        elif rat == -11:  # N F F
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                gaussian.Gaussian_blur(2.0, 3),
                Median_filter.Median_filter(3),
            )
        elif rat == -12:  # N F G
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                Median_filter.Median_filter(3),
                crop1.Crop(0.3),
            )
        elif rat == -13:  # N G G
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                crop1.Crop(0.3),
                grid_crop.grid_crop(0.5),
            )
        elif rat == -14:  # F F G
            self.noise_layers = nn.Sequential(
                Median_filter.Median_filter(3),
                gaussian.Gaussian_blur(2.0, 3),
                crop1.Crop(0.3),
            )
        elif rat == -15:  # F G G
            self.noise_layers = nn.Sequential(
                Median_filter.Median_filter(3),
                crop1.Crop(0.3),
                grid_crop.grid_crop(0.5),
            )
        elif rat == -16:  # -16 -17 证明应对 gaussian和grid_crop哪个鲁棒性差
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                Median_filter.Median_filter(3),
                grid_crop.grid_crop(0.5)
            )
        elif rat == -17:  # -16 -17 证明应对 gaussian和grid_crop哪个鲁棒性差
            self.noise_layers = nn.Sequential(
                Gaussian_noise.Gaussian_Noise(0, 0.08),
                gaussian.Gaussian_blur(2.0, 3),
                crop1.Crop(0.3),
            )


    def forward(self, encoded_and_cover):
        attacked_image= self.noise_layers(encoded_and_cover)
        # print(random_noise_layer.__class__.__name__)  # 打印类名
        encoded_and_cover = attacked_image
        return encoded_and_cover

