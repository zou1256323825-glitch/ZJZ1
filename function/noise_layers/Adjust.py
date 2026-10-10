import torch.nn as nn
import torch
from kornia.enhance.adjust import AdjustHue,AdjustSaturation,AdjustContrast,AdjustBrightness,\
    AdjustGamma, sharpness
import math
from torchvision.transforms import ToPILImage
class Adjust_contrast(nn.Module):
    def __init__(self,factor):
        super(Adjust_contrast, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustContrast(contrast_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover

class Adjust_hue(nn.Module):
    def __init__(self,factor):
        super(Adjust_hue, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustHue(hue_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover

class Adjust_saturation(nn.Module):
    def __init__(self,factor):
        super(Adjust_saturation, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustSaturation(saturation_factor=torch.tensor(self.factor))(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover


class Adjust_brightness(nn.Module):
    def __init__(self,factor):
        super(Adjust_brightness, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustBrightness(brightness_factor=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover


class Adjust_gamma(nn.Module):
    def __init__(self,factor):
        super(Adjust_gamma, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=AdjustGamma(gamma=self.factor)(encoded)
        noised_and_cover[0]=(encoded)
        return noised_and_cover


class Adjust_sharpness(nn.Module):
    def __init__(self,factor):
        super(Adjust_sharpness, self).__init__()
        self.factor=factor

    def forward(self, noised_and_cover):
        encoded=((noised_and_cover[0]).clone())
        encoded=sharpness(encoded, self.factor)
        noised_and_cover[0]=(encoded)
        return noised_and_cover

