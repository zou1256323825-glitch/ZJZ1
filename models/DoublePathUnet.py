from models.ConvBlock import ConvBlock
import torch
import torch.nn as nn
from models.System import BIAM


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownSample, self).__init__()
        self.downsample = nn.Sequential(
            ConvBlock(in_ch, out_ch, 3, 2, 1),
            BIAM(out_ch)
        )

    def forward(self, x):
        out = self.downsample(x)
        return out


class UpSample(nn.Module):
    def __init__(self, ch: int, out_ch: int, use_xattn: bool = True):
        super().__init__()
        self.up = nn.Sequential(
            ConvBlock(ch, ch * 2, 3, 1, 1),
            nn.PixelShuffle(2),
            BIAM(ch//2)
        )
        self.pre = ConvBlock(ch//2 + out_ch, out_ch, 3, 1, 1)

    def forward(self, y: torch.Tensor, skip_spatial: torch.Tensor):
        y = self.up(y)
        y = self.pre(torch.cat([y, skip_spatial], dim=1))
        return y


class UNet(nn.Module):
    def __init__(self, inc=3, outc=3, nf=32):
        super(UNet, self).__init__()
        out_channels = [nf * (2 ** (i + 1)) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # 下采样
        self.pre_layer = nn.Sequential(
            ConvBlock(inc, out_channels[0], 3, 1, 1),
            ConvBlock(out_channels[0], out_channels[0], 3, 1, 1)
        )
        self.d1 = DownSample(out_channels[0], out_channels[1])  # 3-64
        self.d2 = DownSample(out_channels[1], out_channels[2])  # 64-128
        self.d3 = DownSample(out_channels[2], out_channels[3])  # 128-256
        self.d4 = DownSample(out_channels[3], out_channels[4])  # 256-512

        self.mid = ConvBlock(out_channels[4], out_channels[4], 3, 1, 1)
        # 上采样
        self.u1 = UpSample(out_channels[4], out_channels[3], False)  # 512-1024-512
        self.u2 = UpSample(out_channels[3], out_channels[2], False)  # 1024-512-256
        self.u3 = UpSample(out_channels[2], out_channels[1], False)  # 512-256-128
        self.u4 = UpSample(out_channels[1], out_channels[0], False)  # 256-128-64
        # 输出
        self.out = nn.Sequential(
            ConvBlock(out_channels[0], 32, 3, 1, 1),
            ConvBlock(32, outc, 3, 1, 1),
        )

    def forward(self, x):
        x_pre = self.pre_layer(x)
        out1 = self.d1(x_pre)
        out2 = self.d2(out1)
        out3 = self.d3(out2)
        out4 = self.d4(out3)
        out5 = self.mid(out4)
        out6 = self.u1(out5, out3)
        out7 = self.u2(out6, out2)
        out8 = self.u3(out7, out1)
        out9 = self.u4(out8, x_pre)
        out = self.out(out9)

        return out


if __name__ == '__main__':
    import time

    a = torch.rand(16, 32, 128, 128).cuda()
    model = UNet(32).cuda()
    start = time.time()
    model(a)
    print(time.time() - start)
