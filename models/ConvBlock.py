import torch
import torch.nn as nn
# from pyct import nsct


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kcore, stride, padding=0, up=False):
        super(ConvBlock, self).__init__()
        if up:
            conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
        else:
            conv = nn.Conv2d(input_nc, output_nc, kernel_size=kcore, stride=stride, padding=padding)
        norm = nn.BatchNorm2d(output_nc)
        relu = nn.ReLU()

        model = [conv, norm, relu]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class DenseBlock(nn.Module):
    def __init__(self, input_nc, output_nc, nc=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBlock(input_nc, nc, 1, 1, 0),
            ConvBlock(nc, nc, 3, 1, 1),
        )
        self.conv2 = nn.Sequential(
            ConvBlock(nc+input_nc, nc, 1, 1, 0),
            ConvBlock(nc, nc, 3, 1, 1)
        )
        self.conv3 = nn.Sequential(
            ConvBlock(nc*2 + input_nc, nc, 1, 1, 0),
            ConvBlock(nc, nc, 3, 1, 1)
        )
        self.conv4 = nn.Sequential(
            ConvBlock(nc*3 + input_nc, nc, 1, 1, 0),
            ConvBlock(nc, nc, 3, 1, 1)
        )
        self.out = ConvBlock(nc*4 + input_nc, output_nc, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), dim=1))
        x3 = self.conv3(torch.cat((x, x1, x2), dim=1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), dim=1))
        out = self.out(torch.cat((x, x1, x2, x3, x4), dim=1))
        return out


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding=kernel//2,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)


    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


if __name__ == '__main__':
    # from torchsummary import summary
    input = torch.zeros([6, 3, 128, 128]).cuda()
    model = DenseBlock(3, 32).cuda()
    out = model(input)
    print(out.shape)
