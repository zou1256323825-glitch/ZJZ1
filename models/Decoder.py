import torch.nn as nn
from models.ConvBlock import ConvBlock


class Down(nn.Module):
    def __init__(self, in_nc, times, double=True):
        super(Down, self).__init__()
        self.donwlayers = []
        for i in range(times):
            if double:
                self.donwlayer = ConvBlock(in_nc, in_nc * 2, 3, 2, 1)
            else:
                self.donwlayer = ConvBlock(in_nc, in_nc, 3, 2, 1)
            self.donwlayers.append(self.donwlayer)
        self.down = nn.Sequential(*self.donwlayers)

    def forward(self, x):
        return self.down(x)


class Decoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(in_ch, 32, 3, 1, 1), Down(32, 1, False), Down(32, 1), Down(64, 1, False),
                                  Down(64, 1), Down(128, 1), Down(256, 1), Down(512, 1))
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),  # assume input 128x128 -> after 7 stride2 roughly 1-2 dims; adapt if different
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.Sigmoid()
        )
        # self.fc = nn.Linear(64, 30)

    def forward(self, img):
        f = self.conv(img)
        out = f.view(f.shape[0], -1)
        out = self.out(out)

        # out = self.fc(out)
        return out


if __name__ == '__main__':
    import torch
    a = torch.randn(2, 3, 128, 128)
    model = Decoder()
    model(a)


















