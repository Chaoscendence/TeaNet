import torch
import math
import torch.nn as nn

class convs(nn.Module):
    '''Conv => BN =>ReLU => Conv1d => BN => ReLU
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(convs, self).__init__()

        conv_layers = []
        for _ in range(width):
            conv_layers += [nn.Conv1d(in_ch, out_ch,kernel_size = kernel, padding=math.floor(kernel*0.5))]
            conv_layers += [nn.BatchNorm1d(out_ch)]
            conv_layers += [nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

    def forward(self,x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    '''downsample encode input
    '''
    def __init__(self, out_ch, kernel, width):
        super(inconv, self).__init__()
        self.conv = convs(1, out_ch, kernel, width)

    def forward(self, x):
        x = self.conv(x)
        return x

class encode(nn.Module):
    '''downsample encode
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(encode, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            convs(in_ch, out_ch, kernel, width)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class decode(nn.Module):
    '''upsample decode
    '''
    def __init__(self, in_ch, out_ch, kernel, width):
        super(decode, self).__init__()
        # transpose
        self.up = nn.ConvTranspose1d(in_ch, in_ch//2, 2, stride=2)
        self.conv = convs(in_ch, out_ch, kernel, width)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        tmp = nn.ZeroPad2d((0, x2.size(2) - x1.size(2), 0, 0))
        x = torch.cat([x2, tmp(x1)], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, 1, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, settings):
        super(Unet, self).__init__()

        self.inc = inconv(16, settings['kernel_unet'], settings['width'])

        self.down1 = encode(16, 32, settings['kernel_unet'], settings['width'])
        self.down2 = encode(32, 64, settings['kernel_unet'], settings['width'])
        self.down3 = encode(64, 128, settings['kernel_unet'], settings['width'])
        self.down4 = encode(128, 256, settings['kernel_unet'], settings['width'])
        self.down5 = encode(256, 512, settings['kernel_unet'], settings['width'])
        self.up5 = decode(512, 256, settings['kernel_unet'], settings['width'])
        self.up1 = decode(256, 128, settings['kernel_unet'], settings['width'])
        self.up2 = decode(128, 64, settings['kernel_unet'], settings['width'])
        self.up3 = decode(64, 32, settings['kernel_unet'], settings['width'])
        self.up4 = decode(32, 16, settings['kernel_unet'], settings['width'])
        self.outc = outconv(16)
        self.fc = nn.Linear(1024, 50)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up5(x6, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
