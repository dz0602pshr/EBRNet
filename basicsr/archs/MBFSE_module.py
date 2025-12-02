import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class LocalFourierBlock(nn.Module):
    def __init__(self, dim):
        super(LocalFourierBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        x_f = torch.cat([x.real, x.imag], dim=1)
        x_f = self.conv(x_f)
        x_real, x_imag = x_f.chunk(2, dim=1)
        x_f = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x_f, dim=(2, 3), norm='ortho')
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

class SFE(nn.Module):
    def __init__(self, in_channels,h,w):
        super(SFE, self).__init__()
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size = (7,1), padding=(3,0),groups=in_channels)
        self.dwconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1,7), padding=(0,3), groups=in_channels)
        self.sfe_module = LocalFourierBlock(in_channels // 2)     ###------------ModifybyPSR--------------------###
        self.instance_norm = nn.InstanceNorm2d(in_channels//2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.dwconv2(self.lrelu(self.dwconv1(x)))
        [x1,x2] = torch.chunk(x, 2, 1)
        x1 = self.sfe_module(x1)
        x2 = self.instance_norm(x2)
        x = torch.cat([x1,x2],1)
        return x

class MFE(nn.Module):
    def __init__(self, in_channels,h,w):
        super(MFE, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.dwconv = nn.Conv2d(in_channels, in_channels, 5, padding=2,groups=in_channels)
        self.sfe_module = SFE(in_channels, h, w)
        self.efe_module = EFE_Layer(in_channels, c2wh[in_channels], c2wh[in_channels],  reduction=16, freq_sel_method = 'top16')
        self.aux_module = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
            )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1= nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        residual = self.lrelu(self.dwconv(x))
        branch1 = self.sfe_module(residual)
        branch2 = self.efe_module(residual)
        branch3 = self.aux_module(residual)
        residual = self.conv1((branch1 + branch2 + branch3))
        x = x * residual
        return x
####-----------------------------------------------fromDCTlayer---------------------------------------------------###
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class EFE_Layer(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):    #reduction=16
        super(EFE_Layer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        self.dwconv1 = nn.Conv2d(channel, channel, kernel_size=(11, 1), padding=(5, 0),
                                 groups=channel)
        self.dwconv2 = nn.Conv2d(channel, channel, kernel_size=(1, 11), padding=(0, 5), groups=channel)
        self.energy = simam_module()
        self.dct_weight = nn.Parameter(torch.randn(channel, dct_h, dct_w, dtype=torch.float32) * 0.02)
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = self.dwconv2(self.lrelu(self.dwconv1(x)))
        x_pooled = self.energy(x_pooled)
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x_pooled, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled) * self.dct_weight
        if y.shape[2] != h or y.shape[3] != w:
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
        y = torch.permute(y, (0, 2, 3, 1)).contiguous()
        y = self.fc(y)
        y = torch.permute(y, (0, 3, 1, 2)).contiguous()
        return y.expand_as(x)

class simam_module(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))


    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        return x

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

def main():
    model = MFE(64, 64,64)
    vector = torch.randn(4, 64, 256, 256)
    out = model(vector)
    print("Hello World!", out.shape)
if __name__=="__main__":
    main()