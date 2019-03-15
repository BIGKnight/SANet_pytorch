#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        if self.use_bn:
            x = self.bn(x)
        return x


class BasicDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.tconv(x)
        # if self.use_bn:
        #     x = self.bn(x)
        # return F.relu(x, inplace=True)
        x = F.relu(x, inplace=True)
        if self.use_bn:
            x = self.bn(x)
        return x


class SAModule_Head(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule_Head, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=1)
        self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=3, padding=1)
        self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=5, padding=2)
        self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=7, padding=3)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class SAModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=1)
        self.branch3x3 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=5, padding=2),
        )
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=7, padding=3),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class FPN_SA_Net(nn.Module):
    def __init__(self, gray_input=False, use_bn=True):
        super(FPN_SA_Net, self).__init__()
        if gray_input:
            in_channels = 1
        else:
            in_channels = 3

        self.encoder_1 = SAModule_Head(in_channels, 64, use_bn)
        self.encoder_2 = nn.Sequential(nn.MaxPool2d(2, 2), SAModule(64, 128, use_bn))
        self.encoder_3 = nn.Sequential(nn.MaxPool2d(2, 2), SAModule(128, 128, use_bn))
        self.encoder_4 = nn.Sequential(nn.MaxPool2d(2, 2), SAModule(128, 128, use_bn))

        self.decoder_1 = nn.Sequential(
            BasicConv(128, 64, use_bn=use_bn, kernel_size=9, padding=4),
            BasicDeconv(64, 64, use_bn=use_bn, kernel_size=2, stride=2)
        )

        self.decoder_2 = nn.Sequential(
            BasicConv(64, 32, use_bn=use_bn, kernel_size=7, padding=3),
            BasicDeconv(32, 32, use_bn=use_bn, kernel_size=2, stride=2)
        )

        self.decoder_3 = nn.Sequential(
            BasicConv(32, 16, use_bn=use_bn, kernel_size=5, padding=2),
            BasicDeconv(16, 16, use_bn=use_bn, kernel_size=2, stride=2)
        )

        self.connection_1 = nn.Conv2d(128, 64, 1, 1)
        self.connection_2 = nn.Conv2d(128, 32, 1, 1)
        self.connection_3 = nn.Conv2d(64, 16, 1, 1)

        self.output = nn.Sequential(
            BasicConv(16, 16, use_bn=use_bn, kernel_size=3, padding=1),
            BasicConv(16, 1, use_bn=False, kernel_size=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e_1 = self.encoder_1(x)
        e_2 = self.encoder_2(e_1)
        e_3 = self.encoder_3(e_2)
        e_4 = self.encoder_4(e_3)

        d_1 = self.decoder_1(e_4) + self.connection_1(e_3)
        d_2 = self.decoder_2(d_1) + self.connection_2(e_2)
        d_3 = self.decoder_3(d_2) + self.connection_3(e_1)
        output = self.output(d_3)
        return output
