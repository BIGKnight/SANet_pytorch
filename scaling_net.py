import torch.nn as nn
import torch


class scaling_net(nn.Module):
    def __init__(self):
        super(scaling_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 3, 1, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 1)

    def forward(self, x):
        # filter?or not we don't use pixel-wise filter yet
        scale_rate = self.net(x)
        return torch.mul(x, scale_rate)

