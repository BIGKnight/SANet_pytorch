import torch.nn as nn
import torch
from ssim_loss import *


class SANetLoss(nn.Module):
    def __init__(self, in_channels, size=5, sigma=1.5, size_average=True):
        super(SANetLoss, self).__init__()
        self.ssim_loss = SSIM_Loss(in_channels, size, sigma, size_average)

    def forward(self, estimated_density_map, gt_map):
        loss_c = self.ssim_loss(estimated_density_map, gt_map)
        loss_e = torch.mean((estimated_density_map - gt_map) ** 2, dim=(0, 1, 2, 3))
        return torch.mul(torch.add(torch.mul(loss_c, 0.001), loss_e), 384 * 512 * 4)


class AEBatch(nn.Module):
    def __init__(self):
        super(AEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.abs(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)))


class SEBatch(nn.Module):
    def __init__(self):
        super(SEBatch, self).__init__()

    def forward(self, estimated_density_map, gt_map):
        return torch.pow(torch.sum(estimated_density_map - gt_map, dim=(1, 2, 3)), 2)
