import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


def show(origin_map, gt_map, predict, index):
    figure, (origin, gt, pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred.imshow(predict, cmap=plt.cm.jet)
    pred.set_title("prediction")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()


def show_phase2(origin_map, gt_map, predict_1, predict_2, index):
    figure, (origin, gt, pred_1, pred_2) = plt.subplots(1, 4, figsize=(20, 4))
    origin.imshow(origin_map)
    origin.set_title("origin picture")
    gt.imshow(gt_map, cmap=plt.cm.jet)
    gt.set_title("gt map")
    pred_1.imshow(predict_1, cmap=plt.cm.jet)
    pred_1.set_title("prediction_phase_1")
    pred_2.imshow(predict_2, cmap=plt.cm.jet)
    pred_2.set_title("prediction_phase_2")
    plt.suptitle(str(index) + "th sample")
    plt.show()
    plt.close()


class HSI_Calculator(nn.Module):
    def __init__(self):
        super(HSI_Calculator, self).__init__()

    def forward(self, image):
        image = transforms.ToTensor()(image)
        I = torch.mean(image)
        Sum = image.sum(0)
        Min = 3 * image.min(0)[0]
        S = (1 - Min.div(Sum.clamp(1e-6))).mean()
        numerator = (2 * image[0] - image[1] - image[2]) / 2
        denominator = ((image[0] - image[1]) ** 2 + (image[0] - image[2]) * (image[1] - image[2])).sqrt()
        theta = (numerator.div(denominator.clamp(1e-6))).clamp(-1 + 1e-6, 1 - 1e-6).acos()
        logistic_matrix = (image[1] - image[2]).ceil()
        H = (theta * logistic_matrix + (1 - logistic_matrix) * (360 - theta)).mean() / 360
        return H, S, I

#
# test = Image.open("/home/zzn/part_B_final/test_data/images/IMG_100.jpg")
# new = F.adjust_brightness(test, 0.43 / 0.376)
# figure, (origin, new_fig) = plt.subplots(1, 2, figsize=(40, 4))
# origin.imshow(test)
# new_fig.imshow(new)
# plt.show()
# calcu = HSI_Calculator()
# H, S, I = calcu(test)
# print(H, S, I)