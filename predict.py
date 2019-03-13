import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
from utils import *
import torchvision.transforms as transforms
from DataConstructor import DatasetConstructor
import metrics
from PIL import Image
import time
SHANGHAITECH = "B"
# %matplotlib inline
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU
# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/gt_map"
dataset = DatasetConstructor(img_dir, gt_dir, 316, 316, False)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
mae_metrics = []
mse_metrics = []
net = torch.load("/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_rate_b.pkl").to(cuda_device)
net.eval()


ae_batch = metrics.AEBatch().to(cuda_device)
se_batch = metrics.SEBatch().to(cuda_device)
for real_index, test_img, test_gt, test_time_cost in test_loader:
    image_shape = test_img.shape
    patch_height = int(image_shape[3])
    patch_width = int(image_shape[4])
    # B
    eval_x = test_img.view(49, 3, patch_height, patch_width)
    eval_y = test_gt.view(1, 1, patch_height * 4, patch_width * 4).cuda()
    prediction_map = torch.zeros(1, 1, patch_height * 4, patch_width * 4).cuda()
    for i in range(7):
        for j in range(7):
            eval_x_sample = eval_x[i * 7 + j:i * 7 + j + 1].cuda()
            eval_y_sample = eval_y[i * 7 + j:i * 7 + j + 1].cuda()
            eval_prediction = net(eval_x_sample)
            start_h = int(patch_height / 4)
            start_w = int(patch_width / 4)
            valid_h = int(patch_height / 2)
            valid_w = int(patch_width / 2)
            h_pred = 3 * int(patch_height / 4) + 2 * int(patch_height / 4) * (i - 1)
            w_pred = 3 * int(patch_width / 4) + 2 * int(patch_width / 4) * (j - 1)
            if i == 0:
                valid_h = int((3 * patch_height) / 4)
                start_h = 0
                h_pred = 0
            elif i == 6:
                valid_h = int((3 * patch_height) / 4)

            if j == 0:
                valid_w = int((3 * patch_width) / 4)
                start_w = 0
                w_pred = 0
            elif j == 6:
                valid_w = int((3 * patch_width) / 4)

            prediction_map[:, :, h_pred:h_pred + valid_h, w_pred:w_pred + valid_w] += eval_prediction[:, :,
                                                                                      start_h:start_h + valid_h,
                                                                                      start_w:start_w + valid_w]

    batch_ae = ae_batch(prediction_map, eval_y).data.cpu().numpy()
    batch_se = se_batch(prediction_map, eval_y).data.cpu().numpy()
    mae_metrics.append(batch_ae)
    mse_metrics.append(batch_se)
    # to numpy
    numpy_predict_map = prediction_map.permute(0, 2, 3, 1).data.cpu().numpy()
    numpy_gt_map = eval_y.permute(0, 2, 3, 1).data.cpu().numpy()

    # show current prediction
    figure, (origin, dm_gt, dm_pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(Image.open("/home/zzn/part_B_final/test_data/images/IMG_" + str(real_index.numpy()[0]) + ".jpg"))
    origin.set_title('Origin Image')
    dm_gt.imshow(np.squeeze(numpy_gt_map), cmap=plt.cm.jet)
    dm_gt.set_title('ground_truth_1')

    dm_pred.imshow(np.squeeze(numpy_predict_map), cmap=plt.cm.jet)
    dm_pred.set_title('prediction')

    plt.suptitle('The ' + str(real_index.numpy()[0]) + 'th images\'prediction')
    plt.show()
    sys.stdout.write('The grount truth crowd number is:{}, and the predicting number is:{}'.format(np.sum(numpy_gt_map),
                                                                                                   np.sum(
                                                                                                       numpy_predict_map)))
    sys.stdout.flush()
    mae_metrics = np.reshape(mae_metrics, [-1])
    mse_metrics = np.reshape(mse_metrics, [-1])
    MAE = np.mean(mae_metrics)
    MSE = np.sqrt(np.mean(mse_metrics))
    print('MAE:', MAE, 'MSE:', MSE)
