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
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/test_data/gt_map"
transform_a = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
gt_transform_a =  transforms.ToTensor()

dataset = DatasetConstructor(img_dir, gt_dir, 316, 316, transform_a, gt_transform_a, False)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

mae_metrics = []
mse_metrics = []
net = torch.load("/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_mae_b.pkl").to(cuda_device)
net.eval()


ae_batch = metrics.AEBatch().to(cuda_device)
se_batch = metrics.SEBatch().to(cuda_device)

for real_index, test_img, test_gt, test_time_cost in test_loader:
    if True:
        predict_x = test_img.cuda()
        predict_gt = test_gt.cuda()
    predict_predict_map = net(predict_x)
    predict_gt_map = predict_gt
    batch_ae = ae_batch(predict_predict_map, predict_gt_map).data.cpu().numpy()
    batch_se = se_batch(predict_predict_map, predict_gt_map).data.cpu().numpy()
    mae_metrics.append(batch_ae)
    mse_metrics.append(batch_se)
    # to numpy
    numpy_predict_map = predict_predict_map.permute(0, 2, 3, 1).data.cpu().numpy()
    numpy_gt_map = predict_gt_map.permute(0, 2, 3, 1).data.cpu().numpy()

    # show current prediction
    figure, (origin, dm_gt, dm_pred) = plt.subplots(1, 3, figsize=(20, 4))
    origin.imshow(Image.open("/home/zzn/part_B_final/test_data/images/IMG_" + str(real_index.numpy()[0]) + ".jpg"))
    origin.set_title('Origin Image')
    dm_gt.imshow(np.squeeze(numpy_gt_map), cmap=plt.cm.jet)
    dm_gt.set_title('ground_truth')
    dm_pred.imshow(np.squeeze(numpy_predict_map), cmap=plt.cm.jet)
    dm_pred.set_title('prediction')
    plt.suptitle('The ' + str(real_index.numpy()[0]) + 'st images\'prediction')
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