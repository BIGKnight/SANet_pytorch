import torch
from net import *
from PIL import Image
import time
import numpy as np
import random
from ssim_loss import *
from metrics import *
from DataConstructor import *
from metrics import *
from utils import show
import sys
import torchvision.transforms as transforms
MAE = 10240000
MSE = 10240000
RATE = 10000000
SHANGHAITECH = "B"
# %matplotlib inline
# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/gt_map"

dataset = DatasetConstructor(img_dir, gt_dir, 400, 50)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
eval_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = SANet().to(cuda_device)
# net = torch.load("/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_mse_b_0312.pkl").to(cuda_device)
# set optimizer and estimator
criterion = SANetLoss(1).to(cuda_device)
optimizer = torch.optim.Adam(net.parameters(), 1e-6)
ae_batch = AEBatch().to(cuda_device)
se_batch = SEBatch().to(cuda_device)

step = 0
for epoch_index in range(10000):
    dataset = dataset.train_model().shuffle()
    for train_img_index, train_img, train_gt, data_ptc in train_loader:
        # eval per 100 batch
        if step % 100 == 0:
            net.eval()
            dataset = dataset.eval_model().shuffle()
            loss_ = []
            MAE_ = []
            MSE_ = []
            difference_rates = []

            rand_number = random.randint(0, 19)
            counter = 0

            for eval_img_index, eval_img, eval_gt, eval_data_ptc in eval_loader:

                image_shape = eval_img.shape
                patch_height = int(image_shape[3])
                patch_width = int(image_shape[4])
                # B
                eval_x = eval_img.view(49, 3, patch_height, patch_width)
                eval_y = eval_gt.view(1, 1, patch_height * 4, patch_width * 4).cuda()
                prediction_map = torch.zeros(1, 1, patch_height * 4, patch_width * 4).cuda()
                for i in range(7):
                    for j in range(7):
                        eval_x_sample = eval_x[i * 7 + j:i * 7 + j + 1].cuda()
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
                # That’s because numpy doesn’t support CUDA,
                # so there’s no way to make it use GPU memory without a copy to CPU first.
                # Remember that .numpy() doesn’t do any copy,
                # but returns an array that uses the same memory as the tensor
                eval_loss = criterion(prediction_map, eval_y).data.cpu().numpy()
                batch_ae = ae_batch(prediction_map, eval_y).data.cpu().numpy()
                batch_se = se_batch(prediction_map, eval_y).data.cpu().numpy()

                validate_pred_map = np.squeeze(prediction_map.permute(0, 2, 3, 1).data.cpu().numpy())
                validate_gt_map = np.squeeze(eval_y.permute(0, 2, 3, 1).data.cpu().numpy())
                gt_counts = np.sum(validate_gt_map)
                pred_counts = np.sum(validate_pred_map)
                # random show 1 sample
                if rand_number == counter and step % 2000 == 0:
                    origin_image = Image.open("/home/zzn/part_" + SHANGHAITECH + "_final/train_data/images/IMG_" + str(
                        eval_img_index.numpy()[0]) + ".jpg")
                    show(origin_image, validate_gt_map, validate_pred_map, eval_img_index.numpy()[0])
                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

                difference_rates.append(np.abs(gt_counts - pred_counts) / gt_counts)
                loss_.append(eval_loss)
                MAE_.append(batch_ae)
                MSE_.append(batch_se)
                counter += 1

            # calculate the validate loss, validate MAE and validate RMSE
            loss_ = np.reshape(loss_, [-1])
            MAE_ = np.reshape(MAE_, [-1])
            MSE_ = np.reshape(MSE_, [-1])

            validate_loss = np.mean(loss_)
            validate_MAE = np.mean(MAE_)
            validate_RMSE = np.sqrt(np.mean(MSE_))
            validate_rate = np.mean(difference_rates)

            sys.stdout.write(
                'In step {}, epoch {}, with loss {}, rate = {}, MAE = {}, MSE = {}\n'.format(step, epoch_index + 1,
                                                                                             validate_loss,
                                                                                             validate_rate,
                                                                                             validate_MAE,
                                                                                             validate_RMSE))
            sys.stdout.flush()

            if RATE > validate_rate:
                RATE = validate_rate
                torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_1_rate_b.pkl")

            # save model
            if MAE > validate_MAE:
                MAE = validate_MAE
                torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_1_mae_b.pkl")

            # save model
            if MSE > validate_RMSE:
                MSE = validate_RMSE
                torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_1_mse_b.pkl")

            torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_1_in_time.pkl")

            # return train model

        net.train()
        dataset = dataset.train_model()
        optimizer.zero_grad()
        # B
        x = train_img.cuda()
        y = train_gt.cuda()

        prediction = net(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        step += 1