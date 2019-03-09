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
SHANGHAITECH = "B"
# data_load
img_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/images"
gt_dir = "/home/zzn/part_" + SHANGHAITECH + "_final/train_data/gt_map"

dataset = DatasetConstructor(img_dir, gt_dir, 400, 20)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
eval_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
# obtain the gpu device
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")

# model construct
net = SANet().to(cuda_device)

# set optimizer and estimator
criterion = SANetLoss(1).to(cuda_device)
optimizer = torch.optim.Adam(net.parameters(), 1e-5, weight_decay=5*1e-4)
ae_batch = AEBatch().to(cuda_device)
se_batch = SEBatch().to(cuda_device)
for epoch_index in range(500):
    dataset = dataset.train_model().shuffle()
    # train
    step = 0
    for train_img_index, train_img, train_gt, data_ptc in train_loader:
        # eval per 100 batch
        if step % 100 == 0:
            net.eval()
            dataset = dataset.eval_model().shuffle()
            loss_ = []
            MAE_ = []
            MSE_ = []

            rand_number = random.randint(0, 19)
            counter = 0

            for eval_img_index, eval_img, eval_gt, eval_data_ptc in eval_loader:

                # B
                eval_x = eval_img.view(-1, 3, 768, 1024).cuda()
                eval_y = eval_gt.view(-1, 1, 768, 1024).cuda()
                # A
                #                     eval_x = eval_img.cuda()
                #                     eval_y = eval_gt.cuda()
                eval_prediction = net(eval_x)
                # That’s because numpy doesn’t support CUDA,
                # so there’s no way to make it use GPU memory without a copy to CPU first.
                # Remember that .numpy() doesn’t do any copy,
                # but returns an array that uses the same memory as the tensor
                eval_loss = criterion(eval_prediction, eval_y).data.cpu().numpy()
                batch_ae = ae_batch(eval_prediction, eval_y).data.cpu().numpy()
                batch_se = se_batch(eval_prediction, eval_y).data.cpu().numpy()

                # random show 1 sample
                if rand_number == counter:
                    origin_image = Image.open("/home/zzn/part_" + SHANGHAITECH + "_final/train_data/images/IMG_" + str(
                        eval_img_index.numpy()[0]) + ".jpg")
                    validate_pred_map = np.squeeze(eval_prediction.permute(0, 2, 3, 1).data.cpu().numpy())
                    validate_gt_map = np.squeeze(eval_y.permute(0, 2, 3, 1).data.cpu().numpy())

                    show(origin_image, validate_gt_map, validate_pred_map, eval_img_index.numpy()[0])
                    gt_counts = np.sum(validate_gt_map)
                    pred_counts = np.sum(validate_pred_map)
                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))

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

            sys.stdout.write(
                'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, epoch_index + 1, validate_loss,
                                                                                  validate_MAE, validate_RMSE))
            sys.stdout.flush()

            # save model
            if MAE > validate_MAE:
                MAE = validate_MAE
                torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_mae_b.pkl")

            # save model
            if MSE > validate_RMSE:
                MSE = validate_RMSE
                torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_mse_b.pkl")

            torch.save(net, "/home/zzn/PycharmProjects/SANet_pytoch/checkpoints/model_in_time.pkl")

            # return train model
            net.train()
            dataset = dataset.train_model()

        net.train()
        optimizer.zero_grad()
        # B
        x = train_img.view(-1, 3, 384, 512).cuda()
        y = train_gt.view(-1, 1, 384, 512).cuda()

        # A
        #       x = train_img.cuda()
        #       y = train_gt.cuda()
        #       figure, (input_picture, gt_picture) = plt.subplots(1, 2, figsize=(20, 4))
        #       input_picture.imshow(train_img[0].view(3, 384, 512).permute(1, 2, 0).numpy())
        #       input_picture.set_title("origin")
        #       gt_picture.imshow(train_gt[0].view(384, 512).numpy(), cmap=plt.cm.jet)
        #       gt_picture.set_title("gt")
        #       plt.show()
        prediction = net(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        step += 1