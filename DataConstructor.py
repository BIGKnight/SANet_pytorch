from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import time
import torch


class DatasetConstructor(data.Dataset):
    def __init__(self,
                 data_dir_path,
                 gt_dir_path,
                 train_num,
                 validate_num,
                 if_train=True
                 ):
        self.train_num = train_num
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.train = if_train
        self.train_permulation = np.random.permutation(self.train_num)
        self.eval_permulation = random.sample(range(0, self.train_num),  self.validate_num)
        for i in range(self.train_num):
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            img = Image.open(self.data_root + img_name).convert("RGB")
            gt_map = Image.fromarray(np.squeeze(np.load(self.gt_root + gt_map_name)))
            self.imgs.append([img, gt_map])

    def __getitem__(self, index):

        start = time.time()
        if self.train:
            img, gt_map = self.imgs[self.train_permulation[index]]
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            img_shape = img.shape  # C, H, W
            random_h = random.randint(0, (3 / 4) * img_shape[1] - 1)
            random_w = random.randint(0, (3 / 4) * img_shape[2] - 1)
            patch_height = int(img_shape[1] / 4)
            patch_width = int(img_shape[2] / 4)
            img = img[:, random_h:random_h + patch_height, random_w:random_w + patch_width]
            gt_map = gt_map[:, random_h:random_h + patch_height, random_w:random_w + patch_width]
            end = time.time()
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            return self.train_permulation[index] + 1, img, gt_map, (end - start)

        else:
            img, gt_map = self.imgs[self.eval_permulation[index]]
            img = transforms.ToTensor()(img)
            gt_map = transforms.ToTensor()(gt_map)
            img_shape = img.shape  # C, H, W
            patch_height = int(img_shape[1] / 4)
            patch_width = int(img_shape[2] / 4)
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            imgs = []
            for i in range(7):
                for j in range(7):
                    start_h = int(patch_height / 2) * i
                    start_w = int(patch_width / 2) * j
                    # print(img.shape, start_h, start_w, patch_height, patch_width)
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
            imgs = torch.stack(imgs)
            end = time.time()
            return self.eval_permulation[index] + 1, imgs, gt_map, (end - start)

    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return self.validate_num

    def shuffle(self):
        if self.train:
            self.train_permulation = np.random.permutation(self.train_num)
        else:
            self.eval_permulation = random.sample(range(0, self.train_num),  self.validate_num)
        return self

    def eval_model(self):
        self.train = False
        return self

    def train_model(self):
        self.train = True
        return self
