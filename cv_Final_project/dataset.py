# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class CoalDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, img_size=256):
        self.raw_dir = raw_dir
        self.gt_dir = gt_dir
        self.img_size = img_size

        self.images = sorted([os.path.join(raw_dir, f) 
                              for f in os.listdir(raw_dir) 
                              if f.endswith(('.png','.jpg','.jpeg'))])
        self.masks = sorted([os.path.join(gt_dir, f) 
                             for f in os.listdir(gt_dir) 
                             if f.endswith(('.png','.jpg','.jpeg'))])

        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # Groundtruth不归一化，只映射成类别
        self.transform_mask = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)
        ])

        # 颜色映射：固定为你 Groundtruth 的 RGB
        self.color2label = {
            (0, 0, 0): 0,        # 背景
            (128, 0, 0): 1,      # 煤
            (0, 128, 0): 2       # 矸石
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 输入图像
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform_img(img)

        # Groundtruth
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert('RGB')
        mask = self.transform_mask(mask)
        mask = np.array(mask)

        # 直接 RGB 映射到类别
        mask_idx = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for color, label in self.color2label.items():
            matches = (mask[:,:,0]==color[0]) & (mask[:,:,1]==color[1]) & (mask[:,:,2]==color[2])
            mask_idx[matches] = label

        mask_tensor = torch.from_numpy(mask_idx).long()
        return img, mask_tensor
