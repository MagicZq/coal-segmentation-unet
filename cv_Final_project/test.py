import torch
import time
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from dataset import CoalDataset
from model import UNet
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 输出文件夹
os.makedirs("predictions", exist_ok=True)

# 数据集
dataset = CoalDataset("raw_data", "groundtruth", img_size=256)

# 加载模型
model = UNet(num_classes=3).to(device)
model.load_state_dict(torch.load("unet_coal_segmentation.pth", map_location=device))
model.eval()

# 损失函数
criterion = nn.CrossEntropyLoss()

# label->颜色映射，用于可视化
label2color = {
    0: [0, 0, 0],      # 背景
    1: [128, 0, 0],    # 煤
    2: [0, 128, 0]     # 矸石
}

def label_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in label2color.items():
        rgb[mask==label] = color
    return rgb

# 记录平均损失
total_loss = 0

for i in range(len(dataset)):
    img, mask = dataset[i]
    img_tensor = img.unsqueeze(0).to(device)
    mask_tensor = mask.unsqueeze(0).to(device)

    # 推理
    start = time.time()
    with torch.no_grad():
        out = model(img_tensor)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
        loss = criterion(out, mask_tensor)
    end = time.time()

    total_loss += loss.item()
    infer_time = (end - start) * 1000  # ms

    # 统计煤和矸石像素
    coal_pixels = np.sum(pred == 1)
    gangue_pixels = np.sum(pred == 2)
    coal_ratio = coal_pixels / (coal_pixels + gangue_pixels + 1e-8)

    print(f"Image {i+1:03d}: Loss={loss.item():.4f}, 煤占比={coal_ratio:.4f}, 推理时间={infer_time:.2f} ms")

    # 可视化对比图
    input_img = (img.permute(1,2,0).numpy()*255).astype(np.uint8)
    gt_color = label_to_rgb(mask.numpy())
    pred_color = label_to_rgb(pred)

    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(input_img)
    ax[0].set_title("Input")
    ax[0].axis('off')

    ax[1].imshow(gt_color)
    ax[1].set_title("Groundtruth")
    ax[1].axis('off')

    ax[2].imshow(pred_color)
    ax[2].set_title(f"Prediction\nCoal proportion={coal_ratio:.2f}, {infer_time:.1f}ms")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"predictions/pred_{i+1:03d}.png")
    plt.close()

# 平均损失
print(f"Average Loss: {total_loss/len(dataset):.4f}")
