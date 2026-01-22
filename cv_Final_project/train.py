import os
import torch
from torch.utils.data import DataLoader, Subset
from dataset import CoalDataset
from model import UNet
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =======================
# 1. 构建数据集
# =======================
dataset = CoalDataset(
    raw_dir="raw_data",
    gt_dir="groundtruth",
    img_size=256
)

# 1–200 → 训练, 201–236 → 测试
train_indices = list(range(0, 200))
test_indices = list(range(200, len(dataset)))

train_set = Subset(dataset, train_indices)
test_set = Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

# =======================
# 2. 模型
# =======================
model = UNet(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =======================
# 3. 可视化辅助函数
# =======================
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

os.makedirs("vis", exist_ok=True)

# =======================
# 4. 训练循环
# =======================
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        last_images, last_masks, last_outputs = images, masks, outputs

    # 取最后一个 batch 做可视化
    preds = torch.argmax(last_outputs, dim=1)

    plt.figure(figsize=(12, 4))

    # Input image
    plt.subplot(1, 3, 1)
    plt.title("Input")
    img_show = last_images[0].cpu().permute(1, 2, 0).numpy()
    plt.imshow(img_show)
    plt.axis('off')

    # Groundtruth
    plt.subplot(1, 3, 2)
    plt.title("Groundtruth")
    plt.imshow(label_to_rgb(last_masks[0].cpu().numpy()))
    plt.axis('off')

    # Prediction
    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(label_to_rgb(preds[0].cpu().numpy()))
    plt.axis('off')

    plt.savefig(f"vis/epoch_{epoch+1:03d}.png")
    plt.close()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# =======================
# 5. 保存训练好的模型
# =======================
torch.save(model.state_dict(), "unet_coal_segmentation.pth")
print("Model saved as unet_coal_segmentation.pth")
