# coal-segmentation-unet
基于卷积神经网络的煤矸石分割识别,UCAS计算机视觉作业
本项目基于 **PyTorch + UNet** 实现煤炭与矸石的像素级分割，并计算每张图像中煤的占比。支持训练、测试及可视化输出
📂 目录结构
- dataset.py  # 数据集类
- model.py    # UNet模型定义
- train.py    # 模型训练脚本
- test.py     # 测试脚本及可视化
- raw_data/   # 原始输入图像
- groundtruth/# 标注mask
- predictions/# 测试集预测输出
- vis/        # 训练过程可视化输出
- README.md

# 模型训练
python train.py
训练过程中会：
输出每个 epoch 的平均 loss
在 vis/ 文件夹保存每个 epoch 的输入 / Groundtruth / Prediction 对比图

训练完成后，模型会保存为：
unet_coal_segmentation.pth

训练完成后，运行 test.py 对测试集进行推理：
python test.py
