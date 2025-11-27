# Fast-Cartoonization

一种高速图片卡通化的转绘模型

## 📋 项目概述

与传统的端到端"黑盒"卡通化方法不同，白盒卡通化方法将卡通风格分解为三个可解释的表示：

1. **表面表示 (Surface)** - 平滑的颜色块
2. **结构表示 (Structure)** - 清晰的边缘和轮廓  
3. **纹理表示 (Texture)** - 纹理细节

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 推理（使用预训练模型）

```bash
python inference.py \
    --input ./test_images \
    --output ./results \
    --checkpoint ./checkpoints/generator.pth
```

### 单张图片处理

```bash
python inference.py \
    --input photo.jpg \
    --output cartoon.jpg \
    --checkpoint ./checkpoints/generator.pth
```

## 🏋️ 训练

### 1. 准备数据集

组织数据目录结构如下：

```
dataset/
├── photo/           # 真实照片 (训练用)
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── cartoon/         # 卡通参考图片
    ├── cartoon1.jpg
    ├── cartoon2.jpg
    └── ...
```

**推荐数据来源：**
- 风景照片：任意高质量风景图片
- 卡通图片：新海诚、宫崎骏、细田守等动画电影截帧
- 人像照片：高质量人像照片
- 人像卡通：京都动画、PA Works 等作品

### 2. 预训练生成器

首先预训练生成器进行图像重建：

```bash
python pretrain.py \
    --photo_dir ./dataset/photo \
    --save_dir ./pretrain_results \
    --batch_size 16 \
    --num_iters 50000 \
    --lr 2e-4
```

### 3. 完整训练

使用预训练模型进行完整的对抗训练：

```bash
python train.py \
    --photo_dir ./dataset/photo \
    --cartoon_dir ./dataset/cartoon \
    --pretrain_path ./pretrain_results/checkpoints/pretrain_final.pth \
    --save_dir ./train_results \
    --batch_size 16 \
    --num_iters 100000 \
    --lr 2e-4 \
    --lambda_surface 0.1 \
    --lambda_structure 1.0 \
    --lambda_content 200.0 \
    --lambda_tv 10000.0
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 16 | 批次大小 |
| `--num_iters` | 100000 | 训练迭代次数 |
| `--lr` | 2e-4 | 学习率 |
| `--lambda_surface` | 0.1 | 表面损失权重 |
| `--lambda_structure` | 1.0 | 结构损失权重 |
| `--lambda_content` | 200.0 | 内容损失权重 |
| `--lambda_tv` | 10000.0 | 总变差损失权重 |

## 📦 模型导出

### 导出到 ONNX

```bash
python export.py \
    --checkpoint ./train_results/generator_final.pth \
    --output_dir ./exported_models \
    --onnx \
    --input_size 256
```

### 导出到 CoreML (iOS)

```bash
python export.py \
    --checkpoint ./train_results/generator_final.pth \
    --output_dir ./exported_models \
    --coreml \
    --input_size 256
```

### 导出所有格式

```bash
python export.py \
    --checkpoint ./train_results/generator_final.pth \
    --output_dir ./exported_models \
    --all
```

## 📁 项目结构

```
MyCartoonization/
├── models/
│   ├── __init__.py
│   ├── generator.py        # UNet 生成器
│   ├── discriminator.py    # 谱归一化判别器
│   ├── guided_filter.py    # 导向滤波器
│   └── vgg.py              # VGG 特征提取
├── dataset.py              # 数据集类
├── losses.py               # 损失函数
├── utils.py                # 工具函数
├── pretrain.py             # 预训练脚本
├── train.py                # 完整训练脚本
├── inference.py            # 推理脚本
├── export.py               # 模型导出
├── requirements.txt        # 依赖
└── README.md               # 说明文档
```

## 🔧 模型架构

### 生成器 (UNet Generator)

- 编码器：3个下采样块
- 瓶颈层：4个残差块
- 解码器：3个上采样块 + 跳跃连接
- 激活函数：LeakyReLU

### 判别器 (Spectral Norm Discriminator)

- PatchGAN 架构
- 谱归一化稳定训练
- 两个判别器：表面 (3通道) + 结构 (1通道)

### 导向滤波器 (Guided Filter)

- 边缘保持平滑
- 可微分实现
- 参数：r=1, eps=5e-3

## 📊 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| Surface GAN | 0.1 | 模糊后图像的对抗损失 |
| Structure GAN | 1.0 | 灰度图像的对抗损失 |
| Content (VGG) | 200 | VGG conv4_4 感知损失 |
| Total Variation | 10000 | 平滑度约束 |

## 🎯 使用 Python API

```python
from inference import Cartoonizer

# 创建卡通化器
cartoonizer = Cartoonizer(
    checkpoint_path='./checkpoints/generator.pth',
    device='cuda'
)

# 处理图片
import cv2
img = cv2.imread('photo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = cartoonizer.cartoonize(img)

# 保存结果
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('cartoon.jpg', result)
```

## ⚙️ 系统要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (推荐，用于GPU训练)
- 内存 >= 8GB RAM
- 显存 >= 8GB VRAM (用于训练)

## 📚 参考

- 原论文：[Learning to Cartoonize Using White-Box Cartoon Representations](https://systemerrorwang.github.io/White-box-Cartoonization/)

## 📄 许可证

本项目仅供学习和研究使用。商业使用请参考原论文的许可证要求。

```bibtex
@InProceedings{Wang_2020_CVPR,
    author = {Wang, Xinrui and Yu, Jinze},
    title = {Learning to Cartoonize Using White-Box Cartoon Representations},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

