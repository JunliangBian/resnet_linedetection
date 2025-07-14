import numpy as np
import cv2
import glob

# 读取所有训练图像路径
image_paths = glob.glob("datasets_480_480/train/image/*.png")  # 修改为你的训练数据路径
if len(image_paths) == 0:
    raise ValueError("Error: No images found, please check the dataset path!")

# 存储像素值总和
mean_r, mean_g, mean_b = 0, 0, 0
num_pixels = 0  # 总像素数

for img_path in image_paths:
    img = cv2.imread(img_path)  # 读取图像 (BGR 格式)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    h, w, c = img.shape
    num_pixels += h * w  # 计算总像素数
    mean_r += np.sum(img[:, :, 0])
    mean_g += np.sum(img[:, :, 1])
    mean_b += np.sum(img[:, :, 2])

# 计算均值
mean_r /= num_pixels
mean_g /= num_pixels
mean_b /= num_pixels

print(f"Mean values: {mean_r:.4f}, {mean_g:.4f}, {mean_b:.4f}")

std_r, std_g, std_b = 0, 0, 0

for img_path in image_paths:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    std_r += np.sum((img[:, :, 0] - mean_r) ** 2)
    std_g += np.sum((img[:, :, 1] - mean_g) ** 2)
    std_b += np.sum((img[:, :, 2] - mean_b) ** 2)

std_r = (std_r / num_pixels) ** 0.5
std_g = (std_g / num_pixels) ** 0.5
std_b = (std_b / num_pixels) ** 0.5

scale_r = 1.0 / std_r
scale_g = 1.0 / std_g
scale_b = 1.0 / std_b

print(f"Scale values: {scale_r:.7f}, {scale_g:.7f}, {scale_b:.7f}")
