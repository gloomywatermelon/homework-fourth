import numpy as np
from PIL import Image

def otsu_threshold(image):
    # 将图像转换为灰度图
    image = image.convert('L')
    # 转换为NumPy数组
    img_array = np.array(image)
    # 获取图像像素总数
    total_pixels = img_array.shape[0] * img_array.shape[1]

    # 计算灰度直方图
    histogram = np.histogram(img_array, bins=256, range=(0, 255), density=True)[0]

    # 初始化参数
    best_threshold = 0
    max_variance = 0

    # 遍历所有可能的阈值
    for threshold in range(256):
        # 计算前景和背景像素的概率和平均灰度值
        w0 = np.sum(histogram[:threshold])
        w1 = np.sum(histogram[threshold:])
        u0 = np.sum(np.arange(threshold) * histogram[:threshold]) / w0 if w0 > 0 else 0
        u1 = np.sum(np.arange(threshold, 256) * histogram[threshold:]) / w1 if w1 > 0 else 0

        # 计算类内方差
        variance = w0 * w1 * ((u0 - u1) ** 2)

        # 更新最大方差和阈值
        if variance > max_variance:
            max_variance = variance
            best_threshold = threshold

    return best_threshold

# 加载图像
image_path = 'image.jpg'
image = Image.open(image_path)

# 应用大津法获取阈值
threshold = otsu_threshold(image)
print("Otsu's threshold:", threshold)

# 将图像二值化
binary_image = image.convert('L').point(lambda x: 0 if x < threshold else 255)
binary_image.show()
