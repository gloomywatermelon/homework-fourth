import numpy as np
from PIL import Image

def iterative_threshold_segmentation(image, initial_threshold, max_iterations=100, tolerance=1e-5):
    # 将图像转换为灰度图
    image = image.convert('L')
    # 转换为NumPy数组
    img_array = np.array(image)
    # 获取图像像素总数
    total_pixels = img_array.shape[0] * img_array.shape[1]

    # 初始化阈值
    threshold = initial_threshold

    for i in range(max_iterations):
        # 将图像分割为前景和背景
        foreground_pixels = img_array[img_array > threshold]
        background_pixels = img_array[img_array <= threshold]

        # 计算前景和背景的平均灰度值
        foreground_mean = np.mean(foreground_pixels) if foreground_pixels.size > 0 else 0
        background_mean = np.mean(background_pixels) if background_pixels.size > 0 else 0

        # 更新阈值为前景和背景的平均灰度值的平均值
        new_threshold = (foreground_mean + background_mean) / 2

        # 检查阈值的变化是否小于容差
        if abs(new_threshold - threshold) < tolerance:
            break

        threshold = new_threshold

    return threshold

# 加载图像
image_path = 'image.jpg'
image = Image.open(image_path)

# 指定初始阈值并进行迭代阈值分割
initial_threshold = 128
threshold = iterative_threshold_segmentation(image, initial_threshold)
print("Iterative threshold:", threshold)

# 将图像二值化
binary_image = image.convert('L').point(lambda x: 0 if x < threshold else 255)
binary_image.show()
