import os
import random
import shutil


def move_images(train_folder, validation_folder, percentage=0.3):
    """将每个子文件夹中的图像随机抽取指定百分比并移动到目标文件夹"""

    # 确保目标文件夹存在
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    # 遍历 train 文件夹中的所有子文件夹
    for subdir in os.listdir(train_folder):
        subdir_path = os.path.join(train_folder, subdir)

        if os.path.isdir(subdir_path):
            # 获取子文件夹中的所有图像文件
            images = [file for file in os.listdir(subdir_path) if
                      file.lower().endswith((".jpg", ".jpeg", ".webp", ".png"))]

            # 计算要移动的图像数量
            num_images_to_move = int(len(images) * percentage)

            # 随机选择图像文件
            images_to_move = random.sample(images, num_images_to_move)

            # 确保目标子文件夹存在
            target_subfolder = os.path.join(validation_folder, subdir)
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)

            # 移动选中的图像
            for image in images_to_move:
                src_image_path = os.path.join(subdir_path, image)
                dest_image_path = os.path.join(target_subfolder, image)
                shutil.move(src_image_path, dest_image_path)
                print(f"已移动: {src_image_path} 到 {dest_image_path}")


# 使用示例
train_folder = r"..\data\train"  # 替换为实际的 train 文件夹路径
validation_folder = r"..\data\valid"  # 替换为实际的 validation 文件夹路径

move_images(train_folder, validation_folder)
