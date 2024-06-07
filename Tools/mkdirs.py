import os

parent_dir = "../data/train/"

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

for i in range(17):
    folder_name = str(i)
    folder_path = os.path.join(parent_dir, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建文件夹: {folder_path}")
    else:
        print(f"文件夹已存在: {folder_path}")
