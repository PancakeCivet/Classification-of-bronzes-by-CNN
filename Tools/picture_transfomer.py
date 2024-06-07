import os
import json
from PIL import Image
import shutil

def load_bronze_vessel_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {item["name"]: item["id"] for item in data["bronze_vessel"]}

def convert_and_copy_images(input_path, output_path):
    for filename in os.listdir(input_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".webp")):
            input_file = os.path.join(input_path, filename)
            try:
                image = Image.open(input_file)

                if image.mode == 'CMYK':
                    image = image.convert('RGB')
                    print(f"已将图像 {input_file} 从 CMYK 转换为 RGB")

                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_path, base_name + ".png")

                image.save(output_file, "PNG")
                print(f"已转换: {input_file} 到 {output_file}")
            except Exception as e:
                print(f"处理图像 {input_file} 时出错: {e}")

        elif filename.lower().endswith(".png"):
            input_file = os.path.join(input_path, filename)
            output_file = os.path.join(output_path, filename)
            try:
                shutil.copy2(input_file, output_file)
                print(f"已复制: {input_file} 到 {output_file}")
            except Exception as e:
                print(f"复制文件 {input_file} 时出错: {e}")

def main():
    # 路径设置
    json_file_path = r"..\bronze_vessel.json"
    before_classify_folder = r"..\data\before_classifty"
    train_folder = r"..\data\train"

    bronze_vessel_mapping = load_bronze_vessel_data(json_file_path)

    for subdir in os.listdir(before_classify_folder):
        subdir_path = os.path.join(before_classify_folder, subdir)
        if os.path.isdir(subdir_path) and subdir in bronze_vessel_mapping:
            folder_id = bronze_vessel_mapping[subdir]

            target_subfolder = os.path.join(train_folder, str(folder_id))
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)

            convert_and_copy_images(subdir_path, target_subfolder)

if __name__ == "__main__":
    main()
