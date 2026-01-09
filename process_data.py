import os
import shutil

# Thư mục nguồn
src_root = r"F:\Full-Dataset\FisheyeDepthDataset\data_calib\Pos9"

# Thư mục đích
dst_root = r"F:\Full-Dataset\FisheyeDepthDataset\data_calib\Pos9_calib"

# Duyệt tất cả folder con
for root, dirs, files in os.walk(src_root):
    # Tạo tương ứng đường dẫn đích
    relative_path = os.path.relpath(root, src_root)
    dst_folder = os.path.join(dst_root, relative_path)
    os.makedirs(dst_folder, exist_ok=True)

    # Copy các file *_Fisheye.png
    for file in files:
        if file.endswith("_Fisheye.png"):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_folder, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

print("Done!")
