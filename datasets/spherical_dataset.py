import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SphericalMVSDataset(Dataset):
    """
    Dataset cho multi-view spherical MVS
    """
    def __init__(self, root, img_size=128, n_views=3):
        self.root = root
        self.img_size = img_size
        self.n_views = n_views

        # Thư mục các camera
        self.cam_dirs = [os.path.join(root, f"Cam{i+1}") for i in range(n_views)]

        # Lấy danh sách ID từ camera đầu tiên
        self.ids = sorted([
            f.split("_")[0]
            for f in os.listdir(self.cam_dirs[0])
            if f.endswith(".png") and not f.endswith("_depth.png")
        ])

    def __len__(self):
        return len(self.ids)

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1)  # [C,H,W]
        return img

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.img_size, self.img_size))
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth)  # [H,W]
        return depth

    def __getitem__(self, idx):
        fid = self.ids[idx]
        imgs = [self.load_img(os.path.join(cam, f"{fid}.png")) for cam in self.cam_dirs]
        depth = self.load_depth(os.path.join(self.cam_dirs[0], f"{fid}_depth.png"))
        return {"imgs": imgs, "depth": depth}
