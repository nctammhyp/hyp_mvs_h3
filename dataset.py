import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FisheyeDataset(Dataset):
    def __init__(self, root, resize=(256,256)):
        self.root = root
        self.resize = resize

        self.cam1 = os.path.join(root, "Fisheye_CamUB_1")
        self.cam2 = os.path.join(root, "Fisheye_CamUFL_2")
        self.cam3 = os.path.join(root, "Fisheye_CamUFR_3")

        self.ids = sorted([
            f.split("_")[0]
            for f in os.listdir(self.cam1)
            if f.endswith("_Fisheye.png")
        ])

    def __len__(self):
        return len(self.ids)

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, self.resize)
        depth = depth.astype(np.float32)
        depth = torch.from_numpy(depth)
        return depth

    def __getitem__(self, idx):
        fid = self.ids[idx]

        ref = self.load_img(os.path.join(self.cam1, f"{fid}_Fisheye.png"))
        src1 = self.load_img(os.path.join(self.cam2, f"{fid}_Fisheye.png"))
        src2 = self.load_img(os.path.join(self.cam3, f"{fid}_Fisheye.png"))

        depth = self.load_depth(os.path.join(self.cam1, f"{fid}_FisheyeDepth.png"))

        srcs = torch.stack([src1, src2], dim=0)  # [2,3,H,W]

        return {
            "ref": ref,
            "srcs": srcs,
            "depth": depth
        }
