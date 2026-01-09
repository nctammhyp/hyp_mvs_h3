import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FisheyeMVSDatasetV2(Dataset):
    def __init__(self, root, img_size=256, mask_path=None):
        self.root = root
        self.img_size = img_size
        self.mask_path = mask_path

        self.cam1 = os.path.join(root, "Fisheye_CamUB_1")
        self.cam2 = os.path.join(root, "Fisheye_CamUFL_2")
        self.cam3 = os.path.join(root, "Fisheye_CamUFR_3")

        self.ids = sorted([
            f.split("_")[0]
            for f in os.listdir(self.cam1)
            if f.endswith("_Fisheye.png")
        ])

        if self.mask_path:
            mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (img_size, img_size))
            self.mask = torch.from_numpy((mask>0).astype(np.float32))
        else:
            self.mask = torch.ones((img_size,img_size), dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size,self.img_size))
        img = img.astype(np.float32)/255.0
        img = torch.from_numpy(img).permute(2,0,1)
        return img

    def load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = cv2.resize(depth, (self.img_size,self.img_size))
        depth = torch.from_numpy(depth.astype(np.float32))
        return depth

    def __getitem__(self, idx):
        fid = self.ids[idx]
        ref = self.load_img(os.path.join(self.cam1, f"{fid}_Fisheye.png"))
        src1 = self.load_img(os.path.join(self.cam2, f"{fid}_Fisheye.png"))
        src2 = self.load_img(os.path.join(self.cam3, f"{fid}_Fisheye.png"))
        depth = self.load_depth(os.path.join(self.cam1, f"{fid}_FisheyeDepth.png"))

        imgs = [ref, src1, src2]

        return {
            "imgs": imgs,
            "depth": depth,
            "mask": self.mask
        }
