import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from datasets.fisheye_dataset import FisheyeMVSDataset
from models.fisheye_mvs import MultiViewFisheyeMVS, create_fisheye_spherical_grid
from utils.calib_loader import load_extrinsics

# -------------------------------
# Utils: lưu ảnh và depth colormap
# -------------------------------
def save_tensor_as_img(tensor, path):
    """
    tensor: [3,H,W] float32 0~1
    """
    tensor = tensor.detach().cpu()
    img = tensor.permute(1,2,0).numpy()  # [H,W,C]
    img = np.clip(img*255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    pil_img.save(path)

def save_depth_as_cmap(tensor, path, cmap="viridis"):
    """
    tensor: [1,H,W] hoặc [H,W] float32
    """
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  # [H,W]
    elif tensor.ndim == 3:
        tensor = tensor.permute(1,2,0)

    # Normalize 0~1
    min_val = tensor.min()
    max_val = tensor.max()
    norm = (tensor - min_val) / (max_val - min_val + 1e-8)
    norm = norm.numpy()

    cmap_func = plt.get_cmap(cmap)
    colored = cmap_func(norm)[:, :, :3]  # drop alpha
    colored = (colored * 255).astype(np.uint8)

    pil_img = Image.fromarray(colored)
    pil_img.save(path)

# -------------------------------
# Main training
# -------------------------------
if __name__=="__main__":
    DEVICE = "cuda"
    BATCH_SIZE = 16
    EPOCHS = 100
    D = 48
    IMG_SIZE = 256
    ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/"
    CALIB_DIR = "calib_data"

    dataset = FisheyeMVSDataset(ROOT, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        drop_last=True)  # num_workers=0 để tránh lỗi Windows

    model = MultiViewFisheyeMVS().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    depth_hypo = torch.linspace(1,10,D,device=DEVICE)
    grid = create_fisheye_spherical_grid(IMG_SIZE, IMG_SIZE, D, depth_hypo, DEVICE)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_dir = os.path.join("outputs", f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(loader)):
            imgs = [im.to(DEVICE) for im in batch["imgs"]]  # list of 3 tensors [B,3,H,W]
            gt_depth = batch["depth"].to(DEVICE)            # [B,H,W]

            # Extrinsics
            ref_ext, src_exts = load_extrinsics(DEVICE, gt_depth.shape[0], CALIB_DIR)

            # Forward
            pred_depth = model(imgs, grid, ref_ext, src_exts, depth_hypo)  # [B,H,W]

            loss = criterion(pred_depth, gt_depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Lưu input / pred / gt cho batch nhỏ
            for b in range(gt_depth.shape[0]):
                save_tensor_as_img(imgs[0][b], os.path.join(epoch_dir, f"input_{b}.png"))
                save_depth_as_cmap(pred_depth[b], os.path.join(epoch_dir, f"pred_{b}.png"))
                save_depth_as_cmap(gt_depth[b], os.path.join(epoch_dir, f"gt_{b}.png"))

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        # Lưu checkpoint
        torch.save(model.state_dict(), os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth"))

    print("Training done!")
