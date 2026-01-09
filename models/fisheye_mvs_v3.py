import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from datasets.fisheye_dataset import FisheyeMVSDataset
from models.fisheye_mvs_v3 import MultiViewFisheyeMVS, create_fisheye_spherical_grid

# ------------------------
# Utils
# ------------------------
def save_tensor_as_img(tensor, path, cmap="viridis"):
    """
    tensor: [H,W] or [C,H,W] with C=1
    """
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # normalize 0-1
    plt.imsave(path, arr, cmap=cmap)

def save_mask(mask_tensor, path):
    """
    mask_tensor: [H,W] float 0-1
    """
    arr = (mask_tensor > 0.5).float().cpu().numpy()
    Image.fromarray((arr*255).astype('uint8')).save(path)

# ------------------------
# Config
# ------------------------
DEVICE = "cuda"
BATCH_SIZE = 4
EPOCHS = 10
D = 32       # number of depth hypotheses
IMG_SIZE = 256
ROOT = r"F:\Full-Dataset\FisheyeDepthDataset\data_calib\Pos9_calib\train"
MASK_PATH = r"F:\hyp_mvs_o1\masks\mask_fisheye.png"

# ------------------------
# Dataset & Loader
# ------------------------
dataset = FisheyeMVSDataset(ROOT, img_size=IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

# ------------------------
# Load mask
# ------------------------
mask = Image.open(MASK_PATH).convert("L")
mask = torch.from_numpy(np.array(mask)/255.0).float().to(DEVICE)  # [H,W]

# ------------------------
# Model, optimizer, loss
# ------------------------
model = MultiViewFisheyeMVS(n_src=2).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# Depth hypotheses
depth_hypo = torch.linspace(1, 10, D, device=DEVICE)
grid = create_fisheye_spherical_grid(IMG_SIZE, IMG_SIZE, D, depth_hypo, DEVICE)

# ------------------------
# Training
# ------------------------
os.makedirs("checkpoints_v2", exist_ok=True)
os.makedirs("samples_v2", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(loader)):
        imgs = [im.to(DEVICE) for im in batch["imgs"]]  # list of 3
        gt_depth = batch["depth"].to(DEVICE)

        # Predict depth
        pred_depth = model(imgs, grid, None, None, depth_hypo)  # no extrinsics

        # Resize mask to match pred_depth
        mask_small = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                   size=pred_depth.shape[-2:], 
                                   mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        pred_masked = pred_depth * mask_small
        gt_masked = gt_depth * mask_small

        # Loss
        loss = criterion(pred_masked, gt_masked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Save samples (first batch only)
        if batch_idx == 0:
            for b in range(BATCH_SIZE):
                sample_dir = os.path.join("samples_v2", f"epoch{epoch+1}_batch{batch_idx}_sample{b}")
                os.makedirs(sample_dir, exist_ok=True)

                # Save input images
                for i, im in enumerate(imgs):
                    im_np = (im[b].permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                    Image.fromarray(im_np).save(os.path.join(sample_dir, f"input_{i}.png"))

                # Save gt depth
                save_tensor_as_img(gt_masked[b], os.path.join(sample_dir, "gt_depth.png"))

                # Save pred depth with colormap
                save_tensor_as_img(pred_masked[b], os.path.join(sample_dir, "pred_depth.png"))

                # Save mask (small version)
                save_mask(mask_small, os.path.join(sample_dir, "mask.png"))

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"checkpoints_v2/model_epoch_{epoch+1}.pth")

print("Training done!")
