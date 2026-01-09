import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.fisheye_dataset import FisheyeMVSDataset
from models.fisheye_mvs import MultiViewFisheyeMVS, create_fisheye_spherical_grid

# -----------------------------
# Helpers
# -----------------------------
def save_depth_as_cmap(tensor, path, vmax=None):
    tensor = tensor.detach().cpu().numpy()
    if vmax is None:
        vmax = tensor.max()
    plt.imsave(path, tensor, cmap='viridis', vmin=0, vmax=vmax)

def save_img(tensor, path):
    img = tensor.detach().cpu().permute(1,2,0).numpy()
    img = (img*255).astype('uint8')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# -----------------------------
# Main
# -----------------------------
DEVICE = "cuda"
BATCH_SIZE = 4
EPOCHS = 100
D = 48
IMG_SIZE = 256
# ROOT = r"F:\Full-Dataset\FisheyeDepthDataset\data_calib\Pos9_calib\train"
# MASK_PATH = r"F:\hyp_mvs_o1\masks\mask_fisheye.png"
ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/"
# CALIB_DIR = "calib_data"
MASK_PATH = "masks/mask_fisheye.png"

dataset = FisheyeMVSDataset(ROOT, IMG_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = MultiViewFisheyeMVS().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss(reduction='none')

depth_hypo = torch.linspace(1,10,D,device=DEVICE)
grid = create_fisheye_spherical_grid(IMG_SIZE, IMG_SIZE, D, depth_hypo, DEVICE)

mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE))
mask_tensor = torch.from_numpy(mask_img.astype('float32')/255.0).to(DEVICE)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("epoch_visuals", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    epoch_dir = os.path.join("epoch_visuals", f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(loader)):
        imgs = [im.to(DEVICE) for im in batch["imgs"]]
        gt_depth = batch["depth"].to(DEVICE)

        pred_depth = model(imgs, grid, depth_hypo=depth_hypo)

        masked_loss = torch.abs(pred_depth - gt_depth) * mask_tensor.unsqueeze(0)
        loss = masked_loss.sum() / mask_tensor.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        for b in range(pred_depth.shape[0]):
            save_img(imgs[0][b], os.path.join(epoch_dir, f"ref_{batch_idx}_{b}.png"))
            save_depth_as_cmap(pred_depth[b]*mask_tensor, os.path.join(epoch_dir, f"pred_{batch_idx}_{b}.png"))
            save_depth_as_cmap(gt_depth[b]*mask_tensor, os.path.join(epoch_dir, f"gt_{batch_idx}_{b}.png"))

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

print("Training done!")
