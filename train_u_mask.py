import os
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.fisheye_dataset import FisheyeMVSDataset
from models.fisheye_mvs import MultiViewFisheyeMVS, create_fisheye_spherical_grid
from utils.calib_loader import load_extrinsics

# ------------------------
# Helpers: lưu depth / ảnh
# ------------------------
def save_depth_as_cmap(tensor, path, vmax=None):
    tensor = tensor.detach().cpu().numpy()
    if vmax is None:
        vmax = tensor.max()
    plt.imsave(path, tensor, cmap='viridis', vmin=0, vmax=vmax)

def save_img(tensor, path):
    img = tensor.detach().cpu().permute(1,2,0).numpy()
    img = (img*255).astype('uint8')
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ------------------------
# Main
# ------------------------
if __name__=="__main__":
    DEVICE = "cuda"
    BATCH_SIZE = 4
    EPOCHS = 500
    D = 48
    IMG_SIZE = 256
    ROOT = "/home/sw-tamnguyen/Desktop/depth_project/datasets/datasets/hyp_synthetic/"
    CALIB_DIR = "calib_data"
    MASK_PATH = "masks/mask_fisheye.png"

    # Dataset & Loader
    dataset = FisheyeMVSDataset(ROOT, IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    # Model & Optimizer
    model = MultiViewFisheyeMVS().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss(reduction='none')  # áp mask

    # Load checkpoint nếu có
    checkpoint_path = "/home/sw-tamnguyen/Desktop/depth_project/hyp_mvs_h3/checkpoints/model_epoch_45_pretrain.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)  # nếu bạn lưu chỉ model.state_dict()
        # Nếu checkpoint lưu cả optimizer, thì thêm:
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded successfully!")


    # Depth hypothesis + grid
    depth_hypo = torch.linspace(1,10,D,device=DEVICE)
    grid = create_fisheye_spherical_grid(IMG_SIZE,IMG_SIZE,D,depth_hypo,DEVICE)

    # Mask
    mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE))
    mask_tensor = torch.from_numpy(mask_img.astype('float32')/255.0).to(DEVICE)

    # Training
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("epoch_visuals", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        epoch_dir = os.path.join("epoch_visuals", f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)

        for batch_idx, batch in enumerate(tqdm(loader)):
            imgs = [im.to(DEVICE) for im in batch["imgs"]]  # [ref, src1, src2]
            gt_depth = batch["depth"].to(DEVICE)           # từ camera ref

            # Extrinsics
            ref_ext, src_exts = load_extrinsics(DEVICE, gt_depth.shape[0], CALIB_DIR)

            # Forward
            pred_depth = model(imgs, grid, ref_ext, src_exts, depth_hypo)

            # Loss trên mask
            masked_loss = torch.abs(pred_depth - gt_depth) * mask_tensor.unsqueeze(0)
            loss = masked_loss.sum() / mask_tensor.sum()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Lưu ref + pred + GT (1 ảnh GT)
            for b in range(pred_depth.shape[0]):
                save_img(imgs[0][b], os.path.join(epoch_dir, f"ref_{batch_idx}_{b}.png"))
                save_depth_as_cmap(pred_depth[b]*mask_tensor, os.path.join(epoch_dir, f"pred_{batch_idx}_{b}.png"))
                save_depth_as_cmap(gt_depth[b]*mask_tensor, os.path.join(epoch_dir, f"gt_{batch_idx}_{b}.png"))

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

    print("Training done!")
