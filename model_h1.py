import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18

# -----------------------------
# Feature extractor
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=False, out_channels=32):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.conv_reduce = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.conv_reduce(feat)
        feat = F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        return feat


# -----------------------------
# Spherical grid for fisheye
# -----------------------------
def create_fisheye_spherical_grid(H, W, D, depth_hypo, fov_deg=220, device="cpu"):
    fov = torch.deg2rad(torch.tensor(fov_deg, device=device))

    u = torch.linspace(0, W-1, W, device=device)
    v = torch.linspace(0, H-1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    x = (uu / (W-1) - 0.5) * 2
    y = (vv / (H-1) - 0.5) * 2

    r = torch.sqrt(x**2 + y**2)
    theta = r * (fov / 2)
    phi = torch.atan2(y, x)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    rays = torch.stack([
        sin_t * torch.cos(phi),
        sin_t * torch.sin(phi),
        cos_t
    ], dim=-1)

    rays = rays / torch.norm(rays, dim=-1, keepdim=True)
    rays = rays.unsqueeze(0).repeat(D,1,1,1)

    return rays * depth_hypo.view(-1,1,1,1)


# -----------------------------
# Project fisheye
# -----------------------------
def project_fisheye(X, H, W, fov_deg=220):
    fov = torch.deg2rad(torch.tensor(fov_deg, device=X.device))

    x, y, z = X[:,0], X[:,1], X[:,2]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(r, z)
    phi = torch.atan2(y, x)

    rn = theta / (fov / 2)

    u = rn * torch.cos(phi)
    v = rn * torch.sin(phi)

    u = (u + 1) / 2 * (W-1)
    v = (v + 1) / 2 * (H-1)

    u_norm = 2 * u / (W-1) - 1
    v_norm = 2 * v / (H-1) - 1

    return torch.stack([u_norm, v_norm], dim=-1)


# -----------------------------
# Warp fisheye spherical
# -----------------------------
def spherical_fisheye_warp(src_feat, grid, src_ext, ref_ext, fov=220):
    B,C,H,W = src_feat.shape
    D = grid.shape[0]

    warped = []

    for d in range(D):
        pts = grid[d].reshape(1, H*W, 3).permute(0,2,1)

        # ref → world
        Rr = ref_ext[:,:3,:3]
        tr = ref_ext[:,:3,3:4]
        Xw = Rr @ pts + tr

        # world → src
        Rs = src_ext[:,:3,:3]
        ts = src_ext[:,:3,3:4]
        Xs = Rs.transpose(1,2) @ (Xw - ts)

        grid_uv = project_fisheye(Xs, H, W, fov)
        grid_uv = grid_uv.view(B, H, W, 2)

        warp = F.grid_sample(
            src_feat,
            grid_uv,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        warped.append(warp)

    warped = torch.stack(warped, dim=2)
    return warped


# -----------------------------
# Cost volume
# -----------------------------
def build_cost_volume_fisheye(ref_feat, src_feats, grid, ref_ext, src_exts):
    ref_vol = ref_feat.unsqueeze(2).repeat(1,1,grid.shape[0],1,1)

    warped_list = [
        spherical_fisheye_warp(src, grid, se, ref_ext)
        for src, se in zip(src_feats, src_exts)
    ]

    cost_vol = torch.cat([ref_vol] + warped_list, dim=1)
    return cost_vol


# -----------------------------
# 3D CNN
# -----------------------------
class CostReg3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,32,3,1,1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32,32,3,1,1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32,1,3,1,1)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x.squeeze(1)


# -----------------------------
# Soft argmin
# -----------------------------
def soft_argmin(cost_vol, depth_hypo):
    prob = F.softmax(-cost_vol, dim=1)
    depth_hypo = depth_hypo.view(1,-1,1,1).to(cost_vol.device)
    return torch.sum(prob * depth_hypo, dim=1)


# -----------------------------
# Full model
# -----------------------------
class MultiViewSphericalFisheyeMVS(nn.Module):
    def __init__(self, n_src=2):
        super().__init__()
        self.n_src = n_src
        self.feature_extractor = FeatureExtractor()
        self.cost_reg = CostReg3D(in_channels=32*(n_src+1))

    def forward(self, imgs, grid, ref_ext, src_exts, depth_hypo):
        ref_img = imgs[0]
        src_imgs = imgs[1:]

        ref_feat = self.feature_extractor(ref_img)
        src_feats = [self.feature_extractor(im) for im in src_imgs]

        cost_vol = build_cost_volume_fisheye(
            ref_feat, src_feats, grid, ref_ext, src_exts
        )

        cost_vol = self.cost_reg(cost_vol)
        depth_map = soft_argmin(cost_vol, depth_hypo)

        return depth_map


# -----------------------------
# Demo
# -----------------------------
if __name__ == "__main__":
    B,C,H,W = 1,3,128,128
    D = 32

    depth_hypo = torch.linspace(0.5, 10, D)
    grid = create_fisheye_spherical_grid(H,W,D,depth_hypo)

    imgs = [torch.rand(B,C,H,W) for _ in range(3)]
    ref_ext = torch.eye(4).unsqueeze(0)
    src_exts = [torch.eye(4).unsqueeze(0) for _ in range(2)]

    model = MultiViewSphericalFisheyeMVS(n_src=2)
    depth = model(imgs, grid, ref_ext, src_exts, depth_hypo)

    print("Depth:", depth.shape)
