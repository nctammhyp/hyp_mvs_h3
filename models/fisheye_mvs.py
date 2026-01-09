import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        resnet = resnet18(weights=None)
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
        self.conv = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        # Upsample to 256x256 to match GT
        x = F.interpolate(x, size=(256,256), mode="bilinear", align_corners=False)
        return x

def create_fisheye_spherical_grid(H, W, D, depth_hypo, device):
    u = torch.linspace(-1,1,W,device=device)
    v = torch.linspace(-1,1,H,device=device)
    uu, vv = torch.meshgrid(u,v,indexing='xy')
    theta = uu * torch.pi
    phi = vv * torch.pi/2
    x = torch.cos(phi)*torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi)*torch.cos(theta)
    rays = torch.stack([x,y,z],dim=-1).unsqueeze(0).repeat(D,1,1,1)
    rays = rays / torch.norm(rays, dim=-1, keepdim=True)
    return rays * depth_hypo.view(-1,1,1,1)

class CostReg3D(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch,32,3,1,1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,32,3,1,1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32,1,3,1,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(1)

def soft_argmin(cost, depth_hypo):
    prob = F.softmax(-cost, dim=1)
    return torch.sum(prob*depth_hypo.view(1,-1,1,1),dim=1)

class MultiViewFisheyeMVS(nn.Module):
    def __init__(self, n_src=2):
        super().__init__()
        self.n_src = n_src
        self.feat = FeatureExtractor()
        self.cost_reg = CostReg3D(32*(n_src+1))

    def forward(self, imgs, grid, ref_ext, src_exts, depth_hypo):
        ref = imgs[0]; srcs = imgs[1:]
        ref_f = self.feat(ref)
        src_fs = [self.feat(s) for s in srcs]
        B,C,H,W = ref_f.shape
        D = grid.shape[0]

        # Build cost volume
        ref_vol = ref_f.unsqueeze(2).repeat(1,1,D,1,1)
        vols = [ref_vol]
        for sf in src_fs:
            vols.append(sf.unsqueeze(2).repeat(1,1,D,1,1))
        cost_vol = torch.cat(vols,dim=1)
        cost = self.cost_reg(cost_vol)
        depth = soft_argmin(cost, depth_hypo)
        return depth
