import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# -----------------------------
# 1. Feature extractor
# -----------------------------
class ResNet18FullRes(nn.Module):
    def __init__(self, pretrained=True, out_channels=64):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        self.conv1 = nn.Conv2d(3, 64, 7, stride=1, padding=3, bias=False)
        self.conv1.weight.data.copy_(resnet.conv1.weight.data)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu

        self.layer1 = resnet.layer1
        self.layer2 = self._make_layer_fullres(resnet.layer2)
        self.layer3 = self._make_layer_fullres(resnet.layer3)
        self.layer4 = self._make_layer_fullres(resnet.layer4)

        self.out_conv = nn.Conv2d(512, out_channels, 1)

    def _make_layer_fullres(self, layer):
        modules = []
        for block in layer:
            block.conv1.stride = (1,1)
            block.conv2.dilation = (2,2)
            block.conv2.padding = (2,2)
            if block.downsample is not None:
                in_c = block.downsample[0].in_channels
                out_c = block.downsample[0].out_channels
                block.downsample = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
                    nn.BatchNorm2d(out_c)
                )
            modules.append(block)
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_conv(x)
        return x

# -----------------------------
# 2. Spherical grid
# -----------------------------
def create_simple_spherical_grid(H,W,D,device):
    u = torch.linspace(-1,1,W,device=device)
    v = torch.linspace(-1,1,H,device=device)
    uu, vv = torch.meshgrid(u,v, indexing='xy')
    grid = torch.stack([uu,vv], dim=-1)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(D,1,1,1)  # [D,H,W,2]
    return grid

# -----------------------------
# 3. Warp features
# -----------------------------
def warp_features(src_feat, grid):
    B,C,H,W = src_feat.shape
    D = grid.shape[0]
    warped = []
    for d in range(D):
        warp = F.grid_sample(src_feat, grid[d].unsqueeze(0).repeat(B,1,1,1),
                             mode='bilinear', padding_mode='zeros', align_corners=True)
        warped.append(warp)
    return torch.stack(warped, dim=2)  # [B,C,D,H,W]

# -----------------------------
# 4. Cost volume
# -----------------------------
def build_cost_volume(ref_feat, src_feats, grid):
    ref_vol = ref_feat.unsqueeze(2).repeat(1,1,grid.shape[0],1,1)
    warped_list = [warp_features(src, grid) for src in src_feats]
    return torch.cat([ref_vol]+warped_list, dim=1)

# -----------------------------
# 5. 3D CNN regularization
# -----------------------------
class CostReg3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels,64,3,1,1), nn.BatchNorm3d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv3d(64,64,3,1,1), nn.BatchNorm3d(64), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv3d(64,32,3,1,1), nn.BatchNorm3d(32), nn.ReLU())
        self.layer4 = nn.Conv3d(32,1,3,1,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.squeeze(1)

# -----------------------------
# 6. Soft-argmin
# -----------------------------
def soft_argmin(cost_vol, depth_hypo):
    prob = F.softmax(-cost_vol, dim=1)
    depth_hypo = depth_hypo.view(1,-1,1,1).to(cost_vol.device)
    return torch.sum(prob*depth_hypo, dim=1)

# -----------------------------
# 7. Multi-view spherical MVS
# -----------------------------
class MultiViewSphericalMVS_FullRes(nn.Module):
    def __init__(self, n_src=2, feat_channels=64):
        super().__init__()
        self.n_src = n_src
        self.feature_extractor = ResNet18FullRes(out_channels=feat_channels)
        self.cost_reg = CostReg3D(feat_channels*(n_src+1))

    def forward(self, imgs, grid, depth_hypo):
        ref_img = imgs[0]
        src_imgs = imgs[1:]
        ref_feat = self.feature_extractor(ref_img)
        src_feats = [self.feature_extractor(im) for im in src_imgs]
        cost_vol = build_cost_volume(ref_feat, src_feats, grid)
        cost_vol = self.cost_reg(cost_vol)
        depth_map = soft_argmin(cost_vol, depth_hypo)
        return depth_map
