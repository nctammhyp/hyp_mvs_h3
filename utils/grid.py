import torch

def create_fisheye_spherical_grid(H, W, D, depth_hypo, device):
    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    theta = uu * torch.pi
    phi = vv * torch.pi / 2

    x = torch.cos(phi) * torch.sin(theta)
    y = torch.sin(phi)
    z = torch.cos(phi) * torch.cos(theta)

    rays = torch.stack([x, y, z], dim=-1)  # [H,W,3]
    rays = rays.unsqueeze(0).repeat(D,1,1,1)  # [D,H,W,3]
    rays = rays / torch.norm(rays, dim=-1, keepdim=True)

    return rays * depth_hypo.view(-1,1,1,1)
