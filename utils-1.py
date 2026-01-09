import torch
import torch.nn.functional as F
import math

def create_fisheye_spherical_grid(H, W, D, depth_hypo, fov=220, device="cuda"):
    """
    Create spherical rays for fisheye camera
    Return: [D, H, W, 3]
    """
    theta_max = math.radians(fov / 2)

    u = torch.linspace(-1, 1, W, device=device)
    v = torch.linspace(-1, 1, H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing='xy')

    r = torch.sqrt(uu**2 + vv**2)
    theta = r * theta_max
    phi = torch.atan2(vv, uu)

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    rays = torch.stack([x, y, z], dim=-1)  # [H,W,3]
    rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-6)

    rays = rays.unsqueeze(0).repeat(D, 1, 1, 1)
    return rays * depth_hypo.view(-1, 1, 1, 1)
