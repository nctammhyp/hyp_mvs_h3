import os
import numpy as np
import torch

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float32)

def build_extrinsic(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = np.array(tvec, dtype=np.float32).reshape(3, 1)

    extr = np.eye(4, dtype=np.float32)
    extr[:3, :3] = R
    extr[:3, 3:4] = t

    extr = np.linalg.inv(extr)  # cam → world
    return torch.from_numpy(extr)

def read_images_txt(path):
    images = {}
    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            i += 1
            continue

        data = line.split()
        name = data[9]
        qvec = list(map(float, data[1:5]))
        tvec = list(map(float, data[5:8]))

        images[name] = build_extrinsic(qvec, tvec)
        i += 2

    return images
