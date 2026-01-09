import os
import numpy as np
import torch

# -------------------------------
# Quaternion → Rotation matrix
# -------------------------------
def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ], dtype=np.float32)


# -------------------------------
# Build intrinsic (OPENCV_FISHEYE)
# -------------------------------
def build_intrinsic_opencv_fisheye(params):
    fx, fy, cx, cy = params[:4]
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)
    return torch.from_numpy(K).unsqueeze(0)  # [1,3,3]


# -------------------------------
# Build extrinsic
# COLMAP: world → camera
# Model: camera → world
# -------------------------------
def build_extrinsic(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = np.array(tvec, dtype=np.float32).reshape(3, 1)

    extr = np.eye(4, dtype=np.float32)
    extr[:3, :3] = R
    extr[:3, 3:4] = t

    # Invert: camera → world
    extr = np.linalg.inv(extr)

    return torch.from_numpy(extr).unsqueeze(0)  # [1,4,4]


# -------------------------------
# Read cameras.txt
# -------------------------------
def read_cameras_txt(path):
    cams = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            data = line.split()
            cam_id = int(data[0])
            model = data[1]
            width = int(data[2])
            height = int(data[3])
            params = list(map(float, data[4:]))

            cams[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cams


# -------------------------------
# Read images.txt
# -------------------------------
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
        img_id = int(data[0])
        qw, qx, qy, qz = map(float, data[1:5])
        tx, ty, tz = map(float, data[5:8])
        cam_id = int(data[8])
        name = data[9]

        images[name] = {
            'qvec': [qw, qx, qy, qz],
            'tvec': [tx, ty, tz],
            'cam_id': cam_id
        }

        i += 2  # skip keypoints line

    return images


# -------------------------------
# Load COLMAP folder
# -------------------------------
def load_colmap(colmap_txt_dir):
    cams = read_cameras_txt(os.path.join(colmap_txt_dir, "cameras.txt"))
    imgs = read_images_txt(os.path.join(colmap_txt_dir, "images.txt"))

    intrinsics = {}
    extrinsics = {}

    for name, data in imgs.items():
        cam = cams[data['cam_id']]

        if cam['model'] != "OPENCV_FISHEYE":
            raise ValueError(f"Unsupported camera model: {cam['model']}")

        K = build_intrinsic_opencv_fisheye(cam['params'])
        E = build_extrinsic(data['qvec'], data['tvec'])

        intrinsics[name] = K
        extrinsics[name] = E

    return intrinsics, extrinsics


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    colmap_txt = r"F:\hyp_mvs_o1\colmap_output\3\txt"

    intrinsics, extrinsics = load_colmap(colmap_txt)

    names = sorted(intrinsics.keys())

    # Choose reference and sources
    ref_name = names[0]
    src_names = names[1:3]  # pick 2 source views

    ref_intr = intrinsics[ref_name]
    ref_ext = extrinsics[ref_name]

    src_intrs = [intrinsics[n] for n in src_names]
    src_exts = [extrinsics[n] for n in src_names]

    print("Reference:", ref_name)
    print("ref_intr shape:", ref_intr.shape)
    print("ref_ext shape:", ref_ext.shape)

    print("\nSource views:", src_names)
    print("src_intrs:", len(src_intrs))
    print("src_exts:", len(src_exts))

    # Save for model
    torch.save(ref_intr, "calib_data/ref_intr_cam3.pt")
    torch.save(ref_ext, "calib_data/ref_ext_cam3.pt")
    torch.save(src_intrs, "calib_data/src_intrs_cam3.pt")
    torch.save(src_exts, "calib_data/src_exts_cam3.pt")

    print("\nSaved:")
    print("ref_intr_cam3.pt")
    print("ref_ext_cam3.pt")
    print("src_intrs_cam3.pt")
    print("src_exts_cam3.pt")
