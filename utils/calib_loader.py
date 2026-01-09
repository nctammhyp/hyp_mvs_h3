import torch
import os

def load_extrinsics(device, B, calib_dir):
    """
    Load extrinsics and expand to batch size B
    """
    ref_ext = torch.load(os.path.join(calib_dir, "ref_ext_cam1.pt")).to(device)
    src_exts_all = torch.load(os.path.join(calib_dir, "src_exts_cam1.pt"))

    src1_ext = src_exts_all[0].to(device)
    src2_ext = src_exts_all[1].to(device)

    # Đảm bảo shape [1,4,4] trước khi repeat
    if ref_ext.dim() == 2:
        ref_ext = ref_ext.unsqueeze(0)
    if src1_ext.dim() == 2:
        src1_ext = src1_ext.unsqueeze(0)
    if src2_ext.dim() == 2:
        src2_ext = src2_ext.unsqueeze(0)

    ref_ext = ref_ext.repeat(B,1,1)
    src1_ext = src1_ext.repeat(B,1,1)
    src2_ext = src2_ext.repeat(B,1,1)

    return ref_ext, [src1_ext, src2_ext]
