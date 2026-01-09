import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_colormap(tensor, cmap="magma"):
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = (arr*255).astype(np.uint8)
    colored = plt.get_cmap(cmap)(arr)
    colored = (colored[:,:,:3]*255).astype(np.uint8)
    return colored

def save_tensor_as_img(tensor, path, cmap="magma"):
    img = apply_colormap(tensor, cmap=cmap)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
