import matplotlib.pyplot as plt
import numpy as np

def show_depth(depth):
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()

    plt.imshow(depth, cmap="plasma")
    plt.colorbar()
    plt.show()
