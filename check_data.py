import cv2
import matplotlib.pyplot as plt
import os

# Đường dẫn tới thư mục
folder_path = r"F:\Full-Dataset\FisheyeDepthDataset\FisheyeDepthDataset\Pos1\Fisheye_CamUFR"

# ID cần show
ID = "000056"

# Load ảnh
rgb_path = os.path.join(folder_path, f"{ID}_Fisheye.png")
depth_path = os.path.join(folder_path, f"{ID}_FisheyeDepth.png")

rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Hàm xử lý click chuột
def onclick(event):
    if event.inaxes == ax_depth:
        x, y = int(event.xdata), int(event.ydata)
        value = depth[y, x]  # chú ý OpenCV dùng (row=y, col=x)
        print(f"Depth raw at ({x}, {y}): {value}")

# Hiển thị ảnh
fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(12,6))

ax_rgb.imshow(rgb)
ax_rgb.set_title(f"RGB - {ID}")
ax_rgb.axis('off')

ax_depth.imshow(depth, cmap='plasma')
ax_depth.set_title(f"Depth - {ID}")
ax_depth.axis('off')

# Kết nối sự kiện click
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
