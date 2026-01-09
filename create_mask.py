import cv2
import numpy as np

# 1. Đọc ảnh
image = cv2.imread(r"F:\Full-Dataset\FisheyeDepthDataset\data_calib\Pos9_calib\train\Fisheye_CamUB_1\000000_FisheyeDepth.png")
h, w = image.shape[:2]

# 2. Tạo một mask đen hoàn toàn (cùng kích thước với ảnh)
mask = np.zeros((h, w), dtype=np.uint8)

# 3. Vẽ một hình tròn trắng ở giữa
# Tâm hình tròn là trung tâm ảnh, bán kính thường là nửa chiều rộng hoặc chiều cao
center = (w // 2, h // 2)
radius = min(h, w) // 2
cv2.circle(mask, center, radius, 255, -1)

# 4. Áp dụng mask vào ảnh gốc (tùy chọn)
# Chuyển các vùng ngoài vòng tròn thành màu đen (hoặc bất kỳ màu nào bạn muốn)
result = cv2.bitwise_and(image, image, mask=mask)

# 5. Lưu hoặc hiển thị kết quả
cv2.imwrite('mask_fisheye.png', mask)
cv2.imwrite('result_fisheye.png', result)

print("Đã tạo mask thành công!")