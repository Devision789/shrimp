import torch
from ultralytics import YOLO 
import supervision as sv 
import cv2
import numpy as np

# Đặt thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Thiết bị: ", device)

# Tải mô hình YOLO
model = YOLO("last.pt")
model.to(device)

# Tải hình ảnh
image = cv2.imread("2.jpg")

# Thực hiện dự đoán
results = model.predict(image, save=True)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections.with_nmm(threshold=0.5)

# Hệ số chuyển đổi từ pixel sang mm
pixel_to_mm = 0.1333

# Tính và hiển thị chiều dài của từng tôm lên ảnh
for i, box in enumerate(results.boxes.xyxy):
    x1, y1, x2, y2 = box[:4].int().tolist()
    length_in_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    length_in_mm = length_in_pixels * pixel_to_mm
    length_in_cm = length_in_mm / 10
    
    # Hiển thị kích thước và chiều của tôm lên ảnh
    label = f"{length_in_cm:.2f} cm"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Hiển thị hình ảnh
cv2.imshow("Shrimp Size and Direction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
