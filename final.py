import torch
from ultralytics import YOLO, solutions
import supervision as sv
import cv2
import numpy as np

# Đặt thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Thiết bị: ", device)

# Tải mô hình YOLO
model = YOLO("last.pt")
model.to(device)

# Tải video
cap = cv2.VideoCapture("1.MOV")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Hệ số chuyển đổi từ pixel sang mm
pixel_to_mm = 0.1333

# Định nghĩa các điểm vùng (region) và khởi tạo Object Counter
region_points = [(0, h), (w, h), (w, h/2), (0, h/2)]
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

# Khởi tạo video writer
video_writer = cv2.VideoWriter("object.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Thực hiện dự đoán
    results = model.track(im0, persist=True, show=False)
    
    

    # Tính và hiển thị chiều dài của từng đối tượng lên video
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = box[:4].int().tolist()
        length_in_pixels = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        length_in_mm = length_in_pixels * pixel_to_mm
        length_in_cm = length_in_mm / 10

        # Hiển thị kích thước lên video
        label = f"{length_in_cm:.2f} cm"
        cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Đếm đối tượng trong video
    im0 = counter.start_counting(im0, results)
    
    # Ghi lại khung hình đã xử lý vào file video output
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
