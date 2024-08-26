
import torch
from ultralytics import YOLO 
import supervision as sv 
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Thiet bi: ", device)

model = YOLO("last.pt")
model.to(device)
image = cv2.imread("2.jpg")
results = model.predict(image, save=True)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections.with_nmm(
    threshold=0.5
)
# Đếm số lượng đối tượng được phát hiện
object_count = len(detections)
print(f'Số lượng đối tượng được phát hiện: {object_count}')




'''
corner_annotator = sv.BoxCornerAnnotator()
annotated_frame = corner_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
'''
'''
ellipse_annotator = sv.EllipseAnnotator()
annotated_frame = ellipse_annotator.annotate(
    scene=image.copy(),
    detections=detections
)
'''
'''
polygon_annotator = sv.PolygonAnnotator()
annotated_frame = polygon_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

'''

