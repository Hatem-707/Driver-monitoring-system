# from ultralytics import YOLO
# import cv2

# model = YOLO("behavior-ultimate2-v8s-100-epochs-best.pt")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     results = model(frame, conf=0.45)
#     annotated = results[0].plot()
#     cv2.imshow("Test", annotated)
#     if cv2.waitKey(1) == ord('q'):
#         break
from ultralytics import YOLO
import os

print("Current directory:", os.getcwd())
print("Files in directory:", [f for f in os.listdir('.') if f.endswith('.pt')])

# Check if model file exists
if os.path.exists("drowsiness_classifier.pt"):
    model = YOLO("drowsiness_classifier.pt")
    print("Drowsiness model classes:", model.names)
else:
    print("ERROR: drowsiness_classifier.pt not found in current directory!")
