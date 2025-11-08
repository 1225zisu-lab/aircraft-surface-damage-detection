from ultralytics import YOLO
import cv2

#Load model
model = YOLO("best.pt") #or "best.pt" once you remane

#Load test image
img = cv2.imread("test.jpg") #change to image path
results = model(img)

annotated = results[0].plot()
cv2.imwrite("output.jpg", annotated)
print("Detection complete! Saved result -> output.jpg")

