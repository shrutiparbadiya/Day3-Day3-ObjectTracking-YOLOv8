import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('./object.mp4')

while True:
    ret, frame = cap.read()
    result = model.track(frame, persist=True)
    frame = result[0].plot()

    cv2.imshow('frame', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()