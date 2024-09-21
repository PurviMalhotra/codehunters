import cv2
from ultralytics import YOLO 
model = YOLO('yolov8n.pt')  

cap = cv2.VideoCapture('cats.mp4.mp4')  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break
    
    results = model(frame)
    
    annotated_frame = results[0].plot()  
    
    cv2.imshow("Object Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
