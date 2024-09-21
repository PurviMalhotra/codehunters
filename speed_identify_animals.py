import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO




object_history = {}
distance_per_pixel = 0.01  
frameTime = 60  


model = YOLO('path/to/your/trained_model.pt')  
class SimpleTracker:
    def __init__(self):
        self.tracked_objects = []

    def update(self, detections):
        self.tracked_objects = detections
        return self.tracked_objects

tracker = SimpleTracker()


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  

    frame = cv2.resize(frame, (640, 480))


    results = model(frame, stream=True)

    tracked_objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  
            label = model.names[class_id]  


            jungle_animals = ['cheetah', 'leopard', 'tiger', 'jaguar']  

            if label in jungle_animals:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                confidence = box.conf[0]  

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                tracked_objects.append([x1, y1, x2, y2, class_id])  


                if class_id not in object_history:
                    object_history[class_id] = deque(maxlen=2)

                object_history[class_id].append(center)


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if len(object_history[class_id]) == 2:
                    prev_center, curr_center = object_history[class_id]
                    distance_pixels = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                              (curr_center[1] - prev_center[1]) ** 2)
                    distance_real_world = distance_pixels * distance_per_pixel
                    speed = distance_real_world * (1000 / frameTime)  


                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('Animal Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
