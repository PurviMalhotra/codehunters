import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Step 1: Train the model (uncomment to train)
# Load a pre-trained model (YOLOv8n, YOLOv8s, etc.)
# model = YOLO('yolov8n.pt')

# # Train your model with your dataset
# model.train(data='data.yaml', epochs=50, imgsz=640)

# Initialize object history and parameters
object_history = {}
distance_per_pixel = 0.01  # Adjust based on your setup
frameTime = 60  # Frame time in milliseconds

# Load your trained YOLOv8 model for jungle animals
model = YOLO('path/to/your/trained_model.pt')  # Replace with your model's path

class SimpleTracker:
    def __init__(self):
        self.tracked_objects = []

    def update(self, detections):
        self.tracked_objects = detections
        return self.tracked_objects

tracker = SimpleTracker()

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if there are no more frames

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Perform detection using YOLOv8
    results = model(frame, stream=True)

    tracked_objects = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Animal label (name)

            # Specify jungle animal classes you want to track
            jungle_animals = ['cheetah', 'leopard', 'tiger', 'jaguar']  # Add more as needed

            if label in jungle_animals:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                tracked_objects.append([x1, y1, x2, y2, class_id])  # Add bounding box and ID

                # Update object history for speed calculation
                if class_id not in object_history:
                    object_history[class_id] = deque(maxlen=2)

                object_history[class_id].append(center)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if len(object_history[class_id]) == 2:
                    prev_center, curr_center = object_history[class_id]
                    distance_pixels = np.sqrt((curr_center[0] - prev_center[0]) ** 2 +
                                              (curr_center[1] - prev_center[1]) ** 2)
                    distance_real_world = distance_pixels * distance_per_pixel
                    speed = distance_real_world * (1000 / frameTime)  # Speed in meters/second

                    # Display speed information
                    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with tracking and speed information
    cv2.imshow('Animal Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
