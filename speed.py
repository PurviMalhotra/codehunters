import numpy as np
import cv2
from collections import deque
import random


object_history = {}
distance_per_pixel = 0.01  
frameTime = 60 


def yolo_detect(frame):
    num_detections = random.randint(1, 5)  
    detections = []
    for obj_id in range(num_detections):
        x1 = random.randint(0, frame.shape[1] - 1)
        y1 = random.randint(0, frame.shape[0] - 1)
        x2 = random.randint(x1 + 1, frame.shape[1])
        y2 = random.randint(y1 + 1, frame.shape[0])
        detections.append([x1, y1, x2, y2, obj_id])
    return detections


class SimpleTracker:
    def __init__(self):
        self.tracked_objects = []

    def update(self, detections):
        self.tracked_objects = detections
        return self.tracked_objects


tracker = SimpleTracker()

# Start capturing video from a file (change the path as needed)
cap = cv2.VideoCapture('tiger.mp4')  # Replace with your video file path

while True:
    ret, frame = cap.read()
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
        break
    if not ret:
        break  # Exit the loop if there are no more frames

    # Detect objects (placeholder)
    detections = yolo_detect(frame)

    # Update tracker
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj[:5])

        # Calculate the center of the bounding box
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Check if the object is already in the history
        if obj_id not in object_history:
            object_history[obj_id] = deque(maxlen=2)
        
        # Append the current center position to the history
        object_history[obj_id].append(center)

        # Calculate speed if we have at least 2 positions for the object
        if len(object_history[obj_id]) == 2:
            prev_center, curr_center = object_history[obj_id]
            
            # Calculate Euclidean distance between the two centers (pixels)
            distance_pixels = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            
            # Convert pixel distance to real-world units
            distance_real_world = distance_pixels * distance_per_pixel
            
            # Calculate speed in real-world units per second
            speed = distance_real_world * frameTime  # speed in meters/second

            # Draw bounding box and speed on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}, Speed: {speed:.2f} m/s", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with tracking and speed information
    cv2.imshow('Object Tracking', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if cv2.getWindowProperty('Object Tracking', cv2.WND_PROP_VISIBLE) >= 1:
    cv2.destroyWindow('Object Tracking')
