import cv2
import numpy as np
from ultralytics import YOLO


# Load your trained YOLOv8 model for jungle animals
model = YOLO('./yolov8n.pt')  # Replace with your model's path

# Use a set to store unique identified labels
identified_labels = set()

# Start capturing video from a video file
cap = cv2.VideoCapture("bear.mp4")  # Replace with your video source

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if there are no more frames

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Perform detection using YOLOv8
    results = model(frame, stream=True)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            label = model.names[class_id]  # Animal label (name)

            # Specify jungle animal classes you want to track
            jungle_animals = ['cheetah', 'bear', 'leopard', 'tiger', 'jaguar']  # Add more as needed

            if label in jungle_animals:
                identified_labels.add(label)  # Add the label to the set

                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with tracking information
    cv2.imshow('Animal Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the identified labels at the end
if identified_labels:
    print("Identified animals:", ", ".join(identified_labels))

# Release resources
cap.release()
cv2.destroyAllWindows()


