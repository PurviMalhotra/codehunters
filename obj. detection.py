import cv2
from ultralytics import YOLO # Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use other YOLOv8 models like 'yolov8s.pt', etc.

# Start video capture from the webcam (use 0 for default webcam)
cap = cv2.VideoCapture('cats.mp4.mp4')  # Use 'video.mp4' to read from a video file instead

# Loop to continuously get frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break
    
    # Perform object detection on the current frame
    results = model(frame)
    
    # Plot the results on the frame
    annotated_frame = results[0].plot()  # Plot detections on the frame
    
    # Display the frame with detections
    cv2.imshow("Object Detection", annotated_frame)
    
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
