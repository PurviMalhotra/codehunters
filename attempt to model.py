from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # You can also use 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model using the data.yaml file from your dataset
model.train(data='C:/Users/cvedi/OneDrive/Desktop/newpy/myenv/myenv/Scripts/codehunters/data.yaml', epochs=5, imgsz=640, batch=16)
