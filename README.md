# codehunters
# Animal Sighting in Rural Areas

This project aims to enhance safety and awareness in rural areas near forest regions by implementing an advanced animal detection and alert system.

## Features

1. **Camera and Sensor Network**: 
   - Deployment of cameras and sensors at elevated heights near forest regions.
   - Cameras capture visual data.
   - Sensors detect audio and infrared signals to capture movement and images of approaching animals.

2. **AI/ML-Powered Identification and Prediction**:
   - Utilizes Artificial Intelligence and Machine Learning algorithms to:
     - Identify animal species.
     - Predict proximity, speed, and estimated time of arrival based on stored data models.
   - Future plans include incorporating sound and behavioral recognition for more accurate identification.

3. **Alert System**:
   - Notifies both forest officials and local citizens upon animal identification.
   - Ensures safety measures can be taken by both parties.

4. **Real-Time Notifications**:
   - Sends real-time SMS alerts to registered users.
   - Triggers alarm notifications on users' mobile devices.

5. **Historical Tracking and Risk Assessment**:
   - Maintains a database of past animal attacks.
   - Identifies and highlights high-risk regions for increased vigilance.

      ## Getting Started

Follow these steps to set up the project and start detecting animals using YOLO:

1. **Clone the Repository**
   ```
   git clone https://github.com/your-username/animal-sighting-rural-areas.git
   cd animal-sighting-rural-areas
   ```

2. **Set Up a Python Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Prepare Your Dataset**
   - Collect images of animals you want to detect.
   - Annotate the images into YOLO format.
   - Organize your dataset into the following structure:
     ```
     dataset/
     ├── images/
     │   ├── train/
     │   └── val/
     └── labels/
         ├── train/
         └── val/
     ```

5. **Configure YOLO**
   - Create a `data.yaml` file with your dataset and class information:
     ```yaml
     train: dataset/images/train
     val: dataset/images/val
     nc: 5  # number of classes
     names: ['elephant', 'tiger', 'leopard', 'bear', 'wild_boar']  # class names
     ```

6. **Train YOLO Model**
   ```
   python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt
   ```

7. **Run Inference**
   ```
   python detect.py --source path/to/test/images --weights runs/train/exp/weights/best.pt
   ```

8. **Integrate with Alert System**
   - Modify the `detect.py` script to send alerts when animals are detected.

## Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
