import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry
import requests

# URL of the SAM model checkpoint
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Specify the filename for saving the checkpoint
checkpoint_filename = "sam_checkpoint.pth"

# Function to download the checkpoint
def download_checkpoint(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded SAM model checkpoint and saved as {filename}.")
    else:
        print(f"Failed to download the checkpoint. Status code: {response.status_code}")

# Call the function to download the checkpoint
download_checkpoint(url, checkpoint_filename)

# Load the SAM model
model_type = "vit_h"  # You can use "vit_b" or "vit_l" as well
sam_checkpoint = "sam_checkpoint.pth"  # Path to your downloaded checkpoint
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare the image for the model
    image_input = torch.from_numpy(image_rgb).to(device)
    image_input = image_input.permute(2, 0, 1).unsqueeze(0).float()  # Add batch dimension

    # Perform object identification
    masks = sam.predict(image_input)

    # Display the results
    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        cv2.imshow("Mask", mask_np * 255)  # Show the mask

    # Display the original frame
    cv2.imshow("Camera Feed", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
