import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry
import requests


url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


checkpoint_filename = "sam_checkpoint.pth"


def download_checkpoint(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded SAM model checkpoint and saved as {filename}.")
    else:
        print(f"Failed to download the checkpoint. Status code: {response.status_code}")


download_checkpoint(url, checkpoint_filename)


model_type = "vit_h"  
sam_checkpoint = "sam_checkpoint.pth" 
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)


device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break


    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    image_input = torch.from_numpy(image_rgb).to(device)
    image_input = image_input.permute(2, 0, 1).unsqueeze(0).float() 


    masks = sam.predict(image_input)


    for mask in masks:
        mask_np = mask.cpu().numpy().astype(np.uint8)
        cv2.imshow("Mask", mask_np * 255)  


    cv2.imshow("Camera Feed", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
