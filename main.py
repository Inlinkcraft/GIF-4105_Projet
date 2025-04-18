print("START")
import cv2
import mediapipe as mp
import numpy as np
import scipy as sp
import sys
import torch
import urllib.request

# Load Holistics model
print("Loading holistics model")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode = False, model_complexity = 1, smooth_landmarks = True, enable_segmentation = False, smooth_segmentation = False, min_detection_confidence = 0.9, min_tracking_confidence = 0.9)

print("loading depth model")
# Load MiDaS model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


print("Starting camera")
cap = cv2.VideoCapture(0)

print("Executing...")
while cap.isOpened():
    
    # Get image from cam
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    #Change image flag and encoding
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Mediapipe Holistic
    results = holistic.process(image)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
    )
            
    # Convert image encoding back to original
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Pytorch Midas
    input_batch = transform(image).to(device)
        
    with torch.no_grad():
            
        prediction = midas(input_batch)
    
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
        depth_map = prediction.cpu().numpy()
    
        # Normalize for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        #countours = cv2.Canny(depth_vis, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    
    cv2.imshow('MediaPipe Holistic', cv2.resize(cv2.flip(image, 1), (image.shape[1] * 2, image.shape[0]*2)))
    cv2.imshow('Depth', cv2.flip(depth_colored,1))#image, 5) , 1))
        
    if cv2.waitKey(5) & 0xFF == 27:
        break
    
print("Closing")

holistic.close()
cap.release()

print("Good bye")