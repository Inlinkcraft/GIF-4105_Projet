import cv2
import mediapipe as mp
import numpy as np
import scipy as sp
import sys
import torch
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





def fast_depth_region(depth_map, x, y, threshold):
    z = depth_map[y, x]

    # Step 1: Create a mask of all pixels within depth range
    mask = np.logical_and(depth_map >= z - threshold, depth_map <= z + threshold).astype(np.uint8) * 255

    # Step 2: Flood fill from seed point (connected region only)
    h, w = mask.shape
    mask_copy = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)  # Required by cv2.floodFill

    # Flood fill modifies mask_copy in-place
    cv2.floodFill(mask_copy, flood_mask, (x, y), 128)  # Use 128 to mark the region

    # Extract only the filled region
    region_mask = (mask_copy == 128).astype(np.uint8) * 255
    return region_mask

def between_point_region(depth_map, x1, y1, x2, y2, threshold, resolution = 100):
    
    xs = np.linspace(x1, x2, resolution)
    ys = np.linspace(y1, y2, resolution)
    
    v_max = np.max(depth_map[y1, x1], depth_map[y2, x2])
    v_min = np.min(depth_map[y1, x1], depth_map[y2, x2])
    
    mask = np.zeros_like(depth_map)
    for x, y in zip (xs, ys):
        if depth_map[int(y), int(x)] > v_min and depth_map[int(y), int(x)] < v_max:
            mask = np.maximum(mask, fast_depth_region(depth_map, int(x), int(y), threshold))
        
    return mask





mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # or "MiDaS_small"
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in str(type(midas)) else midas_transforms.small_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

cap = cv2.VideoCapture(0)

#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal')
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')

with mp_holistic.Holistic(
    static_image_mode = False,
    model_complexity = 1,
    smooth_landmarks = True,
    enable_segmentation = False,
    smooth_segmentation = False,
    min_detection_confidence = 0.9,
    min_tracking_confidence = 0.9) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignprong empty camera frame.")
            continue
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        input_batch = transform(image).to(device)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        with torch.no_grad():
            prediction = midas(input_batch)

            # Resize to original size
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
        
        """
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        """
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        )
        """
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS#,
            #landmark_drawing_spec = mp_drawing_styles.get_default_right_hand_landmarks_style()
        )
        
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS#,
            #landmark_drawing_spec = mp_drawing_styles.get_default_right_hand_landmarks_style()
        )
        """
        #ax.cla()

        landmark_id = [12, 11, 24, 23]#[11, 12, 13, 14, 15, 16, 24, 23]
        landmarks_data = []

        A = np.empty((1,12))

        region = np.zeros_like(depth_vis)

        # Draw landmarks
        if results.pose_landmarks and results.pose_world_landmarks:
            for i in landmark_id:
                lm_2d = results.pose_landmarks.landmark[i]
                lm_3d = results.pose_world_landmarks.landmark[i]
                
                combine_visibility = lm_2d.visibility# + lm_3d.visibility
                landmark_data = {
                    "index":i,
                    "2d":{"x":lm_2d.x, "y":lm_2d.y},
                    "3d":{"x":lm_3d.x, "y":lm_3d.y, "z":lm_3d.z},
                    "visibility":combine_visibility
                }
                    
                A = np.vstack([A, np.array([[lm_3d.x, lm_3d.y, lm_3d.z, 1, 0, 0, 0, 0, -(lm_3d.x * lm_2d.x), -(lm_3d.y * lm_2d.x), -(lm_3d.z * lm_2d.x), -lm_2d.x],
                                            [0, 0, 0, 0, lm_3d.x, lm_3d.y, lm_3d.z, 1, -(lm_3d.x * lm_2d.y), -(lm_3d.y * lm_2d.y), -(lm_3d.z * lm_2d.y), -lm_2d.y]])])
                    
                    
                landmarks_data.append(landmark_data)
                
            track_start = results.pose_landmarks.landmark[11]
            track_end = results.pose_landmarks.landmark[13]
            if (track_start.x > 0 and track_start.x < 1 and track_start.y > 0 and track_start.y < 1
                and track_end.x > 0 and track_end.x < 1 and track_end.y > 0 and track_end.y < 1):
                region = between_point_region(depth_vis, int(track_start.x * depth_vis.shape[1]), int(track_start.y * depth_vis.shape[0]), int(track_end.x * depth_vis.shape[1]), int(track_end.y * depth_vis.shape[0]), 0.5)
                    
        U, S, VT = np.linalg.svd(A)
        proj_vec = VT[-1]
        
        proj = np.reshape(proj_vec, (3,4))
        
        for lm in landmarks_data:
            proj_pos = proj @ np.array([lm["3d"]["x"], lm["3d"]["y"], lm["3d"]["z"], 1])
            screen_pos = (int((proj_pos[0]/proj_pos[2]) * image.shape[1]), int((proj_pos[1]/proj_pos[2]) * image.shape[0]))
            #print(screen_pos)
            if screen_pos[0] > 0 and screen_pos[1] > 0 and screen_pos[0] < image.shape[1] and screen_pos[1] < image.shape[0]:
                image = cv2.circle(image, (screen_pos[0], screen_pos[1]), radius=3, color=(0, 0, 255), thickness=-1)
            
        
        
        cv2.imshow('MediaPipe Holistic', cv2.resize(cv2.flip(image, 1), (image.shape[1] * 2, image.shape[0]*2)))
        cv2.imshow('Depth', cv2.flip(depth_colored,1))#image, 5) , 1))
        
        tracked_img = np.zeros_like(image)
        tracked_img[:, :, 0] = image[:, :, 0] * region
        tracked_img[:, :, 1] = image[:, :, 1] * region
        tracked_img[:, :, 2] = image[:, :, 2] * region
        
        cv2.imshow('Region Mask', cv2.flip(tracked_img,1))#image, 5) , 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    cap.release()


