import threading
import cv2
import time
import mediapipe as mp

def run_mediapipe(shared):
    
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode = False, 
        model_complexity = 1, 
        smooth_landmarks = True, 
        enable_segmentation = False, 
        smooth_segmentation = False, 
        min_detection_confidence = 0.9, 
        min_tracking_confidence = 0.9
        ) as holistic:
        
        while shared.running:
            
            with shared.lock:
                frame = shared.latest_frame.copy() if shared.latest_frame is not None else None
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            with shared.lock:
                shared.mediapipe_data = results

            
        
