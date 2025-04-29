import threading
import cv2
import mediapipe as mp
import overlay
import numpy as np
from shared_data import SharedData
from mediapipe_thread import run_mediapipe
from midas_thread import run_midas

global DEBUG
DEBUG = False

def main():
    
    # Debugin tools
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
     
    print("Initializing")
    shared = SharedData()
    
    media_thread = threading.Thread(target=run_mediapipe, args=(shared,))
    midas_thread = threading.Thread(target=run_midas, args=(shared,))
    
    media_thread.start()
    midas_thread.start()
    
    ref_t_shirt = {}
    ref_t_shirt["image"] = cv2.imread("../Inputs/t_turtle.jpg")
    if ref_t_shirt["image"] is None:
        print("Failed to load image.")
    ref_t_shirt["inter"] = {
        "r": ref_t_shirt["image"][:,:,0].astype(np.float32),
        "g": ref_t_shirt["image"][:,:,1].astype(np.float32),
        "b": ref_t_shirt["image"][:,:,2].astype(np.float32)
    }
    ref_t_shirt["points"] = loadpoints("../Inputs/t_turtle.txt")


    cap = cv2.VideoCapture(0)
    
    print("Running")
    
    while cap.isOpened():
        
        success, frame = cap.read()
        if not success:
            continue
        
        cv2.imshow('Camera input', cv2.flip(frame, 1))
        
        frame = make_square(frame)
        
        with shared.lock:
            shared.latest_frame = frame.copy()
        
        with shared.lock:
            mp_data = shared.mediapipe_data
            midas_data = shared.midas_data
        
        if DEBUG == True:
            
            cv2.imshow("Ref image", cv2.resize(ref_t_shirt["image"].copy(), (int(ref_t_shirt["image"].shape[1]/4), int(ref_t_shirt["image"].shape[0]/4)))) 
            
            if mp_data is not None and mp_data.pose_landmarks:
                media_pipe_debug_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    media_pipe_debug_frame,
                    mp_data.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
                )
                cv2.imshow("Mediapipe", cv2.flip(media_pipe_debug_frame, 1))
        
            if midas_data is not None:
                
                depth_min = midas_data.min()
                depth_max = midas_data.max()
                depth_vis = (255 * (midas_data - depth_min) / (depth_max - depth_min)).astype(np.uint8)
                
                cv2.imshow("Midas", cv2.flip(cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA), 1))
        
        ### This is where it gets fun
        
        if mp_data is not None and mp_data.pose_landmarks and midas_data is not None:
            compose_image = frame.copy()
            
            overlay.t_shirt(compose_image, ref_t_shirt, mp_data.pose_landmarks.landmark, midas_data)
            
            cv2.imshow('Result', cv2.flip(compose_image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    print("Closing")
    
    shared.running = False
    media_thread.join()
    midas_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    
    
def make_square(frame):
    height, width = frame.shape[:2]
    size = min(height, width)
    remove_side = int((width - size)/2)
    remove_top = int((height - size)/2)
    return frame[remove_top:-(remove_top+1),remove_side:-(remove_side+1),:]

def loadpoints(filepath):
    f = open(filepath, "r")
    text_data = f.read() 

    points = []
    text_points = text_data.split("\n")
    for text_point in text_points:
        if text_point != "":
            data = text_point.split("\t")
            x = float(data[1].strip())
            y = float(data[0].strip())
            p = np.array([x, y])
            points.append(p)
        else:
            print("ATTENTION: " + text_point + " wasn't read")
    
    f.close()
    
    return np.array(points)

    
if __name__ == "__main__":
    main()