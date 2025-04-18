import threading
import cv2
import mediapipe as mp
from shared_data import SharedData
from mediapipe_thread import run_mediapipe
from midas_thread import run_midas

def main(debug):
    
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
    
    cap = cv2.VideoCapture(0)
    
    print("Running")
    
    while cap.isOpened():
        
        success, frame = cap.read()
        if not success:
            continue
        
        with shared.lock:
            shared.latest_frame = frame.copy()
        
        with shared.lock:
            mp_data = shared.mediapipe_data
            midas_data = shared.midas_data
        
        if debug == True:
            
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
                cv2.imshow("Midas", cv2.flip(cv2.applyColorMap(midas_data, cv2.COLORMAP_MAGMA), 1))
        
        ### This is where it gets fun
        
        cv2.imshow('Result', cv2.flip(frame, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    print("Closing")
    
    shared.running = False
    media_thread.join()
    midas_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(True)