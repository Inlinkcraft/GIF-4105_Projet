import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode = False,
    model_complexity = 1,
    smooth_landmarks = True,
    enable_segmentation = True,
    smooth_segmentation = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignprong empty camera frame.")
            continue
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
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
        
        print(len(results.pose_landmarks.landmark))
        
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        cv2.imshow('Segmentation Mask', cv2.flip(results.segmentation_mask, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    cap.release()


