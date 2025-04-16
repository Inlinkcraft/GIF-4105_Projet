import cv2
import mediapipe as mp
import numpy as np
import scipy as sp
import sys
sys.setrecursionlimit(999999)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)


def trouverContour(pos, dir, img, step = 0.1, tol = 0.5):
    
    x = pos[0]
    y = pos[1]
    
    while (x >= 0 and y >= 0 and x < img.shape[1] and y < img.shape[0]):

        if (img[int(y), int(x)] >= tol):
            return(x, y)
        
        x += dir[0] * step
        y += dir[1] * step
        
    return (x, y)
        
    
    

def normalise(vec):
    lenght = np.sqrt(vec[0]**2 + vec[1]**2)
    return (vec[0]/lenght, vec[1]/lenght)



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
        
        """
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
        """
        
        #ref_ids = [11, 12, 13, 14, 23, 24]
        
        coude_gauche = results.pose_landmarks.landmark[13]
        coude_droit = results.pose_landmarks.landmark[14]
        epaule_gauche = results.pose_landmarks.landmark[11]
        epaule_droite = results.pose_landmarks.landmark[12]
        waist_gauche = results.pose_landmarks.landmark[23]
        waist_droite = results.pose_landmarks.landmark[24]
        track_points = [coude_gauche, coude_droit, epaule_gauche, epaule_droite, waist_gauche, waist_droite]
        
        vec_bras_droit = normalise((coude_droit.x - epaule_droite.x, coude_droit.y - epaule_droite.y))
        vec_bras_gauche = normalise((coude_gauche.x - epaule_gauche.x, coude_gauche.y - epaule_gauche.y))
        vec_epaule = normalise((epaule_gauche.x - epaule_droite.x, epaule_gauche.y - epaule_droite.y))
        
        per_vec_epaule = (vec_epaule[1], -vec_epaule[0])
        
        
        
        
        
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.bilateralFilter(gray_image, 11, 75, 75)
        edges = cv2.Canny(blurred_image, threshold1=0, threshold2=150)
        
        
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        min_area = 20  # you might need to adjust this
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        output = np.zeros_like(gray_image)
        cv2.drawContours(output, large_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        
        #start = (epaule_gauche.x * image_filtrer.shape[1], epaule_gauche.y * image_filtrer.shape[0])
        #point_test = trouverContour(start, per_vec_epaule, image_filtrer)
        
        
        
        
        
        image_filtrer = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        
        for point in track_points:
            cv2.circle(image, (int(point.x * image.shape[1]), int(point.y * image.shape[0])), 3, [0,0,255], -1)
            cv2.circle(image_filtrer, (int(point.x * image_filtrer.shape[1]), int(point.y * image_filtrer.shape[0])), 3, [0,0,255], -1)
        
        #if point_test is not None:
        #    cv2.circle(image, (int(point_test[0]), int(point_test[1])), 3, [0,255,0], -1)
        #    cv2.circle(image_filtrer, (int(point_test[0]), int(point_test[1])), 3, [0,255,0], -1)
        
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        cv2.imshow('Segmentation Mask', cv2.flip(image_filtrer,1))#image, 5) , 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
    cap.release()


