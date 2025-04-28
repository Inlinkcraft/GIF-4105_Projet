import cv2
import numpy as np
import scipy as sp
from mesurements import BODY_SIZE

TSHIRT_TRI_INDEX = [
    0,6,5,
    0,1,6,
    1,7,6,
    1,2,7,
    2,8,7,
    2,3,8,
    3,9,8,
    3,4,9,
    22,21,0,
    21,1,0,
    1,21,2,
    2,21,3,
    3,21,20,
    3,20,4,
    22,25,21,
    21,25,24,
    21,24,20,
    20,24,23,
    24,26,23,
    24,27,26,
    28,31,27,
    27,31,30,
    27,30,26,
    26,30,29,
    31,14,30,
    30,14,13,
    30,13,12,
    30,12,11,
    30,11,29,
    29,11,10,
    14,19,18,
    14,18,13,
    13,18,17,
    13,17,12,
    12,17,16,
    12,16,11,
    11,16,15,
    11,15,10,
    34,4,20,
    34,20,33,
    33,20,32,
    20,23,32,
    32,23,26,
    32,26,41,
    41,26,29,
    41,29,42,
    42,29,43,
    43,29,10,
    37,34,33,
    37,33,36,
    36,33,35,
    35,33,32,
    35,32,44,
    32,41,44,
    44,41,42,
    44,42,45,
    45,42,46,
    46,42,43,
    40,37,39,
    39,37,36,
    39,36,35,
    39,35,38,
    38,35,44,
    38,44,47,
    47,44,48,
    48,44,45,
    48,45,46,
    48,46,49
]

def get_vector_lenght(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def normalise_vec(vec):
    return vec / get_vector_lenght(vec)

def t_shirt(compose_image, t_shirt_ref, pose_landmarks, depth):
    
    # Bicept gauche
    epaule_gauche_id = 11
    coude_gauche_id = 13
    epaule_droite_id = 12
    coude_droite_id = 14
    taille_gauche_id = 23
    taille_droite_id = 24
    
    epaule_gauche_lm = pose_landmarks[epaule_gauche_id]
    coude_gauche_lm = pose_landmarks[coude_gauche_id]
    epaule_droite_lm = pose_landmarks[epaule_droite_id]
    coude_droite_lm = pose_landmarks[coude_droite_id]
    taille_gauche_lm = pose_landmarks[taille_gauche_id]
    taille_droite_lm = pose_landmarks[taille_droite_id]
    
    epaule_gauche_pos = extract_3d_position(epaule_gauche_lm, depth)
    coude_gauche_pos = extract_3d_position(coude_gauche_lm, depth)
    epaule_droite_pos = extract_3d_position(epaule_droite_lm, depth)
    coude_droite_pos = extract_3d_position(coude_droite_lm, depth)
    taille_gauche_pos = extract_3d_position(taille_gauche_lm, depth)
    taille_droite_pos = extract_3d_position(taille_droite_lm, depth)
    
    points = []
    
    if coude_gauche_pos is not None and epaule_gauche_pos is not None:
        #estimated_circonference= BODY_SIZE["LEFT_ARM"]["BICEP"],estimated_length= BODY_SIZE["LEFT_ARM"]["BICEP_LENGHT"],
        points.extend(add_cyl_points(epaule_gauche_pos, coude_gauche_pos,n_points=5, d_angle = np.pi/5, angle_offset=np.pi/2,subdivision=[0.25, 0.7]))
    
    if coude_droite_pos is not None and epaule_droite_pos is not None:
        #estimated_circonference= BODY_SIZE["RIGHT_ARM"]["BICEP"],estimated_length= BODY_SIZE["RIGHT_ARM"]["BICEP_LENGHT"],
        points.extend(add_cyl_points(epaule_droite_pos, coude_droite_pos,n_points=5, d_angle = np.pi/5, angle_offset=np.pi/2,subdivision=[0.25, 0.7]))
    
    if epaule_droite_pos is not None and epaule_gauche_pos is not None:
        points.extend(add_cyl_points(epaule_gauche_pos, epaule_droite_pos, n_points=3, angle_offset=np.pi, subdivision=[0.1, 0.33, 0.66, 0.9]))
    
    if taille_gauche_pos is not None and epaule_gauche_pos is not None:
        points.extend(add_cyl_points(epaule_gauche_pos, taille_gauche_pos, n_points=3, angle_offset=0, subdivision=[0.4, 0.6, 1]))
    
    if taille_droite_pos is not None and epaule_droite_pos is not None:
        points.extend(add_cyl_points(epaule_droite_pos, taille_droite_pos, n_points=3, angle_offset=3*np.pi/4, subdivision=[0.4, 0.6, 1]))
    
    
    z_min = np.inf
    z_max = -np.inf
    for p in points:
        if p[2] > z_max:
            z_max = p[2]
        if p[2] < z_min:
            z_min = p[2]
    
    for i, p in enumerate(points):
        z_norm = (p[2] - z_min) / (z_max - z_min)
        cv2.circle(compose_image, (int(p[0]), int(p[1])), 3, [0, 0, int(255* z_norm)] , -1)
    
    if len(points) == 50:
    
        triangles_ids = np.array(TSHIRT_TRI_INDEX).reshape(-1,3)
    
        for tri in triangles_ids:
            p_1 = (int(points[tri[0]][0]),int(points[tri[0]][1]))
            p_2 = (int(points[tri[1]][0]),int(points[tri[1]][1]))
            p_3 = (int(points[tri[2]][0]),int(points[tri[2]][1]))
            cv2.line(compose_image, p_1, p_2, [0, 255, 0], 2) 
            cv2.line(compose_image, p_2, p_3, [0, 255, 0], 2) 
            cv2.line(compose_image, p_3, p_1, [0, 255, 0], 2) 
            
        affine_transforms = getAllTransform(points, t_shirt_ref["points"], TSHIRT_TRI_INDEX)
        
        t_shirt_r = sp.interpolate.RectBivariateSpline(np.arange(t_shirt_ref["image"].shape[0]), np.arange(t_shirt_ref["image"].shape[1]), t_shirt_ref["image"][:,:,0])
        t_shirt_g = sp.interpolate.RectBivariateSpline(np.arange(t_shirt_ref["image"].shape[0]), np.arange(t_shirt_ref["image"].shape[1]), t_shirt_ref["image"][:,:,1])
        t_shirt_b = sp.interpolate.RectBivariateSpline(np.arange(t_shirt_ref["image"].shape[0]), np.arange(t_shirt_ref["image"].shape[1]), t_shirt_ref["image"][:,:,2])
    
        tri_mask = np.zeros(compose_image.shape[:2], dtype = np.uint8)
    
        # for each triangle in triangles_ids
        for i, tri in enumerate(triangles_ids):
            # get the corresponding affine_transforms
            affine_t = affine_transforms[i]
            
            # get all pixel in the triangle and there position
            tri_mask.fill(0)
            
            
            triangle_points = np.array([
                [int(points[tri[0]][0]), int(points[tri[0]][1])],
                [int(points[tri[1]][0]), int(points[tri[1]][1])],
                [int(points[tri[2]][0]), int(points[tri[2]][1])]
            ], dtype=np.int32)

            cv2.fillPoly(tri_mask, [triangle_points], 255)

            py, px = np.where(tri_mask == 255)
            
            pixel_pos = np.array([px, py])
            
            ones = np.ones((1, pixel_pos.shape[1]))
            pixel_pos_hom = np.vstack((pixel_pos, ones))
            pixel_pos_transformed = affine_t @ pixel_pos_hom
            
            sampleR = t_shirt_r(pixel_pos_transformed[1],pixel_pos_transformed[0], grid=False)
            sampleG = t_shirt_g(pixel_pos_transformed[1],pixel_pos_transformed[0], grid=False)
            sampleB = t_shirt_b(pixel_pos_transformed[1],pixel_pos_transformed[0], grid=False)
            
            compose_image[py, px, 0] = sampleR
            compose_image[py, px, 1] = sampleG
            compose_image[py, px, 2] = sampleB
           
            
    
        
def extract_3d_position(landmark, depth):
    if landmark.x > 0 and landmark.y > 0 and landmark.x < 1 and landmark.y < 1:
        x = landmark.x * depth.shape[1]
        y = landmark.y * depth.shape[0]
        z = depth[int(x), int(y)]
        return np.array([x, y, z])
    else:
        return None
    
def add_cyl_points(start_pos, end_pos, estimated_circonference = None, estimated_length = None, n_points = 8, angle_offset = 0, d_angle = np.pi/4, subdivision = [0, 0.5, 1]):
    points = []
    
    radius = 25
    
    z_anchor = np.array([0, 0, 1])
    segment_vector = end_pos - start_pos
    
    if estimated_length is not None and estimated_circonference is not None:
        scale = get_vector_lenght(segment_vector)/estimated_length
        radius = (scale * estimated_circonference)/(2*np.pi)
    
    y = np.cross(segment_vector, z_anchor)
    x = np.cross(segment_vector, y)
    
    ny = normalise_vec(y)
    nx = normalise_vec(x)
    
    for alpha in subdivision:
        pos = start_pos + alpha * segment_vector
        for i in range(n_points):
            rot = angle_offset + i * d_angle
            dir = normalise_vec(nx * np.cos(rot) + ny * np.sin(rot))
            points.append(pos + dir * radius)
    
    return points

def getAllTransform(start_points, end_points, tris):
    assert len(start_points) == len(end_points), "Start and end points should be the same lenght"
    
    affine_transform = []
    
    for tri in np.array(tris).reshape(-1,3):
        p_start = np.array([
            [start_points[tri[0]][0], start_points[tri[0]][1]],
            [start_points[tri[1]][0], start_points[tri[1]][1]],
            [start_points[tri[2]][0], start_points[tri[2]][1]],
        ], dtype=np.float32)

        p_end = np.array([
            [end_points[tri[0]][0], end_points[tri[0]][1]],
            [end_points[tri[1]][0], end_points[tri[1]][1]],
            [end_points[tri[2]][0], end_points[tri[2]][1]],
        ], dtype=np.float32)
        
        M = cv2.getAffineTransform(p_start, p_end)
        affine_transform.append(M)
        
    return affine_transform