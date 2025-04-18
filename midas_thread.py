import torch
import urllib.request
import numpy as np


def run_midas(shared):
    # Setup
    model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
        
    while shared.running:
        
        with shared.lock:
            frame = shared.latest_frame.copy() if shared.latest_frame is not None else None 
    
        if frame is None:
            time.sleep(0.01)
            continue
        
        input_batch = transform(frame).to(device)
        
        with torch.no_grad():
            
            prediction = midas(input_batch)
    
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    
            depth_map = prediction.cpu().numpy()
    
            # Normalize for visualization
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
            
            with shared.lock:
                shared.midas_data = depth_vis
        
        