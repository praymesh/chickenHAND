import h5py
import cv2
import numpy as np

session_path = "/home/pranay23/VPR_model_tests/dino/chickenHAND/data/session_data_20260405_092950"  
def extract_session_data(session_path):
    # 1. Read Annotations (6-DoF Poses, MANO Coordinates)
    # The HDF5 file typically contains keys for 'camera_poses' and 'mango_params'
    h5_file = h5py.File(f"{session_path}/annotation.hdf5", 'r')
    
    # Example pseudo-keys (you will need to verify the exact Stera-10m keys)
    # 6-DoF poses might be stored as rotation matrices/quaternions + translations
    camera_poses = h5_file['camera_pose'][:] # Shape: (Total_Frames, 4, 4) 
    mano_params = h5_file['mano_params'][:]  # Shape: (Total_Frames, ... )
    
    # 2. Extract Video Frames
    cap = cv2.VideoCapture(f"{session_path}/rgb.mp4")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    return np.array(frames), camera_poses, mano_params