import os
import h5py
import cv2
import numpy as np
from torch.utils.data import Dataset
import json 


class SteraSpatialDataset(Dataset):
    def __init__(self, session_path, obs_lens=5, pred_lens=50):
        self.session_path = session_path #path to the session folder containing annotation.hdf5 and rgb.mp4
        self.obs_lens = obs_lens
        self.pred_lens = pred_lens
        
        # Load HDF5 annotations
        self.h5_path = os.path.join(session_path, "annotation.hdf5")
        self.vid_path = os.path.join(session_path, "rgb.mp4")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.cam_rot = f['cam-pose/rotations'][:]
            self.total_frames = self.cam_rot.shape[0]
            
            
            #for one data session , there were 8514 hand pose instances but only 7435 frames 
            self.hand_