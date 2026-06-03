import os
import h5py
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SteraDataset(Dataset):
    def __init__(self, session_path, history_len=5, future_horizon=50):
        self.session_path = session_path
        self.history_len = history_len
        self.future_horizon = future_horizon
        
        # Load HDF5 annotations
        self.h5_path = os.path.join(session_path, "annotation.hdf5")
        self.vid_path = os.path.join(session_path, "rgb.mp4")
        
        with h5py.File(self.h5_path, 'r') as f:
            self.cam_rot = f['cam-pose/rotations'][:]
            self.cam_trans = f['cam-pose/translations'][:]
            self.timestamps = f['cam-pose/timestamps'][:]
            
            # Text goals
            self.texts = f['text-annotations/text'][:]
            self.text_start = f['text-annotations/start_time'][:]
            self.text_end = f['text-annotations/end_time'][:]
            
            # Depth Maps
            self.depth_maps = f['depth/frames'][:]
            
            # Hand poses & 3D Joints
            hand_idx = f['hand-pose/frame_idx'][:]
            raw_hand_poses = f['hand-pose/hand_pose'][:] # (H, 15, 3, 3)
            raw_joints = f['hand-pose/joints_cam_rel'][:] # (H, 21, 3) - standard 3D joints
            
            self.total_frames = len(self.timestamps)
            
            # Dense hand poses mapped precisely to sequential video frames
            self.dense_hand_poses = np.zeros((self.total_frames, 15, 3, 3), dtype=np.float32)
            self.dense_joints = np.zeros((self.total_frames, 21, 3), dtype=np.float32)
            valid_mask = np.zeros(self.total_frames, dtype=bool)
            
            for i, f_idx in enumerate(hand_idx):
                if not valid_mask[f_idx]: # Just keep the first detected hand for simplicity
                    self.dense_hand_poses[f_idx] = raw_hand_poses[i]
                    self.dense_joints[f_idx] = raw_joints[i]
                    valid_mask[f_idx] = True

        # Pre-load video object mapping
        self.cap = cv2.VideoCapture(self.vid_path)

    def __len__(self):
        # Must have room for history and the projection window
        return self.total_frames - self.history_len - self.future_horizon - 1

    def get_text_prompt(self, current_time):
        # Check which text annotation overlaps with the current timestamp
        for i in range(len(self.texts)):
            if self.text_start[i] <= current_time <= self.text_end[i]:
                # Decode bytes to string natively
                val = self.texts[i]
                return val.decode('utf-8') if hasattr(val, 'decode') else str(val)
        return "continue current action"

    def __getitem__(self, idx):
        # Shift idx forward to account for the required history window
        curr_idx = idx + self.history_len
        
        # 1. Image extraction using sequence indexing
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
        ret, frame = self.cap.read()
        if not ret:
            # Fallback for EOF edge cases
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        depth = self.depth_maps[curr_idx]
        
        # 2. Text Prompt Extraction
        curr_time = self.timestamps[curr_idx]
        prompt = self.get_text_prompt(curr_time)
        
        # 3. Past Context [t-history : t]
        hist_cam_rot = self.cam_rot[idx : curr_idx]
        hist_cam_trans = self.cam_trans[idx : curr_idx]
        hist_hand_pose = self.dense_hand_poses[idx : curr_idx]
        hist_joints = self.dense_joints[idx : curr_idx]
        
        # 4. Target Horizon (The "Guess") [t+1 : t+1 + horizon]
        fut_start = curr_idx + 1
        fut_end = curr_idx + 1 + self.future_horizon
        fut_cam_rot = self.cam_rot[fut_start : fut_end]
        fut_cam_trans = self.cam_trans[fut_start : fut_end]
        fut_hand_pose = self.dense_hand_poses[fut_start : fut_end]
        fut_joints = self.dense_joints[fut_start : fut_end]
        
        # In a real setup, we would normalize fut_* poses relative to the cam_rot[curr_idx].
        
        return {
            "time_idx": curr_idx,
            "image": torch.tensor(frame).permute(2, 0, 1) / 255.0, # Target shape: (C, H, W) normalized
            "depth": torch.tensor(depth).unsqueeze(0).float() / 1000.0, # Normalize depth
            "prompt": prompt,
            "past_cam_rot": torch.tensor(hist_cam_rot),
            "past_cam_trans": torch.tensor(hist_cam_trans),
            "past_hand_pose": torch.tensor(hist_hand_pose),
            "past_joints": torch.tensor(hist_joints),
            "future_cam_rot": torch.tensor(fut_cam_rot),
            "future_cam_trans": torch.tensor(fut_cam_trans),
            "future_hand_pose": torch.tensor(fut_hand_pose),
            "future_joints": torch.tensor(fut_joints)
        }

if __name__ == "__main__":
    # Smoke test
    ds = SteraDataset("../data/session_data_20260405_092950", history_len=5, future_horizon=50)
    print(f"Dataset length: {len(ds)}")
    sample = ds[500]
    print(f"Index 500 timeframe: {sample['time_idx']}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Depth shape: {sample['depth'].shape}")
    print(f"Past Hand Pose shape: {sample['past_hand_pose'].shape}")
    print(f"Future Hand Pose shape: {sample['future_hand_pose'].shape}")
