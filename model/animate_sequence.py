import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataloader import SteraDataset

def plot_3d_hand_to_image(joints, title="3D Hand"):
    """
    Renders a 3D hand to a NumPy RGB image array (H, W, 3).
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
    ]
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', s=20)
    for u, v in connections:
        ax.plot([joints[u, 0], joints[v, 0]],
                [joints[u, 1], joints[v, 1]],
                [joints[u, 2], joints[v, 2]], 'b-', lw=2)
                
    ax.set_title(title)
    
    # Standardize scale
    all_vals = joints.flatten()
    max_range = (all_vals.max() - all_vals.min()) / 2.0
    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Render canvas to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3] # Drop Alpha channel to get RGB
    plt.close(fig)
    return img

def create_animation(dataset, start_idx=500, frames_to_render=50, output_path="sequence_eval.mp4"):
    print(f"Generating video sequence for {frames_to_render} frames...")
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    for i in range(frames_to_render):
        # We step forward through the dataset manually
        sample = dataset[start_idx + i]
        
        # 1. Get original RGB Frame
        # Re-permute to H, W, C and scale to 0-255
        rgb_frame = (sample['image'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 2. Get current ground truth 3D joints (Current T=0 is the last of the past joints)
        current_joints = sample['past_joints'][-1].numpy()
        
        # 3. Render 3D plot to image
        plot_img = plot_3d_hand_to_image(current_joints, title=f"Frame {i}")
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        
        # Resize plot to match video height
        h, w = rgb_frame.shape[:2]
        plot_resized = cv2.resize(plot_img, (int(plot_img.shape[1] * (h / plot_img.shape[0])), h))
        
        # Concatenate side-by-side
        combined_frame = np.hstack((rgb_frame, plot_resized))
        
        if out is None:
            out = cv2.VideoWriter(output_path, fourcc, 10.0, (combined_frame.shape[1], combined_frame.shape[0]))
            
        out.write(combined_frame)
        
    out.release()
    print(f"[✓] Validation video saved to '{output_path}'")

if __name__ == "__main__":
    session_path = "../data/session_data_20260405_092950"
    ds = SteraDataset(session_path, history_len=5, future_horizon=50)
    create_animation(ds, start_idx=500, frames_to_render=50)
