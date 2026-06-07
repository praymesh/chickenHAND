import matplotlib.pyplot as plt
import numpy as np
from dataloader import SteraDataset

def plot_3d_hand(ax, joints, title):
    # MANO Joint Connections standard format: 
    # [Wrist:0, Thumb:1-4, Index:5-8, Middle:9-12, Ring:13-16, Pinky:17-20]
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio for better visualization
    all_vals = joints.flatten()
    max_range = (all_vals.max() - all_vals.min()) / 2.0
    mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
    mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
    mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

if __name__ == "__main__":
    session_path = "../data/session_data_20260405_092950"
    ds = SteraDataset(session_path, history_len=5, future_horizon=50)
    
    # Get a sample
    sample = ds[500]
    
    # Grab the last known frame in history, and the 25th frame into the future
    past_joint = sample['past_joints'][-1].numpy()  # Extract the t-1 position
    fut_joint = sample['future_joints'][25].numpy() # Extract the future guess
    
    fig = plt.figure(figsize=(12, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_hand(ax1, past_joint, "Observation (t=0)")
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_hand(ax2, fut_joint, "Future Target (t=25)")
    
    plt.suptitle(f"Task: {sample['prompt']}")
    plt.tight_layout()
    plt.savefig("3d_hand_visualization.png")
    print("[✓] 3D Hand rendered to 3d_hand_visualization.png")
