import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import SteraDataset
import numpy as np
import torch

def visualize_sample(sample):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Render Image
    img = sample['image'][0].permute(1, 2, 0).numpy() # Grab batch 0, reformat to (H,W,C)
    axes[0].imshow(img)
    axes[0].set_title(f"RGB | T={sample['time_idx'][0].item()}")
    axes[0].axis('off')
    
    # Render Depth
    depth = sample['depth'][0][0].numpy() # Grab batch 0, channel 0
    im = axes[1].imshow(depth, cmap='jet')
    axes[1].set_title(f"Depth Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Print Title/Prompt
    plt.suptitle(f"Task Prompt: \"{sample['prompt'][0]}\"", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualization_output.png")
    print("\n[✓] Saved frame breakdown to 'visualization_output.png'")
    
    print("-" * 50)
    print("VLA Tensors Configured:")
    print(f"Past History    ==> {sample['past_hand_pose'].shape} (Batch, Horizon, Joints, 3, 3)")
    print(f"Future Target   ==> {sample['future_hand_pose'].shape} (Batch, Horizon, Joints, 3, 3)")
    print(f"Visual Tensors  ==> {sample['image'].shape} (Batch, C, H, W)")
    print("-" * 50)

if __name__ == "__main__":
    session_path = "../data/session_data_20260405_092950"
    dataset = SteraDataset(session_path, history_len=5, future_horizon=50)
    
    # Initialize PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch random batch
    sample_batch = next(iter(dataloader))
    
    # Visualize index 0 of the batch
    visualize_sample(sample_batch)