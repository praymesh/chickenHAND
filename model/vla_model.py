import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, visual_dim=512, text_dim=512, joint_input_dim=63, hidden_dim=768, num_layers=4, nhead=8):
        """
        Args:
            visual_dim: Dimension of embedded RGB+Depth
            text_dim: Dimension of embedded Text Prompt
            joint_input_dim: Raw 3D joints shape (21 * 3 = 63)
            hidden_dim: Transformer latent dimension
        """
        super().__init__()
        
        # 1. Projectors to bring everything to the common latent `hidden_dim` space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.joint_proj = nn.Linear(joint_input_dim, hidden_dim)
        
        # 2. Transformer Decoder configuration (causal temporal reasoning)
        # We use a transformer encoder strictly on the temporal axis.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Action Decoder Head (Predicts 50 future frames in one shot, or auto-regressive)
        # For simplicity in this base architecture, let's output a continuous tensor of size `future_horizon * 63`
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 50 * joint_input_dim) # Predicting 50 steps of 63-dim vectors
        )

    def forward(self, visual_emb, text_emb, past_joints):
        """
        visual_emb: Tensor of shape (Batch, visual_dim)
        text_emb: Tensor of shape (Batch, text_dim)
        past_joints: Tensor of shape (Batch, history_len, 21, 3)
        """
        B, hist_len, num_joints, coords = past_joints.shape
        
        # Flatten joints out of (21, 3) space to 63-dim
        past_joints_flat = past_joints.view(B, hist_len, num_joints * coords)
        
        # Embed past trajectories (B, hist_len, hidden_dim)
        traj_tokens = self.joint_proj(past_joints_flat)
        
        # Embed conditions (B, 1, hidden_dim)
        v_tokens = self.visual_proj(visual_emb).unsqueeze(1)
        t_tokens = self.text_proj(text_emb).unsqueeze(1)
        
        # Concat context with trajectories: [Text, Image, T-5, T-4... T-1]
        # Shape: (B, 2 + hist_len, hidden_dim)
        sequence = torch.cat([t_tokens, v_tokens, traj_tokens], dim=1)
        
        # Pass through Transformer
        encoded_sequence = self.transformer(sequence)
        
        # We only care about predicting from the context of the last trajectory token
        # Get the representation of the last token
        last_hidden_state = encoded_sequence[:, -1, :] 
        
        # Output 50 frames into the future
        future_traj_flat = self.action_head(last_hidden_state)  # (Batch, 50 * 63)
        
        # Reshape to (Batch, 50, 21, 3) for the loss function
        future_traj = future_traj_flat.view(B, 50, num_joints, coords)
        
        return future_traj

if __name__ == "__main__":
    # Scaffold run simulation!
    print("Testing base ActionEncoder architecture...")
    model = ActionEncoder()
    
    # Mock data directly reflecting our dataloader limits
    B = 4
    mock_visual = torch.randn(B, 512)
    mock_text = torch.randn(B, 512)
    mock_past_joints = torch.randn(B, 5, 21, 3)  # history_len=5
    
    out = model(mock_visual, mock_text, mock_past_joints)
    
    print(f"Mock Visual Embed: {mock_visual.shape}")
    print(f"Mock Past Joints: {mock_past_joints.shape}")
    print(f"Model Forward Output: {out.shape}")
