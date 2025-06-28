import torch
import torch.nn.functional as F

def combine_with_ptv(ptv, encoder_outputs):
    """
    Combine the PTV tensor with the encoder outputs at each stage.

    Args:
        ptv (torch.Tensor): The PTV tensor of shape [1, 1, 128, 128, 128].
        encoder_outputs (list of torch.Tensor): List of encoder outputs at different stages, each of shape [1, C, D, H, W].

    Returns:
        List of torch.Tensor: List of combined tensors.
    """
    combined_outputs = []
    for enc_out in encoder_outputs:
        # Resize PTV to match encoder output's spatial size
        ptv_resized = F.interpolate(ptv, size=enc_out.shape[2:], mode='trilinear', align_corners=False)
        
        # Concatenate along the channel dimension
        combined = ptv_resized + enc_out
        #combined = torch.cat((ptv_resized, enc_out), dim=1)
        
        combined_outputs.append(combined)

    return combined_outputs

# Example usage:

# PTV tensor
PTV = torch.randn(1, 1, 128, 128, 128)  # Example PTV tensor

# Encoder outputs at different stages
encoder_outputs = [
    torch.randn(1, 16, 128, 128, 128),  # Encoder 1 output
    torch.randn(1, 32, 64, 64, 64),    # Encoder 2 output
    torch.randn(1, 64, 32, 32, 32),    # Encoder 3 output
    torch.randn(1, 128, 16, 16, 16),   # Encoder 4 output
    torch.randn(1, 256, 8, 8, 8)       # Encoder 5 output
]

# Call the function
combined_outputs = combine_with_ptv(PTV, encoder_outputs)

# Check the shapes of the combined outputs
for i, combined in enumerate(combined_outputs):
    print(f"Combined output {i+1} shape: {combined.shape}")
