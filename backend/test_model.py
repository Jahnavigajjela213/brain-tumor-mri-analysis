import torch
import numpy as np
from app.models.unet_light import AttentionUNet

def test_inference():
    model = AttentionUNet(in_channels=4, out_channels=3, base=16)
    model.eval()
    
    # Simulate 4-channel MRI input (1, 4, 128, 128)
    x = torch.randn(1, 4, 128, 128)
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"Success! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"Inference failed: {e}")
        return False

if __name__ == "__main__":
    test_inference()
