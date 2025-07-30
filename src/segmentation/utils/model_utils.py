import os
import torch
from cellSAM.model import get_model

def load_cellsam_model(model_path=None, preeval=False, device=None):
    """
    Load model weights and send to device.
    
    Args:
        model_path (str): path to model checkpoint
        preeval (bool): load the original weights of the model
        device (torch.device, optional): defaults to GPU if available
    
    Returns:
        torch.nn.Module: model with weights loaded on the correct device
        device: device utilized
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()

    if preeval or model_path is None:
        return model.to(device), device

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model path not found: {model_path}")

    return model.to(device), device
