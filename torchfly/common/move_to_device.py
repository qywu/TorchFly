import os
import torch

def move_to_device(data, device, exclude_keys=None):
    """
    Args:
        data: a list, dict, or torch.Tensor
        device: the target torch.device
        exclude_keys: remove unwanted keys
    """
    # send data to device
    if isinstance(data, list):
        data = [
            item.to(device, non_blocking=True) for item in data
            if isinstance(item, torch.Tensor)
        ]
    if isinstance(data, dict):
        if exclude_keys is None:
            exclude_keys = []
        data = {
            k: v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            and all([key not in k for key in exclude_keys]) else v
            for k, v in data.items()
        }
    elif isinstance(data, torch.Tensor) or isinstance(data, torch.nn.Module):
        data = data.to(device, non_blocking=True)
    else:
        raise NotImplementedError

    return data