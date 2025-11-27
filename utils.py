"""
White-box Cartoonization - PyTorch Implementation
Utility Functions

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
"""

import os
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, filepath: str):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, device: torch.device = None) -> dict:
    """Load training checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filepath, map_location=device)
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1]
    """
    return (tensor + 1) / 2


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image
    
    Args:
        tensor: [C, H, W] or [B, C, H, W] tensor in range [-1, 1]
        
    Returns:
        [H, W, C] uint8 numpy array in range [0, 255]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
        
    # Denormalize
    tensor = denormalize(tensor)
    
    # Clamp and convert
    tensor = tensor.clamp(0, 1)
    tensor = tensor.cpu().detach().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))  # CHW to HWC
    tensor = (tensor * 255).astype(np.uint8)
    
    return tensor


def save_image(tensor: torch.Tensor, filepath: str):
    """
    Save tensor as image
    
    Args:
        tensor: [C, H, W] or [B, C, H, W] tensor in range [-1, 1]
        filepath: Output path
    """
    img = tensor_to_image(tensor)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, img)


def save_image_grid(tensors: List[torch.Tensor], filepath: str, 
                    nrow: int = 4, padding: int = 2):
    """
    Save a grid of images
    
    Args:
        tensors: List of [C, H, W] tensors
        filepath: Output path
        nrow: Number of images per row
        padding: Padding between images
    """
    images = [tensor_to_image(t) for t in tensors]
    
    n = len(images)
    h, w, c = images[0].shape
    
    ncol = nrow
    nrow = (n + ncol - 1) // ncol
    
    grid_h = nrow * h + (nrow - 1) * padding
    grid_w = ncol * w + (ncol - 1) * padding
    
    grid = np.ones((grid_h, grid_w, c), dtype=np.uint8) * 255
    
    for idx, img in enumerate(images):
        i = idx // ncol
        j = idx % ncol
        
        y = i * (h + padding)
        x = j * (w + padding)
        
        grid[y:y+h, x:x+w] = img
        
    grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, grid)


def load_image(filepath: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Load image and convert to tensor
    
    Args:
        filepath: Image path
        size: Optional (height, width) to resize to
        
    Returns:
        [1, C, H, W] tensor in range [-1, 1]
    """
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if size is not None:
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        
    # Make dimensions divisible by 8
    h, w = img.shape[:2]
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    img = img[:new_h, :new_w]
    
    # Convert to tensor
    img = img.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    
    return tensor


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop


def get_lr_scheduler(optimizer: torch.optim.Optimizer,
                     scheduler_type: str = 'step',
                     **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 50000),
            gamma=kwargs.get('gamma', 0.5)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100000),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def init_weights(model: nn.Module, init_type: str = 'kaiming'):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model
        init_type: Initialization type ('kaiming', 'xavier', 'normal')
    """
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
                
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    model.apply(init_func)


if __name__ == '__main__':
    # Test utilities
    print("Utils module loaded successfully")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter: avg={meter.avg:.2f}, count={meter.count}")
    
    # Test parameter counting
    from models.generator import UNetGenerator
    model = UNetGenerator()
    print(f"UNetGenerator parameters: {count_parameters(model):,}")

