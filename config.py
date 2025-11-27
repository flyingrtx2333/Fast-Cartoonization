"""
White-box Cartoonization - PyTorch Implementation
Configuration File

Default hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Generator and Discriminator model configuration"""
    
    # Generator
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    num_blocks: int = 4
    
    # Discriminator
    disc_channels: int = 32
    disc_layers: int = 3
    use_spectral_norm: bool = True
    use_patch: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Data
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    
    # Training
    num_iters: int = 100000
    pretrain_iters: int = 50000
    
    # Optimizer
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.99
    
    # Loss weights
    lambda_surface: float = 0.1      # Surface (blur) discriminator
    lambda_structure: float = 1.0    # Structure (gray) discriminator
    lambda_content: float = 200.0    # VGG perceptual loss
    lambda_tv: float = 10000.0       # Total variation loss
    
    # Guided filter (eps=1e-2 matches original TensorFlow implementation)
    guided_filter_r: int = 1
    guided_filter_eps: float = 1e-2
    
    # Smooth filter for surface representation
    smooth_filter_r: int = 5
    smooth_filter_eps: float = 0.2
    
    # Superpixel
    use_superpixel: bool = False
    superpixel_segments: int = 200
    
    # Logging
    log_interval: int = 50
    save_interval: int = 5000
    sample_interval: int = 500
    
    # Misc
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration"""
    
    max_size: int = 720
    use_guided_filter: bool = True
    guided_filter_r: int = 1
    guided_filter_eps: float = 1e-2  # Matches original TensorFlow implementation


@dataclass  
class ExportConfig:
    """Model export configuration"""
    
    input_size: int = 256
    opset_version: int = 14
    dynamic_axes: bool = True
    include_guided_filter: bool = True


def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
DEFAULT_EXPORT_CONFIG = ExportConfig()

