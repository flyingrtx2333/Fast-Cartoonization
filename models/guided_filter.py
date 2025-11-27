"""
White-box Cartoonization - PyTorch Implementation
Guided Filter for Edge-preserving Smoothing

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'

Reference:
He, K., Sun, J., & Tang, X. (2010). Guided image filtering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """
    Box filter (mean filter) implemented using depthwise convolution
    
    Args:
        x: Input tensor [B, C, H, W]
        r: Radius of the box filter
        
    Returns:
        Filtered tensor [B, C, H, W]
    """
    channels = x.shape[1]
    kernel_size = 2 * r + 1
    
    # Create box kernel
    weight = torch.ones(channels, 1, kernel_size, kernel_size, 
                        device=x.device, dtype=x.dtype) / (kernel_size ** 2)
    
    # Apply depthwise convolution with padding
    output = F.conv2d(x, weight, padding=r, groups=channels)
    
    return output


def guided_filter(guide: torch.Tensor, src: torch.Tensor, 
                  r: int = 1, eps: float = 1e-2) -> torch.Tensor:
    """
    Guided Filter for edge-aware smoothing
    
    The guided filter smooths the source image while preserving edges
    from the guidance image. This is a key component in white-box cartoonization
    for creating cartoon-like flat regions with sharp edges.
    
    Args:
        guide: Guidance image [B, C, H, W], usually the input photo
        src: Source image [B, C, H, W], usually the network output
        r: Filter radius (default: 1)
        eps: Regularization parameter (default: 1e-2)
             - Smaller eps = sharper edges, less smoothing
             - Larger eps = more smoothing, softer edges
             
    Returns:
        Filtered output [B, C, H, W]
    """
    # Get image dimensions
    b, c, h, w = guide.shape
    
    # Create a normalization tensor (handles boundaries)
    ones = torch.ones(b, 1, h, w, device=guide.device, dtype=guide.dtype)
    N = box_filter(ones, r)
    
    # Mean of guide and source
    mean_guide = box_filter(guide, r) / N
    mean_src = box_filter(src, r) / N
    
    # Covariance of (guide, src) in each local patch
    cov_guide_src = box_filter(guide * src, r) / N - mean_guide * mean_src
    
    # Variance of guide in each local patch
    var_guide = box_filter(guide * guide, r) / N - mean_guide * mean_guide
    
    # Linear coefficients
    A = cov_guide_src / (var_guide + eps)
    b = mean_src - A * mean_guide
    
    # Mean of A and b
    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N
    
    # Output
    output = mean_A * guide + mean_b
    
    return output


class GuidedFilter(nn.Module):
    """
    Guided Filter Module
    
    Differentiable module for guided filtering during training.
    """
    
    def __init__(self, r: int = 1, eps: float = 1e-2):
        super().__init__()
        self.r = r
        self.eps = eps
        
    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return guided_filter(guide, src, self.r, self.eps)


class FastGuidedFilter(nn.Module):
    """
    Fast Guided Filter with downsampling for efficiency
    
    Performs guided filtering at a lower resolution and upsamples the result.
    Useful for high-resolution images.
    """
    
    def __init__(self, r: int = 1, eps: float = 1e-2, scale: int = 4):
        super().__init__()
        self.r = r
        self.eps = eps
        self.scale = scale
        
    def forward(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        h, w = guide.shape[2:]
        
        # Downsample
        guide_small = F.interpolate(guide, scale_factor=1/self.scale, 
                                    mode='bilinear', align_corners=False)
        src_small = F.interpolate(src, scale_factor=1/self.scale,
                                  mode='bilinear', align_corners=False)
        
        # Apply guided filter at lower resolution
        output_small = guided_filter(guide_small, src_small, self.r, self.eps)
        
        # Upsample
        output = F.interpolate(output_small, size=(h, w), 
                               mode='bilinear', align_corners=False)
        
        return output


class LearnableGuidedFilter(nn.Module):
    """
    Learnable Guided Filter
    
    Uses small CNNs to predict the filtering parameters locally.
    """
    
    def __init__(self, channels: int = 3, hidden_channels: int = 16):
        super().__init__()
        
        # Network to predict local eps (regularization)
        self.eps_net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 3, 1, 1),
            nn.Softplus()  # Ensure positive eps
        )
        
        self.base_eps = 1e-2
        
    def forward(self, guide: torch.Tensor, src: torch.Tensor, r: int = 1) -> torch.Tensor:
        # Predict local eps
        local_eps = self.eps_net(guide) + self.base_eps
        
        # Get dimensions
        b, c, h, w = guide.shape
        ones = torch.ones(b, 1, h, w, device=guide.device, dtype=guide.dtype)
        N = box_filter(ones, r)
        
        mean_guide = box_filter(guide, r) / N
        mean_src = box_filter(src, r) / N
        cov_guide_src = box_filter(guide * src, r) / N - mean_guide * mean_src
        var_guide = box_filter(guide * guide, r) / N - mean_guide * mean_guide
        
        A = cov_guide_src / (var_guide + local_eps)
        b_coef = mean_src - A * mean_guide
        
        mean_A = box_filter(A, r) / N
        mean_b = box_filter(b_coef, r) / N
        
        output = mean_A * guide + mean_b
        
        return output


def smooth_filter(img: torch.Tensor, r: int = 5, eps: float = 0.2) -> torch.Tensor:
    """
    Self-guided smoothing filter for surface representation
    
    Used to create smooth, flat color regions characteristic of cartoon style.
    
    Args:
        img: Input image [B, C, H, W]
        r: Filter radius (larger = more smoothing)
        eps: Regularization (larger = more smoothing)
        
    Returns:
        Smoothed image [B, C, H, W]
    """
    return guided_filter(img, img, r, eps)


if __name__ == '__main__':
    # Test guided filter
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test images
    guide = torch.randn(1, 3, 256, 256).to(device)
    src = torch.randn(1, 3, 256, 256).to(device)
    
    # Test basic guided filter
    output = guided_filter(guide, src, r=1, eps=1e-2)
    print(f"Guided filter output shape: {output.shape}")
    
    # Test module
    gf = GuidedFilter(r=1, eps=1e-2).to(device)
    output_module = gf(guide, src)
    print(f"GuidedFilter module output shape: {output_module.shape}")
    
    # Test smooth filter
    smooth = smooth_filter(guide, r=5, eps=0.2)
    print(f"Smooth filter output shape: {smooth.shape}")
    
    # Verify differentiability
    guide.requires_grad = True
    src.requires_grad = True
    output = guided_filter(guide, src)
    loss = output.sum()
    loss.backward()
    print(f"Gradients computed successfully!")
    print(f"Guide grad shape: {guide.grad.shape}")
    print(f"Src grad shape: {src.grad.shape}")

