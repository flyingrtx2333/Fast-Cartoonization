"""
White-box Cartoonization - PyTorch Implementation
Discriminator Networks with Spectral Normalization

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SpectralNormDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization
    
    Used for adversarial training with stable gradients.
    Spectral normalization constrains the Lipschitz constant of the discriminator.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32, 
                 num_layers: int = 3, use_patch: bool = True):
        super().__init__()
        
        self.use_patch = use_patch
        layers = []
        
        # Build discriminator layers
        ch_in = in_channels
        ch_out = base_channels
        
        for i in range(num_layers):
            # Strided conv (downsampling)
            layers.append(
                spectral_norm(nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=True))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Regular conv
            layers.append(
                spectral_norm(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            ch_in = ch_out
            ch_out = min(ch_out * 2, 256)
            
        self.features = nn.Sequential(*layers)
        
        # Output layer
        if use_patch:
            self.out = spectral_norm(nn.Conv2d(ch_in, 1, 1, 1, 0, bias=True))
        else:
            self.out = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch_in, 1)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.out(features)
        return out


class BatchNormDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Batch Normalization
    
    Alternative discriminator using batch normalization instead of spectral norm.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_layers: int = 3, use_patch: bool = True):
        super().__init__()
        
        self.use_patch = use_patch
        layers = []
        
        ch_in = in_channels
        ch_out = base_channels
        
        for i in range(num_layers):
            # Strided conv (downsampling)
            layers.append(nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            # Regular conv
            layers.append(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            ch_in = ch_out
            ch_out = min(ch_out * 2, 256)
            
        self.features = nn.Sequential(*layers)
        
        if use_patch:
            self.out = nn.Conv2d(ch_in, 1, 1, 1, 0, bias=True)
        else:
            self.out = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch_in, 1)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.out(features)
        return out


class LayerNormDiscriminator(nn.Module):
    """
    PatchGAN Discriminator with Layer Normalization
    
    Uses layer normalization which is instance-independent.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_layers: int = 3, use_patch: bool = True):
        super().__init__()
        
        self.use_patch = use_patch
        self.num_layers = num_layers
        
        # We need to track channels for layer norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        ch_in = in_channels
        ch_out = base_channels
        
        for i in range(num_layers):
            # Strided conv
            self.convs.append(nn.Conv2d(ch_in, ch_out, 3, 2, 1, bias=True))
            # Regular conv  
            self.convs.append(nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=True))
            
            ch_in = ch_out
            ch_out = min(ch_out * 2, 256)
            
        if use_patch:
            self.out = nn.Conv2d(ch_in, 1, 1, 1, 0, bias=True)
        else:
            self.out = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(ch_in, 1)
            )
            
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(0, len(self.convs), 2):
            x = self.convs[i](x)
            # Layer norm over spatial and channel dims
            x = F.layer_norm(x, x.shape[1:])
            x = self.leaky_relu(x)
            
            x = self.convs[i + 1](x)
            x = F.layer_norm(x, x.shape[1:])
            x = self.leaky_relu(x)
            
        out = self.out(x)
        return out


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale Discriminator
    
    Uses multiple discriminators at different scales for better gradient flow.
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_scales: int = 3, use_patch: bool = True):
        super().__init__()
        
        self.num_scales = num_scales
        
        self.discriminators = nn.ModuleList([
            SpectralNormDiscriminator(in_channels, base_channels, 3, use_patch)
            for _ in range(num_scales)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
    def forward(self, x: torch.Tensor) -> list:
        outputs = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
            
        return outputs


if __name__ == '__main__':
    # Test discriminators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test SpectralNorm Discriminator
    disc_sn = SpectralNormDiscriminator().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        out = disc_sn(x)
    
    print(f"SpectralNorm Discriminator:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {sum(p.numel() for p in disc_sn.parameters()):,}")
    
    # Test MultiScale Discriminator
    disc_ms = MultiScaleDiscriminator().to(device)
    
    with torch.no_grad():
        outs = disc_ms(x)
    
    print(f"\nMultiScale Discriminator:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shapes: {[o.shape for o in outs]}")
    print(f"  Parameters: {sum(p.numel() for p in disc_ms.parameters()):,}")

