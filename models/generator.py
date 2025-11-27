"""
White-box Cartoonization - PyTorch Implementation
Generator Networks (UNet-based)

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block with LeakyReLU"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.leaky_relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class UNetGenerator(nn.Module):
    """
    U-Net Generator for White-box Cartoonization
    
    Architecture:
    - Encoder: 3 downsampling blocks
    - Bottleneck: 4 residual blocks  
    - Decoder: 3 upsampling blocks with skip connections
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 base_channels: int = 32, num_blocks: int = 4):
        super().__init__()
        
        self.base_channels = base_channels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 7, 1, 3, bias=True)
        
        # Encoder (Downsampling)
        # Block 1: base_channels -> base_channels -> base_channels*2
        self.down1_conv1 = nn.Conv2d(base_channels, base_channels, 3, 2, 1, bias=True)
        self.down1_conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1, bias=True)
        
        # Block 2: base_channels*2 -> base_channels*2 -> base_channels*4
        self.down2_conv1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 2, 1, bias=True)
        self.down2_conv2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 1, 1, bias=True)
        
        # Residual Blocks (Bottleneck)
        self.res_blocks = nn.ModuleList([
            ResBlock(base_channels * 4) for _ in range(num_blocks)
        ])
        
        # Post-residual conv
        self.post_res_conv = nn.Conv2d(base_channels * 4, base_channels * 2, 3, 1, 1, bias=True)
        
        # Decoder (Upsampling) with skip connections
        # Block 1: Upsample + skip from down1
        self.up1_conv1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1, bias=True)
        self.up1_conv2 = nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1, bias=True)
        
        # Block 2: Upsample + skip from input
        self.up2_conv1 = nn.Conv2d(base_channels, base_channels, 3, 1, 1, bias=True)
        
        # Output convolution
        self.conv_out = nn.Conv2d(base_channels, out_channels, 7, 1, 3, bias=True)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x0 = self.leaky_relu(self.conv_in(x))  # [B, 32, H, W]
        
        # Encoder
        x1 = self.leaky_relu(self.down1_conv1(x0))   # [B, 32, H/2, W/2]
        x1 = self.leaky_relu(self.down1_conv2(x1))   # [B, 64, H/2, W/2]
        
        x2 = self.leaky_relu(self.down2_conv1(x1))   # [B, 64, H/4, W/4]
        x2 = self.leaky_relu(self.down2_conv2(x2))   # [B, 128, H/4, W/4]
        
        # Residual blocks
        for res_block in self.res_blocks:
            x2 = res_block(x2)
            
        # Post-residual
        x2 = self.leaky_relu(self.post_res_conv(x2))  # [B, 64, H/4, W/4]
        
        # Decoder with skip connections
        # Upsample x2 to match x1 size
        h1, w1 = x1.shape[2], x1.shape[3]
        x3 = F.interpolate(x2, size=(h1, w1), mode='bilinear', align_corners=False)
        x3 = x3 + x1  # Skip connection
        x3 = self.leaky_relu(self.up1_conv1(x3))  # [B, 64, H/2, W/2]
        x3 = self.leaky_relu(self.up1_conv2(x3))  # [B, 32, H/2, W/2]
        
        # Upsample x3 to match x0 size
        h0, w0 = x0.shape[2], x0.shape[3]
        x4 = F.interpolate(x3, size=(h0, w0), mode='bilinear', align_corners=False)
        x4 = x4 + x0  # Skip connection
        x4 = self.leaky_relu(self.up2_conv1(x4))  # [B, 32, H, W]
        
        # Output
        out = self.conv_out(x4)  # [B, 3, H, W]
        
        return out


class SimpleGenerator(nn.Module):
    """
    Simple Generator without U-Net skip connections
    For comparison or simpler use cases
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 32, num_blocks: int = 4):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, 1, 3),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(*[
            ResBlock(base_channels * 4) for _ in range(num_blocks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, 1),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels, out_channels, 7, 1, 3),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Test the generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNetGenerator().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

