"""
White-box Cartoonization - PyTorch Implementation
Loss Functions

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'

This module contains all loss functions used for training:
- Reconstruction Loss (L1)
- VGG Perceptual Loss  
- GAN Losses (LSGAN, WGAN-GP, Standard GAN)
- Total Variation Loss
- Color Shift for structure representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.vgg import VGG19FeatureExtractor


class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss (L1 or L2)
    
    Used for pretraining the generator to reconstruct input images.
    """
    
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, target)


class VGGPerceptualLoss(nn.Module):
    """
    VGG Perceptual Loss
    
    Computes feature-level loss using VGG19 conv4_4 features.
    This helps preserve semantic content during cartoonization.
    """
    
    def __init__(self, layer: str = 'conv4_4'):
        super().__init__()
        
        self.vgg = VGG19FeatureExtractor([layer], requires_grad=False)
        self.layer = layer
        self.criterion = nn.L1Loss(reduction='mean')
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute VGG perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Normalized perceptual loss
        """
        pred_feat = self.vgg(pred)[self.layer]
        target_feat = self.vgg(target)[self.layer]
        
        # Normalize by feature dimensions
        _, c, h, w = pred_feat.shape
        loss = self.criterion(pred_feat, target_feat)
        
        return loss / (c * h * w)


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for smoothness
    
    Encourages spatial smoothness in the output by penalizing
    high-frequency variations (large gradients between adjacent pixels).
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation loss
        
        Args:
            x: Image tensor [B, C, H, W]
            
        Returns:
            TV loss value
        """
        b, c, h, w = x.shape
        
        # Compute horizontal and vertical differences
        tv_h = torch.mean((x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2)
        tv_w = torch.mean((x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2)
        
        loss = (tv_h + tv_w) / (3 * h * w)
        
        return self.weight * loss


class LSGANLoss(nn.Module):
    """
    Least Squares GAN Loss
    
    More stable than standard GAN loss, provides smoother gradients.
    D_loss = 0.5 * (E[(D(real) - 1)^2] + E[D(fake)^2])
    G_loss = 0.5 * E[(D(fake) - 1)^2]
    """
    
    def __init__(self):
        super().__init__()
        
    def discriminator_loss(self, real_logits: torch.Tensor, 
                           fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator loss: real should be 1, fake should be 0"""
        real_loss = torch.mean((real_logits - 1) ** 2)
        fake_loss = torch.mean(fake_logits ** 2)
        return 0.5 * (real_loss + fake_loss)
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator loss: fake should be 1"""
        return torch.mean((fake_logits - 1) ** 2)


class StandardGANLoss(nn.Module):
    """
    Standard GAN Loss (Binary Cross Entropy)
    
    D_loss = -E[log(D(real))] - E[log(1 - D(fake))]
    G_loss = -E[log(D(fake))]
    """
    
    def __init__(self, smooth_labels: bool = True):
        super().__init__()
        self.smooth = smooth_labels
        
    def discriminator_loss(self, real_logits: torch.Tensor,
                           fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator loss"""
        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)
        
        if self.smooth:
            real_labels = real_labels * 0.9  # Label smoothing
            
        real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
        
        return real_loss + fake_loss
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator loss"""
        return F.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )


class WGANGPLoss(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty
    
    Provides better training stability and avoids mode collapse.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def discriminator_loss(self, discriminator: nn.Module,
                           real: torch.Tensor, fake: torch.Tensor,
                           real_logits: torch.Tensor,
                           fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Discriminator loss with gradient penalty
        """
        d_loss = torch.mean(fake_logits) - torch.mean(real_logits)
        
        # Gradient penalty
        gp = self._gradient_penalty(discriminator, real, fake)
        
        return d_loss + self.lambda_gp * gp
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator loss"""
        return -torch.mean(fake_logits)
    
    def _gradient_penalty(self, discriminator: nn.Module,
                          real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Compute penalty
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


class HingeLoss(nn.Module):
    """
    Hinge Loss for GANs
    
    Used in SAGAN and BigGAN, provides stable training.
    """
    
    def discriminator_loss(self, real_logits: torch.Tensor,
                           fake_logits: torch.Tensor) -> torch.Tensor:
        """Discriminator hinge loss"""
        real_loss = torch.mean(F.relu(1.0 - real_logits))
        fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        return real_loss + fake_loss
    
    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Generator hinge loss"""
        return -torch.mean(fake_logits)


def color_shift(image1: torch.Tensor, image2: torch.Tensor,
                mode: str = 'uniform') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert RGB images to grayscale with random channel weights
    
    This is used to create the structure representation by removing color
    information while preserving edges and structure.
    
    Args:
        image1: First image [B, 3, H, W]
        image2: Second image [B, 3, H, W]  
        mode: Weight sampling mode ('uniform' or 'normal')
        
    Returns:
        Tuple of grayscale images
    """
    device = image1.device
    
    if mode == 'uniform':
        # Random weights with uniform distribution
        r_weight = torch.empty(1, device=device).uniform_(0.199, 0.399)
        g_weight = torch.empty(1, device=device).uniform_(0.487, 0.687)
        b_weight = torch.empty(1, device=device).uniform_(0.014, 0.214)
    elif mode == 'normal':
        # Random weights with normal distribution
        r_weight = torch.empty(1, device=device).normal_(0.299, 0.1).clamp(0.1, 0.5)
        g_weight = torch.empty(1, device=device).normal_(0.587, 0.1).clamp(0.4, 0.8)
        b_weight = torch.empty(1, device=device).normal_(0.114, 0.1).clamp(0.0, 0.3)
    else:
        # Standard grayscale weights
        r_weight = torch.tensor([0.299], device=device)
        g_weight = torch.tensor([0.587], device=device)
        b_weight = torch.tensor([0.114], device=device)
    
    # Split channels (assuming RGB order)
    r1, g1, b1 = image1[:, 0:1], image1[:, 1:2], image1[:, 2:3]
    r2, g2, b2 = image2[:, 0:1], image2[:, 1:2], image2[:, 2:3]
    
    # Compute weighted grayscale
    total_weight = r_weight + g_weight + b_weight
    gray1 = (r_weight * r1 + g_weight * g1 + b_weight * b1) / total_weight
    gray2 = (r_weight * r2 + g_weight * g2 + b_weight * b2) / total_weight
    
    return gray1, gray2


class WhiteBoxCartoonLoss(nn.Module):
    """
    Combined Loss for White-box Cartoonization
    
    Combines all loss components used in the white-box cartoonization method:
    - Surface loss (blur discriminator)
    - Structure loss (gray discriminator)  
    - Content/Reconstruction loss (VGG perceptual)
    - Total variation loss
    """
    
    def __init__(self,
                 lambda_surface: float = 0.1,
                 lambda_structure: float = 1.0,
                 lambda_content: float = 200.0,
                 lambda_tv: float = 10000.0):
        super().__init__()
        
        self.lambda_surface = lambda_surface
        self.lambda_structure = lambda_structure
        self.lambda_content = lambda_content
        self.lambda_tv = lambda_tv
        
        self.vgg_loss = VGGPerceptualLoss('conv4_4')
        self.tv_loss = TotalVariationLoss()
        self.gan_loss = LSGANLoss()
        
    def generator_loss(self,
                       output: torch.Tensor,
                       input_photo: torch.Tensor,
                       superpixel: torch.Tensor,
                       surface_fake_logits: torch.Tensor,
                       structure_fake_logits: torch.Tensor) -> dict:
        """
        Compute generator loss
        
        Args:
            output: Generator output
            input_photo: Original input photo
            superpixel: Superpixel representation of output
            surface_fake_logits: Surface discriminator output for fake
            structure_fake_logits: Structure discriminator output for fake
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # GAN losses
        g_loss_surface = self.gan_loss.generator_loss(surface_fake_logits)
        g_loss_structure = self.gan_loss.generator_loss(structure_fake_logits)
        
        # Content loss
        photo_loss = self.vgg_loss(output, input_photo)
        superpixel_loss = self.vgg_loss(output, superpixel)
        content_loss = photo_loss + superpixel_loss
        
        # TV loss
        tv_loss = self.tv_loss(output)
        
        # Total generator loss
        total_loss = (
            self.lambda_surface * g_loss_surface +
            self.lambda_structure * g_loss_structure +
            self.lambda_content * content_loss +
            self.lambda_tv * tv_loss
        )
        
        return {
            'total': total_loss,
            'surface': g_loss_surface,
            'structure': g_loss_structure,
            'content': content_loss,
            'tv': tv_loss
        }
    
    def discriminator_loss(self,
                           real_logits: torch.Tensor,
                           fake_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss
        """
        return self.gan_loss.discriminator_loss(real_logits, fake_logits)


if __name__ == '__main__':
    # Test losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test tensors
    pred = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randn(2, 3, 256, 256, device=device)
    
    # Test reconstruction loss
    recon_loss = ReconstructionLoss('l1')
    loss = recon_loss(pred, target)
    print(f"Reconstruction loss: {loss.item():.6f}")
    
    # Test VGG loss
    vgg_loss = VGGPerceptualLoss().to(device)
    loss = vgg_loss(pred, target)
    print(f"VGG perceptual loss: {loss.item():.6f}")
    
    # Test TV loss
    tv_loss = TotalVariationLoss()
    loss = tv_loss(pred)
    print(f"Total variation loss: {loss.item():.6f}")
    
    # Test color shift
    gray1, gray2 = color_shift(pred, target)
    print(f"Color shift output shapes: {gray1.shape}, {gray2.shape}")
    
    # Test LSGAN loss
    lsgan = LSGANLoss()
    real_logits = torch.randn(2, 1, 32, 32, device=device)
    fake_logits = torch.randn(2, 1, 32, 32, device=device)
    d_loss = lsgan.discriminator_loss(real_logits, fake_logits)
    g_loss = lsgan.generator_loss(fake_logits)
    print(f"LSGAN D loss: {d_loss.item():.6f}, G loss: {g_loss.item():.6f}")

