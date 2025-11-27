"""
White-box Cartoonization - PyTorch Implementation
Full Training Script

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'

This script trains the full cartoonization model with:
- Surface representation (blur discriminator)
- Structure representation (gray discriminator)  
- Texture representation (superpixel loss)
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.generator import UNetGenerator
from models.discriminator import SpectralNormDiscriminator
from models.guided_filter import guided_filter, smooth_filter
from dataset import CartoonDataset, create_dataloader, batch_superpixel
from losses import (
    VGGPerceptualLoss, TotalVariationLoss, LSGANLoss, color_shift
)
from utils import (
    set_seed, count_parameters, save_checkpoint, load_checkpoint,
    AverageMeter, save_image_grid, denormalize
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train White-box Cartoonization')
    
    # Data
    parser.add_argument('--photo_dir', type=str, default='dataset/photo',
                        help='Directory containing training photos')
    parser.add_argument('--cartoon_dir', type=str, default='dataset/cartoon',
                        help='Directory containing cartoon reference images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Training image size')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_iters', type=int, default=100000,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Adam beta2')
    
    # Loss weights
    parser.add_argument('--lambda_surface', type=float, default=0.1,
                        help='Weight for surface (blur) loss')
    parser.add_argument('--lambda_structure', type=float, default=1.0,
                        help='Weight for structure (gray) loss')
    parser.add_argument('--lambda_content', type=float, default=200.0,
                        help='Weight for content (VGG) loss')
    parser.add_argument('--lambda_tv', type=float, default=10000.0,
                        help='Weight for total variation loss')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of residual blocks')
    
    # Pretrained model
    parser.add_argument('--pretrain_path', type=str, default='pretrain_results/checkpoints/pretrain_final.pth',
                        help='Path to pretrained generator checkpoint')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training (e.g., train_results/checkpoints/model_5000.pth)')
    
    # Superpixel (enabled by default as in original paper)
    parser.add_argument('--use_superpixel', action='store_true', default=True,
                        help='Use superpixel segmentation for texture (default: True)')
    parser.add_argument('--no_superpixel', action='store_true',
                        help='Disable superpixel segmentation')
    parser.add_argument('--superpixel_segments', type=int, default=200,
                        help='Number of superpixel segments')
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='train_results',
                        help='Directory to save results')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log every N iterations')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--sample_interval', type=int, default=500,
                        help='Save sample images every N iterations')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def train(args):
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'samples'), exist_ok=True)
    
    # Tensorboard
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # Dataset and dataloader
    dataset = CartoonDataset(
        args.photo_dir,
        args.cartoon_dir,
        image_size=args.image_size,
        augment=True
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Models
    generator = UNetGenerator(
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    ).to(device)
    
    # Two discriminators for surface and structure
    disc_surface = SpectralNormDiscriminator(
        in_channels=3, base_channels=32, use_patch=True
    ).to(device)
    
    disc_structure = SpectralNormDiscriminator(
        in_channels=1, base_channels=32, use_patch=True
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Surface Discriminator parameters: {count_parameters(disc_surface):,}")
    print(f"Structure Discriminator parameters: {count_parameters(disc_structure):,}")
    
    # Resume from checkpoint or load pretrained generator
    start_iteration = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = load_checkpoint(args.resume, device)
        generator.load_state_dict(checkpoint['generator'])
        disc_surface.load_state_dict(checkpoint['disc_surface'])
        disc_structure.load_state_dict(checkpoint['disc_structure'])
        start_iteration = checkpoint.get('iteration', 0)
        print(f"Resumed from checkpoint: {args.resume} (iteration {start_iteration})")
    elif args.pretrain_path and os.path.exists(args.pretrain_path):
        checkpoint = load_checkpoint(args.pretrain_path, device)
        generator.load_state_dict(checkpoint['generator'])
        print(f"Loaded pretrained generator from {args.pretrain_path}")
    
    # Losses
    vgg_loss = VGGPerceptualLoss('conv4_4').to(device)
    tv_loss = TotalVariationLoss()
    gan_loss = LSGANLoss()
    
    # Optimizers
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    
    d_optimizer = torch.optim.Adam(
        list(disc_surface.parameters()) + list(disc_structure.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    
    # Load optimizer states if resuming
    if args.resume and os.path.exists(args.resume):
        checkpoint = load_checkpoint(args.resume, device)
        if 'g_optimizer' in checkpoint:
            g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        if 'd_optimizer' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Loaded optimizer states from checkpoint")
    
    # Training loop
    generator.train()
    disc_surface.train()
    disc_structure.train()
    
    data_iter = iter(dataloader)
    
    # Loss meters
    g_loss_meter = AverageMeter()
    d_loss_meter = AverageMeter()
    
    pbar = tqdm(range(start_iteration, args.num_iters), desc='Training', initial=start_iteration, total=args.num_iters)
    
    for iteration in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        photos = batch['photo'].to(device)
        cartoons = batch['cartoon'].to(device)
        
        # ==================
        # Generate output (first pass - for superpixel calculation)
        # This matches the original TensorFlow implementation
        # ==================
        with torch.no_grad():
            output_for_superpixel = generator(photos)
            output_for_superpixel = guided_filter(photos, output_for_superpixel, r=1, eps=1e-2)
        
        # Texture representation (superpixel) - computed from first pass output
        use_superpixel = args.use_superpixel and not getattr(args, 'no_superpixel', False)
        if use_superpixel:
            try:
                superpixel = batch_superpixel(output_for_superpixel, args.superpixel_segments)
                superpixel = superpixel.to(device)
            except Exception as e:
                print(f"Superpixel failed: {e}, using output directly")
                superpixel = output_for_superpixel
        else:
            superpixel = output_for_superpixel
        
        # ==================
        # Generate output (second pass - for training)
        # ==================
        output = generator(photos)
        output = guided_filter(photos, output, r=1, eps=1e-2)
        
        # ==================
        # Prepare representations
        # ==================
        
        # Surface representation (smoothed/blurred)
        blur_output = smooth_filter(output, r=5, eps=0.2)
        blur_cartoon = smooth_filter(cartoons, r=5, eps=0.2)
        
        # Structure representation (grayscale with color shift)
        gray_output, gray_cartoon = color_shift(output, cartoons)
        
        # ==================
        # Train Discriminators
        # ==================
        d_optimizer.zero_grad()
        
        # Surface discriminator
        real_surface_logits = disc_surface(blur_cartoon)
        fake_surface_logits = disc_surface(blur_output.detach())
        d_loss_surface = gan_loss.discriminator_loss(real_surface_logits, fake_surface_logits)
        
        # Structure discriminator
        real_structure_logits = disc_structure(gray_cartoon)
        fake_structure_logits = disc_structure(gray_output.detach())
        d_loss_structure = gan_loss.discriminator_loss(real_structure_logits, fake_structure_logits)
        
        # Total discriminator loss
        d_loss = d_loss_surface + d_loss_structure
        d_loss.backward()
        d_optimizer.step()
        
        # ==================
        # Train Generator
        # ==================
        g_optimizer.zero_grad()
        
        # GAN losses
        fake_surface_logits = disc_surface(blur_output)
        fake_structure_logits = disc_structure(gray_output)
        
        g_loss_surface = gan_loss.generator_loss(fake_surface_logits)
        g_loss_structure = gan_loss.generator_loss(fake_structure_logits)
        
        # Content loss (photo + superpixel)
        content_loss_photo = vgg_loss(output, photos)
        content_loss_superpixel = vgg_loss(output, superpixel)
        content_loss = content_loss_photo + content_loss_superpixel
        
        # TV loss
        tv = tv_loss(output)
        
        # Total generator loss
        g_loss = (
            args.lambda_surface * g_loss_surface +
            args.lambda_structure * g_loss_structure +
            args.lambda_content * content_loss +
            args.lambda_tv * tv
        )
        
        g_loss.backward()
        g_optimizer.step()
        
        # Update meters
        g_loss_meter.update(g_loss.item())
        d_loss_meter.update(d_loss.item())
        
        # ==================
        # Logging
        # ==================
        if (iteration + 1) % args.log_interval == 0:
            pbar.set_postfix({
                'G': f'{g_loss_meter.avg:.4f}',
                'D': f'{d_loss_meter.avg:.4f}'
            })
            
            # Tensorboard
            writer.add_scalar('loss/g_total', g_loss_meter.avg, iteration + 1)
            writer.add_scalar('loss/d_total', d_loss_meter.avg, iteration + 1)
            writer.add_scalar('loss/g_surface', g_loss_surface.item(), iteration + 1)
            writer.add_scalar('loss/g_structure', g_loss_structure.item(), iteration + 1)
            writer.add_scalar('loss/content', content_loss.item(), iteration + 1)
            writer.add_scalar('loss/tv', tv.item(), iteration + 1)
            
            g_loss_meter.reset()
            d_loss_meter.reset()
            
        # ==================
        # Save samples
        # ==================
        if (iteration + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                sample_output = generator(photos[:8])
                sample_output = guided_filter(photos[:8], sample_output, r=1, eps=1e-2)
                
            # Save comparison grid
            samples = []
            for i in range(min(4, photos.size(0))):
                samples.extend([photos[i], sample_output[i]])
                
            save_image_grid(
                samples,
                os.path.join(args.save_dir, 'samples', f'iter_{iteration+1}.jpg'),
                nrow=4
            )
            generator.train()
            
        # ==================
        # Save checkpoint
        # ==================
        if (iteration + 1) % args.save_interval == 0:
            save_checkpoint(
                {
                    'iteration': iteration + 1,
                    'generator': generator.state_dict(),
                    'disc_surface': disc_surface.state_dict(),
                    'disc_structure': disc_structure.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                },
                os.path.join(args.save_dir, 'checkpoints', f'model_{iteration+1}.pth')
            )
            
    # Save final model
    save_checkpoint(
        {
            'iteration': args.num_iters,
            'generator': generator.state_dict(),
            'disc_surface': disc_surface.state_dict(),
            'disc_structure': disc_structure.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict(),
        },
        os.path.join(args.save_dir, 'checkpoints', 'model_final.pth')
    )
    
    # Save generator only for inference
    torch.save(
        generator.state_dict(),
        os.path.join(args.save_dir, 'generator_final.pth')
    )
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    args = parse_args()
    train(args)

