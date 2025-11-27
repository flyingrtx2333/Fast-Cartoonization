"""
White-box Cartoonization - PyTorch Implementation
Pretraining Script

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'

This script pretrains the generator to reconstruct input images.
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
from dataset import PhotoDataset, create_dataloader
from losses import ReconstructionLoss
from utils import (
    set_seed, count_parameters, save_checkpoint, 
    AverageMeter, save_image_grid, denormalize
)


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain Generator')
    
    # Data
    parser.add_argument('--photo_dir', type=str, default='dataset/photo',
                        help='Directory containing training photos')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Training image size')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_iters', type=int, default=50000,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Adam beta2')
    
    # Model
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in generator')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of residual blocks')
    
    # Logging
    parser.add_argument('--save_dir', type=str, default='pretrain_results',
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
    dataset = PhotoDataset(
        args.photo_dir,
        image_size=args.image_size,
        augment=True
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Model
    generator = UNetGenerator(
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    ).to(device)
    
    print(f"Generator parameters: {count_parameters(generator):,}")
    
    # Loss
    recon_loss = ReconstructionLoss('l1')
    
    # Optimizer
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )
    
    # Training loop
    generator.train()
    data_iter = iter(dataloader)
    loss_meter = AverageMeter()
    
    pbar = tqdm(range(args.num_iters), desc='Pretraining')
    
    for iteration in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        photos = batch['photo'].to(device)
        
        # Forward pass
        output = generator(photos)
        
        # Compute loss
        loss = recon_loss(output, photos)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update meter
        loss_meter.update(loss.item())
        
        # Logging
        if (iteration + 1) % args.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
            writer.add_scalar('pretrain/loss', loss_meter.avg, iteration + 1)
            loss_meter.reset()
            
        # Save samples
        if (iteration + 1) % args.sample_interval == 0:
            generator.eval()
            with torch.no_grad():
                sample_output = generator(photos[:8])
                
            # Save grid
            samples = []
            for i in range(min(4, photos.size(0))):
                samples.extend([photos[i], sample_output[i]])
                
            save_image_grid(
                samples,
                os.path.join(args.save_dir, 'samples', f'iter_{iteration+1}.jpg'),
                nrow=4
            )
            generator.train()
            
        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            save_checkpoint(
                {
                    'iteration': iteration + 1,
                    'generator': generator.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.save_dir, 'checkpoints', f'pretrain_{iteration+1}.pth')
            )
            
    # Save final model
    save_checkpoint(
        {
            'iteration': args.num_iters,
            'generator': generator.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        os.path.join(args.save_dir, 'checkpoints', 'pretrain_final.pth')
    )
    
    writer.close()
    print("Pretraining complete!")


if __name__ == '__main__':
    args = parse_args()
    train(args)

