"""
White-box Cartoonization - PyTorch Implementation
Inference Script

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'

This script runs inference on images using a trained model.
"""

import os
import argparse
from typing import Optional

import torch
import cv2
import numpy as np
from tqdm import tqdm

from models.generator import UNetGenerator
from models.guided_filter import guided_filter
from utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Cartoonize Images')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image or directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model settings
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in generator')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of residual blocks')
    
    # Processing settings
    parser.add_argument('--max_size', type=int, default=720,
                        help='Maximum image dimension')
    parser.add_argument('--guided_filter', action='store_true', default=True,
                        help='Apply guided filter post-processing')
    parser.add_argument('--filter_r', type=int, default=1,
                        help='Guided filter radius')
    parser.add_argument('--filter_eps', type=float, default=5e-3,
                        help='Guided filter epsilon')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()


def load_and_preprocess(image_path: str, max_size: int = 720) -> tuple:
    """
    Load and preprocess image for inference
    
    Returns:
        tensor: Preprocessed tensor [1, 3, H, W]
        original_size: (H, W) of original image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    
    # Resize if too large
    h, w = img.shape[:2]
    if min(h, w) > max_size:
        if h > w:
            new_h = int(max_size * h / w)
            new_w = max_size
        else:
            new_h = max_size
            new_w = int(max_size * w / h)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
    # Make dimensions divisible by 8
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    img = img[:new_h, :new_w]
    
    # Convert to tensor
    img = img.astype(np.float32) / 127.5 - 1
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0)
    
    return tensor, (original_h, original_w)


def postprocess_and_save(tensor: torch.Tensor, output_path: str,
                         original_size: Optional[tuple] = None):
    """
    Postprocess tensor and save as image
    """
    # Convert to numpy
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    
    # Denormalize
    img = (img + 1) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Resize back to original if needed
    if original_size is not None:
        img = cv2.resize(img, (original_size[1], original_size[0]),
                        interpolation=cv2.INTER_LANCZOS4)
        
    # Convert to BGR and save
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img)


class Cartoonizer:
    """
    Cartoonization inference class
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 base_channels: int = 32,
                 num_blocks: int = 4,
                 device: str = 'cuda',
                 use_guided_filter: bool = True,
                 filter_r: int = 1,
                 filter_eps: float = 5e-3):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_guided_filter = use_guided_filter
        self.filter_r = filter_r
        self.filter_eps = filter_eps
        
        # Load model
        self.generator = UNetGenerator(
            base_channels=base_channels,
            num_blocks=num_blocks
        ).to(self.device)
        
        # Load weights
        if checkpoint_path.endswith('.pth'):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # Handle different checkpoint formats
            if 'generator' in state_dict:
                self.generator.load_state_dict(state_dict['generator'])
            else:
                self.generator.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unknown checkpoint format: {checkpoint_path}")
            
        self.generator.eval()
        print(f"Model loaded from {checkpoint_path}")
        
    @torch.no_grad()
    def cartoonize(self, image: np.ndarray) -> np.ndarray:
        """
        Cartoonize a single image
        
        Args:
            image: RGB image as numpy array [H, W, 3], range [0, 255]
            
        Returns:
            Cartoonized image as numpy array [H, W, 3], range [0, 255]
        """
        original_h, original_w = image.shape[:2]
        
        # Preprocess
        h, w = image.shape[:2]
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        if new_h != h or new_w != w:
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
            
        # To tensor
        img_tensor = image_resized.astype(np.float32) / 127.5 - 1
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)
        
        # Generate
        output = self.generator(img_tensor)
        
        # Apply guided filter
        if self.use_guided_filter:
            output = guided_filter(img_tensor, output, 
                                   r=self.filter_r, eps=self.filter_eps)
            
        # To numpy
        output = output.squeeze(0).cpu().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output + 1) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Resize back
        if output.shape[:2] != (original_h, original_w):
            output = cv2.resize(output, (original_w, original_h),
                              interpolation=cv2.INTER_LANCZOS4)
            
        return output
    
    def process_image(self, input_path: str, output_path: str, max_size: int = 720):
        """
        Process a single image file
        """
        # Load
        img = cv2.imread(input_path)
        if img is None:
            print(f"Failed to load: {input_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        h, w = img.shape[:2]
        if min(h, w) > max_size:
            if h > w:
                new_h = int(max_size * h / w)
                new_w = max_size
            else:
                new_h = max_size
                new_w = int(max_size * w / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Cartoonize
        result = self.cartoonize(img)
        
        # Save
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result)
        
        return True
    
    def process_directory(self, input_dir: str, output_dir: str, max_size: int = 720):
        """
        Process all images in a directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in os.listdir(input_dir) 
                if os.path.splitext(f)[1].lower() in valid_extensions]
        
        print(f"Processing {len(files)} images...")
        
        for filename in tqdm(files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                self.process_image(input_path, output_path, max_size)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


def main():
    args = parse_args()
    
    # Create cartoonizer
    cartoonizer = Cartoonizer(
        checkpoint_path=args.checkpoint,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        device=args.device,
        use_guided_filter=args.guided_filter,
        filter_r=args.filter_r,
        filter_eps=args.filter_eps
    )
    
    # Process input
    if os.path.isdir(args.input):
        cartoonizer.process_directory(args.input, args.output, args.max_size)
    else:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        cartoonizer.process_image(args.input, args.output, args.max_size)
        
    print("Done!")


if __name__ == '__main__':
    main()

