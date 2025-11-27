"""
White-box Cartoonization - PyTorch Implementation
Dataset Classes

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
"""

import os
import random
from typing import Optional, Tuple, List, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Optional: scikit-image for superpixel segmentation
try:
    from skimage import segmentation, color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class CartoonDataset(Dataset):
    """
    Dataset for White-box Cartoonization Training
    
    Loads pairs of photos and cartoon reference images for training.
    """
    
    def __init__(self,
                 photo_dir: str,
                 cartoon_dir: str,
                 image_size: int = 256,
                 augment: bool = True):
        """
        Args:
            photo_dir: Directory containing real photos
            cartoon_dir: Directory containing cartoon images
            image_size: Target image size (square)
            augment: Whether to apply data augmentation
        """
        self.photo_dir = photo_dir
        self.cartoon_dir = cartoon_dir
        self.image_size = image_size
        self.augment = augment
        
        # Get image lists
        self.photo_list = self._get_image_list(photo_dir)
        self.cartoon_list = self._get_image_list(cartoon_dir)
        
        if len(self.photo_list) == 0:
            raise ValueError(f"No images found in {photo_dir}")
        if len(self.cartoon_list) == 0:
            raise ValueError(f"No images found in {cartoon_dir}")
            
        print(f"Loaded {len(self.photo_list)} photos and {len(self.cartoon_list)} cartoon images")
        
        # Image normalization (to [-1, 1])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                              std=[0.5, 0.5, 0.5])
        
    def _get_image_list(self, directory: str) -> List[str]:
        """Get list of image files in directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_list = []
        
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_list.append(os.path.join(directory, filename))
                
        return sorted(image_list)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        # Read image with OpenCV (BGR format)
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _resize_crop(self, img: np.ndarray) -> np.ndarray:
        """Resize and random crop to target size"""
        h, w = img.shape[:2]
        
        # Resize shorter side to image_size
        if h < w:
            new_h = self.image_size
            new_w = int(w * self.image_size / h)
        else:
            new_w = self.image_size
            new_h = int(h * self.image_size / w)
            
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Random crop (or center crop during inference)
        if self.augment:
            start_h = random.randint(0, new_h - self.image_size) if new_h > self.image_size else 0
            start_w = random.randint(0, new_w - self.image_size) if new_w > self.image_size else 0
        else:
            start_h = (new_h - self.image_size) // 2
            start_w = (new_w - self.image_size) // 2
            
        img = img[start_h:start_h + self.image_size, 
                  start_w:start_w + self.image_size]
        
        return img
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return img
            
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        return img
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor"""
        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor
        tensor = torch.from_numpy(img)
        
        # Normalize to [-1, 1]
        tensor = self.normalize(tensor)
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.photo_list)
    
    def __getitem__(self, idx: int) -> dict:
        # Load photo
        photo_path = self.photo_list[idx]
        photo = self._load_image(photo_path)
        photo = self._resize_crop(photo)
        photo = self._augment(photo)
        photo_tensor = self._to_tensor(photo)
        
        # Load random cartoon image
        cartoon_idx = random.randint(0, len(self.cartoon_list) - 1)
        cartoon_path = self.cartoon_list[cartoon_idx]
        cartoon = self._load_image(cartoon_path)
        cartoon = self._resize_crop(cartoon)
        cartoon = self._augment(cartoon)
        cartoon_tensor = self._to_tensor(cartoon)
        
        return {
            'photo': photo_tensor,
            'cartoon': cartoon_tensor,
            'photo_path': photo_path,
            'cartoon_path': cartoon_path
        }


class PhotoDataset(Dataset):
    """
    Simple Photo Dataset for pretraining or inference
    """
    
    def __init__(self,
                 photo_dir: str,
                 image_size: int = 256,
                 augment: bool = True):
        self.photo_dir = photo_dir
        self.image_size = image_size
        self.augment = augment
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.photo_list = []
        
        for filename in os.listdir(photo_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                self.photo_list.append(os.path.join(photo_dir, filename))
                
        self.photo_list.sort()
        print(f"Loaded {len(self.photo_list)} photos")
        
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])
        
    def __len__(self) -> int:
        return len(self.photo_list)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.photo_list[idx]
        
        # Load image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Resize
        if h < w:
            new_h = self.image_size
            new_w = int(w * self.image_size / h)
        else:
            new_w = self.image_size
            new_h = int(h * self.image_size / w)
            
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Crop
        if self.augment:
            start_h = random.randint(0, max(0, new_h - self.image_size))
            start_w = random.randint(0, max(0, new_w - self.image_size))
        else:
            start_h = (new_h - self.image_size) // 2
            start_w = (new_w - self.image_size) // 2
            
        img = img[start_h:start_h + self.image_size,
                  start_w:start_w + self.image_size]
        
        # Augmentation
        if self.augment and random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        # To tensor
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img)
        tensor = self.normalize(tensor)
        
        return {
            'photo': tensor,
            'path': path
        }


class InferenceDataset(Dataset):
    """
    Dataset for inference with variable-sized images
    """
    
    def __init__(self,
                 image_dir: str,
                 max_size: int = 720):
        """
        Args:
            image_dir: Directory containing images to process
            max_size: Maximum dimension (larger images will be resized)
        """
        self.image_dir = image_dir
        self.max_size = max_size
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_list = []
        
        for filename in os.listdir(image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                self.image_list.append(os.path.join(image_dir, filename))
                
        self.image_list.sort()
        print(f"Found {len(self.image_list)} images for inference")
        
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> dict:
        path = self.image_list[idx]
        
        # Load image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Resize if too large
        if min(h, w) > self.max_size:
            if h > w:
                new_h = int(self.max_size * h / w)
                new_w = self.max_size
            else:
                new_h = self.max_size
                new_w = int(self.max_size * w / h)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            
        # Make dimensions divisible by 8 (for network)
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        img = img[:new_h, :new_w]
        
        # To tensor
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img)
        tensor = tensor * 2 - 1  # Normalize to [-1, 1]
        
        return {
            'image': tensor,
            'path': path,
            'filename': os.path.basename(path)
        }


def simple_superpixel(image: np.ndarray, n_segments: int = 200) -> np.ndarray:
    """
    Create superpixel representation for surface loss
    
    Args:
        image: Input image [H, W, C] in range [0, 255]
        n_segments: Number of superpixel segments
        
    Returns:
        Superpixel averaged image
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for superpixel segmentation")
        
    # SLIC superpixel segmentation
    segments = segmentation.slic(image, n_segments=n_segments, sigma=1,
                                  compactness=10, convert2lab=True, start_label=0)
    
    # Average colors within each superpixel
    output = skcolor.label2rgb(segments, image, kind='avg', bg_label=-1)
    
    return (output * 255).astype(np.uint8)


def batch_superpixel(images: torch.Tensor, n_segments: int = 200) -> torch.Tensor:
    """
    Apply superpixel segmentation to a batch of images
    
    Args:
        images: Batch of images [B, C, H, W] in range [-1, 1]
        n_segments: Number of superpixel segments
        
    Returns:
        Superpixel averaged images [B, C, H, W] in range [-1, 1]
    """
    # Convert to numpy
    batch = images.detach().cpu().numpy()
    batch = (batch + 1) * 127.5  # To [0, 255]
    batch = np.transpose(batch, (0, 2, 3, 1))  # BCHW to BHWC
    
    outputs = []
    for img in batch:
        sp = simple_superpixel(img.astype(np.uint8), n_segments)
        outputs.append(sp)
        
    outputs = np.array(outputs)
    outputs = np.transpose(outputs, (0, 3, 1, 2))  # BHWC to BCHW
    outputs = outputs.astype(np.float32) / 127.5 - 1  # To [-1, 1]
    
    return torch.from_numpy(outputs).to(images.device)


def create_dataloader(dataset: Dataset,
                      batch_size: int = 16,
                      shuffle: bool = True,
                      num_workers: int = 4,
                      pin_memory: bool = True) -> DataLoader:
    """
    Create a DataLoader with recommended settings
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )


if __name__ == '__main__':
    # Test dataset loading
    print("Dataset module loaded successfully")
    print(f"scikit-image available: {HAS_SKIMAGE}")
    
    # Test superpixel if available
    if HAS_SKIMAGE:
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        sp = simple_superpixel(test_img, n_segments=100)
        print(f"Superpixel output shape: {sp.shape}")

