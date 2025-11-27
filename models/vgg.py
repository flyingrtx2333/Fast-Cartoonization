"""
White-box Cartoonization - PyTorch Implementation
VGG19 Feature Extractor for Perceptual Loss

Author: Reimplemented from CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Optional


# ImageNet normalization constants (PyTorch standard)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# VGG mean values (original Caffe format, BGR order, 0-255 range)
# Used in the original TensorFlow implementation
VGG_MEAN_BGR = [103.939, 116.779, 123.68]


class VGG19FeatureExtractor(nn.Module):
    """
    VGG19 Feature Extractor for computing perceptual loss
    
    Extracts intermediate features from VGG19 pretrained on ImageNet.
    Commonly used layers:
    - relu1_2: Low-level features (edges, colors)
    - relu2_2: Textures
    - relu3_4: Object parts  
    - relu4_4: Semantic features (used in white-box cartoonization)
    - relu5_4: High-level semantic features
    """
    
    # VGG19 layer name to index mapping
    LAYER_MAP = {
        'conv1_1': 0, 'relu1_1': 1, 'conv1_2': 2, 'relu1_2': 3, 'pool1': 4,
        'conv2_1': 5, 'relu2_1': 6, 'conv2_2': 7, 'relu2_2': 8, 'pool2': 9,
        'conv3_1': 10, 'relu3_1': 11, 'conv3_2': 12, 'relu3_2': 13,
        'conv3_3': 14, 'relu3_3': 15, 'conv3_4': 16, 'relu3_4': 17, 'pool3': 18,
        'conv4_1': 19, 'relu4_1': 20, 'conv4_2': 21, 'relu4_2': 22,
        'conv4_3': 23, 'relu4_3': 24, 'conv4_4': 25, 'relu4_4': 26, 'pool4': 27,
        'conv5_1': 28, 'relu5_1': 29, 'conv5_2': 30, 'relu5_2': 31,
        'conv5_3': 32, 'relu5_3': 33, 'conv5_4': 34, 'relu5_4': 35, 'pool5': 36,
    }
    
    def __init__(self, 
                 feature_layers: List[str] = ['conv4_4'],
                 use_input_norm: bool = True,
                 requires_grad: bool = False,
                 use_caffe_preprocessing: bool = True):
        """
        Args:
            feature_layers: List of layer names to extract features from
            use_input_norm: If True, normalize input
            requires_grad: If False, freeze VGG weights
            use_caffe_preprocessing: If True, use original Caffe/TF preprocessing (BGR, 0-255)
                                    This matches the original white-box cartoonization paper.
        """
        super().__init__()
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        self.use_caffe_preprocessing = use_caffe_preprocessing
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Get the maximum layer index we need
        max_idx = max(self.LAYER_MAP[layer] for layer in feature_layers)
        
        # Extract only the layers we need
        self.features = nn.Sequential(*list(vgg.features.children())[:max_idx + 1])
        
        # Register normalization buffers
        if use_input_norm:
            if use_caffe_preprocessing:
                # Original Caffe/TensorFlow preprocessing (BGR, 0-255 range)
                # VGG_MEAN_BGR = [103.939, 116.779, 123.68] for B, G, R
                self.register_buffer('mean_bgr', torch.tensor(VGG_MEAN_BGR).view(1, 3, 1, 1))
            else:
                # PyTorch ImageNet preprocessing
                self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        
        # Freeze weights if not training
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        # Store layer indices for extraction
        self.layer_indices = {layer: self.LAYER_MAP[layer] for layer in feature_layers}
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input from [-1, 1] to VGG expected format
        
        For Caffe preprocessing (original paper):
            1. Convert from [-1, 1] to [0, 255]
            2. Convert RGB to BGR
            3. Subtract VGG mean [103.939, 116.779, 123.68]
            
        For PyTorch preprocessing:
            1. Convert from [-1, 1] to [0, 1]
            2. Subtract ImageNet mean and divide by std
        """
        if self.use_caffe_preprocessing:
            # Convert from [-1, 1] to [0, 255]
            x = (x + 1) * 127.5
            # Convert RGB to BGR by flipping channel dimension
            x = x[:, [2, 1, 0], :, :]  # RGB -> BGR
            # Subtract VGG mean (BGR order)
            x = x - self.mean_bgr
        else:
            # PyTorch standard preprocessing
            x = (x + 1) / 2
            x = (x - self.mean) / self.std
        return x
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers
        
        Args:
            x: Input tensor [B, 3, H, W], expected range [-1, 1]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        if self.use_input_norm:
            x = self._normalize(x)
            
        features = {}
        
        for name, module in self.features._modules.items():
            x = module(x)
            
            idx = int(name)
            for layer_name, layer_idx in self.layer_indices.items():
                if idx == layer_idx:
                    features[layer_name] = x
                    
        return features
    
    def extract_single(self, x: torch.Tensor, layer: str = 'conv4_4') -> torch.Tensor:
        """
        Extract features from a single layer (convenience method)
        
        Args:
            x: Input tensor [B, 3, H, W]
            layer: Layer name to extract
            
        Returns:
            Feature tensor
        """
        if self.use_input_norm:
            x = self._normalize(x)
            
        target_idx = self.LAYER_MAP[layer]
        
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx == target_idx:
                return x
                
        raise ValueError(f"Layer {layer} not found in network")


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features
    
    Computes the difference between VGG features of two images.
    """
    
    def __init__(self, 
                 feature_layers: List[str] = ['conv4_4'],
                 weights: Optional[List[float]] = None,
                 criterion: str = 'l1'):
        super().__init__()
        
        self.vgg = VGG19FeatureExtractor(feature_layers, requires_grad=False)
        self.feature_layers = feature_layers
        
        if weights is None:
            weights = [1.0] * len(feature_layers)
        self.weights = weights
        
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            
        Returns:
            Scalar loss value
        """
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        
        loss = 0.0
        for layer, weight in zip(self.feature_layers, self.weights):
            # Normalize by feature dimensions
            feat_pred = pred_features[layer]
            feat_target = target_features[layer]
            
            _, c, h, w = feat_pred.shape
            loss += weight * self.criterion(feat_pred, feat_target) / (c * h * w)
            
        return loss


class StyleLoss(nn.Module):
    """
    Style Loss using Gram matrix of VGG features
    
    Captures texture and style information.
    """
    
    def __init__(self, feature_layers: List[str] = ['relu2_2', 'relu3_4', 'relu4_4']):
        super().__init__()
        
        self.vgg = VGG19FeatureExtractor(feature_layers, requires_grad=False)
        self.feature_layers = feature_layers
        
    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style representation"""
        b, c, h, w = x.shape
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
        
    def forward(self, pred: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Compute style loss"""
        pred_features = self.vgg(pred)
        style_features = self.vgg(style)
        
        loss = 0.0
        for layer in self.feature_layers:
            pred_gram = self.gram_matrix(pred_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += torch.mean((pred_gram - style_gram) ** 2)
            
        return loss / len(self.feature_layers)


if __name__ == '__main__':
    # Test VGG feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vgg = VGG19FeatureExtractor(['conv4_4']).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    
    features = vgg(x)
    print("VGG19 Feature Extractor:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
        
    # Test perceptual loss
    perc_loss = PerceptualLoss(['conv4_4']).to(device)
    pred = torch.randn(1, 3, 256, 256).to(device)
    target = torch.randn(1, 3, 256, 256).to(device)
    
    loss = perc_loss(pred, target)
    print(f"\nPerceptual loss: {loss.item():.6f}")
    
    # Test style loss
    style_loss = StyleLoss().to(device)
    loss = style_loss(pred, target)
    print(f"Style loss: {loss.item():.6f}")

