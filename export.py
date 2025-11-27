"""
White-box Cartoonization - PyTorch Implementation
Model Export Script

Export trained model to ONNX and CoreML formats for deployment.
"""

import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

from models.generator import UNetGenerator
from models.guided_filter import guided_filter


class CartoonizeModel(nn.Module):
    """
    Wrapper model for export that includes guided filter post-processing
    """
    
    def __init__(self, generator: nn.Module, 
                 use_guided_filter: bool = True,
                 filter_r: int = 1,
                 filter_eps: float = 5e-3):
        super().__init__()
        self.generator = generator
        self.use_guided_filter = use_guided_filter
        self.filter_r = filter_r
        self.filter_eps = filter_eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.generator(x)
        
        if self.use_guided_filter:
            # Simplified guided filter for export
            output = self._simple_guided_filter(x, output)
            
        return output
    
    def _simple_guided_filter(self, guide: torch.Tensor, 
                               src: torch.Tensor) -> torch.Tensor:
        """Simplified guided filter compatible with export"""
        r = self.filter_r
        eps = self.filter_eps
        
        # Box filter using average pooling
        kernel_size = 2 * r + 1
        padding = r
        
        # Create box kernel manually for better compatibility
        b, c, h, w = guide.shape
        
        # Compute means using convolution
        ones = torch.ones(1, 1, h, w, device=guide.device, dtype=guide.dtype)
        
        # Use unfold for box filtering (more export-friendly)
        mean_guide = torch.nn.functional.avg_pool2d(
            guide, kernel_size, stride=1, padding=padding
        )
        mean_src = torch.nn.functional.avg_pool2d(
            src, kernel_size, stride=1, padding=padding
        )
        mean_guide_src = torch.nn.functional.avg_pool2d(
            guide * src, kernel_size, stride=1, padding=padding
        )
        mean_guide_guide = torch.nn.functional.avg_pool2d(
            guide * guide, kernel_size, stride=1, padding=padding
        )
        
        cov = mean_guide_src - mean_guide * mean_src
        var = mean_guide_guide - mean_guide * mean_guide
        
        a = cov / (var + eps)
        b = mean_src - a * mean_guide
        
        mean_a = torch.nn.functional.avg_pool2d(
            a, kernel_size, stride=1, padding=padding
        )
        mean_b = torch.nn.functional.avg_pool2d(
            b, kernel_size, stride=1, padding=padding
        )
        
        output = mean_a * guide + mean_b
        
        return output


def export_to_onnx(model: nn.Module, 
                   output_path: str,
                   input_size: Tuple[int, int] = (256, 256),
                   dynamic_axes: bool = True,
                   opset_version: int = 14):
    """
    Export model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_size: (height, width) of input
        dynamic_axes: If True, allow dynamic input sizes
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Configure dynamic axes
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_axes_config = None
        
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config
    )
    
    print(f"ONNX model saved to {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
    except ImportError:
        print("Install 'onnx' package to verify the exported model")
    except Exception as e:
        print(f"ONNX verification warning: {e}")


def export_to_coreml(model: nn.Module,
                     output_path: str,
                     input_size: Tuple[int, int] = (256, 256)):
    """
    Export model to CoreML format for iOS deployment
    
    Args:
        model: PyTorch model
        output_path: Output .mlmodel or .mlpackage path
        input_size: (height, width) of input
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Please install coremltools: pip install coremltools")
        return
        
    model.eval()
    
    # Trace the model
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, input_size[0], input_size[1]),
            scale=1/127.5,
            bias=[-1, -1, -1]
        )],
        outputs=[ct.ImageType(
            name="output",
            scale=127.5,
            bias=[127.5, 127.5, 127.5]
        )],
        minimum_deployment_target=ct.target.iOS15
    )
    
    # Add metadata
    mlmodel.author = "White-box Cartoonization"
    mlmodel.short_description = "Cartoonize images using white-box cartoon representations"
    mlmodel.version = "1.0"
    
    # Save
    mlmodel.save(output_path)
    print(f"CoreML model saved to {output_path}")


def export_to_torchscript(model: nn.Module,
                          output_path: str,
                          input_size: Tuple[int, int] = (256, 256),
                          use_script: bool = False):
    """
    Export model to TorchScript format
    
    Args:
        model: PyTorch model
        output_path: Output .pt file path
        input_size: (height, width) for tracing
        use_script: If True, use torch.jit.script instead of trace
    """
    model.eval()
    
    if use_script:
        scripted_model = torch.jit.script(model)
    else:
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
        scripted_model = torch.jit.trace(model, dummy_input)
        
    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Export Cartoonization Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='exported_models',
                        help='Output directory')
    parser.add_argument('--name', type=str, default='cartoonizer',
                        help='Model name prefix')
    
    # Model settings
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--input_size', type=int, default=256,
                        help='Input image size (square)')
    
    # Export formats
    parser.add_argument('--onnx', action='store_true',
                        help='Export to ONNX format')
    parser.add_argument('--coreml', action='store_true',
                        help='Export to CoreML format')
    parser.add_argument('--torchscript', action='store_true',
                        help='Export to TorchScript format')
    parser.add_argument('--all', action='store_true',
                        help='Export to all formats')
    
    # Options
    parser.add_argument('--no_guided_filter', action='store_true',
                        help='Disable guided filter in exported model')
    parser.add_argument('--dynamic_onnx', action='store_true', default=True,
                        help='Enable dynamic axes for ONNX')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generator
    generator = UNetGenerator(
        base_channels=args.base_channels,
        num_blocks=args.num_blocks
    )
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
        
    print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create export model
    export_model = CartoonizeModel(
        generator,
        use_guided_filter=not args.no_guided_filter
    )
    export_model.eval()
    
    input_size = (args.input_size, args.input_size)
    
    # Export to requested formats
    if args.all or args.onnx:
        onnx_path = os.path.join(args.output_dir, f'{args.name}.onnx')
        export_to_onnx(export_model, onnx_path, input_size, args.dynamic_onnx)
        
    if args.all or args.coreml:
        coreml_path = os.path.join(args.output_dir, f'{args.name}.mlpackage')
        export_to_coreml(export_model, coreml_path, input_size)
        
    if args.all or args.torchscript:
        ts_path = os.path.join(args.output_dir, f'{args.name}.pt')
        export_to_torchscript(export_model, ts_path, input_size)
        
    print("Export complete!")


if __name__ == '__main__':
    main()

