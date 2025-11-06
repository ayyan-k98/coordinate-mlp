"""
Fourier Positional Encoding for Coordinate-Based Networks

Converts normalized coordinates (x, y) ∈ [-1, 1]² into high-frequency Fourier features.

Based on:
- Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields" (2020)
- Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions" (2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class FourierPositionalEncoding(nn.Module):
    """
    Maps coordinates to higher-dimensional space using sinusoidal functions.
    
    Input:  (x, y) ∈ [-1, 1]²  (2D)
    Output: [x, y, sin(2⁰πx), cos(2⁰πx), sin(2¹πx), cos(2¹πx), ...]
    
    where L = number of frequency bands
    Output dimension: 2 + 4*L
    """
    
    def __init__(self, num_freq_bands: int = 6):
        """
        Args:
            num_freq_bands: Number of frequency octaves
                            L=6 → frequencies [1, 2, 4, 8, 16, 32]
                            Output dimension: 2 + 4*L = 2 + 24 = 26
        """
        super().__init__()
        
        self.num_freq_bands = num_freq_bands
        
        # Frequency bands: [2^0, 2^1, 2^2, ..., 2^(L-1)]
        frequencies = torch.tensor([2.0 ** i for i in range(num_freq_bands)])
        self.register_buffer('frequencies', frequencies)
        
        # Output dimension: original coords (2) + sin/cos for each freq (4*L)
        self.output_dim = 2 + 4 * num_freq_bands
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [N, 2] or [B, N, 2] - normalized coordinates in [-1, 1]
        
        Returns:
            features: [..., N, output_dim] - Fourier features
        """
        # Handle both [N, 2] and [B, N, 2] inputs
        input_shape = coords.shape
        if len(input_shape) == 2:
            coords = coords.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, _ = coords.shape
        
        # Start with original coordinates
        features = [coords]  # [B, N, 2]
        
        # Add sinusoidal features for each frequency
        for freq in self.frequencies:
            # Compute 2π * freq * coords for both x and y
            scaled_coords = 2 * np.pi * freq * coords  # [B, N, 2]
            
            # Add sin and cos features
            features.append(torch.sin(scaled_coords))  # [B, N, 2]
            features.append(torch.cos(scaled_coords))  # [B, N, 2]
        
        # Concatenate: [B, N, 2 + 4*L]
        output = torch.cat(features, dim=-1)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.output_dim


def generate_normalized_coords(H: int, W: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate normalized coordinates for H×W grid.
    
    Args:
        H: Grid height
        W: Grid width
        device: Device to create tensor on
    
    Returns:
        coords: [H*W, 2] where each row is (x, y) ∈ [-1, 1]²
    
    Example for 3×3 grid:
        [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0),
         (-1.0,  0.0), (0.0,  0.0), (1.0,  0.0),
         (-1.0,  1.0), (0.0,  1.0), (1.0,  1.0)]
    """
    # Linspace in [-1, 1]
    y_coords = torch.linspace(-1, 1, H, device=device)  # [H]
    x_coords = torch.linspace(-1, 1, W, device=device)  # [W]
    
    # Meshgrid (indexing='ij' for row-major order)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W] each
    
    # Stack and flatten
    coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    coords = coords.reshape(-1, 2)  # [H*W, 2]
    
    return coords


if __name__ == "__main__":
    # Unit test
    print("="*60)
    print("Testing Fourier Positional Encoding")
    print("="*60)
    
    # Test 1: Basic encoding
    encoder = FourierPositionalEncoding(num_freq_bands=6)
    print(f"\nEncoder output dimension: {encoder.output_dim}")
    
    # Test 2: Single coordinate
    coord = torch.tensor([[0.0, 0.0]])  # Origin
    encoded = encoder(coord)
    print(f"\nSingle coordinate {coord.shape} → {encoded.shape}")
    print(f"First 10 features: {encoded[0, :10].tolist()}")
    
    # Test 3: Batch of coordinates
    coords = torch.tensor([
        [0.0, 0.0],
        [0.5, 0.5],
        [-1.0, 1.0],
        [1.0, -1.0]
    ])
    encoded = encoder(coords)
    print(f"\nBatch of coordinates {coords.shape} → {encoded.shape}")
    
    # Test 4: Generate grid coordinates
    H, W = 5, 5
    grid_coords = generate_normalized_coords(H, W)
    print(f"\nGenerated {H}×{W} grid: {grid_coords.shape}")
    print(f"Min coords: {grid_coords.min(dim=0).values}")
    print(f"Max coords: {grid_coords.max(dim=0).values}")
    print(f"First few coords:\n{grid_coords[:5]}")
    
    # Test 5: Encode entire grid
    encoded_grid = encoder(grid_coords)
    print(f"\nEncoded grid: {encoded_grid.shape}")
    
    # Test 6: Batched grid encoding
    batch_size = 4
    batched_coords = grid_coords.unsqueeze(0).expand(batch_size, -1, -1)
    batched_encoded = encoder(batched_coords)
    print(f"\nBatched encoding: {batched_coords.shape} → {batched_encoded.shape}")
    
    print("\n✓ All tests passed!")
