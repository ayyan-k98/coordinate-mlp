"""
Coordinate-Based Coverage Network

Main architecture for scale-invariant coverage planning using coordinate-based
representations and Fourier features.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .positional_encoding import FourierPositionalEncoding, generate_normalized_coords
from .cell_encoder import CellFeatureMLP
from .attention import AttentionPooling
from .q_network import DuelingQNetwork


class CoordinateCoverageNetwork(nn.Module):
    """
    Scale-invariant network using coordinate-based representations.
    
    Pipeline:
    1. Grid[H, W, C] → sample coordinates for each cell
    2. Coordinates → Fourier features
    3. Fourier features + grid values → per-cell embeddings
    4. Attention aggregation over cells
    5. Q-values for actions
    """
    
    def __init__(
        self,
        input_channels: int = 5,      # Grid channels (visited, coverage, agent, frontier, obstacles)
        num_freq_bands: int = 6,      # Fourier feature bands (output: 2 + 4*6 = 26)
        hidden_dim: int = 256,        # Hidden layer size
        num_actions: int = 9,         # Action space size
        num_attention_heads: int = 4, # Number of attention heads
        dropout: float = 0.1          # Dropout probability
    ):
        """
        Args:
            input_channels: Number of channels in input grid
            num_freq_bands: Number of frequency bands for Fourier encoding
            hidden_dim: Hidden dimension for feature processing
            num_actions: Number of discrete actions
            num_attention_heads: Number of attention heads for pooling
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_freq_bands = num_freq_bands
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Sub-modules
        self.positional_encoder = FourierPositionalEncoding(num_freq_bands=num_freq_bands)
        coord_dim = self.positional_encoder.get_output_dim()
        
        self.cell_encoder = CellFeatureMLP(
            input_dim=coord_dim + input_channels,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.attention_pool = AttentionPooling(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.q_head = DuelingQNetwork(
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            dropout=dropout
        )
        
        # Cache for coordinate features (to avoid recomputing)
        self._coord_cache = {}
    
    def _get_coord_features(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Get or compute coordinate features for given grid size.
        
        Args:
            H: Grid height
            W: Grid width
            device: Device to create tensors on
        
        Returns:
            coord_features: [H*W, coord_dim]
        """
        cache_key = (H, W, str(device))
        
        if cache_key not in self._coord_cache:
            # Generate normalized coordinates
            coords = generate_normalized_coords(H, W, device=device)  # [H*W, 2]
            
            # Encode with Fourier features
            coord_features = self.positional_encoder(coords)  # [H*W, coord_dim]
            
            # Cache for future use
            self._coord_cache[cache_key] = coord_features
        
        return self._coord_cache[cache_key]
    
    def forward(
        self,
        grid: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            grid: [batch, channels, H, W] - Input grid
            return_attention: If True, also return attention weights
        
        Returns:
            q_values: [batch, num_actions] - Q-values for each action
            attention_weights: [batch, num_heads, 1, H*W] (optional)
        """
        B, C, H, W = grid.shape
        device = grid.device
        
        # STEP 1: Get coordinate features for this grid size
        coord_features = self._get_coord_features(H, W, device)  # [H*W, coord_dim]
        
        # STEP 2: Get grid values for each cell
        # Rearrange from [B, C, H, W] to [B, H*W, C]
        grid_values = grid.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # STEP 3: Combine coordinate features with grid values
        # Expand coord_features to batch: [B, H*W, coord_dim]
        coord_features_batched = coord_features.unsqueeze(0).expand(B, -1, -1)
        
        # Concatenate: [B, H*W, coord_dim + C]
        combined = torch.cat([coord_features_batched, grid_values], dim=-1)
        
        # STEP 4: Per-cell feature extraction
        cell_features = self.cell_encoder(combined)  # [B, H*W, hidden_dim]
        
        # STEP 5: Attention-based aggregation
        if return_attention:
            aggregated, attention_weights = self.attention_pool(
                cell_features, return_attention_weights=True
            )
        else:
            aggregated = self.attention_pool(cell_features)
            attention_weights = None
        
        # STEP 6: Q-values
        q_values = self.q_head(aggregated)  # [B, num_actions]
        
        if return_attention:
            return q_values, attention_weights
        else:
            return q_values
    
    def clear_coord_cache(self):
        """Clear the coordinate feature cache (e.g., when changing devices)."""
        self._coord_cache.clear()
    
    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Unit test
    print("="*70)
    print("Testing Coordinate Coverage Network")
    print("="*70)
    
    # Configuration
    input_channels = 5
    num_freq_bands = 6
    hidden_dim = 256
    num_actions = 9
    
    # Create model
    model = CoordinateCoverageNetwork(
        input_channels=input_channels,
        num_freq_bands=num_freq_bands,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        num_attention_heads=4
    )
    
    print(f"\nModel created:")
    print(f"  Input channels: {input_channels}")
    print(f"  Fourier bands: {num_freq_bands}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num actions: {num_actions}")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # Test 1: Single grid
    batch_size = 1
    H, W = 20, 20
    grid = torch.randn(batch_size, input_channels, H, W)
    q_values = model(grid)
    print(f"\nTest 1 - Single 20×20 grid:")
    print(f"  Input: {grid.shape}")
    print(f"  Q-values: {q_values.shape}")
    print(f"  Q-values: {q_values[0].tolist()}")
    assert q_values.shape == (batch_size, num_actions)
    
    # Test 2: Batch of grids
    batch_size = 8
    grid = torch.randn(batch_size, input_channels, H, W)
    q_values = model(grid)
    print(f"\nTest 2 - Batch of 20×20 grids:")
    print(f"  Batch size: {batch_size}")
    print(f"  Q-values: {q_values.shape}")
    assert q_values.shape == (batch_size, num_actions)
    
    # Test 3: Scale invariance - different grid sizes
    print(f"\nTest 3 - Scale invariance:")
    for grid_size in [15, 20, 25, 30, 35, 40]:
        grid = torch.randn(4, input_channels, grid_size, grid_size)
        q_values = model(grid)
        num_cells = grid_size * grid_size
        print(f"  {grid_size:2d}×{grid_size:2d} grid ({num_cells:4d} cells): "
              f"{q_values.shape} ✓")
        assert q_values.shape == (4, num_actions)
    
    # Test 4: Attention weights
    batch_size = 2
    H, W = 20, 20
    grid = torch.randn(batch_size, input_channels, H, W)
    q_values, attn_weights = model(grid, return_attention=True)
    print(f"\nTest 4 - Attention weights:")
    print(f"  Q-values: {q_values.shape}")
    print(f"  Attention: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, 4, 1, H * W)
    
    # Test 5: Gradient flow
    grid = torch.randn(2, input_channels, 20, 20, requires_grad=True)
    q_values = model(grid)
    loss = q_values.sum()
    loss.backward()
    print(f"\nTest 5 - Gradient flow:")
    print(f"  Input grad: {grid.grad is not None}")
    print(f"  Grad norm: {grid.grad.norm().item():.4f}")
    
    # Test 6: Coordinate cache
    print(f"\nTest 6 - Coordinate cache:")
    print(f"  Cache size before: {len(model._coord_cache)}")
    
    # Forward pass with different sizes
    for size in [20, 25, 20, 30, 20]:
        grid = torch.randn(1, input_channels, size, size)
        _ = model(grid)
    
    print(f"  Cache size after: {len(model._coord_cache)}")
    print(f"  Cached sizes: {[(k[0], k[1]) for k in model._coord_cache.keys()]}")
    
    # Clear cache
    model.clear_coord_cache()
    print(f"  Cache size after clear: {len(model._coord_cache)}")
    
    # Test 7: Model summary
    print(f"\nTest 7 - Model architecture summary:")
    print(f"  Positional Encoder: {model.positional_encoder.output_dim} dims")
    print(f"  Cell Encoder: {model.cell_encoder.input_dim} → {model.cell_encoder.hidden_dim}")
    print(f"  Attention Pooling: {model.attention_pool.num_heads} heads")
    print(f"  Q-Network: {model.q_head.hidden_dim} → {model.q_head.num_actions}")
    
    # Count parameters per module
    print(f"\n  Parameters breakdown:")
    for name, module in [
        ("Positional Encoder", model.positional_encoder),
        ("Cell Encoder", model.cell_encoder),
        ("Attention Pooling", model.attention_pool),
        ("Q-Network", model.q_head)
    ]:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"    {name:20s}: {num_params:8,} params")
    
    # Test 8: Non-square grids
    print(f"\nTest 8 - Non-square grids:")
    for H, W in [(15, 20), (20, 25), (10, 30)]:
        grid = torch.randn(2, input_channels, H, W)
        q_values = model(grid)
        print(f"  {H}×{W} grid: {q_values.shape} ✓")
        assert q_values.shape == (2, num_actions)
    
    print("\n✓ All tests passed!")
    print("="*70)
