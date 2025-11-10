"""
Cell Feature Encoder

Processes each cell independently to extract features from combined
coordinate and grid value information.
"""

import torch
import torch.nn as nn


class CellFeatureMLP(nn.Module):
    """
    MLP that processes each cell's combined features.
    
    Input:  [batch, num_cells, coord_dim + grid_channels]
    Output: [batch, num_cells, hidden_dim]
    
    This is applied independently to each cell (permutation-invariant across cells).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: coord_dim + grid_channels (e.g., 26 + 5 = 31)
            hidden_dim: Output feature dimension (e.g., 256)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Three-layer MLP with stable ordering: Linear → LayerNorm → ReLU → Dropout
        # This order prevents gradient explosions in mixed precision training
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),  # Normalize BEFORE activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize BEFORE activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_cells, input_dim]
        
        Returns:
            features: [batch, num_cells, hidden_dim]
        """
        # Apply MLP to each cell independently
        # The network automatically broadcasts over the num_cells dimension
        return self.network(x)


if __name__ == "__main__":
    # Unit test
    print("="*60)
    print("Testing Cell Feature MLP")
    print("="*60)
    
    # Configuration
    coord_dim = 26  # Fourier features
    grid_channels = 5  # Coverage grid channels
    input_dim = coord_dim + grid_channels
    hidden_dim = 256
    
    # Create model
    model = CellFeatureMLP(input_dim=input_dim, hidden_dim=hidden_dim)
    print(f"\nModel created:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: Single batch
    batch_size = 1
    num_cells = 400  # 20x20 grid
    x = torch.randn(batch_size, num_cells, input_dim)
    output = model(x)
    print(f"\nTest 1 - Single batch:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (batch_size, num_cells, hidden_dim)
    
    # Test 2: Multiple batches
    batch_size = 8
    output = model(torch.randn(batch_size, num_cells, input_dim))
    print(f"\nTest 2 - Multiple batches:")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output.shape}")
    assert output.shape == (batch_size, num_cells, hidden_dim)
    
    # Test 3: Different grid sizes
    for grid_size in [15, 20, 25, 30]:
        num_cells = grid_size * grid_size
        x = torch.randn(4, num_cells, input_dim)
        output = model(x)
        print(f"\nTest 3 - Grid size {grid_size}×{grid_size}:")
        print(f"  Num cells: {num_cells}")
        print(f"  Output: {output.shape}")
        assert output.shape == (4, num_cells, hidden_dim)
    
    # Test 4: Gradient flow
    x = torch.randn(2, 100, input_dim, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    print(f"\nTest 4 - Gradient flow:")
    print(f"  Input grad: {x.grad is not None}")
    print(f"  Grad norm: {x.grad.norm().item():.4f}")
    
    print("\n✓ All tests passed!")
