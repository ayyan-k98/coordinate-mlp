"""
Fully Convolutional Network (FCN) for Coverage Planning

A stable CNN-based architecture that replaces the Coordinate MLP's attention
mechanism with proven convolutional layers and spatial softmax aggregation.

Architecture:
1. Spatial encoder: 3 conv layers with residual connections
2. Spatial softmax: Aggregate to fixed-size vector
3. Simple dueling Q-head: 2-layer head for actions

Key differences from Coordinate MLP:
- NO Fourier features (raw grid input)
- NO attention mechanism (spatial softmax instead)
- NO per-cell MLPs (convolutional processing)
- Simpler, battle-tested components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpatialEncoder(nn.Module):
    """
    Encode grid with convolutional layers and residual connections.

    Architecture:
        Conv Block 1: 5 → 64 channels
        Conv Block 2: 64 → 128 channels (+ residual from block 1)
        Conv Block 3: 128 → 256 channels (+ residual from block 2)

    Each block: Conv3x3 → BatchNorm → ReLU → Dropout
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_channels: list = [64, 128, 256],
        dropout: float = 0.1
    ):
        """
        Args:
            input_channels: Number of input grid channels (default: 5)
            hidden_channels: Channel progression [64, 128, 256]
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # Block 1: 5 → 64
        self.conv1 = nn.Conv2d(
            input_channels, hidden_channels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.dropout1 = nn.Dropout2d(dropout)

        # Block 2: 64 → 128
        self.conv2 = nn.Conv2d(
            hidden_channels[0], hidden_channels[1],
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels[1])
        self.dropout2 = nn.Dropout2d(dropout)

        # Residual connection: 64 → 128 (1x1 conv for dimension matching)
        self.residual_1_to_2 = nn.Conv2d(
            hidden_channels[0], hidden_channels[1],
            kernel_size=1, stride=1, bias=False
        )

        # Block 3: 128 → 256
        self.conv3 = nn.Conv2d(
            hidden_channels[1], hidden_channels[2],
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(hidden_channels[2])
        self.dropout3 = nn.Dropout2d(dropout)

        # Residual connection: 128 → 256 (1x1 conv for dimension matching)
        self.residual_2_to_3 = nn.Conv2d(
            hidden_channels[1], hidden_channels[2],
            kernel_size=1, stride=1, bias=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 5, H, W] - Input grid

        Returns:
            features: [B, 256, H, W] - Encoded features
        """
        # Block 1: 5 → 64
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, inplace=True)
        out1 = self.dropout1(out1)

        # Block 2: 64 → 128 (with residual)
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        residual = self.residual_1_to_2(out1)
        out2 = out2 + residual  # Residual connection
        out2 = F.relu(out2, inplace=True)
        out2 = self.dropout2(out2)

        # Block 3: 128 → 256 (with residual)
        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        residual = self.residual_2_to_3(out2)
        out3 = out3 + residual  # Residual connection
        out3 = F.relu(out3, inplace=True)
        out3 = self.dropout3(out3)

        return out3


class SpatialSoftmax(nn.Module):
    """
    Spatial softmax aggregation for converting spatial features to fixed-size vector.

    Computes expected (x, y) position for each feature channel, producing
    a scale-invariant representation.

    Method:
        1. Compute softmax over spatial dimensions
        2. Compute expected (x, y) coordinates
        3. Output: [B, C*2] where C is number of channels

    Reference:
        Levine et al. "End-to-End Training of Deep Visuomotor Policies" (2016)
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Softmax temperature (higher = softer, default: 1.0)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W] - Spatial features

        Returns:
            aggregated: [B, C*2] - Expected (x, y) per channel
        """
        B, C, H, W = features.shape

        # Create normalized coordinate grids
        # x_coords: [H, W] with values in [-1, 1]
        # y_coords: [H, W] with values in [-1, 1]
        device = features.device
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords, y_coords = torch.meshgrid(x_coords, y_coords, indexing='xy')

        # Reshape for broadcasting: [1, 1, H, W]
        x_coords = x_coords.unsqueeze(0).unsqueeze(0)
        y_coords = y_coords.unsqueeze(0).unsqueeze(0)

        # Flatten spatial dimensions: [B, C, H*W]
        features_flat = features.reshape(B, C, H * W)

        # Apply softmax with temperature: [B, C, H*W]
        softmax_attention = F.softmax(features_flat / self.temperature, dim=2)

        # Reshape back: [B, C, H, W]
        softmax_attention = softmax_attention.reshape(B, C, H, W)

        # Compute expected x coordinate per channel: [B, C]
        expected_x = (softmax_attention * x_coords).sum(dim=[2, 3])

        # Compute expected y coordinate per channel: [B, C]
        expected_y = (softmax_attention * y_coords).sum(dim=[2, 3])

        # Concatenate: [B, C*2]
        output = torch.cat([expected_x, expected_y], dim=1)

        return output


class SimpleDuelingQHead(nn.Module):
    """
    Simplified 2-layer dueling Q-network.

    Architecture:
        Shared: Linear(input_dim, hidden_dim) → BatchNorm → ReLU → Dropout
        Value:  Linear(hidden_dim, 1)
        Advantage: Linear(hidden_dim, num_actions)
        Q(s,a) = V(s) + (A(s,a) - mean(A))

    Key simplifications from Coordinate MLP:
    - 2 layers instead of 4 (removed intermediate layers)
    - BatchNorm instead of LayerNorm
    - Smaller weight initialization for final layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_actions: int = 9,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension (typically C*2 from spatial softmax)
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Value stream (scalar output)
        self.value_stream = nn.Linear(hidden_dim, 1)

        # Advantage stream (per-action output)
        self.advantage_stream = nn.Linear(hidden_dim, num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights with special attention to final layers.

        Strategy:
        - Shared layer: Xavier uniform (standard)
        - Value/Advantage streams: Xavier uniform but scaled down by 10×
          to prevent initial Q-value explosions
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Scale down final layers for stability
        with torch.no_grad():
            self.value_stream.weight *= 0.1
            self.advantage_stream.weight *= 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] - Aggregated features

        Returns:
            q_values: [B, num_actions]
        """
        # Shared processing
        shared = self.shared(x)  # [B, hidden_dim]

        # Value stream
        value = self.value_stream(shared)  # [B, 1]

        # Advantage stream
        advantage = self.advantage_stream(shared)  # [B, num_actions]

        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        advantage_mean = advantage.mean(dim=-1, keepdim=True)  # [B, 1]
        q_values = value + (advantage - advantage_mean)  # [B, num_actions]

        return q_values


class FCNCoverageNetwork(nn.Module):
    """
    Fully Convolutional Network for coverage planning.

    Complete architecture:
        1. Input: [B, 5, H, W] grid
        2. Spatial encoder: 3 conv blocks → [B, 256, H, W]
        3. Spatial softmax: → [B, 512]
        4. Dueling Q-head: → [B, 9]

    Key properties:
    - Scale-aware (not scale-invariant like Coordinate MLP)
    - Stable gradient flow (no attention, no Fourier features)
    - Proven components (CNNs since 2013, spatial softmax since 2016)
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_channels: list = None,
        num_actions: int = 9,
        dropout: float = 0.1,
        spatial_softmax_temperature: float = 1.0
    ):
        """
        Args:
            input_channels: Number of grid channels (5 for standard obs)
            hidden_channels: Channel progression for encoder [64, 128, 256]
            num_actions: Number of discrete actions (9 for 8-way + STAY)
            dropout: Dropout probability for regularization
            spatial_softmax_temperature: Temperature for spatial softmax
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128, 256]

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions

        # Sub-modules
        self.spatial_encoder = SpatialEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            dropout=dropout
        )

        self.spatial_softmax = SpatialSoftmax(
            temperature=spatial_softmax_temperature
        )

        # Input to Q-head: 256 channels × 2 coords = 512
        q_head_input_dim = hidden_channels[-1] * 2

        self.q_head = SimpleDuelingQHead(
            input_dim=q_head_input_dim,
            hidden_dim=256,
            num_actions=num_actions,
            dropout=dropout
        )

    def forward(
        self,
        grid: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            grid: [B, 5, H, W] - Input grid
                  Channel 0: visited (binary)
                  Channel 1: coverage probability
                  Channel 2: agent position (binary)
                  Channel 3: frontiers (binary)
                  Channel 4: obstacle belief (probability)
            return_features: If True, also return intermediate features

        Returns:
            q_values: [B, num_actions] - Q-values for each action
            features: dict (optional) - Intermediate features for analysis
        """
        B, C, H, W = grid.shape

        # Step 1: Spatial encoding
        spatial_features = self.spatial_encoder(grid)  # [B, 256, H, W]

        # Step 2: Spatial aggregation
        aggregated = self.spatial_softmax(spatial_features)  # [B, 512]

        # Step 3: Q-values
        q_values = self.q_head(aggregated)  # [B, num_actions]

        if return_features:
            features = {
                'spatial_features': spatial_features,
                'aggregated': aggregated
            }
            return q_values, features
        else:
            return q_values

    def get_num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Unit Tests
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Testing FCN Coverage Network")
    print("="*70)

    # Configuration
    input_channels = 5
    num_actions = 9

    # Create model
    model = FCNCoverageNetwork(
        input_channels=input_channels,
        hidden_channels=[64, 128, 256],
        num_actions=num_actions
    )

    print(f"\nModel created:")
    print(f"  Input channels: {input_channels}")
    print(f"  Hidden channels: [64, 128, 256]")
    print(f"  Num actions: {num_actions}")
    print(f"  Total parameters: {model.get_num_parameters():,}")

    # Test 1: Single grid
    print(f"\n{'='*70}")
    print("Test 1: Single grid forward pass")
    print("="*70)
    batch_size = 1
    H, W = 20, 20
    grid = torch.randn(batch_size, input_channels, H, W)
    q_values = model(grid)
    print(f"  Input: {grid.shape}")
    print(f"  Q-values: {q_values.shape}")
    print(f"  Q-value range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    assert q_values.shape == (batch_size, num_actions)
    print("  ✓ PASS")

    # Test 2: Batch processing
    print(f"\n{'='*70}")
    print("Test 2: Batch processing")
    print("="*70)
    batch_size = 8
    grid = torch.randn(batch_size, input_channels, H, W)
    q_values = model(grid)
    print(f"  Batch size: {batch_size}")
    print(f"  Q-values: {q_values.shape}")
    assert q_values.shape == (batch_size, num_actions)
    print("  ✓ PASS")

    # Test 3: Different grid sizes
    print(f"\n{'='*70}")
    print("Test 3: Different grid sizes")
    print("="*70)
    for grid_size in [15, 20, 25, 30, 40]:
        grid = torch.randn(4, input_channels, grid_size, grid_size)
        q_values = model(grid)
        num_cells = grid_size * grid_size
        print(f"  {grid_size:2d}×{grid_size:2d} grid ({num_cells:4d} cells): "
              f"{q_values.shape} ✓")
        assert q_values.shape == (4, num_actions)
    print("  ✓ PASS")

    # Test 4: Feature extraction
    print(f"\n{'='*70}")
    print("Test 4: Intermediate features")
    print("="*70)
    grid = torch.randn(2, input_channels, 20, 20)
    q_values, features = model(grid, return_features=True)
    print(f"  Q-values: {q_values.shape}")
    print(f"  Spatial features: {features['spatial_features'].shape}")
    print(f"  Aggregated features: {features['aggregated'].shape}")
    assert features['spatial_features'].shape == (2, 256, 20, 20)
    assert features['aggregated'].shape == (2, 512)
    print("  ✓ PASS")

    # Test 5: Gradient flow
    print(f"\n{'='*70}")
    print("Test 5: Gradient flow")
    print("="*70)
    grid = torch.randn(2, input_channels, 20, 20, requires_grad=True)
    q_values = model(grid)
    loss = q_values.sum()
    loss.backward()
    print(f"  Input grad: {grid.grad is not None}")
    print(f"  Grad norm: {grid.grad.norm().item():.4f}")
    print(f"  Grad finite: {torch.isfinite(grid.grad).all()}")
    assert torch.isfinite(grid.grad).all()
    print("  ✓ PASS")

    # Test 6: No NaN/Inf in forward pass
    print(f"\n{'='*70}")
    print("Test 6: Numerical stability")
    print("="*70)
    for i in range(10):
        grid = torch.randn(8, input_channels, 20, 20)
        q_values = model(grid)
        assert torch.isfinite(q_values).all(), f"Non-finite Q-values at iteration {i}"
    print(f"  Tested 10 random forward passes")
    print(f"  All outputs finite: ✓")
    print("  ✓ PASS")

    # Test 7: Parameter count comparison
    print(f"\n{'='*70}")
    print("Test 7: Parameter breakdown")
    print("="*70)
    for name, module in [
        ("Spatial Encoder", model.spatial_encoder),
        ("Spatial Softmax", model.spatial_softmax),
        ("Q-Head", model.q_head)
    ]:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}: {num_params:8,} params")
    print(f"  {'Total':20s}: {model.get_num_parameters():8,} params")
    print("  ✓ PASS")

    print(f"\n{'='*70}")
    print("✓ ALL TESTS PASSED!")
    print("="*70)
