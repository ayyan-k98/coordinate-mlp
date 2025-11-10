"""
Dueling Q-Network Head

Maps aggregated features to Q-values for each action using dueling architecture.
"""

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.
    
    Splits into:
    - Value stream: V(s) - how good is this state?
    - Advantage stream: A(s, a) - how much better is action a?
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
    
    Reference:
    Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
    """
    
    def __init__(self, hidden_dim: int, num_actions: int, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Input feature dimension
            num_actions: Number of actions (typically 9 for grid movement)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Shared layers - stable ordering: Linear → LayerNorm → ReLU → Dropout
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalize BEFORE activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Value stream (scalar output) - stable ordering
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Normalize BEFORE activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # V(s)
        )

        # Advantage stream (per-action output) - stable ordering
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),  # Normalize BEFORE activation
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions)  # A(s, a)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, hidden_dim] - Aggregated features
        
        Returns:
            q_values: [batch, num_actions]
        """
        # Shared processing
        shared = self.shared(x)  # [batch, hidden_dim]
        
        # Value stream
        value = self.value_stream(shared)  # [batch, 1]
        
        # Advantage stream
        advantage = self.advantage_stream(shared)  # [batch, num_actions]
        
        # Combine using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        advantage_mean = advantage.mean(dim=-1, keepdim=True)  # [batch, 1]
        q_values = value + (advantage - advantage_mean)  # [batch, num_actions]
        
        return q_values
    
    def get_value_and_advantage(self, x: torch.Tensor):
        """
        Get separate value and advantage estimates (for analysis).
        
        Args:
            x: [batch, hidden_dim]
        
        Returns:
            value: [batch, 1]
            advantage: [batch, num_actions]
        """
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value, advantage


if __name__ == "__main__":
    # Unit test
    print("="*60)
    print("Testing Dueling Q-Network")
    print("="*60)
    
    # Configuration
    hidden_dim = 256
    num_actions = 9
    
    # Create model
    model = DuelingQNetwork(hidden_dim=hidden_dim, num_actions=num_actions)
    print(f"\nModel created:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num actions: {num_actions}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: Single batch
    batch_size = 1
    x = torch.randn(batch_size, hidden_dim)
    q_values = model(x)
    print(f"\nTest 1 - Single batch:")
    print(f"  Input: {x.shape}")
    print(f"  Q-values: {q_values.shape}")
    print(f"  Q-values: {q_values[0].tolist()}")
    assert q_values.shape == (batch_size, num_actions)
    
    # Test 2: Multiple batches
    batch_size = 8
    x = torch.randn(batch_size, hidden_dim)
    q_values = model(x)
    print(f"\nTest 2 - Multiple batches:")
    print(f"  Batch size: {batch_size}")
    print(f"  Q-values: {q_values.shape}")
    assert q_values.shape == (batch_size, num_actions)
    
    # Test 3: Action selection
    x = torch.randn(4, hidden_dim)
    q_values = model(x)
    best_actions = q_values.argmax(dim=-1)
    print(f"\nTest 3 - Action selection:")
    print(f"  Batch size: 4")
    print(f"  Best actions: {best_actions.tolist()}")
    print(f"  Max Q-values: {q_values.max(dim=-1).values.tolist()}")
    
    # Test 4: Dueling decomposition
    x = torch.randn(2, hidden_dim)
    q_values = model(x)
    value, advantage = model.get_value_and_advantage(x)
    print(f"\nTest 4 - Dueling decomposition:")
    print(f"  Value: {value.shape} -> {value[0, 0].item():.4f}")
    print(f"  Advantage: {advantage.shape}")
    print(f"  Advantage mean: {advantage.mean(dim=-1)[0].item():.6f}")
    
    # Verify dueling equation
    advantage_centered = advantage - advantage.mean(dim=-1, keepdim=True)
    reconstructed = value + advantage_centered
    print(f"  Reconstruction error: {(reconstructed - q_values).abs().max().item():.8f}")
    assert torch.allclose(reconstructed, q_values, atol=1e-6)
    
    # Test 5: Gradient flow
    x = torch.randn(2, hidden_dim, requires_grad=True)
    q_values = model(x)
    loss = q_values.sum()
    loss.backward()
    print(f"\nTest 5 - Gradient flow:")
    print(f"  Input grad: {x.grad is not None}")
    print(f"  Grad norm: {x.grad.norm().item():.4f}")
    
    # Test 6: Value vs Advantage
    # Create scenarios to test if value and advantage decompose correctly
    good_state = torch.randn(1, hidden_dim) + 2.0  # Bias towards positive
    bad_state = torch.randn(1, hidden_dim) - 2.0   # Bias towards negative
    
    q_good = model(good_state)
    q_bad = model(bad_state)
    v_good, a_good = model.get_value_and_advantage(good_state)
    v_bad, a_bad = model.get_value_and_advantage(bad_state)
    
    print(f"\nTest 6 - Value estimation:")
    print(f"  Good state - Value: {v_good[0, 0].item():.4f}, "
          f"Q-range: [{q_good.min().item():.4f}, {q_good.max().item():.4f}]")
    print(f"  Bad state  - Value: {v_bad[0, 0].item():.4f}, "
          f"Q-range: [{q_bad.min().item():.4f}, {q_bad.max().item():.4f}]")
    
    print("\n✓ All tests passed!")
