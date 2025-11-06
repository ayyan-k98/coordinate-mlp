"""
Attention-Based Aggregation

Learns which cells are important for decision-making using multi-head attention.
"""

import torch
import torch.nn as nn
import math


class AttentionPooling(nn.Module):
    """
    Multi-head attention pooling over cells.
    
    Key idea: Network learns to attend to important cells
    (e.g., frontiers, agent position, uncovered areas)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Learnable query (what to attend to)
        # This is what the network learns to look for in the cell features
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Expect [batch, seq, features]
        )
        
        # Post-attention processing
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, cell_features: torch.Tensor, 
                return_attention_weights: bool = False):
        """
        Args:
            cell_features: [batch, num_cells, hidden_dim]
            return_attention_weights: If True, also return attention weights
        
        Returns:
            aggregated: [batch, hidden_dim]
            attention_weights: [batch, num_heads, 1, num_cells] (optional)
        """
        B, N, D = cell_features.shape
        
        # Expand query for batch
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        
        # Attention: query attends to all cells
        # Q: [B, 1, D]  (what we're looking for)
        # K, V: [B, N, D]  (cells to attend to)
        attended, attention_weights = self.attention(
            query=query,
            key=cell_features,
            value=cell_features,
            need_weights=return_attention_weights,
            average_attn_weights=False  # Return per-head weights
        )  # attended: [B, 1, D], weights: [B, num_heads, 1, N]
        
        # Remove singleton sequence dimension
        attended = attended.squeeze(1)  # [B, D]
        
        # Post-process
        output = self.output_mlp(attended)  # [B, D]
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output


if __name__ == "__main__":
    # Unit test
    print("="*60)
    print("Testing Attention Pooling")
    print("="*60)
    
    # Configuration
    hidden_dim = 256
    num_heads = 4
    
    # Create model
    model = AttentionPooling(hidden_dim=hidden_dim, num_heads=num_heads)
    print(f"\nModel created:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {model.head_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: Single batch
    batch_size = 1
    num_cells = 400
    cell_features = torch.randn(batch_size, num_cells, hidden_dim)
    output, attn_weights = model(cell_features, return_attention_weights=True)
    print(f"\nTest 1 - Single batch:")
    print(f"  Input: {cell_features.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    assert output.shape == (batch_size, hidden_dim)
    assert attn_weights.shape == (batch_size, num_heads, 1, num_cells)
    
    # Test 2: Attention weights sum to 1
    attn_sum = attn_weights.sum(dim=-1)
    print(f"\nTest 2 - Attention normalization:")
    print(f"  Sum of attention weights: {attn_sum[0, 0, 0].item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
    
    # Test 3: Multiple batches
    batch_size = 8
    cell_features = torch.randn(batch_size, num_cells, hidden_dim)
    output = model(cell_features)
    print(f"\nTest 3 - Multiple batches:")
    print(f"  Batch size: {batch_size}")
    print(f"  Output: {output.shape}")
    assert output.shape == (batch_size, hidden_dim)
    
    # Test 4: Different grid sizes
    for grid_size in [15, 20, 25, 30, 40]:
        num_cells = grid_size * grid_size
        cell_features = torch.randn(4, num_cells, hidden_dim)
        output = model(cell_features)
        print(f"\nTest 4 - Grid size {grid_size}×{grid_size}:")
        print(f"  Num cells: {num_cells}")
        print(f"  Output: {output.shape}")
        assert output.shape == (4, hidden_dim)
    
    # Test 5: Gradient flow
    cell_features = torch.randn(2, 100, hidden_dim, requires_grad=True)
    output = model(cell_features)
    loss = output.sum()
    loss.backward()
    print(f"\nTest 5 - Gradient flow:")
    print(f"  Input grad: {cell_features.grad is not None}")
    print(f"  Grad norm: {cell_features.grad.norm().item():.4f}")
    print(f"  Query grad: {model.query.grad is not None}")
    print(f"  Query grad norm: {model.query.grad.norm().item():.4f}")
    
    # Test 6: Attention visualization (simple check)
    batch_size = 1
    num_cells = 100
    cell_features = torch.randn(batch_size, num_cells, hidden_dim)
    output, attn_weights = model(cell_features, return_attention_weights=True)
    
    # Average over heads
    avg_attn = attn_weights.mean(dim=1).squeeze()  # [num_cells]
    top_5_indices = torch.topk(avg_attn, k=5).indices
    print(f"\nTest 6 - Attention pattern:")
    print(f"  Top 5 attended cells: {top_5_indices.tolist()}")
    print(f"  Their weights: {avg_attn[top_5_indices].tolist()}")
    
    print("\n✓ All tests passed!")
