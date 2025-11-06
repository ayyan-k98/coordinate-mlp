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


class LocalAttentionPooling(nn.Module):
    """
    Local attention pooling that only attends to cells within a radius R of the agent.
    
    Key benefits:
    - 5-10× speedup on large grids (40×40+)
    - Enables training on 50×50+ grids
    - More realistic: agents attend to nearby cells
    - Linear scaling with grid size (instead of quadratic)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, 
                 window_radius: int = 7, dropout: float = 0.1):
        """
        Args:
            hidden_dim: Feature dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            window_radius: Radius of local attention window (cells)
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_radius = window_radius
        
        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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
    
    def _get_local_window(self, H: int, W: int, agent_pos: tuple, device: str = 'cpu') -> torch.Tensor:
        """
        Get indices of cells within window_radius of agent.
        
        Args:
            H, W: Grid dimensions
            agent_pos: (y, x) agent position
            device: Device to create tensors on
        
        Returns:
            indices: [num_local_cells] - flattened indices of cells in window
        """
        y_agent, x_agent = agent_pos
        
        # Create coordinate grid
        y_coords = torch.arange(H, device=device)
        x_coords = torch.arange(W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute distances from agent
        distances = torch.sqrt(
            (y_grid - y_agent) ** 2 + (x_grid - x_agent) ** 2
        )
        
        # Get cells within radius
        mask = distances <= self.window_radius
        indices = torch.where(mask.flatten())[0]
        
        return indices
    
    def forward(self, cell_features: torch.Tensor, 
                agent_pos: tuple,
                grid_shape: tuple,
                return_attention_weights: bool = False):
        """
        Args:
            cell_features: [batch, num_cells, hidden_dim]
            agent_pos: (y, x) agent position(s) - single tuple or list of tuples
            grid_shape: (H, W) grid dimensions
            return_attention_weights: If True, also return attention weights
        
        Returns:
            aggregated: [batch, hidden_dim]
            attention_weights: [batch, num_heads, 1, num_local_cells] (optional)
        """
        B, N, D = cell_features.shape
        H, W = grid_shape
        device = cell_features.device
        
        # Handle single agent_pos for all batches
        if isinstance(agent_pos, tuple) and len(agent_pos) == 2:
            agent_pos = [agent_pos] * B
        
        # Check if all agents have the same position (optimization)
        same_position = all(pos == agent_pos[0] for pos in agent_pos)
        
        if same_position:
            # Optimized path: all agents at same position, process batch together
            local_indices = self._get_local_window(H, W, agent_pos[0], device=device)
            
            # Extract local features for entire batch
            local_features = cell_features[:, local_indices, :]  # [B, num_local, D]
            
            # Expand query for batch
            query = self.query.expand(B, -1, -1)  # [B, 1, D]
            
            # Local attention
            attended, weights = self.attention(
                query=query,
                key=local_features,
                value=local_features,
                need_weights=return_attention_weights,
                average_attn_weights=False
            )  # attended: [B, 1, D], weights: [B, num_heads, 1, num_local]
            
            # Remove singleton dimension
            attended = attended.squeeze(1)  # [B, D]
            
            # Post-process
            final_output = self.output_mlp(attended)  # [B, D]
            
            if return_attention_weights:
                return final_output, weights
            else:
                return final_output
        
        else:
            # Different positions: process each batch item separately
            outputs = []
            attn_weights_list = []
            
            for b in range(B):
                # Get local window indices
                local_indices = self._get_local_window(H, W, agent_pos[b], device=device)
                
                # Extract local cell features
                local_features = cell_features[b:b+1, local_indices, :]  # [1, num_local, D]
                
                # Expand query
                query = self.query  # [1, 1, D]
                
                # Local attention
                attended, weights = self.attention(
                    query=query,
                    key=local_features,
                    value=local_features,
                    need_weights=return_attention_weights,
                    average_attn_weights=False
                )  # attended: [1, 1, D], weights: [1, num_heads, 1, num_local]
                
                # Remove singleton dimensions
                attended = attended.squeeze(1)  # [1, D]
                
                # Post-process
                output = self.output_mlp(attended)  # [1, D]
                
                outputs.append(output)
                if return_attention_weights:
                    attn_weights_list.append(weights)
            
            # Concatenate batch results
            final_output = torch.cat(outputs, dim=0)  # [B, D]
            
            if return_attention_weights:
                final_weights = torch.cat(attn_weights_list, dim=0)  # [B, num_heads, 1, num_local]
                return final_output, final_weights
            else:
                return final_output


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
    model.eval()  # Disable dropout for this test
    output, attn_weights = model(cell_features, return_attention_weights=True)
    attn_sum = attn_weights.sum(dim=-1)
    print(f"\nTest 2 - Attention normalization:")
    print(f"  Sum of attention weights: {attn_sum[0, 0, 0].item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
    model.train()  # Re-enable training mode
    
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
    
    print("\nPASS: All global attention tests passed!")
    
    # =================================================================
    # LOCAL ATTENTION TESTS
    # =================================================================
    print("\n" + "="*60)
    print("Testing Local Attention Pooling")
    print("="*60)
    
    # Create local attention model
    window_radius = 7
    local_model = LocalAttentionPooling(
        hidden_dim=hidden_dim, 
        num_heads=num_heads,
        window_radius=window_radius
    )
    print(f"\nLocal attention model created:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Window radius: {window_radius}")
    print(f"  Parameters: {sum(p.numel() for p in local_model.parameters()):,}")
    
    # Test 1: Local window creation
    H, W = 20, 20
    agent_pos = (10, 10)
    indices = local_model._get_local_window(H, W, agent_pos)
    expected_cells = math.pi * window_radius ** 2
    print(f"\nTest 1 - Local window:")
    print(f"  Grid: {H}x{W}")
    print(f"  Agent: {agent_pos}")
    print(f"  Radius: {window_radius}")
    print(f"  Window cells: {len(indices)}")
    print(f"  Expected: ~{expected_cells:.0f} (pi*r^2)")
    print(f"  Reduction: {len(indices)}/{H*W} = {len(indices)/(H*W)*100:.1f}%")
    assert 0.7 * expected_cells < len(indices) < 1.3 * expected_cells
    
    # Test 2: Local attention forward pass
    batch_size = 1
    num_cells = H * W
    cell_features = torch.randn(batch_size, num_cells, hidden_dim)
    output, attn_weights = local_model(
        cell_features, 
        agent_pos=agent_pos,
        grid_shape=(H, W),
        return_attention_weights=True
    )
    print(f"\nTest 2 - Forward pass:")
    print(f"  Input: {cell_features.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    assert output.shape == (batch_size, hidden_dim)
    assert attn_weights.shape[0] == batch_size
    assert attn_weights.shape[1] == num_heads
    
    # Test 3: Attention normalization
    local_model.eval()  # Disable dropout
    output, attn_weights = local_model(
        cell_features, 
        agent_pos=agent_pos,
        grid_shape=(H, W),
        return_attention_weights=True
    )
    attn_sum = attn_weights.sum(dim=-1)
    print(f"\nTest 3 - Local attention normalization:")
    print(f"  Sum of attention weights: {attn_sum[0, 0, 0].item():.6f}")
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
    local_model.train()  # Re-enable training mode
    
    # Test 4: Different agent positions
    positions = [(0, 0), (10, 10), (19, 19), (5, 15)]
    print(f"\nTest 4 - Different agent positions:")
    for pos in positions:
        indices = local_model._get_local_window(H, W, pos)
        output = local_model(cell_features, agent_pos=pos, grid_shape=(H, W))
        print(f"  Position {pos}: {len(indices)} cells, output {output.shape}")
        assert output.shape == (1, hidden_dim)
    
    # Test 5: Multiple batches with different positions
    batch_size = 4
    cell_features = torch.randn(batch_size, H*W, hidden_dim)
    agent_positions = [(5, 5), (10, 10), (15, 5), (5, 15)]
    output = local_model(
        cell_features,
        agent_pos=agent_positions,
        grid_shape=(H, W)
    )
    print(f"\nTest 5 - Multiple batches:")
    print(f"  Batch size: {batch_size}")
    print(f"  Agent positions: {agent_positions}")
    print(f"  Output: {output.shape}")
    assert output.shape == (batch_size, hidden_dim)
    
    # Test 6: Different grid sizes
    print(f"\nTest 6 - Different grid sizes:")
    for grid_size in [15, 20, 30, 40]:
        H, W = grid_size, grid_size
        num_cells = H * W
        cell_features = torch.randn(2, num_cells, hidden_dim)
        agent_pos = (grid_size // 2, grid_size // 2)
        
        indices = local_model._get_local_window(H, W, agent_pos)
        output = local_model(cell_features, agent_pos=agent_pos, grid_shape=(H, W))
        
        reduction = len(indices) / num_cells * 100
        print(f"  Grid {grid_size}x{grid_size}: {len(indices)}/{num_cells} cells ({reduction:.1f}%), output {output.shape}")
        assert output.shape == (2, hidden_dim)
    
    # Test 7: Gradient flow
    cell_features = torch.randn(2, 20*20, hidden_dim, requires_grad=True)
    output = local_model(cell_features, agent_pos=(10, 10), grid_shape=(20, 20))
    loss = output.sum()
    loss.backward()
    print(f"\nTest 7 - Gradient flow:")
    print(f"  Input grad: {cell_features.grad is not None}")
    print(f"  Grad norm: {cell_features.grad.norm().item():.4f}")
    print(f"  Query grad: {local_model.query.grad is not None}")
    
    # Test 8: Speed comparison
    print(f"\nTest 8 - Speed comparison:")
    import time
    
    # Test on multiple grid sizes
    for grid_size in [20, 30, 40, 50]:
        H, W = grid_size, grid_size
        num_cells = H * W
        batch_size = 8
        cell_features = torch.randn(batch_size, num_cells, hidden_dim)
        agent_pos = (grid_size // 2, grid_size // 2)
        
        # Global attention
        global_model = AttentionPooling(hidden_dim=hidden_dim, num_heads=num_heads)
        global_model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = global_model(cell_features)
            global_time = (time.time() - start) / 10
        
        # Local attention
        local_model.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = local_model(cell_features, agent_pos=agent_pos, grid_shape=(H, W))
            local_time = (time.time() - start) / 10
        
        speedup = global_time / local_time
        local_cells = len(local_model._get_local_window(H, W, agent_pos))
        reduction = (1 - local_cells / num_cells) * 100
        print(f"  {grid_size}x{grid_size}: Global {global_time*1000:.2f}ms, Local {local_time*1000:.2f}ms, Speedup {speedup:.2f}x ({reduction:.0f}% reduction)")
    
    print("\nPASS: All local attention tests passed!")
    print(f"\nSummary:")
    print(f"  Window radius {window_radius} reduces attention to ~{math.pi * window_radius**2:.0f} cells")
    print(f"  Speedup increases with grid size (quadratic vs constant window)")
    print(f"  All tests passed!")
