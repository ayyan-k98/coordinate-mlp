"""
Coordinate-Based DQN Agent

Wraps the coordinate network with RL training logic including experience replay,
target networks, and epsilon-greedy exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
import random

from coordinate_network import CoordinateCoverageNetwork
from replay_buffer import ReplayMemory


class CoordinateDQNAgent:
    """
    DQN agent using coordinate-based network.
    
    Implements:
    - Double DQN for stable training
    - Target network with soft updates
    - Epsilon-greedy exploration
    - Experience replay
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        num_actions: int = 9,
        hidden_dim: int = 256,
        num_freq_bands: int = 6,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        memory_capacity: int = 50000,
        target_update_tau: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        sensor_range: Optional[float] = None,
        use_pomdp: bool = False,
        use_local_attention: bool = False,
        attention_window_radius: int = 7,
        use_mixed_precision: bool = True
    ):
        """
        Args:
            input_channels: Number of grid channels (5 for full obs, 6 for POMDP with visibility mask)
            num_actions: Number of discrete actions
            hidden_dim: Hidden dimension size
            num_freq_bands: Number of Fourier frequency bands
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exponential decay rate for epsilon
            batch_size: Batch size for training
            memory_capacity: Replay buffer capacity
            target_update_tau: Soft update coefficient (Polyak averaging)
            device: Device to run on ('cuda' or 'cpu')
            sensor_range: Sensor range for POMDP (None = full observability)
            use_pomdp: Whether to use POMDP with visibility masking
            use_local_attention: Whether to use local attention (faster for large grids)
            attention_window_radius: Radius for local attention window
            use_mixed_precision: Whether to use mixed precision training (AMP)
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau

        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=2.**10,  # Start with smaller scale (default is 2^16)
                growth_interval=2000  # Grow scale less frequently
            )
            print("  Mixed Precision (AMP): Enabled")
        else:
            self.scaler = None
        
        # POMDP settings
        self.sensor_range = sensor_range
        self.use_pomdp = use_pomdp
        
        # Local attention settings
        self.use_local_attention = use_local_attention
        self.attention_window_radius = attention_window_radius
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = CoordinateCoverageNetwork(
            input_channels=input_channels,
            num_freq_bands=num_freq_bands,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            use_local_attention=use_local_attention,
            attention_window_radius=attention_window_radius
        ).to(self.device)
        
        self.target_net = CoordinateCoverageNetwork(
            input_channels=input_channels,
            num_freq_bands=num_freq_bands,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            use_local_attention=use_local_attention,
            attention_window_radius=attention_window_radius
        ).to(self.device)
        
        # Copy policy network weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Replay buffers - separate buffer for each grid size
        # This allows multi-scale training without shape mismatch errors
        self.memories = {
            15: ReplayMemory(capacity=memory_capacity // 4),
            20: ReplayMemory(capacity=memory_capacity // 4),
            25: ReplayMemory(capacity=memory_capacity // 4),
            30: ReplayMemory(capacity=memory_capacity // 4),
        }
        # For backward compatibility with code that accesses self.memory
        self.memory = self.memories[20]  # Default to 20x20
        # Training stats
        self.training_steps = 0
        self.episodes = 0
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: Optional[float] = None,
        valid_actions: Optional[np.ndarray] = None,
        agent_pos: Optional[Tuple[int, int]] = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state [C, H, W] (without visibility mask)
            epsilon: Exploration rate (use self.epsilon if None)
            valid_actions: Boolean mask of valid actions [num_actions]
            agent_pos: Agent position (y, x) for POMDP visibility mask
        
        Returns:
            action: Integer in [0, num_actions-1]
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Get valid action indices
        if valid_actions is not None:
            valid_indices = np.where(valid_actions)[0]
            if len(valid_indices) == 0:
                raise ValueError("No valid actions available!")
        else:
            valid_indices = np.arange(self.num_actions)
        
        # Epsilon-greedy
        if random.random() < epsilon:
            # Random valid action
            return np.random.choice(valid_indices)
        
        # Greedy action
        with torch.no_grad():
            # Add visibility mask if POMDP
            if self.use_pomdp and agent_pos is not None:
                state_with_mask = self._add_visibility_mask(state, agent_pos)
            else:
                state_with_mask = state
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state_with_mask).unsqueeze(0).to(self.device)
            
            # Forward pass
            if self.use_local_attention:
                if agent_pos is None:
                    raise ValueError("agent_pos is required when use_local_attention=True")
                q_values = self.policy_net(state_tensor, agent_pos=agent_pos).squeeze(0).cpu().numpy()
            else:
                q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()  # [num_actions]
            
            # Mask invalid actions
            masked_q = np.full(self.num_actions, -np.inf)
            masked_q[valid_indices] = q_values[valid_indices]
            action = np.argmax(masked_q)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done, grid_size: int = 20):
        """
        Store transition in grid-specific replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            grid_size: Grid size (15, 20, 25, or 30)
        """
        memory = self.memories.get(grid_size, self.memories[20])
        memory.push(state, action, reward, next_state, done)
    def _add_visibility_mask(self, state: np.ndarray, agent_pos: Tuple[int, int]) -> np.ndarray:
        """
        Add visibility mask as additional channel for POMDP.
        
        Args:
            state: [C, H, W] state without visibility mask
            agent_pos: (y, x) agent position
        
        Returns:
            state_with_mask: [C+1, H, W] state with visibility mask
        """
        C, H, W = state.shape
        
        # Create visibility mask
        visibility = self._create_visibility_mask(H, W, agent_pos)
        
        # Concatenate: [C+1, H, W]
        state_with_mask = np.concatenate([state, visibility[np.newaxis, :, :]], axis=0)
        
        return state_with_mask
    
    def _create_visibility_mask(self, H: int, W: int, agent_pos: Tuple[int, int]) -> np.ndarray:
        """
        Create binary visibility mask based on sensor range.
        
        Args:
            H: Grid height
            W: Grid width
            agent_pos: (y, x) agent position
        
        Returns:
            mask: [H, W] binary mask (1=visible, 0=not visible)
        """
        if self.sensor_range is None:
            # Full observability
            return np.ones((H, W), dtype=np.float32)
        
        y, x = agent_pos
        
        # Create coordinate grids
        y_coords = np.arange(H)[:, np.newaxis]
        x_coords = np.arange(W)[np.newaxis, :]
        
        # Compute distances from agent
        distances = np.sqrt((y_coords - y)**2 + (x_coords - x)**2)
        
        # Binary mask based on sensor range
        mask = (distances <= self.sensor_range).astype(np.float32)
        
        return mask
        
    def update(self, grid_size: int = 20) -> Optional[dict]:
        """
        Perform one step of DQN training using Double DQN.

        Args:
            grid_size: Grid size to sample from (15, 20, 25, or 30)

        Returns:
            info: Dictionary with training metrics (loss, q_values, etc.)
                Returns None if not enough samples in memory
        """
        # Get the appropriate memory buffer for this grid size
        memory = self.memories.get(grid_size, self.memories[20])
        
        if len(memory) < self.batch_size:
            return None

        # Sample batch from grid-specific buffer
        states, actions, rewards, next_states, dones = memory.sample(self.batch_size)
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        if self.use_mixed_precision:
            # Compute targets in FP32 FIRST (prevents numerical instability)
            with torch.no_grad():
                # Forward passes can use autocast
                with torch.cuda.amp.autocast():
                    next_q_policy = self.policy_net(next_states)
                    next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                    
                    next_q_target = self.target_net(next_states)
                    next_q_values = next_q_target.gather(1, next_actions)
                
                # Compute Bellman targets in FP32
                targets = rewards + self.gamma * next_q_values.float() * (1 - dones)
                #targets = torch.clamp(targets, min=-100.0, max=100.0)

            # Now compute Q-values and loss with autocast
            with torch.cuda.amp.autocast():
                q_values = self.policy_net(states)
                q_values = q_values.gather(1, actions)
                
                # Loss computation
                loss = nn.functional.smooth_l1_loss(q_values, targets)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(self.optimizer)
            
            # ========== COMPREHENSIVE GRADIENT HEALTH CHECKS ==========
            has_nan = False
            has_inf = False
            bad_layers = []
            
            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan = True
                        bad_layers.append(f"NaN in {name}")
                    if torch.isinf(param.grad).any():
                        has_inf = True
                        bad_layers.append(f"Inf in {name}")
            
            if has_nan or has_inf:
                print(f"\n{'='*70}")
                print(f"⚠️  GRADIENT EXPLOSION DETECTED ⚠️")
                print(f"{'='*70}")
                print(f"Training Step: {self.training_steps}")
                
                # Show which layers exploded
                print(f"\nExploded Layers ({len(bad_layers)} total):")
                for layer_info in bad_layers[:5]:  # First 5
                    print(f"  • {layer_info}")
                if len(bad_layers) > 5:
                    print(f"  ... and {len(bad_layers)-5} more")
                
                # Network statistics
                print(f"\nNetwork Statistics:")
                print(f"  Q-values: min={q_values.min().item():.2f}, max={q_values.max().item():.2f}, mean={q_values.mean().item():.2f}, std={q_values.std().item():.2f}")
                print(f"  Targets:  min={targets.min().item():.2f}, max={targets.max().item():.2f}, mean={targets.mean().item():.2f}, std={targets.std().item():.2f}")
                print(f"  Rewards:  min={rewards.min().item():.2f}, max={rewards.max().item():.2f}, mean={rewards.mean().item():.2f}")
                print(f"  Loss:     {loss.item():.4f}")
                
                # Test for forward pass issues
                print(f"\nForward Pass Check:")
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        test_out = self.policy_net(states[:1])
                    if torch.isnan(test_out).any():
                        print(f"  ⚠️  NaN in network output (forward pass corruption!)")
                    elif torch.isinf(test_out).any():
                        print(f"  ⚠️  Inf in network output (activation overflow!)")
                    else:
                        print(f"  ✓  Forward pass output is finite")
                        print(f"     Output range: [{test_out.min().item():.2f}, {test_out.max().item():.2f}]")
                
                print(f"{'='*70}\n")
                
                for param in self.policy_net.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                # Reset and skip this update
                self.scaler.update()
                return None
            
            # Gradient clipping (only if healthy)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)

            # Optimizer step with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Standard FP32 training path
            # Current Q-values
            q_values = self.policy_net(states)  # [batch, num_actions]
            q_values = q_values.gather(1, actions)  # [batch, 1]

            # Target Q-values (Double DQN)
            with torch.no_grad():
                # Select best actions using policy network
                next_q_policy = self.policy_net(next_states)
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)

                # Evaluate using target network
                next_q_target = self.target_net(next_states)
                next_q_values = next_q_target.gather(1, next_actions)

                # Bellman target
                targets = rewards + self.gamma * next_q_values * (1 - dones)

            # Compute loss (Huber loss for robustness)
            loss = nn.functional.smooth_l1_loss(q_values, targets)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

            self.optimizer.step()

        # Update training steps
        self.training_steps += 1

        # Compute TD error
        td_error = (q_values - targets).abs()

        # Return enhanced training metrics
        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_max': q_values.max().item(),
            'q_min': q_values.min().item(),
            'q_std': q_values.std().item(),
            'target_mean': targets.mean().item(),
            'target_std': targets.std().item(),
            'td_error_mean': td_error.mean().item(),
            'td_error_max': td_error.max().item(),
            'grad_norm': grad_norm.item(),
        }
    
    def update_target_network(self):
        """
        Soft update of target network using Polyak averaging.
        
        θ_target = τ * θ_policy + (1 - τ) * θ_target
        """
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.target_update_tau * policy_param.data + 
                (1 - self.target_update_tau) * target_param.data
            )
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """Save agent state to file."""
        # Save all replay buffers
        memory_states = {size: list(mem.memory) for size, mem in self.memories.items()}
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'memories': memory_states,  # Save all buffers
        }, path)
    
    def load(self, path: str):
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint.get('training_steps', 0)
        
        # Load all replay buffers if available
        if 'memories' in checkpoint:
            for size, memory_list in checkpoint['memories'].items():
                self.memories[size].memory = deque(memory_list, maxlen=self.memories[size].capacity)

if __name__ == "__main__":
    # Unit test
    print("="*70)
    print("Testing Coordinate DQN Agent")
    print("="*70)
    
    # Create agent
    agent = CoordinateDQNAgent(
        input_channels=5,
        num_actions=9,
        hidden_dim=256,
        device='cpu'
    )
    
    print(f"\nAgent created:")
    print(f"  Device: {agent.device}")
    print(f"  Epsilon: {agent.epsilon}")
    print(f"  Memory capacity: {agent.memory.capacity}")
    print(f"  Parameters: {agent.policy_net.get_num_parameters():,}")
    
    # Test 1: Action selection
    state = np.random.randn(5, 20, 20)
    action = agent.select_action(state, epsilon=0.0)
    print(f"\nTest 1 - Greedy action selection:")
    print(f"  State shape: {state.shape}")
    print(f"  Selected action: {action}")
    assert 0 <= action < 9
    
    # Test 2: Epsilon-greedy
    actions = [agent.select_action(state, epsilon=1.0) for _ in range(100)]
    unique_actions = len(set(actions))
    print(f"\nTest 2 - Random action selection:")
    print(f"  100 random actions")
    print(f"  Unique actions: {unique_actions}/9")
    assert unique_actions >= 5  # Should explore
    
    # Test 3: Valid actions mask
    valid_actions = np.array([True, False, False, True, True, False, False, False, True])
    action = agent.select_action(state, epsilon=0.0, valid_actions=valid_actions)
    print(f"\nTest 3 - Valid actions mask:")
    print(f"  Valid actions: {np.where(valid_actions)[0].tolist()}")
    print(f"  Selected action: {action}")
    assert valid_actions[action]
    
    # Test 4: Store transitions
    for i in range(50):
        state = np.random.randn(5, 20, 20)
        action = random.randint(0, 8)
        reward = random.random()
        next_state = np.random.randn(5, 20, 20)
        done = i % 10 == 9
        agent.memory.push(state, action, reward, next_state, done)
    
    print(f"\nTest 4 - Experience replay:")
    print(f"  Memory size: {len(agent.memory)}")
    
    # Test 5: Training update
    info = agent.update()
    print(f"\nTest 5 - Training update:")
    if info:
        print(f"  Loss: {info['loss']:.4f}")
        print(f"  Q mean: {info['q_mean']:.4f}")
        print(f"  Q max: {info['q_max']:.4f}")
        print(f"  Target mean: {info['target_mean']:.4f}")
    
    # Test 6: Target network update
    old_params = list(agent.target_net.parameters())[0].clone()
    agent.update_target_network()
    new_params = list(agent.target_net.parameters())[0]
    param_diff = (old_params - new_params).abs().mean().item()
    print(f"\nTest 6 - Target network update:")
    print(f"  Parameter change: {param_diff:.6f}")
    assert param_diff > 0  # Parameters should change
    
    # Test 7: Epsilon decay
    old_epsilon = agent.epsilon
    agent.decay_epsilon()
    print(f"\nTest 7 - Epsilon decay:")
    print(f"  Old epsilon: {old_epsilon:.4f}")
    print(f"  New epsilon: {agent.epsilon:.4f}")
    assert agent.epsilon < old_epsilon
    
    # Test 8: Different grid sizes
    print(f"\nTest 8 - Multi-scale compatibility:")
    for size in [15, 20, 25, 30]:
        state = np.random.randn(5, size, size)
        action = agent.select_action(state, epsilon=0.0)
        print(f"  {size}×{size} grid: action={action} ✓")
    
    # Test 9: Save and load
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'agent.pt')
        
        # Save
        old_epsilon = agent.epsilon
        old_steps = agent.training_steps
        agent.save(save_path)
        print(f"\nTest 9 - Save and load:")
        print(f"  Saved to: {save_path}")
        
        # Modify agent
        agent.epsilon = 0.123
        agent.training_steps = 999
        
        # Load
        agent.load(save_path)
        print(f"  Loaded successfully")
        print(f"  Epsilon restored: {agent.epsilon}")
        print(f"  Steps restored: {agent.training_steps}")
        
        assert agent.epsilon == old_epsilon
        assert agent.training_steps == old_steps
    
    print("\n✓ All tests passed!")
    print("="*70)
