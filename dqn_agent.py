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

from ..models.coordinate_network import CoordinateCoverageNetwork
from .replay_buffer import ReplayMemory


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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            input_channels: Number of grid channels
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
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = CoordinateCoverageNetwork(
            input_channels=input_channels,
            num_freq_bands=num_freq_bands,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(self.device)
        
        self.target_net = CoordinateCoverageNetwork(
            input_channels=input_channels,
            num_freq_bands=num_freq_bands,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(self.device)
        
        # Copy policy network weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Replay buffer
        self.memory = ReplayMemory(capacity=memory_capacity)
        
        # Training stats
        self.training_steps = 0
        self.episodes = 0
    
    def select_action(
        self,
        state: np.ndarray,
        epsilon: Optional[float] = None,
        valid_actions: Optional[np.ndarray] = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state [C, H, W]
            epsilon: Exploration rate (use self.epsilon if None)
            valid_actions: Boolean mask of valid actions [num_actions]
        
        Returns:
            action: Integer in [0, num_actions-1]
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy
        if random.random() < epsilon:
            # Random valid action
            if valid_actions is not None:
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return random.randint(0, self.num_actions - 1)
        
        # Greedy action
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Forward pass
            q_values = self.policy_net(state_tensor).squeeze(0)  # [num_actions]
            
            # Mask invalid actions
            if valid_actions is not None:
                q_values = q_values.cpu().numpy()
                q_values[~valid_actions] = -np.inf
                action = np.argmax(q_values)
            else:
                action = q_values.argmax().item()
        
        return action
    
    def update(self) -> Optional[dict]:
        """
        Perform one step of DQN training using Double DQN.
        
        Returns:
            info: Dictionary with training metrics (loss, q_values, etc.)
                  Returns None if not enough samples in memory
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
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
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update training steps
        self.training_steps += 1
        
        # Return training metrics
        return {
            'loss': loss.item(),
            'q_mean': q_values.mean().item(),
            'q_max': q_values.max().item(),
            'target_mean': targets.mean().item(),
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
        """
        Save agent state to file.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
        }, path)
    
    def load(self, path: str):
        """
        Load agent state from file.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']


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
