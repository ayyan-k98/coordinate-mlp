"""
Replay Buffer for Experience Replay

Stores and samples transitions for DQN training.
"""

import random
from collections import deque, namedtuple
from typing import List
import torch
import numpy as np


# Transition tuple for storing experience
Transition = namedtuple('Transition', 
                       ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    """
    Fixed-size buffer to store experience tuples.
    
    Implements uniform random sampling for DQN training.
    """
    
    def __init__(self, capacity: int = 50000):
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Save a transition.
        
        Args:
            state: Current state [C, H, W]
            action: Action taken
            reward: Reward received
            next_state: Next state [C, H, W]
            done: Whether episode ended
        """
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        """
        Randomly sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            states: [batch, C, H, W]
            actions: [batch, 1]
            rewards: [batch, 1]
            next_states: [batch, C, H, W]
            dones: [batch, 1]
        """
        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        
        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of internal memory."""
        return len(self.memory)
    
    def clear(self):
        """Clear all stored transitions."""
        self.memory.clear()


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (optional advanced feature).
    
    Samples transitions with probability proportional to TD error.
    Reference: Schaul et al. "Prioritized Experience Replay" (2016)
    """
    
    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001  # Anneal beta to 1.0 over time
        
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Save transition with maximum priority."""
        self.memory.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> tuple:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        
        # Get transitions
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        weights = torch.FloatTensor(weights).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: Absolute TD errors for each transition
        """
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.memory)


if __name__ == "__main__":
    # Unit test
    print("="*60)
    print("Testing Replay Memory")
    print("="*60)
    
    # Test 1: Basic functionality
    memory = ReplayMemory(capacity=1000)
    print(f"\nTest 1 - Basic operations:")
    print(f"  Initial size: {len(memory)}")
    
    # Add some transitions
    for i in range(10):
        state = np.random.randn(5, 20, 20)
        action = np.random.randint(0, 9)
        reward = np.random.randn()
        next_state = np.random.randn(5, 20, 20)
        done = i == 9
        memory.push(state, action, reward, next_state, done)
    
    print(f"  Size after adding 10: {len(memory)}")
    
    # Test 2: Sampling
    if len(memory) >= 4:
        batch = memory.sample(4)
        states, actions, rewards, next_states, dones = batch
        print(f"\nTest 2 - Sampling:")
        print(f"  States: {states.shape}")
        print(f"  Actions: {actions.shape}")
        print(f"  Rewards: {rewards.shape}")
        print(f"  Next states: {next_states.shape}")
        print(f"  Dones: {dones.shape}")
    
    # Test 3: Capacity limit
    memory = ReplayMemory(capacity=100)
    for i in range(150):
        state = np.random.randn(5, 20, 20)
        memory.push(state, 0, 0.0, state, False)
    
    print(f"\nTest 3 - Capacity limit:")
    print(f"  Added 150 transitions")
    print(f"  Current size: {len(memory)} (max 100)")
    assert len(memory) == 100
    
    # Test 4: Prioritized replay
    print(f"\nTest 4 - Prioritized replay:")
    per_memory = PrioritizedReplayMemory(capacity=1000)
    
    for i in range(20):
        state = np.random.randn(5, 20, 20)
        per_memory.push(state, 0, 0.0, state, False)
    
    print(f"  Size: {len(per_memory)}")
    print(f"  Beta: {per_memory.beta:.3f}")
    
    # Sample and update priorities
    batch = per_memory.sample(5)
    states, actions, rewards, next_states, dones, indices, weights = batch
    print(f"  Sampled indices: {indices}")
    print(f"  IS weights: {weights.squeeze().tolist()}")
    
    # Update priorities
    td_errors = np.random.rand(5)
    per_memory.update_priorities(indices, td_errors)
    print(f"  Updated priorities for indices {indices}")
    
    print("\n✓ All tests passed!")
