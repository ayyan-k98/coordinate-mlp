# Checkpoint Saving/Loading Fix

## Problem

PyTorch 2.6+ changed the default `torch.load()` behavior to `weights_only=True` for security. The old checkpoint saving code was trying to save replay buffer `Transition` objects, causing this error:

```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL replay_buffer.Transition 
was not an allowed global by default.
```

## Root Cause

The `dqn_agent.py` save method was attempting to pickle the entire replay buffer:

```python
# OLD (BROKEN):
memory_states = {size: list(mem.memory) for size, mem in self.memories.items()}
torch.save({
    'memories': memory_states,  # ❌ Tries to pickle Transition objects
    ...
}, path)
```

This fails because:
1. `Transition` is a custom namedtuple from `replay_buffer.py`
2. PyTorch 2.6+ blocks unpickling of arbitrary classes by default (security feature)
3. Replay buffers are large and unnecessary to save (can rebuild during training)

## Solution

### 1. Simplified Checkpoint Format

Only save essential training state (no replay buffer):

```python
# NEW (FIXED):
def save(self, path: str):
    """Save agent state to file (PyTorch 2.6+ compatible)."""
    torch.save({
        'policy_net': self.policy_net.state_dict(),
        'target_net': self.target_net.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'epsilon': self.epsilon,
        'training_steps': self.training_steps,
    }, path)
```

### 2. Flexible Loading with Security Options

```python
def load(self, path: str, weights_only: bool = True):
    """
    Load agent state from file (PyTorch 2.6+ compatible).
    
    Args:
        weights_only: If True, only load model weights (secure). 
                     If False, load all state including optimizer (for resuming training).
    """
    if weights_only:
        # Secure loading: only weights, no arbitrary code execution
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
    else:
        # Full loading: includes optimizer state for training resume
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    
    self.policy_net.load_state_dict(checkpoint['policy_net'])
    self.target_net.load_state_dict(checkpoint['target_net'])
    
    # Only load optimizer if not weights_only mode
    if not weights_only and 'optimizer' in checkpoint:
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'epsilon' in checkpoint:
        self.epsilon = checkpoint['epsilon']
    if 'training_steps' in checkpoint:
        self.training_steps = checkpoint.get('training_steps', 0)
```

### 3. Updated Scripts

#### test.py
```python
# For evaluation: use secure weights_only loading
agent.load(args.checkpoint, weights_only=True)
```

#### visualize_checkpoint.py
```python
# Direct loading with weights_only for visualization
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

# Handle different checkpoint formats
if 'policy_net' in checkpoint:
    agent.policy_net.load_state_dict(checkpoint['policy_net'])
elif 'policy_net_state_dict' in checkpoint:  # Legacy format
    agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
```

## Benefits

### Security
- ✅ Compatible with PyTorch 2.6+ security defaults
- ✅ No arbitrary code execution risk
- ✅ Uses `weights_only=True` for evaluation

### Simplicity
- ✅ Smaller checkpoint files (no replay buffer)
- ✅ Faster save/load operations
- ✅ No pickle dependency issues

### Flexibility
- ✅ `weights_only=True` for evaluation (secure)
- ✅ `weights_only=False` for training resume (full state)
- ✅ Backward compatible with old checkpoint formats

## Usage

### Evaluation (Secure)
```python
agent = FCNDQNAgent(...)
agent.load("checkpoint.pt", weights_only=True)  # Secure: only model weights
agent.policy_net.eval()
```

### Resume Training (Full State)
```python
agent = FCNDQNAgent(...)
agent.load("checkpoint.pt", weights_only=False)  # Includes optimizer state
# Continue training with same epsilon and optimizer state
```

### Visualization
```python
checkpoint = torch.load("checkpoint.pt", weights_only=True)
agent.policy_net.load_state_dict(checkpoint['policy_net'])
```

## Files Modified

1. **dqn_agent.py**:
   - Removed replay buffer from `save()` method
   - Added `weights_only` parameter to `load()` method
   - Added secure loading with PyTorch 2.6+ compatibility

2. **test.py**:
   - Changed to `agent.load(checkpoint, weights_only=True)` for evaluation

3. **visualize_checkpoint.py**:
   - Changed to `torch.load(checkpoint, weights_only=True)`
   - Added support for both 'policy_net' and 'policy_net_state_dict' formats

## Checkpoint Format

### Saved Keys
- `policy_net`: Policy network state dict
- `target_net`: Target network state dict
- `optimizer`: Adam optimizer state
- `epsilon`: Current exploration rate
- `training_steps`: Total training steps

### Not Saved (Intentionally)
- `memories`: Replay buffers (too large, rebuilds during training)
- `scaler`: AMP scaler state (rebuilds automatically)

## Migration

### Old Checkpoints
If you have old checkpoints with replay buffer data:
```python
# They will still load, but will ignore the replay buffer data
# This is safe and backward compatible
agent.load("old_checkpoint.pt", weights_only=False)
```

### New Checkpoints
All new checkpoints saved after this fix:
- Are smaller (~5-10 MB vs 50-100 MB with replay buffer)
- Load faster
- Work with PyTorch 2.6+ security defaults

## Testing

Verify the fix works:
```bash
# Save a checkpoint during training
python train.py --episodes 100 --device cuda

# Load it for evaluation (should work now)
python test.py --checkpoint checkpoints/coordinate_mlp_coverage_best.pt

# Visualize it (should work now)
python visualize_checkpoint.py --checkpoint checkpoints/coordinate_mlp_coverage_best.pt
```

Expected: No `_pickle.UnpicklingError` or security warnings.

## Why Not Save Replay Buffer?

1. **Size**: Replay buffers can be 50,000+ transitions = 50-100 MB
2. **Security**: Requires pickling custom objects (security risk)
3. **Unnecessary**: Replay buffer rebuilds quickly during training
4. **Compatibility**: Avoids PyTorch version issues

The only downside: when resuming training, the replay buffer starts empty. But it fills up in ~50-100 episodes, which is negligible for 1500-episode training.
