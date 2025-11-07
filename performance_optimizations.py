"""
Performance Optimizations Module

Provides performance enhancements including:
- Mixed precision training
- JIT compilation
- Efficient data loading
- CUDA optimizations
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings


class PerformanceConfig:
    """Configuration for performance optimizations."""

    def __init__(self,
                 use_amp: bool = True,  # Automatic Mixed Precision
                 use_jit: bool = False,  # TorchScript JIT compilation
                 compile_model: bool = True,  # torch.compile (PyTorch 2.0+)
                 use_cudnn_benchmark: bool = True,  # cuDNN autotuner
                 use_tf32: bool = True,  # TF32 for Ampere GPUs
                 pin_memory: bool = True,  # Pin memory for faster GPU transfer
                 num_workers: int = 4):  # DataLoader workers

        self.use_amp = use_amp
        self.use_jit = use_jit
        self.compile_model = compile_model
        self.use_cudnn_benchmark = use_cudnn_benchmark
        self.use_tf32 = use_tf32
        self.pin_memory = pin_memory
        self.num_workers = num_workers


def setup_performance_optimizations(config: PerformanceConfig, device: torch.device):
    """
    Set up performance optimizations.

    Args:
        config: Performance configuration
        device: Target device
    """
    # Enable cuDNN benchmark mode for faster convolutions
    if config.use_cudnn_benchmark and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN benchmark mode enabled")

    # Enable TF32 for Ampere GPUs (3x speedup on A100)
    if config.use_tf32 and device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster matmul")

    # Set default tensor type for GPU
    if device.type == 'cuda':
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


def optimize_model(model: nn.Module,
                   config: PerformanceConfig,
                   device: torch.device,
                   example_input: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Apply optimizations to a model.

    Args:
        model: Model to optimize
        config: Performance configuration
        device: Target device
        example_input: Example input for JIT tracing

    Returns:
        Optimized model
    """
    optimized_model = model

    # PyTorch 2.0+ torch.compile (fastest option)
    if config.compile_model and hasattr(torch, 'compile'):
        try:
            optimized_model = torch.compile(model, mode='reduce-overhead')
            print("✓ Model compiled with torch.compile (PyTorch 2.0+)")
            return optimized_model
        except Exception as e:
            warnings.warn(f"torch.compile failed: {e}, falling back to regular model")

    # JIT compilation (TorchScript) - alternative
    if config.use_jit and example_input is not None:
        try:
            optimized_model = torch.jit.trace(model, example_input)
            print("✓ Model JIT compiled with TorchScript")
            return optimized_model
        except Exception as e:
            warnings.warn(f"JIT compilation failed: {e}, using regular model")

    return optimized_model


class MixedPrecisionTrainer:
    """
    Wrapper for mixed precision training using torch.cuda.amp.

    Provides 2-3x speedup with minimal accuracy loss on modern GPUs.
    """

    def __init__(self, enabled: bool = True, device: torch.device = None):
        self.enabled = enabled and device is not None and device.type == 'cuda'

        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast = torch.cuda.amp.autocast
            print("✓ Mixed precision training enabled (AMP)")
        else:
            # Dummy scaler for CPU/disabled mode
            self.scaler = None
            self.autocast = self._dummy_autocast
            if device is not None and device.type == 'cuda':
                print("⚠ Mixed precision disabled")

    @staticmethod
    def _dummy_autocast():
        """Dummy context manager when AMP is disabled."""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()

    def backward_and_step(self, loss, optimizer, clip_grad: Optional[float] = None, parameters=None):
        """
        Perform backward pass and optimizer step with mixed precision.

        Args:
            loss: Loss tensor
            optimizer: Optimizer
            clip_grad: Gradient clipping value (optional)
            parameters: Model parameters for gradient clipping
        """
        if self.enabled:
            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            if clip_grad is not None and parameters is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)

            # Optimizer step with scaled gradients
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Regular backward pass
            loss.backward()

            # Gradient clipping
            if clip_grad is not None and parameters is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)

            optimizer.step()


def get_optimal_batch_size(device: torch.device, model: nn.Module) -> int:
    """
    Heuristically determine optimal batch size based on available memory.

    Args:
        device: Target device
        model: Model to train

    Returns:
        Recommended batch size
    """
    if device.type == 'cuda':
        # Get GPU memory in GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9

        # Heuristic: larger batch for more memory
        if total_memory >= 40:  # A100 80GB
            return 256
        elif total_memory >= 20:  # A100 40GB, V100 32GB
            return 128
        elif total_memory >= 10:  # RTX 3080, 2080 Ti
            return 64
        else:  # Smaller GPUs
            return 32
    else:
        # CPU: smaller batch
        return 16


# Vectorized coordinate generation (faster than loops)
@torch.jit.script
def generate_coords_vectorized(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Vectorized coordinate generation (JIT compiled for speed).

    Args:
        H: Height
        W: Width
        device: Target device

    Returns:
        Normalized coordinates [H*W, 2] in range [-1, 1]
    """
    # Create meshgrid (vectorized)
    y_coords = torch.linspace(-1.0, 1.0, H, device=device)
    x_coords = torch.linspace(-1.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Stack and reshape
    coords = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
    coords = coords.reshape(-1, 2)  # [H*W, 2]

    return coords


def profile_model(model: nn.Module, input_shape: tuple, device: torch.device, num_runs: int = 100):
    """
    Profile model inference speed.

    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Target device
        num_runs: Number of profiling runs
    """
    import time

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Profile
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start
    avg_time = elapsed / num_runs * 1000  # ms
    throughput = num_runs / elapsed  # samples/sec

    print(f"\n{'='*60}")
    print(f"Model Performance Profile")
    print(f"{'='*60}")
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test performance optimizations
    print("="*70)
    print("Testing Performance Optimizations")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")

    # Test optimizations
    config = PerformanceConfig(
        use_amp=True,
        compile_model=True,
        use_cudnn_benchmark=True,
        use_tf32=True
    )

    setup_performance_optimizations(config, device)

    # Test vectorized coordinate generation
    print("\n" + "-"*70)
    print("Testing Vectorized Coordinate Generation")
    print("-"*70)

    import time
    H, W = 32, 32

    # Warmup
    for _ in range(10):
        coords = generate_coords_vectorized(H, W, device)

    # Benchmark
    start = time.time()
    for _ in range(1000):
        coords = generate_coords_vectorized(H, W, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Generated {1000} coordinate grids ({H}x{W})")
    print(f"Total time: {elapsed*1000:.2f} ms")
    print(f"Per-generation: {elapsed/1000*1000:.4f} ms")

    # Test mixed precision
    print("\n" + "-"*70)
    print("Testing Mixed Precision Trainer")
    print("-"*70)

    mp_trainer = MixedPrecisionTrainer(enabled=True, device=device)
    print(f"AMP enabled: {mp_trainer.enabled}")

    # Optimal batch size
    print("\n" + "-"*70)
    print("Optimal Batch Size Recommendation")
    print("-"*70)

    # Create dummy model
    dummy_model = nn.Linear(100, 100).to(device)
    optimal_bs = get_optimal_batch_size(device, dummy_model)
    print(f"Recommended batch size: {optimal_bs}")

    print("\n✓ All performance optimizations tested successfully!")
