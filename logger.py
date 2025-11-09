"""
Logging Utilities

Provides logging functionality for training and evaluation.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


class Logger:
    """
    Simple logger for training metrics.
    
    Logs to console and JSON file.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{timestamp}.jsonl"
        )
        
        self.metrics_history = []
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step/episode number
        """
        # Add timestamp and step
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        
        # Save to history
        self.metrics_history.append(log_entry)
        
        # Write to file (one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_episode(self, episode: int, metrics: Dict[str, float]):
        """
        Log episode metrics to console and file.
        
        Args:
            episode: Episode number
            metrics: Dictionary of metrics
        """
        # Log to file
        self.log(metrics, step=episode)
        
        # Log to console with type-aware formatting
        metric_parts = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_parts.append(f"{k}={v:.4f}")
            else:
                metric_parts.append(f"{k}={v}")
        metric_str = ', '.join(metric_parts)
        print(f"Episode {episode:4d}: {metric_str}")
    
    def save_summary(self, filepath: Optional[str] = None):
        """
        Save summary statistics to file.
        
        Args:
            filepath: Path to save summary (default: log_dir/summary.json)
        """
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")
        
        summary = {
            'experiment_name': self.experiment_name,
            'num_entries': len(self.metrics_history),
            'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to {filepath}")


class TensorBoardLogger:
    """
    TensorBoard logger for rich visualization.
    
    Requires tensorboard package.
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save TensorBoard logs
            experiment_name: Name of experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
            
            self.writer = SummaryWriter(log_path)
            self.enabled = True
            print(f"TensorBoard logging enabled: {log_path}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """Log histogram of values."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int):
        """Log image (H, W, C) or (C, H, W)."""
        if self.enabled:
            self.writer.add_image(tag, image, step, dataformats='HWC')
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


if __name__ == "__main__":
    import tempfile
    import os
    
    print("="*60)
    print("Testing Logger")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test basic logger
        logger = Logger(tmpdir, "test_experiment")
        print(f"\nLogger created: {logger.log_file}")
        
        # Log some metrics
        for i in range(5):
            metrics = {
                'loss': np.random.rand(),
                'reward': np.random.rand() * 100,
                'epsilon': 1.0 - i * 0.1
            }
            logger.log_episode(i, metrics)
        
        # Save summary
        logger.save_summary()
        
        # Check files
        print(f"\nGenerated files:")
        for f in os.listdir(tmpdir):
            print(f"  {f}")
        
        # Test TensorBoard logger
        print(f"\nTesting TensorBoard logger:")
        tb_logger = TensorBoardLogger(tmpdir, "test_tb")
        
        if tb_logger.enabled:
            for i in range(10):
                tb_logger.log_scalar('train/loss', np.random.rand(), i)
                tb_logger.log_scalars('train/metrics', {
                    'metric1': np.random.rand(),
                    'metric2': np.random.rand()
                }, i)
            
            tb_logger.close()
            print("  TensorBoard logs created successfully")
        else:
            print("  TensorBoard not available")
    
    print("\n✓ Logger test complete!")
