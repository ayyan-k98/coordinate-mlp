"""
Run All Unit Tests

Execute all module tests to verify implementation correctness.
"""

import sys
import subprocess


def run_test(module_path, description):
    """Run a single test module."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", module_path],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Running All Unit Tests")
    print("="*70)
    
    # FIXED: Removed 'src.' prefixes - using flat directory structure
    tests = [
        ("positional_encoding", "Fourier Positional Encoding"),
        ("cell_encoder", "Cell Feature MLP"),
        ("attention", "Attention Pooling"),
        ("q_network", "Dueling Q-Network"),
        ("coordinate_network", "Coordinate Coverage Network"),
        ("replay_buffer", "Replay Buffer"),
        ("dqn_agent", "DQN Agent"),
        ("config", "Configuration"),
        ("logger", "Logger"),
        ("metrics", "Metrics"),
        ("visualization", "Visualization"),
    ]
    
    results = []
    for module_path, description in tests:
        success = run_test(module_path, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    failed = total - passed
    
    for description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {description}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if failed > 0:
        print(f"\n⚠ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
