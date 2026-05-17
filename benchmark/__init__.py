"""
Benchmark package for running experiments on external repositories

Usage:
    from benchmark import BenchmarkRunner
    
    runner = BenchmarkRunner("cider", agent=coder)
    results = runner.run()
"""

from .runner import BenchmarkRunner
from .executor import BenchmarkExecutor

__all__ = ['BenchmarkRunner', 'BenchmarkExecutor']
__version__ = '1.0.0'