"""
Minecraft Agent Benchmarking Suite

This package provides comprehensive benchmarking tools for comparing
VPT, STEVE-1, and ALEX agents on various Minecraft tasks.

Modules:
    base_benchmark: Core benchmark classes and utilities
    task_success_benchmark: Task completion benchmarks
    dirt_mining_benchmark: Resource gathering benchmarks
    run_benchmarks: Master runner for all benchmarks
    analyze_results: Result analysis and visualization

Example usage:
    # Run complete benchmark suite
    from benchmarks.run_benchmarks import run_full_benchmark_suite
    
    models_config = {
        'vpt': {'model_path': '/path/to/vpt/model', 'weights_path': '/path/to/weights'},
        'steve': {'model_path': '/path/to/steve/checkpoint'},
        'alex': {'model_path': '/path/to/steve/checkpoint'}
    }
    
    results = run_full_benchmark_suite(models_config)
    
    # Analyze results
    from benchmarks.analyze_results import analyze_benchmark_results
    analyze_benchmark_results('./benchmark_results/benchmark_suite_TIMESTAMP/')
"""

from benchmarks.base_benchmark import (
    BenchmarkResult,
    TaskChecker,
    CraftingTableChecker,
    StoneAxeChecker,
    IronOreChecker,
    DirtMiningChecker,
    get_task_checker
)

__all__ = [
    'BenchmarkResult',
    'TaskChecker',
    'CraftingTableChecker',
    'StoneAxeChecker',
    'IronOreChecker',
    'DirtMiningChecker',
    'get_task_checker',
]

__version__ = '1.0.0'
