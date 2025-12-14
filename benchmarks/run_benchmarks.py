"""
Master Benchmark Runner

Runs all benchmarks for VPT, STEVE, and ALEX models and generates comparison reports.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.task_success_benchmark import run_task_benchmark
from benchmarks.dirt_mining_benchmark import run_dirt_mining_benchmark


def run_full_benchmark_suite(
    models_config: Dict[str, Dict[str, str]],
    output_dir: str = "./benchmark_results",
    task_trials: int = 10,
    task_max_steps: int = 6000,
    dirt_trials: int = 5,
    dirt_max_steps: int = 3000,
    device: str = "cuda"
):
    """
    Run complete benchmark suite for all models.
    
    Args:
        models_config: Dict mapping model name to config with 'model_path' and optional 'weights_path'
                      Example: {
                          'vpt': {'model_path': '/path/to/model', 'weights_path': '/path/to/weights'},
                          'steve': {'model_path': '/path/to/checkpoint'},
                          'alex': {'model_path': '/path/to/checkpoint'}
                      }
        output_dir: Base directory for all results
        task_trials: Number of trials per task
        task_max_steps: Max steps per task trial
        dirt_trials: Number of dirt mining trials
        dirt_max_steps: Max steps per dirt mining trial
        device: Device to run on
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = os.path.join(output_dir, f"benchmark_suite_{timestamp}")
    os.makedirs(suite_dir, exist_ok=True)
    
    print("="*80)
    print("BENCHMARK SUITE")
    print(f"Models: {list(models_config.keys())}")
    print(f"Output: {suite_dir}")
    print("="*80)
    
    all_results = {
        "timestamp": timestamp,
        "models": list(models_config.keys()),
        "task_success": {},
        "dirt_mining": {}
    }
    
    tasks = ["crafting_table", "stone_axe", "iron_ore"]
    
    # Run task success benchmarks
    print("\n" + "="*80)
    print("TASK SUCCESS BENCHMARKS")
    print("="*80)
    
    for model_name, model_config in models_config.items():
        print(f"\n{'='*80}")
        print(f"Running benchmarks for {model_name.upper()}")
        print(f"{'='*80}")
        
        all_results["task_success"][model_name] = {}
        
        for task in tasks:
            print(f"\n--- Task: {task} ---\n")
            
            try:
                result = run_task_benchmark(
                    model_name=model_name,
                    task_name=task,
                    num_trials=task_trials,
                    max_steps=task_max_steps,
                    output_dir=suite_dir,
                    model_path=model_config.get("model_path"),
                    weights_path=model_config.get("weights_path"),
                    device=device
                )
                
                all_results["task_success"][model_name][task] = result.compute_statistics()
                
            except Exception as e:
                print(f"ERROR running {model_name} on {task}: {e}")
                import traceback
                traceback.print_exc()
                all_results["task_success"][model_name][task] = {"error": str(e)}
    
    # Run dirt mining benchmarks
    print("\n" + "="*80)
    print("DIRT MINING BENCHMARKS")
    print("="*80)
    
    for model_name, model_config in models_config.items():
        print(f"\n{'='*80}")
        print(f"Running dirt mining for {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            result = run_dirt_mining_benchmark(
                model_name=model_name,
                num_trials=dirt_trials,
                max_steps=dirt_max_steps,
                output_dir=suite_dir,
                model_path=model_config.get("model_path"),
                weights_path=model_config.get("weights_path"),
                device=device
            )
            
            stats = result.compute_statistics()
            # Add dirt-specific stats
            import numpy as np
            dirt_counts = [t["final_dirt_count"] for t in result.trials]
            mining_rates = [t["avg_mining_rate"] for t in result.trials]
            stats["best_dirt_count"] = max(dirt_counts) if dirt_counts else 0
            stats["avg_dirt_count"] = float(np.mean(dirt_counts))
            stats["std_dirt_count"] = float(np.std(dirt_counts))
            stats["avg_mining_rate"] = float(np.mean(mining_rates))
            stats["std_mining_rate"] = float(np.std(mining_rates))
            
            all_results["dirt_mining"][model_name] = stats
            
        except Exception as e:
            print(f"ERROR running {model_name} dirt mining: {e}")
            import traceback
            traceback.print_exc()
            all_results["dirt_mining"][model_name] = {"error": str(e)}
    
    # Save comprehensive results
    summary_file = os.path.join(suite_dir, "benchmark_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison report
    generate_comparison_report(all_results, suite_dir)
    
    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print(f"Results saved to: {suite_dir}")
    print("="*80)
    
    return all_results


def generate_comparison_report(results: Dict[str, Any], output_dir: str):
    """Generate a human-readable comparison report."""
    
    report_file = os.path.join(output_dir, "comparison_report.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK COMPARISON REPORT\n")
        f.write(f"Generated: {results['timestamp']}\n")
        f.write("="*80 + "\n\n")
        
        # Task Success Comparison
        f.write("TASK SUCCESS RATES\n")
        f.write("-"*80 + "\n\n")
        
        tasks = ["crafting_table", "stone_axe", "iron_ore"]
        task_names = {
            "crafting_table": "Crafting Table",
            "stone_axe": "Stone Axe",
            "iron_ore": "Iron Ore"
        }
        
        for task in tasks:
            f.write(f"\n{task_names[task]}:\n")
            f.write("-" * 40 + "\n")
            
            for model in results['models']:
                task_data = results['task_success'].get(model, {}).get(task, {})
                
                if 'error' in task_data:
                    f.write(f"  {model.upper():8s}: ERROR - {task_data['error']}\n")
                elif 'success_rate' in task_data:
                    success_rate = task_data['success_rate'] * 100
                    num_successes = task_data.get('num_successes', 0)
                    num_trials = task_data.get('num_trials', 0)
                    avg_steps = task_data.get('avg_steps', 0)
                    
                    f.write(f"  {model.upper():8s}: {success_rate:5.1f}% ({num_successes}/{num_trials})")
                    
                    if task_data.get('avg_completion_time'):
                        avg_time = task_data['avg_completion_time']
                        f.write(f" | Avg Time: {avg_time:6.1f}s")
                    
                    f.write(f" | Avg Steps: {avg_steps:6.1f}\n")
                else:
                    f.write(f"  {model.upper():8s}: No data\n")
        
        # Dirt Mining Comparison
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write("DIRT MINING PERFORMANCE\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'Model':<10s} {'Best':<10s} {'Avg':<15s} {'Rate (dirt/s)':<15s}\n")
        f.write("-" * 60 + "\n")
        
        for model in results['models']:
            dirt_data = results['dirt_mining'].get(model, {})
            
            if 'error' in dirt_data:
                f.write(f"{model.upper():<10s} ERROR: {dirt_data['error']}\n")
            elif 'best_dirt_count' in dirt_data:
                best = dirt_data['best_dirt_count']
                avg = dirt_data['avg_dirt_count']
                std = dirt_data['std_dirt_count']
                rate = dirt_data['avg_mining_rate']
                rate_std = dirt_data['std_mining_rate']
                
                f.write(f"{model.upper():<10s} {best:<10d} {avg:6.1f} ± {std:4.1f}    {rate:5.2f} ± {rate_std:4.2f}\n")
            else:
                f.write(f"{model.upper():<10s} No data\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\nComparison report saved to: {report_file}")
    
    # Also print to console
    with open(report_file, 'r') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite")
    
    # Model paths
    parser.add_argument("--vpt-model", type=str, help="Path to VPT model file")
    parser.add_argument("--vpt-weights", type=str, help="Path to VPT weights file")
    parser.add_argument("--steve-model", type=str, help="Path to STEVE checkpoint (also used by ALEX)")
    
    # Benchmark parameters
    parser.add_argument("--task-trials", type=int, default=10, help="Trials per task")
    parser.add_argument("--task-max-steps", type=int, default=6000, help="Max steps per task trial")
    parser.add_argument("--dirt-trials", type=int, default=5, help="Dirt mining trials")
    parser.add_argument("--dirt-max-steps", type=int, default=3000, help="Max steps per dirt trial")
    
    # Other options
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=["vpt", "steve", "alex"],
                        choices=["vpt", "steve", "alex"],
                        help="Models to benchmark")
    
    args = parser.parse_args()
    
    # Build models config
    models_config = {}
    
    if "vpt" in args.models:
        if not args.vpt_model:
            print("ERROR: --vpt-model required for VPT benchmark")
            return
        models_config["vpt"] = {
            "model_path": args.vpt_model,
            "weights_path": args.vpt_weights
        }
    
    if "steve" in args.models:
        if not args.steve_model:
            print("ERROR: --steve-model required for STEVE benchmark")
            return
        models_config["steve"] = {
            "model_path": args.steve_model
        }
    
    if "alex" in args.models:
        # ALEX uses STEVE-1 as backbone (no separate checkpoint)
        if not args.steve_model:
            print("ERROR: --steve-model required for ALEX benchmark (ALEX uses STEVE-1 as backbone)")
            return
        models_config["alex"] = {
            "model_path": args.steve_model
        }
    
    if not models_config:
        print("ERROR: No models configured. Provide model paths.")
        return
    
    print(f"Running benchmarks for: {list(models_config.keys())}")
    
    run_full_benchmark_suite(
        models_config=models_config,
        output_dir=args.output_dir,
        task_trials=args.task_trials,
        task_max_steps=args.task_max_steps,
        dirt_trials=args.dirt_trials,
        dirt_max_steps=args.dirt_max_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
