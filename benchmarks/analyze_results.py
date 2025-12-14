"""
Benchmark Analysis and Visualization

Tools for analyzing and visualizing benchmark results.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_benchmark_results(results_dir: str) -> Dict[str, Any]:
    """Load benchmark results from a directory."""
    
    summary_file = os.path.join(results_dir, "benchmark_summary.json")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"No benchmark_summary.json found in {results_dir}")
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_task_success_comparison(results: Dict[str, Any], output_dir: str):
    """Create bar chart comparing task success rates."""
    
    tasks = ["crafting_table", "stone_axe", "iron_ore"]
    task_names = {
        "crafting_table": "Crafting Table",
        "stone_axe": "Stone Axe",
        "iron_ore": "Iron Ore"
    }
    models = results['models']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(tasks))
    width = 0.25
    
    colors = {
        'vpt': '#FF6B6B',
        'steve': '#4ECDC4',
        'alex': '#45B7D1'
    }
    
    for i, model in enumerate(models):
        success_rates = []
        for task in tasks:
            task_data = results['task_success'].get(model, {}).get(task, {})
            rate = task_data.get('success_rate', 0) * 100
            success_rates.append(rate)
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, success_rates, width, 
                     label=model.upper(), 
                     color=colors.get(model, '#95E1D3'),
                     alpha=0.8)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{rate:.0f}%',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Task Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([task_names[t] for t in tasks])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'task_success_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved task success comparison to: {output_file}")


def plot_completion_time_comparison(results: Dict[str, Any], output_dir: str):
    """Create bar chart comparing average completion times for successful trials."""
    
    tasks = ["crafting_table", "stone_axe", "iron_ore"]
    task_names = {
        "crafting_table": "Crafting Table",
        "stone_axe": "Stone Axe",
        "iron_ore": "Iron Ore"
    }
    models = results['models']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(tasks))
    width = 0.25
    
    colors = {
        'vpt': '#FF6B6B',
        'steve': '#4ECDC4',
        'alex': '#45B7D1'
    }
    
    for i, model in enumerate(models):
        avg_times = []
        std_times = []
        for task in tasks:
            task_data = results['task_success'].get(model, {}).get(task, {})
            avg = task_data.get('avg_completion_time', 0)
            std = task_data.get('std_completion_time', 0)
            avg_times.append(avg if avg else 0)
            std_times.append(std if std else 0)
        
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, avg_times, width, 
                     yerr=std_times,
                     label=model.upper(), 
                     color=colors.get(model, '#95E1D3'),
                     alpha=0.8,
                     capsize=5)
        
        # Add value labels on bars
        for bar, avg in zip(bars, avg_times):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max(std_times) + 5,
                       f'{avg:.0f}s',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Completion Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Task Completion Time Comparison (Successful Trials)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([task_names[t] for t in tasks])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'completion_time_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved completion time comparison to: {output_file}")


def plot_dirt_mining_comparison(results: Dict[str, Any], output_dir: str):
    """Create bar chart comparing dirt mining performance."""
    
    models = results['models']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {
        'vpt': '#FF6B6B',
        'steve': '#4ECDC4',
        'alex': '#45B7D1'
    }
    
    # Best dirt count
    best_counts = []
    avg_counts = []
    std_counts = []
    mining_rates = []
    rate_stds = []
    
    for model in models:
        dirt_data = results['dirt_mining'].get(model, {})
        best_counts.append(dirt_data.get('best_dirt_count', 0))
        avg_counts.append(dirt_data.get('avg_dirt_count', 0))
        std_counts.append(dirt_data.get('std_dirt_count', 0))
        mining_rates.append(dirt_data.get('avg_mining_rate', 0))
        rate_stds.append(dirt_data.get('std_mining_rate', 0))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Dirt count plot
    bars1 = ax1.bar(x - width/2, best_counts, width, label='Best', 
                    color=[colors.get(m, '#95E1D3') for m in models], alpha=0.9)
    bars2 = ax1.bar(x + width/2, avg_counts, width, 
                    yerr=std_counts, label='Average', 
                    color=[colors.get(m, '#95E1D3') for m in models], 
                    alpha=0.6, capsize=5)
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Dirt Count', fontsize=12, fontweight='bold')
    ax1.set_title('Dirt Mining: Total Dirt Collected', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.0f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Mining rate plot
    bars3 = ax2.bar(x, mining_rates, 
                    yerr=rate_stds,
                    color=[colors.get(m, '#95E1D3') for m in models], 
                    alpha=0.8, capsize=5)
    
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mining Rate (dirt/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Dirt Mining: Average Mining Rate', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, rate in zip(bars3, mining_rates):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(rate_stds) + 0.05,
                    f'{rate:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'dirt_mining_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved dirt mining comparison to: {output_file}")


def create_summary_table(results: Dict[str, Any], output_dir: str):
    """Create a summary table in HTML format."""
    
    html_content = """
    <html>
    <head>
        <title>Benchmark Results Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #333; text-align: center; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }
            th { background-color: #4CAF50; color: white; font-weight: bold; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .best { background-color: #d4edda; font-weight: bold; }
            .timestamp { text-align: center; color: #666; font-style: italic; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
        </style>
    </head>
    <body>
    """
    
    html_content += f"<h1>Minecraft Agent Benchmark Results</h1>\n"
    html_content += f"<p class='timestamp'>Generated: {results['timestamp']}</p>\n"
    
    # Task success table
    html_content += "<h2>Task Success Rates</h2>\n"
    html_content += "<table>\n"
    html_content += "<tr><th>Task</th>"
    for model in results['models']:
        html_content += f"<th>{model.upper()}</th>"
    html_content += "</tr>\n"
    
    tasks = ["crafting_table", "stone_axe", "iron_ore"]
    task_names = {
        "crafting_table": "Crafting Table",
        "stone_axe": "Stone Axe",
        "iron_ore": "Iron Ore"
    }
    
    for task in tasks:
        html_content += f"<tr><td><b>{task_names[task]}</b></td>"
        
        rates = []
        for model in results['models']:
            task_data = results['task_success'].get(model, {}).get(task, {})
            rate = task_data.get('success_rate', 0) * 100
            rates.append((model, rate))
        
        best_rate = max(rates, key=lambda x: x[1])[1] if rates else 0
        
        for model, rate in rates:
            css_class = "best" if rate == best_rate and rate > 0 else ""
            html_content += f"<td class='{css_class}'>{rate:.1f}%</td>"
        
        html_content += "</tr>\n"
    
    html_content += "</table>\n"
    
    # Dirt mining table
    html_content += "<h2>Dirt Mining Performance</h2>\n"
    html_content += "<table>\n"
    html_content += "<tr><th>Model</th><th>Best Dirt Count</th><th>Avg Dirt Count</th><th>Avg Mining Rate (dirt/s)</th></tr>\n"
    
    best_counts = []
    for model in results['models']:
        dirt_data = results['dirt_mining'].get(model, {})
        best = dirt_data.get('best_dirt_count', 0)
        avg = dirt_data.get('avg_dirt_count', 0)
        std = dirt_data.get('std_dirt_count', 0)
        rate = dirt_data.get('avg_mining_rate', 0)
        rate_std = dirt_data.get('std_mining_rate', 0)
        
        best_counts.append((model, best))
        
        html_content += f"<tr><td><b>{model.upper()}</b></td>"
        html_content += f"<td>{best}</td>"
        html_content += f"<td>{avg:.1f} ± {std:.1f}</td>"
        html_content += f"<td>{rate:.2f} ± {rate_std:.2f}</td>"
        html_content += "</tr>\n"
    
    html_content += "</table>\n"
    
    # Add images
    html_content += "<h2>Visualizations</h2>\n"
    html_content += "<img src='task_success_comparison.png' alt='Task Success Comparison'>\n"
    html_content += "<img src='completion_time_comparison.png' alt='Completion Time Comparison'>\n"
    html_content += "<img src='dirt_mining_comparison.png' alt='Dirt Mining Comparison'>\n"
    
    html_content += "</body>\n</html>"
    
    output_file = os.path.join(output_dir, 'benchmark_summary.html')
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Saved HTML summary to: {output_file}")


def analyze_benchmark_results(results_dir: str):
    """
    Analyze and visualize benchmark results.
    
    Args:
        results_dir: Directory containing benchmark_summary.json
    """
    
    print(f"Loading results from: {results_dir}")
    results = load_benchmark_results(results_dir)
    
    print(f"Found results for models: {results['models']}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_task_success_comparison(results, results_dir)
    plot_completion_time_comparison(results, results_dir)
    plot_dirt_mining_comparison(results, results_dir)
    
    # Create summary table
    print("\nGenerating HTML summary...")
    create_summary_table(results, results_dir)
    
    print(f"\nAnalysis complete! Check {results_dir} for outputs.")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("results_dir", type=str, 
                       help="Directory containing benchmark_summary.json")
    
    args = parser.parse_args()
    
    analyze_benchmark_results(args.results_dir)


if __name__ == "__main__":
    main()
