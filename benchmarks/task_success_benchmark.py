"""
Task Success Benchmark

Measures the success rate of agents completing specific tasks within a time limit.
Tasks: make crafting table, make stone axe, find iron ore
Each task is repeated 10 times.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.models import VPTPolicy, load_vpt_policy, SteveOnePolicy, load_steve_one_policy
from minestudio.inference import EpisodePipeline, MineGenerator

from alex.agent import Agent
from alex.core.extractor import extract_state
from benchmarks.base_benchmark import BenchmarkResult, get_task_checker


class TaskSuccessBenchmarkCallback(MinecraftCallback):
    """Callback for task success benchmark."""
    
    def __init__(self, task_name: str, max_steps: int = 6000):
        super().__init__()
        self.task_name = task_name
        self.max_steps = max_steps
        self.checker = get_task_checker(task_name)
        
        self.current_step = 0
        self.task_completed = False
        self.completion_step = None
        self.start_time = None
        self.completion_time = None
        self.progress_log = []
        
    def after_reset(self, sim, obs, info):
        self.current_step = 0
        self.task_completed = False
        self.completion_step = None
        self.start_time = time.time()
        self.completion_time = None
        self.progress_log = []
        
        # Log initial state
        self.progress_log.append({
            "step": 0,
            "progress": self.checker.get_progress(info),
            "completed": False
        })
        
        return obs, info
        
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.current_step += 1
        
        # Check task completion
        if not self.task_completed and self.checker.check(info):
            self.task_completed = True
            self.completion_step = self.current_step
            self.completion_time = time.time() - self.start_time
            print(f"[BENCHMARK] Task '{self.task_name}' completed at step {self.current_step}!")
            
        # Log progress every 50 steps
        if self.current_step % 50 == 0:
            progress = self.checker.get_progress(info)
            self.progress_log.append({
                "step": self.current_step,
                "progress": progress,
                "completed": self.task_completed
            })
            print(f"[BENCHMARK] Step {self.current_step}/{self.max_steps}, Progress: {progress:.2%}, Completed: {self.task_completed}")
            
        # Terminate if task completed or max steps reached
        if self.task_completed or self.current_step >= self.max_steps:
            terminated = True
            
        return obs, reward, terminated, truncated, info
        
    def get_trial_result(self) -> Dict[str, Any]:
        """Get results for this trial."""
        return {
            "success": self.task_completed,
            "steps": self.current_step,
            "completion_step": self.completion_step,
            "completion_time": self.completion_time,
            "progress_log": self.progress_log,
            "final_progress": self.progress_log[-1]["progress"] if self.progress_log else 0.0
        }


class VPTBenchmarkCallback(TaskSuccessBenchmarkCallback):
    """Callback for VPT baseline benchmark."""
    
    def __init__(self, task_name: str, max_steps: int = 6000):
        super().__init__(task_name, max_steps)


class STEVEBenchmarkCallback(TaskSuccessBenchmarkCallback):
    """Callback for STEVE-1 benchmark."""
    
    def __init__(self, task_name: str, policy, max_steps: int = 6000, update_interval: int = 100):
        super().__init__(task_name, max_steps)
        self.policy = policy
        self.update_interval = update_interval
        self.current_command = self._get_task_command(task_name)
        
    def _get_task_command(self, task_name: str) -> str:
        """Get natural language command for task."""
        commands = {
            "crafting_table": "chop trees and craft a crafting table",
            "stone_axe": "gather resources and craft a stone axe",
            "iron_ore": "mine underground and find iron ore"
        }
        return commands.get(task_name, "complete the task")
        
    def after_reset(self, sim, obs, info):
        obs, info = super().after_reset(sim, obs, info)
        
        # Set STEVE-1 text condition
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition']['text'] = self.current_command
        obs['condition']['cond_scale'] = 6.0
        
        print(f"[STEVE-1] Command: '{self.current_command}'")
        return obs, info
        
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        # Update command periodically based on progress
        if self.current_step % self.update_interval == 0 and self.current_step > 0:
            progress = self.checker.get_progress(info)
            # Could adapt command based on progress, but keeping it simple for now
            
        obs, reward, terminated, truncated, info = super().after_step(
            sim, obs, reward, terminated, truncated, info
        )
        
        # Ensure condition stays set
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition']['text'] = self.current_command
        obs['condition']['cond_scale'] = 6.0
        
        return obs, reward, terminated, truncated, info


class ALEXBenchmarkCallback(TaskSuccessBenchmarkCallback):
    """Callback for ALEX agent benchmark."""
    
    def __init__(self, task_name: str, max_steps: int = 6000, update_interval: int = 100):
        super().__init__(task_name, max_steps)
        self.agent = Agent()
        self.update_interval = update_interval
        self.current_command = "explore around"
        
    def after_reset(self, sim, obs, info):
        obs, info = super().after_reset(sim, obs, info)
        
        # Get initial command from ALEX agent
        try:
            state = extract_state(info)
            action = self.agent.step(obs, state)
            
            if hasattr(action, 'steve_prompt') and action.steve_prompt:
                self.current_command = action.steve_prompt
            elif hasattr(action, 'info') and action.info and 'steve_prompt' in action.info:
                self.current_command = action.info['steve_prompt']
                
            print(f"[ALEX] Initial command: '{self.current_command}'")
        except Exception as e:
            print(f"[ALEX] Error getting initial command: {e}")
            self.current_command = "explore around"
            
        # Set condition for STEVE-1
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition']['text'] = self.current_command
        obs['condition']['cond_scale'] = 6.0
        
        return obs, info
        
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        # Update command periodically using ALEX agent
        if self.current_step % self.update_interval == 0 and self.current_step > 0:
            try:
                state = extract_state(info)
                action = self.agent.step(obs, state)
                
                if hasattr(action, 'steve_prompt') and action.steve_prompt:
                    self.current_command = action.steve_prompt
                elif hasattr(action, 'info') and action.info and 'steve_prompt' in action.info:
                    self.current_command = action.info['steve_prompt']
                    
                print(f"[ALEX] Updated command: '{self.current_command}'")
            except Exception as e:
                print(f"[ALEX] Error updating command: {e}")
                
        obs, reward, terminated, truncated, info = super().after_step(
            sim, obs, reward, terminated, truncated, info
        )
        
        # Set condition for STEVE-1
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition']['text'] = self.current_command
        obs['condition']['cond_scale'] = 6.0
        
        return obs, reward, terminated, truncated, info


def run_task_benchmark(
    model_name: str,
    task_name: str,
    num_trials: int = 10,
    max_steps: int = 6000,
    output_dir: str = "./benchmark_results",
    model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: str = "cuda"
):
    """
    Run task success benchmark for a model.
    
    Args:
        model_name: One of 'vpt', 'steve', 'alex'
        task_name: One of 'crafting_table', 'stone_axe', 'iron_ore'
        num_trials: Number of times to repeat the task
        max_steps: Maximum steps per trial
        output_dir: Directory to save results
        model_path: Path to model file (for VPT/STEVE)
        weights_path: Path to weights file (for VPT/STEVE)
        device: Device to run on
    """
    
    print("="*80)
    print(f"TASK SUCCESS BENCHMARK")
    print(f"Model: {model_name.upper()}")
    print(f"Task: {task_name}")
    print(f"Trials: {num_trials}")
    print(f"Max Steps: {max_steps}")
    print("="*80)
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
        
    try:
        result = BenchmarkResult(model_name, task_name)
        result.start_time = time.time()
        
        for trial in range(num_trials):
            print(f"\n{'='*80}")
            print(f"TRIAL {trial + 1}/{num_trials}")
            print(f"{'='*80}\n")
            
            # Create callback
            if model_name.lower() == "vpt":
                callback = VPTBenchmarkCallback(task_name, max_steps)
                # Use from_pretrained for HuggingFace models, load_vpt_policy for local files
                if model_path and (model_path.startswith("CraftJarvis/") or "/" in model_path and not os.path.exists(model_path)):
                    # HuggingFace model ID
                    policy = VPTPolicy.from_pretrained(model_path).to(device)
                else:
                    # Local file path
                    policy = load_vpt_policy(
                        model_path=model_path,
                        weights_path=weights_path
                    ).to(device)
            elif model_name.lower() == "steve":
                policy = load_steve_one_policy(
                    ckpt_path=model_path
                ).to(device)
                callback = STEVEBenchmarkCallback(task_name, policy, max_steps)
            elif model_name.lower() == "alex":
                policy = load_steve_one_policy(
                    ckpt_path=model_path
                ).to(device)
                callback = ALEXBenchmarkCallback(task_name, max_steps)
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Run episode
            worker_kwargs = {
                "env_type": "sim",
                "sim_name": f"{model_name}_task_bench_{trial}",
                "policy": policy,
                "callbacks": [callback],
                "env_kwargs": {
                    "obs_size": (128, 128),
                    "preferred_spawn_biome": "forest",
                },
                "max_steps": max_steps + 10,  # Small buffer
                "reset_flag": True,
            }
            
            pipeline = EpisodePipeline(
                episode_generator=MineGenerator(
                    num_workers=1,
                    num_gpus=0.25 if device == "cuda" else 0,
                    max_restarts=3,
                    **worker_kwargs,
                ),
            )
            
            pipeline.run()
            
            # Get trial results
            trial_result = callback.get_trial_result()
            trial_result["trial_id"] = trial
            result.add_trial(trial_result)
            
            print(f"\nTrial {trial + 1} complete:")
            print(f"  Success: {trial_result['success']}")
            print(f"  Steps: {trial_result['steps']}")
            if trial_result['success']:
                print(f"  Completion Time: {trial_result['completion_time']:.2f}s")
                
        result.end_time = time.time()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, "task_success", timestamp)
        result.save(save_dir)
        
        # Print summary
        stats = result.compute_statistics()
        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETE - {model_name.upper()} - {task_name}")
        print(f"{'='*80}")
        print(f"Success Rate: {stats.get('success_rate', 0):.1%} ({stats.get('num_successes', 0)}/{num_trials})")
        if stats.get('avg_completion_time'):
            print(f"Avg Completion Time: {stats['avg_completion_time']:.2f}s ± {stats['std_completion_time']:.2f}s")
        print(f"Avg Steps: {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
        print(f"Results saved to: {save_dir}")
        print(f"{'='*80}\n")
        
        return result
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run task success benchmark")
    parser.add_argument("--model", type=str, required=True, choices=["vpt", "steve", "alex"],
                        help="Model to benchmark")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["crafting_table", "stone_axe", "iron_ore"],
                        help="Task to benchmark")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--max-steps", type=int, default=6000, help="Max steps per trial")
    parser.add_argument("--model-path", type=str, help="Path to model file")
    parser.add_argument("--weights-path", type=str, help="Path to weights file (VPT only)")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    run_task_benchmark(
        model_name=args.model,
        task_name=args.task,
        num_trials=args.trials,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        model_path=args.model_path,
        weights_path=args.weights_path,
        device=args.device
    )
