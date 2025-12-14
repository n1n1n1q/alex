"""
Base classes and utilities for benchmarking Minecraft agents.
"""

import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, model_name: str, task_name: str):
        self.model_name = model_name
        self.task_name = task_name
        self.trials: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None
        
    def add_trial(self, trial_data: Dict[str, Any]):
        """Add a trial result."""
        self.trials.append(trial_data)
        
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all trials."""
        if not self.trials:
            return {}
            
        stats = {
            "model": self.model_name,
            "task": self.task_name,
            "num_trials": len(self.trials),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time - self.start_time) if self.start_time and self.end_time else None
        }
        
        # Collect success flags if available
        successes = [t.get("success", False) for t in self.trials]
        if successes:
            stats["success_rate"] = sum(successes) / len(successes)
            stats["num_successes"] = sum(successes)
            
        # Collect completion times
        completion_times = [t.get("completion_time") for t in self.trials if t.get("completion_time") is not None]
        if completion_times:
            stats["avg_completion_time"] = np.mean(completion_times)
            stats["std_completion_time"] = np.std(completion_times)
            stats["min_completion_time"] = np.min(completion_times)
            stats["max_completion_time"] = np.max(completion_times)
            
        # Collect steps taken
        steps = [t.get("steps") for t in self.trials if t.get("steps") is not None]
        if steps:
            stats["avg_steps"] = np.mean(steps)
            stats["std_steps"] = np.std(steps)
            stats["min_steps"] = np.min(steps)
            stats["max_steps"] = np.max(steps)
            
        return stats
        
    def save(self, output_dir: str):
        """Save results to JSON files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed trials
        trials_file = os.path.join(output_dir, f"{self.model_name}_{self.task_name}_trials.json")
        with open(trials_file, 'w') as f:
            json.dump({
                "model": self.model_name,
                "task": self.task_name,
                "trials": self.trials
            }, f, indent=2)
            
        # Save statistics
        stats_file = os.path.join(output_dir, f"{self.model_name}_{self.task_name}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.compute_statistics(), f, indent=2)
            
        print(f"Saved results to {output_dir}")


class TaskChecker(ABC):
    """Abstract base class for checking task completion."""
    
    @abstractmethod
    def check(self, info: Dict[str, Any]) -> bool:
        """
        Check if task is completed.
        
        Args:
            info: Environment info dict containing inventory, observations, etc.
            
        Returns:
            True if task is completed, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_progress(self, info: Dict[str, Any]) -> float:
        """
        Get progress towards task completion.
        
        Args:
            info: Environment info dict
            
        Returns:
            Progress as float between 0.0 and 1.0
        """
        pass


class CraftingTableChecker(TaskChecker):
    """Check if agent has crafted a crafting table."""
    
    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("crafting_table", 0) > 0
        
    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})
        
        # Progress milestones:
        # 0.0: nothing
        # 0.25: has wood logs
        # 0.5: has planks
        # 0.75: near crafting table or has crafting materials
        # 1.0: has crafting table
        
        if inventory.get("crafting_table", 0) > 0:
            return 1.0
        
        planks = inventory.get("oak_planks", 0) + inventory.get("planks", 0)
        if planks >= 4:
            return 0.75
        elif planks > 0:
            return 0.5
            
        logs = inventory.get("oak_log", 0) + inventory.get("log", 0)
        if logs > 0:
            return 0.25
            
        return 0.0


class StoneAxeChecker(TaskChecker):
    """Check if agent has crafted a stone axe."""
    
    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("stone_axe", 0) > 0
        
    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})
        
        # Progress milestones:
        # 0.0: nothing
        # 0.2: has wood
        # 0.4: has planks or sticks
        # 0.5: has crafting table
        # 0.6: has wooden pickaxe
        # 0.7: has cobblestone
        # 0.9: has sticks + cobblestone
        # 1.0: has stone axe
        
        if inventory.get("stone_axe", 0) > 0:
            return 1.0
            
        cobblestone = inventory.get("cobblestone", 0)
        sticks = inventory.get("stick", 0)
        
        if cobblestone >= 3 and sticks >= 2:
            return 0.9
        if cobblestone > 0:
            return 0.7
        if inventory.get("wooden_pickaxe", 0) > 0:
            return 0.6
        if inventory.get("crafting_table", 0) > 0:
            return 0.5
        if sticks > 0 or inventory.get("planks", 0) > 0 or inventory.get("oak_planks", 0) > 0:
            return 0.4
        if inventory.get("log", 0) > 0 or inventory.get("oak_log", 0) > 0:
            return 0.2
            
        return 0.0


class IronOreChecker(TaskChecker):
    """Check if agent has found iron ore."""
    
    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("iron_ore", 0) > 0 or inventory.get("raw_iron", 0) > 0
        
    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})
        
        # Progress milestones:
        # 0.0: nothing
        # 0.2: has wood
        # 0.3: has crafting table
        # 0.4: has wooden pickaxe
        # 0.5: has cobblestone
        # 0.6: has stone pickaxe
        # 0.7: mining underground (y < 60)
        # 0.9: very deep (y < 20)
        # 1.0: has iron ore
        
        if inventory.get("iron_ore", 0) > 0 or inventory.get("raw_iron", 0) > 0:
            return 1.0
            
        # Check depth (if available in info)
        y_pos = info.get("location_stats", {}).get("ypos", 100)
        if y_pos < 20:
            return 0.9
        elif y_pos < 60:
            return 0.7
            
        if inventory.get("stone_pickaxe", 0) > 0:
            return 0.6
        if inventory.get("cobblestone", 0) > 0:
            return 0.5
        if inventory.get("wooden_pickaxe", 0) > 0:
            return 0.4
        if inventory.get("crafting_table", 0) > 0:
            return 0.3
        if inventory.get("log", 0) > 0 or inventory.get("oak_log", 0) > 0:
            return 0.2
            
        return 0.0


class DirtMiningChecker:
    """Track amount of dirt mined."""
    
    def get_dirt_count(self, info: Dict[str, Any]) -> int:
        """Get current dirt count from inventory."""
        inventory = info.get("inventory", {})
        return inventory.get("dirt", 0)
        
    def get_mining_rate(self, dirt_counts: List[int], timestamps: List[float]) -> float:
        """Calculate dirt mining rate (dirt/second)."""
        if len(dirt_counts) < 2 or len(timestamps) < 2:
            return 0.0
        
        dirt_delta = dirt_counts[-1] - dirt_counts[0]
        time_delta = timestamps[-1] - timestamps[0]
        
        if time_delta == 0:
            return 0.0
            
        return dirt_delta / time_delta


def get_task_checker(task_name: str) -> TaskChecker:
    """Factory function to get appropriate task checker."""
    checkers = {
        "crafting_table": CraftingTableChecker,
        "stone_axe": StoneAxeChecker,
        "iron_ore": IronOreChecker,
    }
    
    checker_class = checkers.get(task_name)
    if checker_class is None:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(checkers.keys())}")
    
    return checker_class()
