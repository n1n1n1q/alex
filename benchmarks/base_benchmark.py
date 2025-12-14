import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class BenchmarkResult:

    def __init__(self, model_name: str, task_name: str):
        self.model_name = model_name
        self.task_name = task_name
        self.trials: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None

    def add_trial(self, trial_data: Dict[str, Any]):
        self.trials.append(trial_data)

    def compute_statistics(self) -> Dict[str, Any]:
        if not self.trials:
            return {}

        stats = {
            "model": self.model_name,
            "task": self.task_name,
            "num_trials": len(self.trials),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (
                (self.end_time - self.start_time)
                if self.start_time and self.end_time
                else None
            ),
        }

        successes = [t.get("success", False) for t in self.trials]
        if successes:
            stats["success_rate"] = sum(successes) / len(successes)
            stats["num_successes"] = sum(successes)

        completion_times = [
            t.get("completion_time")
            for t in self.trials
            if t.get("completion_time") is not None
        ]
        if completion_times:
            stats["avg_completion_time"] = float(np.mean(completion_times))
            stats["std_completion_time"] = float(np.std(completion_times))
            stats["min_completion_time"] = float(np.min(completion_times))
            stats["max_completion_time"] = float(np.max(completion_times))

        steps = [t.get("steps") for t in self.trials if t.get("steps") is not None]
        if steps:
            stats["avg_steps"] = float(np.mean(steps))
            stats["std_steps"] = float(np.std(steps))
            stats["min_steps"] = int(np.min(steps))
            stats["max_steps"] = int(np.max(steps))

        return stats

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        trials_file = os.path.join(
            output_dir, f"{self.model_name}_{self.task_name}_trials.json"
        )
        with open(trials_file, "w") as f:
            json.dump(
                {
                    "model": self.model_name,
                    "task": self.task_name,
                    "trials": self.trials,
                },
                f,
                indent=2,
            )

        stats_file = os.path.join(
            output_dir, f"{self.model_name}_{self.task_name}_stats.json"
        )
        with open(stats_file, "w") as f:
            json.dump(self.compute_statistics(), f, indent=2)

        print(f"Saved results to {output_dir}")


class TaskChecker(ABC):

    @abstractmethod
    def check(self, info: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_progress(self, info: Dict[str, Any]) -> float:
        pass


class CraftingTableChecker(TaskChecker):

    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("crafting_table", 0) > 0

    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})

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

    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("stone_axe", 0) > 0

    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})

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
        if (
            sticks > 0
            or inventory.get("planks", 0) > 0
            or inventory.get("oak_planks", 0) > 0
        ):
            return 0.4
        if inventory.get("log", 0) > 0 or inventory.get("oak_log", 0) > 0:
            return 0.2

        return 0.0


class IronOreChecker(TaskChecker):

    def check(self, info: Dict[str, Any]) -> bool:
        inventory = info.get("inventory", {})
        return inventory.get("iron_ore", 0) > 0 or inventory.get("raw_iron", 0) > 0

    def get_progress(self, info: Dict[str, Any]) -> float:
        inventory = info.get("inventory", {})

        if inventory.get("iron_ore", 0) > 0 or inventory.get("raw_iron", 0) > 0:
            return 1.0

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

    def get_dirt_count(self, info: Dict[str, Any]) -> int:
        inventory = info.get("inventory", {})
        return inventory.get("dirt", 0)

    def get_mining_rate(self, dirt_counts: List[int], timestamps: List[float]) -> float:
        if len(dirt_counts) < 2 or len(timestamps) < 2:
            return 0.0

        dirt_delta = dirt_counts[-1] - dirt_counts[0]
        time_delta = timestamps[-1] - timestamps[0]

        if time_delta == 0:
            return 0.0

        return dirt_delta / time_delta


def get_task_checker(task_name: str) -> TaskChecker:
    checkers = {
        "crafting_table": CraftingTableChecker,
        "stone_axe": StoneAxeChecker,
        "iron_ore": IronOreChecker,
    }

    checker_class = checkers.get(task_name)
    if checker_class is None:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(checkers.keys())}"
        )

    return checker_class()
