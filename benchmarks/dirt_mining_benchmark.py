import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))

import ray
import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import MinecraftCallback
from minestudio.models import (
    VPTPolicy,
    load_vpt_policy,
    SteveOnePolicy,
    load_steve_one_policy,
)
from minestudio.inference import EpisodePipeline, MineGenerator

from alex.agent import Agent, VerboseAgent
from alex.core.extractor import extract_state
from benchmark_config import BenchmarkConfig


class BenchmarkResult:

    def __init__(self, model_name: str, task_name: str):
        self.model_name = model_name
        self.task_name = task_name
        self.trials = []
        self.start_time = None
        self.end_time = None

    def add_trial(self, trial_data):
        self.trials.append(trial_data)

    def compute_statistics(self):
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

        steps = [t.get("steps") for t in self.trials if t.get("steps") is not None]
        if steps:
            stats["avg_steps"] = float(np.mean(steps))
            stats["std_steps"] = float(np.std(steps))

        dirt_mined = [t.get("dirt_mined", 0) for t in self.trials]
        if dirt_mined:
            stats["avg_dirt_mined"] = float(np.mean(dirt_mined))
            stats["std_dirt_mined"] = float(np.std(dirt_mined))
            stats["total_dirt_mined"] = sum(dirt_mined)

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


def get_dirt_count(info: Dict[str, Any]) -> int:
    if "inventory" not in info:
        return 0

    inventory = info["inventory"]

    if isinstance(inventory, dict):
        dirt_count = 0
        for slot_num, item in inventory.items():
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if "dirt" in str(item_type).lower():
                    quantity = item.get("quantity", item.get("count", 1))
                    dirt_count += quantity
        return dirt_count

    return 0


def get_mining_rate(dirt_counts: List[int], timestamps: List[float]) -> float:
    if len(dirt_counts) < 2 or len(timestamps) < 2:
        return 0.0

    total_dirt = dirt_counts[-1] - dirt_counts[0]
    total_time = timestamps[-1] - timestamps[0]

    if total_time <= 0:
        return 0.0

    return total_dirt / total_time


class DirtMiningBenchmarkCallback(MinecraftCallback):

    def __init__(self, max_steps: int = 3000, sample_interval: int = 20):
        super().__init__()
        self.max_steps = max_steps
        self.sample_interval = sample_interval

        self.current_step = 0
        self.start_time = None
        self.dirt_log = []

    def after_reset(self, sim, obs, info):
        self.current_step = 0
        self.start_time = time.time()
        self.dirt_log = []

        dirt_count = get_dirt_count(info)
        self.dirt_log.append({"step": 0, "timestamp": 0.0, "dirt_count": dirt_count})

        print(f"[BENCHMARK] Starting dirt mining benchmark")
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        self.current_step += 1

        if self.current_step % self.sample_interval == 0:
            elapsed_time = time.time() - self.start_time
            dirt_count = get_dirt_count(info)

            self.dirt_log.append(
                {
                    "step": self.current_step,
                    "timestamp": elapsed_time,
                    "dirt_count": dirt_count,
                }
            )

            if len(self.dirt_log) >= 2:
                dirt_counts = [log["dirt_count"] for log in self.dirt_log]
                timestamps = [log["timestamp"] for log in self.dirt_log]
                rate = get_mining_rate(dirt_counts, timestamps)
                print(
                    f"[BENCHMARK] Step {self.current_step}/{self.max_steps}, "
                    f"Dirt: {dirt_count}, Rate: {rate:.2f} dirt/sec"
                )
            else:
                print(
                    f"[BENCHMARK] Step {self.current_step}/{self.max_steps}, Dirt: {dirt_count}"
                )

        if self.current_step >= self.max_steps:
            terminated = True

        return obs, reward, terminated, truncated, info

    def get_trial_result(self) -> Dict[str, Any]:
        dirt_counts = [log["dirt_count"] for log in self.dirt_log]
        timestamps = [log["timestamp"] for log in self.dirt_log]

        final_dirt = dirt_counts[-1] if dirt_counts else 0
        avg_rate = get_mining_rate(dirt_counts, timestamps)

        return {
            "steps": self.current_step,
            "final_dirt_count": final_dirt,
            "avg_mining_rate": avg_rate,
            "dirt_log": self.dirt_log,
            "peak_dirt": max(dirt_counts) if dirt_counts else 0,
        }


class VPTDirtMiningCallback(DirtMiningBenchmarkCallback):
    pass


class STEVEDirtMiningCallback(DirtMiningBenchmarkCallback):

    def __init__(self, policy, max_steps: int = 3000, sample_interval: int = 20):
        super().__init__(max_steps, sample_interval)
        self.policy = policy
        self.command = "collect many dirt"

    def after_reset(self, sim, obs, info):
        obs, info = super().after_reset(sim, obs, info)

        if "condition" not in obs:
            obs["condition"] = {}
        obs["condition"]["text"] = self.command
        obs["condition"]["cond_scale"] = 10.0

        print(f"[STEVE-1] Command: '{self.command}'")
        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs, reward, terminated, truncated, info = super().after_step(
            sim, obs, reward, terminated, truncated, info
        )

        if "condition" not in obs:
            obs["condition"] = {}
        obs["condition"]["text"] = self.command
        obs["condition"]["cond_scale"] = 10.0

        return obs, reward, terminated, truncated, info


class ALEXDirtMiningCallback(DirtMiningBenchmarkCallback):

    def __init__(
        self,
        max_steps: int = 3000,
        sample_interval: int = 20,
        update_interval: int = 100,
    ):
        super().__init__(max_steps, sample_interval)
        self.agent = VerboseAgent()
        self.update_interval = update_interval
        self.current_command = "explore around"

    def after_reset(self, sim, obs, info):
        obs, info = super().after_reset(sim, obs, info)

        try:
            state = extract_state(info)
            action = self.agent.step(obs, state)

            if hasattr(action, "steve_prompt") and action.steve_prompt:
                self.current_command = action.steve_prompt
            elif (
                hasattr(action, "info")
                and action.info
                and "steve_prompt" in action.info
            ):
                self.current_command = action.info["steve_prompt"]

            print(f"[ALEX] Initial command: '{self.current_command}'")
        except Exception as e:
            print(f"[ALEX] Error getting initial command: {e}")
            self.current_command = "mine dirt"

        if "condition" not in obs:
            obs["condition"] = {}
        obs["condition"]["text"] = self.current_command
        obs["condition"]["cond_scale"] = 10.0

        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if self.current_step % self.update_interval == 0 and self.current_step > 0:
            try:
                state = extract_state(info)
                action = self.agent.step(obs, state)

                if hasattr(action, "steve_prompt") and action.steve_prompt:
                    self.current_command = action.steve_prompt
                elif (
                    hasattr(action, "info")
                    and action.info
                    and "steve_prompt" in action.info
                ):
                    self.current_command = action.info["steve_prompt"]

                print(f"[ALEX] Updated command: '{self.current_command}'")
            except Exception as e:
                print(f"[ALEX] Error updating command: {e}")

        obs, reward, terminated, truncated, info = super().after_step(
            sim, obs, reward, terminated, truncated, info
        )

        if "condition" not in obs:
            obs["condition"] = {}
        obs["condition"]["text"] = self.current_command
        obs["condition"]["cond_scale"] = 10.0

        return obs, reward, terminated, truncated, info


def run_dirt_mining_benchmark(
    model_name: str,
    num_trials: int = 5,
    max_steps: int = 3000,
    output_dir: str = "./benchmark_results",
    model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    device: str = "cuda",
):

    print("=" * 80)
    print(f"DIRT MINING BENCHMARK")
    print(f"Model: {model_name.upper()}")
    print(f"Trials: {num_trials}")
    print(f"Max Steps: {max_steps}")
    print("=" * 80)

    if not ray.is_initialized():
        ray.init()

    try:
        result = BenchmarkResult(model_name, "dirt_mining")
        result.start_time = time.time()

        best_trial = None
        best_dirt_count = 0

        for trial in range(num_trials):
            print(f"\n{'='*80}")
            print(f"TRIAL {trial + 1}/{num_trials}")
            print(f"{'='*80}\n")

            if model_name.lower() == "vpt":
                callback = VPTDirtMiningCallback(max_steps)
                if model_path and (
                    model_path.startswith("CraftJarvis/")
                    or "/" in model_path
                    and not os.path.exists(model_path)
                ):
                    policy = VPTPolicy.from_pretrained(model_path).to(device)
                else:
                    policy = load_vpt_policy(
                        model_path=model_path, weights_path=weights_path
                    ).to(device)
            elif model_name.lower() == "steve":
                policy = load_steve_one_policy(ckpt_path=model_path).to(device)
                callback = STEVEDirtMiningCallback(policy, max_steps)
            elif model_name.lower() == "alex":
                policy = load_steve_one_policy(ckpt_path=model_path).to(device)
                callback = ALEXDirtMiningCallback(max_steps)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            env_generator = partial(
                MinecraftSim,
                obs_size=(128, 128),
                preferred_spawn_biome="plains",
                callbacks=[callback],
            )

            agent_generator = lambda: policy

            worker_kwargs = dict(
                env_generator=env_generator,
                agent_generator=agent_generator,
                num_max_steps=max_steps + 10,
                num_episodes=1,
                tmpdir="./benchmark_results",
                image_media="h264",
            )

            pipeline = EpisodePipeline(
                episode_generator=MineGenerator(
                    num_workers=1,
                    num_gpus=0.25 if device == "cuda" else 0,
                    max_restarts=3,
                    **worker_kwargs,
                ),
            )

            pipeline.run()

            trial_result = callback.get_trial_result()
            trial_result["trial_id"] = trial
            result.add_trial(trial_result)

            dirt_count = trial_result["final_dirt_count"]
            print(f"\nTrial {trial + 1} complete:")
            print(f"  Final Dirt Count: {dirt_count}")
            print(f"  Avg Mining Rate: {trial_result['avg_mining_rate']:.2f} dirt/sec")

            if dirt_count > best_dirt_count:
                best_dirt_count = dirt_count
                best_trial = trial_result

        result.end_time = time.time()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, "dirt_mining", timestamp)
        result.save(save_dir)

        if best_trial:
            best_file = os.path.join(save_dir, f"{model_name}_dirt_mining_best.json")
            with open(best_file, "w") as f:
                json.dump(best_trial, f, indent=2)

        for idx, trial in enumerate(result.trials):
            if "dirt_log" in trial:
                csv_file = os.path.join(
                    save_dir, f"{model_name}_trial_{idx}_timeseries.csv"
                )
                with open(csv_file, "w") as f:
                    f.write("step,timestamp,dirt_count\n")
                    for log_entry in trial["dirt_log"]:
                        f.write(
                            f"{log_entry['step']},{log_entry['timestamp']:.2f},{log_entry['dirt_count']}\n"
                        )

        stats = result.compute_statistics()
        dirt_counts = [t["final_dirt_count"] for t in result.trials]
        mining_rates = [t["avg_mining_rate"] for t in result.trials]

        print(f"\n{'='*80}")
        print(f"BENCHMARK COMPLETE - {model_name.upper()} - Dirt Mining")
        print(f"{'='*80}")
        print(f"Best Dirt Count: {best_dirt_count}")
        print(f"Avg Dirt Count: {np.mean(dirt_counts):.1f} ± {np.std(dirt_counts):.1f}")
        print(
            f"Avg Mining Rate: {np.mean(mining_rates):.2f} ± {np.std(mining_rates):.2f} dirt/sec"
        )
        print(f"Results saved to: {save_dir}")
        print(f"{'='*80}\n")

        return result

    finally:
        ray.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dirt mining benchmark")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vpt", "steve", "alex"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: benchmark_config.yaml)",
    )
    parser.add_argument(
        "--trials", type=int, default=None, help="Number of trials (overrides config)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per trial (overrides config)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file (overrides config)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Path to weights file (VPT only, overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, overrides config)"
    )

    args = parser.parse_args()

    config = BenchmarkConfig(args.config)

    model_path = (
        args.model_path if args.model_path else config.get_model_path(args.model)
    )
    weights_path = (
        args.weights_path if args.weights_path else config.get_model_weights(args.model)
    )
    output_dir = args.output_dir if args.output_dir else config.get_output_dir()
    device = args.device if args.device else config.get_device()
    trials = args.trials if args.trials else config.get_dirt_trials(args.model)
    max_steps = (
        args.max_steps if args.max_steps else config.get_dirt_max_steps(args.model)
    )

    run_dirt_mining_benchmark(
        model_name=args.model,
        num_trials=trials,
        max_steps=max_steps,
        output_dir=output_dir,
        model_path=model_path,
        weights_path=weights_path,
        device=device,
    )
