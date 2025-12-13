import os
import sys
import json
import cv2
from pathlib import Path
from datetime import datetime
from functools import partial
import ray
import textwrap

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    MinecraftCallback,
    SpeedTestCallback,
    RecordCallback,
)
from minestudio.models import SteveOnePolicy
from minestudio.inference import EpisodePipeline, MineGenerator

from alex.agent import Agent, VerboseAgent
from alex.core.extractor import extract_state
from alex.utils.serialization import to_serializable


class AlexAgentCallback(MinecraftCallback):

    def __init__(
        self,
        update_interval: int = 50,
        cond_scale: float = 5.0,
        verbose: bool = True,
        output_dir: str = None,
        debug_panel_width: int = 512,
    ):

        super().__init__()
        self.agent = VerboseAgent() if verbose else Agent()
        self.update_interval = update_interval
        self.cond_scale = cond_scale
        self.verbose = verbose
        self.timestep = 0
        self.current_command = "explore around"
        self.output_dir = output_dir
        self.states_log = []
        self.prompts_log = []
        self.responses_log = []
        self.debug_panel_width = debug_panel_width

        self.current_llm_prompt = "Initializing..."
        self.current_steve_prompt = "explore around"
        self.last_llm_response = ""

        self.debug_frame = None

    def _add_debug_panel_to_frame(self, frame):
        """Add debug panel to right side of frame with LLM and Steve prompts."""
        h, w = frame.shape[:2]

        new_width = w + self.debug_panel_width
        new_frame = np.zeros((h, new_width, 3), dtype=np.uint8)
        new_frame[:h, :w] = frame
        new_frame[:, w:] = (30, 30, 30)

        cv2.line(new_frame, (w, 0), (w, h), (100, 100, 100), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        color = (220, 220, 220)
        line_height = 15

        y_pos = 20

        cv2.putText(
            new_frame, "DEBUG INFO", (w + 10, y_pos), font, 0.5, (100, 255, 100), 2
        )
        y_pos += 25

        cv2.putText(
            new_frame,
            "STEVE-1 PROMPT:",
            (w + 10, y_pos),
            font,
            font_scale,
            (100, 200, 255),
            thickness,
        )
        y_pos += line_height + 5

        steve_text = self.current_steve_prompt or "None"
        wrapped_steve = textwrap.wrap(steve_text, width=55)
        for line in wrapped_steve[:10]:
            cv2.putText(
                new_frame,
                line,
                (w + 10, y_pos),
                font,
                font_scale * 0.9,
                color,
                thickness,
            )
            y_pos += line_height

        y_pos += 15

        cv2.putText(
            new_frame,
            "LLM PROMPT:",
            (w + 10, y_pos),
            font,
            font_scale,
            (255, 200, 100),
            thickness,
        )
        y_pos += line_height + 5

        llm_text = self.current_llm_prompt or "None"
        wrapped_llm = textwrap.wrap(llm_text, width=55)
        for line in wrapped_llm[:15]:
            cv2.putText(
                new_frame,
                line,
                (w + 10, y_pos),
                font,
                font_scale * 0.9,
                color,
                thickness,
            )
            y_pos += line_height
            if y_pos > h - 30:
                break

        cv2.putText(
            new_frame,
            f"Step: {self.timestep}",
            (w + 10, h - 10),
            font,
            0.4,
            (150, 150, 150),
            thickness,
        )

        return new_frame

    def _save_state(self, state, step: int, phase: str):
        """Save extracted game state to log."""
        state_entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "state": to_serializable(state),
        }
        self.states_log.append(state_entry)

        if self.output_dir:
            states_file = os.path.join(self.output_dir, "game_states.json")
            with open(states_file, "w", encoding="utf-8") as f:
                json.dump(self.states_log, f, indent=2, ensure_ascii=False)

    def _save_prompt(self, prompt: str, step: int, phase: str):
        """Save STEVE-1 prompt to log."""
        prompt_entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
        }
        self.prompts_log.append(prompt_entry)

        if self.output_dir:
            prompts_file = os.path.join(self.output_dir, "prompts.json")
            with open(prompts_file, "w", encoding="utf-8") as f:
                json.dump(self.prompts_log, f, indent=2, ensure_ascii=False)

    def _save_model_response(
        self, response_data: dict, step: int, phase: str, response_type: str
    ):
        """Save model response (from planner or reflex manager) to log."""
        response_entry = {
            "step": step,
            "phase": phase,
            "response_type": response_type,  # 'planner' or 'reflex'
            "timestamp": datetime.now().isoformat(),
            "response": response_data,
        }
        self.responses_log.append(response_entry)

        if self.output_dir:
            responses_file = os.path.join(self.output_dir, "model_responses.json")
            with open(responses_file, "w", encoding="utf-8") as f:
                json.dump(self.responses_log, f, indent=2, ensure_ascii=False)

    def after_reset(self, sim, obs, info):

        self.timestep = 0

        try:
            print("\n" + "=" * 80)
            print("AGENT RESET - Getting Initial Command")
            print("=" * 80)

            state = extract_state(info)

            self._save_state(state, self.timestep, "reset")

            print(
                f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}"
            )
            health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
            hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
            print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
            print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
            print(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")

            print(f"[AGENT] Processing state through pipeline...")
            action = self.agent.step(obs, state)

            if hasattr(action, "info") and action.info:
                if "raw_model_response" in action.info:
                    self.last_llm_response = str(action.info["raw_model_response"])[
                        :500
                    ]
                    self.current_llm_prompt = (
                        f"Planner response: {self.last_llm_response}"
                    )
                if "reflex_response" in action.info:
                    reflex_data = action.info["reflex_response"]
                    if isinstance(reflex_data, dict) and "raw" in reflex_data:
                        self.current_llm_prompt = (
                            f"Reflex: {str(reflex_data['raw'])[:500]}"
                        )

            if hasattr(action, "info") and action.info:
                model_response = {}
                if "raw_model_response" in action.info:
                    model_response["raw"] = action.info["raw_model_response"]
                if "parsed_plan" in action.info:
                    model_response["parsed_plan"] = action.info["parsed_plan"]
                if "reflex_response" in action.info:
                    model_response["reflex"] = action.info["reflex_response"]
                if model_response:
                    response_type = (
                        "reflex" if "reflex_response" in action.info else "planner"
                    )
                    self._save_model_response(
                        model_response, self.timestep, "reset", response_type
                    )

            print(f"[AGENT] Action status: {action.status}")
            if hasattr(action, "info") and action.info:
                print(f"[AGENT] Action info: {action.info}")
                if "steve_prompt" in action.info:
                    self.current_command = action.info["steve_prompt"]
                    self.current_steve_prompt = self.current_command
                    print(f"\n[STEVE-1] Command from action: '{self.current_command}'")

            if hasattr(action, "steve_prompt") and action.steve_prompt:
                self.current_command = action.steve_prompt
                self.current_steve_prompt = self.current_command
                print(f"\n[STEVE-1] Command: '{self.current_command}'")
            elif self.current_command == "explore around":
                print(f"\n[STEVE-1] Default command: '{self.current_command}'")
                self.current_steve_prompt = self.current_command

            self._save_prompt(self.current_command, self.timestep, "reset")

            print("=" * 80 + "\n")

        except Exception as e:
            print(f"\n[ERROR] Failed to get initial command: {e}")
            import traceback

            traceback.print_exc()
            self.current_command = "explore around"
            self.current_steve_prompt = self.current_command
            self.current_llm_prompt = f"Error: {str(e)[:200]}"
            self._save_prompt(self.current_command, self.timestep, "reset_fallback")

        if "condition" not in obs:
            obs["condition"] = {}

        obs["condition"] = {"cond_scale": self.cond_scale, "text": "chop a tree"}

        if info is not None:
            self.debug_frame = self._add_debug_panel_to_frame(info["pov"])

        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):

        self.timestep += 1

        if self.timestep % self.update_interval == 0:
            try:
                print("\n" + "=" * 80)
                print(f"AGENT REPLANNING - Step {self.timestep}")
                print("=" * 80)

                state = extract_state(info)

                self._save_state(state, self.timestep, "replan")

                print(
                    f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}"
                )
                health_str = (
                    f"{state.health:.1f}" if state.health is not None else "N/A"
                )
                hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
                print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
                if state.player_pos and isinstance(state.player_pos, dict):
                    x = state.player_pos.get("x", 0)
                    y = state.player_pos.get("y", 0)
                    z = state.player_pos.get("z", 0)
                    print(f"[STATE] Position: ({x:.1f}, {y:.1f}, {z:.1f})")
                print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
                print(
                    f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}"
                )

                print(f"\n[AGENT] Running planner...")
                action = self.agent.step(obs, state)

                if hasattr(action, "info") and action.info:
                    if "raw_model_response" in action.info:
                        self.last_llm_response = str(action.info["raw_model_response"])[
                            :500
                        ]
                        self.current_llm_prompt = (
                            f"Planner response: {self.last_llm_response}"
                        )
                    if "reflex_response" in action.info:
                        reflex_data = action.info["reflex_response"]
                        if isinstance(reflex_data, dict) and "raw" in reflex_data:
                            self.current_llm_prompt = (
                                f"Reflex: {str(reflex_data['raw'])[:500]}"
                            )

                if hasattr(action, "info") and action.info:
                    model_response = {}
                    if "raw_model_response" in action.info:
                        model_response["raw"] = action.info["raw_model_response"]
                    if "parsed_plan" in action.info:
                        model_response["parsed_plan"] = action.info["parsed_plan"]
                    if "reflex_response" in action.info:
                        model_response["reflex"] = action.info["reflex_response"]
                    if model_response:
                        response_type = (
                            "reflex" if "reflex_response" in action.info else "planner"
                        )
                        self._save_model_response(
                            model_response, self.timestep, "replan", response_type
                        )

                print(f"[AGENT] Action status: {action.status}")
                if hasattr(action, "info") and action.info:
                    print(f"[AGENT] Action info: {action.info}")
                    if "steve_prompt" in action.info:
                        old_command = self.current_command
                        self.current_command = action.info["steve_prompt"]
                        self.current_steve_prompt = self.current_command
                        if old_command != self.current_command:
                            print(f"\n[STEVE-1] COMMAND CHANGED (from info)")
                            print(f"[STEVE-1]   Old: '{old_command}'")
                            print(f"[STEVE-1]   New: '{self.current_command}'")

                if hasattr(action, "steve_prompt") and action.steve_prompt:
                    old_command = self.current_command
                    self.current_command = action.steve_prompt
                    self.current_steve_prompt = self.current_command
                    if old_command != self.current_command:
                        print(f"\n[STEVE-1] COMMAND CHANGED")
                        print(f"[STEVE-1]   Old: '{old_command}'")
                        print(f"[STEVE-1]   New: '{self.current_command}'")
                    else:
                        print(f"\n[STEVE-1] Continuing: '{self.current_command}'")
                else:
                    if action.status == "FAILED":
                        print(
                            f"\n[STEVE-1] Execution failed, keeping: '{self.current_command}'"
                        )
                    else:
                        print(
                            f"\n[STEVE-1] No new command, continuing: '{self.current_command}'"
                        )

                self._save_prompt(self.current_command, self.timestep, "replan")

                print("=" * 80 + "\n")

            except Exception as e:
                print(f"\n[ERROR t={self.timestep}] Failed to update agent: {e}")
                import traceback

                traceback.print_exc()
                self.current_llm_prompt = (
                    f"Error at step {self.timestep}: {str(e)[:200]}"
                )

        obs["condition"] = {"cond_scale": self.cond_scale, "text": self.current_command}

        if info is not None:
            self.debug_frame = self._add_debug_panel_to_frame(info["pov"])

        return obs, reward, terminated, truncated, info



class DebugRecordCallback(MinecraftCallback):
    """Custom recording callback that uses debug frames from AlexAgentCallback."""

    def __init__(
        self, record_path: str, fps: int = 20, alex_callback: AlexAgentCallback = None
    ):
        super().__init__()
        self.record_path = record_path
        self.fps = fps
        self.alex_callback = alex_callback
        self.video_writer = None
        self.frame_count = 0
        self.episode_num = 0

    def after_reset(self, sim, obs, info):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.episode_num += 1
        video_path = os.path.join(
            self.record_path, f"episode_{self.episode_num}_debug.mp4"
        )

        if self.alex_callback and self.alex_callback.debug_frame is not None:
            h, w = self.alex_callback.debug_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))

            frame_bgr = cv2.cvtColor(self.alex_callback.debug_frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            self.frame_count = 1

            print(f"[DEBUG RECORDER] Started recording to {video_path} ({w}x{h})")

        return obs, info

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        if (
            self.video_writer
            and self.alex_callback
            and self.alex_callback.debug_frame is not None
        ):
            frame_bgr = cv2.cvtColor(self.alex_callback.debug_frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame_bgr)
            self.frame_count += 1

        return obs, reward, terminated, truncated, info

    def before_close(self, sim):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[DEBUG RECORDER] Saved {self.frame_count} frames")
            self.video_writer = None


def run_agent_with_recording(
    num_episodes: int = 3,
    max_steps: int = 1000,
    update_interval: int = 50,
    cond_scale: float = 1.0,
    output_dir: str = None,
    description: str = "agent_loop",
    verbose: bool = True,
):

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30])
        output_dir = f"./recordings/{safe_desc}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ALEX AGENT WITH VIDEO RECORDING")
    print("=" * 80)
    print(f"Description:       {description}")
    print(f"Episodes:          {num_episodes}")
    print(f"Max Steps:         {max_steps}")
    print(f"Update Interval:   {update_interval} steps")
    print(f"Cond Scale:        {cond_scale}")
    print(f"Output Directory:  {output_dir}")
    print("=" * 80)
    print()

    print("Initializing Ray...")
    ray.init()

    try:

        print("Setting up environment...")

        alex_callback = AlexAgentCallback(
            update_interval=update_interval,
            cond_scale=cond_scale,
            verbose=verbose,
            output_dir=output_dir,
        )

        env_generator = partial(
            MinecraftSim,
            obs_size=(128, 128),
            preferred_spawn_biome="forest",
            callbacks=[
                SpeedTestCallback(50),
                alex_callback,
                RecordCallback(
                    record_path=output_dir,
                    fps=20,
                    frame_type="pov",
                ),
                DebugRecordCallback(
                    record_path=output_dir, fps=20, alex_callback=alex_callback
                ),
            ],
        )

        print("Loading STEVE-1 model from CraftJarvis/MineStudio_STEVE-1.official...")
        agent_generator = lambda: SteveOnePolicy.from_pretrained(
            "CraftJarvis/MineStudio_STEVE-1.official"
        )

        worker_kwargs = dict(
            env_generator=env_generator,
            agent_generator=agent_generator,
            num_max_steps=max_steps,
            num_episodes=num_episodes,
            tmpdir=output_dir,
            image_media="h264",
        )

        print("Setting up inference pipeline...")
        pipeline = EpisodePipeline(
            episode_generator=MineGenerator(
                num_workers=1,
                num_gpus=0.25,
                max_restarts=3,
                **worker_kwargs,
            ),
        )

        print("\n" + "=" * 80)
        print("RUNNING AGENT LOOP...")
        print("=" * 80)
        print()

        summary = pipeline.run()

        print("\n" + "=" * 80)
        print("AGENT LOOP COMPLETE!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Total Episodes:    {summary.get('num_episodes', 0)}")
        print(f"  Avg Episode Length: {summary.get('avg_length', 'N/A')}")
        print(f"\nRecordings saved to: {output_dir}")
        print("=" * 80)

        info_file = os.path.join(output_dir, "experiment_info.txt")
        with open(info_file, "w") as f:
            f.write(f"Alex Agent Recording Info\n")
            f.write(f"==========================\n\n")
            f.write(f"Description:       {description}\n")
            f.write(f"Episodes:          {num_episodes}\n")
            f.write(f"Max Steps:         {max_steps}\n")
            f.write(f"Update Interval:   {update_interval}\n")
            f.write(f"Cond Scale:        {cond_scale}\n")
            f.write(f"\nResults:\n")
            f.write(f"  Total Episodes:    {summary.get('num_episodes', 0)}\n")
            f.write(f"  Avg Episode Length: {summary.get('avg_length', 'N/A')}\n")

        print(f"Experiment info saved to: {info_file}\n")

        return summary

    finally:
        ray.shutdown()
        print("Ray shutdown complete")


if __name__ == "__main__":
    run_agent_with_recording(
        description="wood_and_crafting_table",
        num_episodes=1,
        max_steps=2000,
        update_interval=250,
        cond_scale=10.0,
        verbose=True,
    )
