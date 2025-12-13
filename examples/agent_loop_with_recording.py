import os
import sys
import json
import cv2 
from pathlib import Path
from datetime import datetime
from functools import partial
import ray

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import MinecraftCallback, SpeedTestCallback, RecordCallback
from minestudio.models import SteveOnePolicy
from minestudio.inference import EpisodePipeline, MineGenerator

from alex.agent import Agent, VerboseAgent
from alex.core.extractor import extract_state
from alex.utils.serialization import to_serializable



class AlexAgentCallback(MinecraftCallback):

    def __init__(self, 
                 update_interval: int = 50,
                 cond_scale: float = 5.0,
                 verbose: bool = True,
                 output_dir: str = None):

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
        
    def _save_state(self, state, step: int, phase: str):
        """Save extracted game state to log."""
        state_entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "state": to_serializable(state)
        }
        self.states_log.append(state_entry)
        
        # Save incrementally if output_dir is set
        if self.output_dir:
            states_file = os.path.join(self.output_dir, "game_states.json")
            with open(states_file, 'w', encoding='utf-8') as f:
                json.dump(self.states_log, f, indent=2, ensure_ascii=False)
    
    def _save_prompt(self, prompt: str, step: int, phase: str):
        """Save STEVE-1 prompt to log."""
        prompt_entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt
        }
        self.prompts_log.append(prompt_entry)
        
        # Save incrementally if output_dir is set
        if self.output_dir:
            prompts_file = os.path.join(self.output_dir, "prompts.json")
            with open(prompts_file, 'w', encoding='utf-8') as f:
                json.dump(self.prompts_log, f, indent=2, ensure_ascii=False)
    
    def _save_model_response(self, response_data: dict, step: int, phase: str, response_type: str):
        """Save model response (from planner or reflex manager) to log."""
        response_entry = {
            "step": step,
            "phase": phase,
            "response_type": response_type,  # 'planner' or 'reflex'
            "timestamp": datetime.now().isoformat(),
            "response": response_data
        }
        self.responses_log.append(response_entry)
        
        # Save incrementally if output_dir is set
        if self.output_dir:
            responses_file = os.path.join(self.output_dir, "model_responses.json")
            with open(responses_file, 'w', encoding='utf-8') as f:
                json.dump(self.responses_log, f, indent=2, ensure_ascii=False)
        
    def after_reset(self, sim, obs, info):

        self.timestep = 0
        
        try:
            print("\n" + "="*80)
            print("AGENT RESET - Getting Initial Command")
            print("="*80)
            
            state = extract_state(info)
            
            # Save extracted state
            self._save_state(state, self.timestep, "reset")
            
            print(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
            health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
            hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
            print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
            print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
            print(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
            
            print(f"[AGENT] Processing state through pipeline...")
            action = self.agent.step(obs, state)
            
            # Save model response if available
            if hasattr(action, 'info') and action.info:
                model_response = {}
                if 'raw_model_response' in action.info:
                    model_response['raw'] = action.info['raw_model_response']
                if 'parsed_plan' in action.info:
                    model_response['parsed_plan'] = action.info['parsed_plan']
                if 'reflex_response' in action.info:
                    model_response['reflex'] = action.info['reflex_response']
                if model_response:
                    response_type = 'reflex' if 'reflex_response' in action.info else 'planner'
                    self._save_model_response(model_response, self.timestep, "reset", response_type)
            
            print(f"[AGENT] Action status: {action.status}")
            if hasattr(action, 'info') and action.info:
                print(f"[AGENT] Action info: {action.info}")
                if 'steve_prompt' in action.info:
                    self.current_command = action.info['steve_prompt']
                    print(f"\n[STEVE-1] Command from action: '{self.current_command}'")
            
            if hasattr(action, 'steve_prompt') and action.steve_prompt:
                self.current_command = action.steve_prompt
                print(f"\n[STEVE-1] Command: '{self.current_command}'")
            elif self.current_command == "explore around":
                print(f"\n[STEVE-1] Default command: '{self.current_command}'")
            
            # Save the prompt
            self._save_prompt(self.current_command, self.timestep, "reset")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to get initial command: {e}")
            import traceback
            traceback.print_exc()
            self.current_command = "explore around"
            self._save_prompt(self.current_command, self.timestep, "reset_fallback")
        
        if 'condition' not in obs:
            obs['condition'] = {}

        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": "chop a tree"
        }
        
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):

        self.timestep += 1
        
        if self.timestep % self.update_interval == 0:
            try:
                print("\n" + "="*80)
                print(f"AGENT REPLANNING - Step {self.timestep}")
                print("="*80)
                
                state = extract_state(info)
                
                # Save extracted state
                self._save_state(state, self.timestep, "replan")
                
                print(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
                health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
                hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
                print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
                if state.player_pos and isinstance(state.player_pos, dict):
                    x = state.player_pos.get('x', 0)
                    y = state.player_pos.get('y', 0)
                    z = state.player_pos.get('z', 0)
                    print(f"[STATE] Position: ({x:.1f}, {y:.1f}, {z:.1f})")
                print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
                print(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
                
                print(f"\n[AGENT] Running planner...")
                action = self.agent.step(obs, state)
                
                if hasattr(action, 'info') and action.info:
                    model_response = {}
                    if 'raw_model_response' in action.info:
                        model_response['raw'] = action.info['raw_model_response']
                    if 'parsed_plan' in action.info:
                        model_response['parsed_plan'] = action.info['parsed_plan']
                    if 'reflex_response' in action.info:
                        model_response['reflex'] = action.info['reflex_response']
                    if model_response:
                        response_type = 'reflex' if 'reflex_response' in action.info else 'planner'
                        self._save_model_response(model_response, self.timestep, "replan", response_type)

                print(f"[AGENT] Action status: {action.status}")
                if hasattr(action, 'info') and action.info:
                    print(f"[AGENT] Action info: {action.info}")
                    if 'steve_prompt' in action.info:
                        old_command = self.current_command
                        self.current_command = action.info['steve_prompt']
                        if old_command != self.current_command:
                            print(f"\n[STEVE-1] COMMAND CHANGED (from info)")
                            print(f"[STEVE-1]   Old: '{old_command}'")
                            print(f"[STEVE-1]   New: '{self.current_command}'")
                
                if hasattr(action, 'steve_prompt') and action.steve_prompt:
                    old_command = self.current_command
                    self.current_command = action.steve_prompt
                    if old_command != self.current_command:
                        print(f"\n[STEVE-1] COMMAND CHANGED")
                        print(f"[STEVE-1]   Old: '{old_command}'")
                        print(f"[STEVE-1]   New: '{self.current_command}'")
                    else:
                        print(f"\n[STEVE-1] Continuing: '{self.current_command}'")
                else:
                    if action.status == "FAILED":
                        print(f"\n[STEVE-1] Execution failed, keeping: '{self.current_command}'")
                    else:
                        print(f"\n[STEVE-1] No new command, continuing: '{self.current_command}'")
                
                # Save the prompt
                self._save_prompt(self.current_command, self.timestep, "replan")
                
                print("="*80 + "\n")
                        
            except Exception as e:
                print(f"\n[ERROR t={self.timestep}] Failed to update agent: {e}")
                import traceback
                traceback.print_exc()

        
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": "chop a tree"
        }
        
        return obs, reward, terminated, truncated, info
    

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
        env_generator = partial(
            MinecraftSim,
            obs_size=(128, 128),
            preferred_spawn_biome="forest",
            callbacks=[
                SpeedTestCallback(50),
                AlexAgentCallback(
                    update_interval=update_interval,
                    cond_scale=cond_scale,
                    verbose=verbose,
                    output_dir=output_dir
                ),
                RecordCallback(
                    record_path=output_dir,
                    fps=20,
                    frame_type="pov",
                ),
            ]
        )
        
        print("Loading STEVE-1 model from CraftJarvis/MineStudio_STEVE-1.official...")
        agent_generator = lambda: SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")

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
        with open(info_file, 'w') as f:
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
        max_steps=1000,
        update_interval=1000,
        cond_scale=10.0,
        verbose=True,
    )
