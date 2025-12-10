import os
import sys
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

from alex.agent import Agent
from alex.core.extractor import extract_state


class VerboseAgent(Agent):

    def step(self, raw_obs):

        state = extract_state(raw_obs)
        
        print(f"  [Reflex] Checking for urgent situations...")
        reflex_goal = self.reflex.detect(state)
        if reflex_goal is not None:
            print(f"  [Reflex] TRIGGERED: {reflex_goal}")
            skill_req = self.router.to_skill(reflex_goal)
            print(f"  [Router] Mapped to skill: {skill_req}")
            from alex.execution.policy_executor import execute_policy_skill
            result = execute_policy_skill(skill_req, env_obs=raw_obs)
            print(f"  [Executor] Result: {result.get('status', 'unknown')}")
            from alex.core.types import SkillResult
            return SkillResult(**result)
        
        print(f"  [Reflex] No urgent situations detected")
        
        print(f"  [Planner] Analyzing state and generating subgoals...")
        subgoals = self.planner.plan(state)
        print(f"  [Planner] Generated {len(subgoals)} subgoal(s):")
        for sg in subgoals:
            print(f"    - {sg.name} (priority={sg.priority}, params={sg.params})")
        
        print(f"  [MetaPlanner] Updating backlog...")
        backlog = self.metaplanner.update(subgoals)
        print(f"  [MetaPlanner] Current backlog ({len(backlog)} items):")
        for i, sg in enumerate(backlog[:5], 1):
            print(f"    {i}. {sg.name} (priority={sg.priority})")
        if len(backlog) > 5:
            print(f"    ... and {len(backlog) - 5} more")
        
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            print(f"  [MetaPlanner] No goals in backlog - nothing to do")
            from alex.core.types import SkillResult
            return SkillResult(status="OK", info={"note": "nothing to do"})
        
        print(f"  [MetaPlanner] Selected next goal: {next_goal.name}")
        
        print(f"  [Router] Routing goal to skill...")
        skill_req = self.router.to_skill(next_goal)
        print(f"  [Router] Skill request: {skill_req}")
        
        print(f"  [Executor] Executing skill...")
        from alex.execution.policy_executor import execute_policy_skill
        result = execute_policy_skill(skill_req, env_obs=raw_obs)
        print(f"  [Executor] Status: {result.get('status', 'unknown')}")
        if 'steve_prompt' in result:
            print(f"  [Executor] STEVE-1 Prompt: '{result['steve_prompt']}'")
        
        from alex.core.types import SkillResult
        return SkillResult(**result)


class AlexAgentCallback(MinecraftCallback):

    def __init__(self, 
                 update_interval: int = 50,
                 cond_scale: float = 5.0,
                 verbose: bool = True):

        super().__init__()
        self.agent = VerboseAgent() if verbose else Agent()
        self.update_interval = update_interval
        self.cond_scale = cond_scale
        self.verbose = verbose
        self.timestep = 0
        self.current_command = "explore around"
        
    def after_reset(self, sim, obs, info):

        self.timestep = 0
        
        try:
            print("\n" + "="*80)
            print("AGENT RESET - Getting Initial Command")
            print("="*80)
            
            state = extract_state(info)
            
            print(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
            health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
            hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
            print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
            print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
            print(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
            
            print(f"\n[AGENT] Processing state through pipeline...")
            action = self.agent.step(obs)
            
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
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to get initial command: {e}")
            import traceback
            traceback.print_exc()
            self.current_command = "explore around"
        
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.current_command
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
                action = self.agent.step(obs)
                
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
                
                print("="*80 + "\n")
                        
            except Exception as e:
                print(f"\n[ERROR t={self.timestep}] Failed to update agent: {e}")
                import traceback
                traceback.print_exc()
        
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.current_command
        }
        
        return obs, reward, terminated, truncated, info


def run_agent_with_recording(
    num_episodes: int = 3,
    max_steps: int = 1000,
    update_interval: int = 50,
    cond_scale: float = 5.0,
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
                    verbose=verbose
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


def run_simple_loop_no_pipeline(
    max_steps: int = 500,
    update_interval: int = 50,
    cond_scale: float = 5.0,
    output_dir: str = None,
    description: str = "simple_loop",
    verbose: bool = True,
):

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_desc = "".join(c if c.isalnum() else "_" for c in description[:30])
        output_dir = f"./recordings/{safe_desc}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SIMPLE ALEX AGENT LOOP")
    print("=" * 80)
    print(f"Description: {description}")
    print(f"Steps:       {max_steps}")
    print(f"Output:      {output_dir}")
    print("=" * 80)
    print()
    
    print("Creating environment...")
    env = MinecraftSim(
        obs_size=(128, 128),
        preferred_spawn_biome="forest",
        callbacks=[
            SpeedTestCallback(50),
            AlexAgentCallback(
                update_interval=update_interval,
                cond_scale=cond_scale,
                verbose=verbose
            ),
            RecordCallback(
                record_path=output_dir,
                fps=20,
                frame_type="pov",
            ),
        ]
    )
    
    print("Loading STEVE-1 model...")
    policy = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")
    
    print("\n" + "=" * 80)
    print("RUNNING EPISODE...")
    print("=" * 80)
    print()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    step = 0
    
    try:
        while not (terminated or truncated) and step < max_steps:
            action = policy.get_action(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            if step % 100 == 0:
                print(f"Step {step}/{max_steps}")
                
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping episode...")
    
    finally:
        env.close()
        
    print("\n" + "=" * 80)
    print("EPISODE COMPLETE!")
    print("=" * 80)
    print(f"Total steps: {step}")
    print(f"Recording saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    run_agent_with_recording(
        description="wood_and_crafting_table",
        num_episodes=3,
        max_steps=3000,
        update_interval=50,
        cond_scale=5.0,
        verbose=True,
    )
