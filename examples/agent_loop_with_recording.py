"""
Alex Agent Loop with Video Recording

Runs the full Alex agent (vision + planning + STEVE-1 execution) in a loop
and records all episodes as videos for analysis.

Requirements:
- export USE_STEVE_EXECUTOR=true
- export GEMINI_API_KEY=your_key
- pip install git+https://github.com/annastasyshyn/MineStudio.git

Usage:
    python examples/agent_loop_with_recording.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from functools import partial
import ray

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import MinecraftCallback, SpeedTestCallback, RecordCallback
from minestudio.models import SteveOnePolicy
from minestudio.inference import EpisodePipeline, MineGenerator

# Import alex components
from alex.agent import Agent
from alex.extractor import extract_state


class VerboseAgent(Agent):
    """Agent wrapper that prints its internal decision-making process."""
    
    def step(self, raw_obs):
        """Override step to add verbose logging."""
        state = extract_state(raw_obs)
        
        # Check reflex
        print(f"  [Reflex] Checking for urgent situations...")
        reflex_goal = self.reflex.detect(state)
        if reflex_goal is not None:
            print(f"  [Reflex] 🚨 TRIGGERED: {reflex_goal}")
            skill_req = self.router.to_skill(reflex_goal)
            print(f"  [Router] Mapped to skill: {skill_req}")
            from alex.policy_executor import execute_policy_skill
            result = execute_policy_skill(skill_req, env_obs=raw_obs)
            print(f"  [Executor] Result: {result.get('status', 'unknown')}")
            from alex.types import SkillResult
            return SkillResult(**result)
        
        print(f"  [Reflex] No urgent situations detected")
        
        # Run planner
        print(f"  [Planner] Analyzing state and generating subgoals...")
        subgoals = self.planner.plan(state)
        print(f"  [Planner] Generated {len(subgoals)} subgoal(s):")
        for sg in subgoals:
            print(f"    - {sg.name} (priority={sg.priority}, params={sg.params})")
        
        # Update metaplanner
        print(f"  [MetaPlanner] Updating backlog...")
        backlog = self.metaplanner.update(subgoals)
        print(f"  [MetaPlanner] Current backlog ({len(backlog)} items):")
        for i, sg in enumerate(backlog[:5], 1):  # Show first 5
            print(f"    {i}. {sg.name} (priority={sg.priority})")
        if len(backlog) > 5:
            print(f"    ... and {len(backlog) - 5} more")
        
        # Pop next goal
        next_goal = self.metaplanner.pop_next()
        if next_goal is None:
            print(f"  [MetaPlanner] ⚠️ No goals in backlog - nothing to do")
            from alex.types import SkillResult
            return SkillResult(status="OK", info={"note": "nothing to do"})
        
        print(f"  [MetaPlanner] 🎯 Selected next goal: {next_goal.name}")
        
        # Route to skill
        print(f"  [Router] Routing goal to skill...")
        skill_req = self.router.to_skill(next_goal)
        print(f"  [Router] Skill request: {skill_req}")
        
        # Execute
        print(f"  [Executor] Executing skill...")
        from alex.policy_executor import execute_policy_skill
        result = execute_policy_skill(skill_req, env_obs=raw_obs)
        print(f"  [Executor] Status: {result.get('status', 'unknown')}")
        if 'steve_prompt' in result:
            print(f"  [Executor] 💬 STEVE-1 Prompt: '{result['steve_prompt']}'")
        
        from alex.types import SkillResult
        return SkillResult(**result)


class AlexAgentCallback(MinecraftCallback):
    """Integrates Alex agent with STEVE-1 execution."""
    
    def __init__(self, 
                 update_interval: int = 50,
                 cond_scale: float = 5.0,
                 verbose: bool = True):
        """
        Args:
            update_interval: How often to update agent planning (steps)
            cond_scale: STEVE-1 conditioning scale
            verbose: Whether to print detailed agent thinking process
        """
        super().__init__()
        # Use VerboseAgent for detailed logging
        self.agent = VerboseAgent() if verbose else Agent()
        self.update_interval = update_interval
        self.cond_scale = cond_scale
        self.verbose = verbose
        self.timestep = 0
        self.current_command = "explore around"  # Default command
        
    def after_reset(self, sim, obs, info):
        """Reset agent state and get initial command."""
        self.timestep = 0
        
        # Get initial state and command
        try:
            print("\n" + "="*80)
            print("AGENT RESET - Getting Initial Command")
            print("="*80)
            
            state = extract_state(info)
            
            # Show extracted state
            print(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
            health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
            hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
            print(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
            print(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
            print(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
            
            # Get agent action (pass obs dict which has the image for STEVE-1)
            print(f"\n[AGENT] Processing state through pipeline...")
            action = self.agent.step(obs)
            
            # Show action details
            print(f"[AGENT] Action status: {action.status}")
            if hasattr(action, 'info') and action.info:
                print(f"[AGENT] Action info: {action.info}")
                # Check if execution returned steve_prompt in info
                if 'steve_prompt' in action.info:
                    self.current_command = action.info['steve_prompt']
                    print(f"\n[STEVE-1] Command from action: '{self.current_command}'")
            
            # If agent provides a STEVE prompt attribute, use it
            if hasattr(action, 'steve_prompt') and action.steve_prompt:
                self.current_command = action.steve_prompt
                print(f"\n[STEVE-1] Command: '{self.current_command}'")
            elif self.current_command == "explore around":
                # Still no command set
                print(f"\n[STEVE-1] Default command: '{self.current_command}'")
            
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to get initial command: {e}")
            import traceback
            traceback.print_exc()
            self.current_command = "explore around"
        
        # Add command to observation
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.current_command
        }
        
        return obs, info
    
    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Update agent planning periodically and inject commands."""
        self.timestep += 1
        
        # Update agent planning at intervals
        if self.timestep % self.update_interval == 0:
            try:
                print("\n" + "="*80)
                print(f"AGENT REPLANNING - Step {self.timestep}")
                print("="*80)
                
                state = extract_state(info)
                
                # Show current state (safely)
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
                
                # Get agent decision (pass obs dict which has the image for STEVE-1)
                print(f"\n[AGENT] Running planner...")
                action = self.agent.step(obs)
                
                # Show planning results
                print(f"[AGENT] Action status: {action.status}")
                if hasattr(action, 'info') and action.info:
                    print(f"[AGENT] Action info: {action.info}")
                    # Check if execution returned steve_prompt in info
                    if 'steve_prompt' in action.info:
                        old_command = self.current_command
                        self.current_command = action.info['steve_prompt']
                        if old_command != self.current_command:
                            print(f"\n[STEVE-1] 🔄 COMMAND CHANGED (from info)")
                            print(f"[STEVE-1]   Old: '{old_command}'")
                            print(f"[STEVE-1]   New: '{self.current_command}'")
                
                # Update command if agent provides new STEVE prompt attribute
                if hasattr(action, 'steve_prompt') and action.steve_prompt:
                    old_command = self.current_command
                    self.current_command = action.steve_prompt
                    if old_command != self.current_command:
                        print(f"\n[STEVE-1] 🔄 COMMAND CHANGED")
                        print(f"[STEVE-1]   Old: '{old_command}'")
                        print(f"[STEVE-1]   New: '{self.current_command}'")
                    else:
                        print(f"\n[STEVE-1] ✓ Continuing: '{self.current_command}'")
                else:
                    if action.status == "FAILED":
                        print(f"\n[STEVE-1] ⚠️ Execution failed, keeping: '{self.current_command}'")
                    else:
                        print(f"\n[STEVE-1] No new command, continuing: '{self.current_command}'")
                
                print("="*80 + "\n")
                        
            except Exception as e:
                print(f"\n[ERROR t={self.timestep}] Failed to update agent: {e}")
                import traceback
                traceback.print_exc()
        
        # Always inject current command into observation
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
    """
    Run Alex agent with STEVE-1 execution and record episodes.
    
    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        update_interval: How often agent updates planning (steps)
        cond_scale: STEVE-1 conditioning scale (2.0-8.0)
        output_dir: Directory to save recordings (auto-generated if None)
        description: Description for output directory naming
        verbose: Print detailed agent thinking process
    
    Returns:
        Summary dictionary with results
    """
    
    # Create output directory with timestamp
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
    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init()
    
    try:
        
        # Create environment generator
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
        
        # Create STEVE-1 agent generator
        print("Loading STEVE-1 model from CraftJarvis/MineStudio_STEVE-1.official...")
        agent_generator = lambda: SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")
        
        # Worker configuration
        worker_kwargs = dict(
            env_generator=env_generator,
            agent_generator=agent_generator,
            num_max_steps=max_steps,
            num_episodes=num_episodes,
            tmpdir=output_dir,
            image_media="h264",  # Save as H.264 MP4 videos
        )
        
        # Create pipeline
        print("Setting up inference pipeline...")
        pipeline = EpisodePipeline(
            episode_generator=MineGenerator(
                num_workers=1,
                num_gpus=0.25,  # Use 1/4 of a GPU
                max_restarts=3,
                **worker_kwargs,
            ),
        )
        
        # Run the experiment
        print("\n" + "=" * 80)
        print("RUNNING AGENT LOOP...")
        print("=" * 80)
        print()
        
        summary = pipeline.run()
        
        # Print results
        print("\n" + "=" * 80)
        print("AGENT LOOP COMPLETE!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Total Episodes:    {summary.get('num_episodes', 0)}")
        print(f"  Avg Episode Length: {summary.get('avg_length', 'N/A')}")
        print(f"\nRecordings saved to: {output_dir}")
        print("=" * 80)
        
        # Save experiment info
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
    """
    Simple agent loop without Ray pipeline (easier for debugging).
    
    Args:
        max_steps: Maximum steps to run
        update_interval: How often agent updates planning (steps)
        cond_scale: STEVE-1 conditioning scale
        output_dir: Directory to save recordings
        description: Description for output directory naming
        verbose: Print detailed agent thinking process
    """
    
    # Create output directory
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
    
    # Create environment with callbacks
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
    
    # Load STEVE-1 policy
    print("Loading STEVE-1 model...")
    policy = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official")
    
    # Run episode
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
            # Get action from STEVE-1
            action = policy.get_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Print progress
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
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Option 1: Full pipeline with multiple episodes (recommended for experiments)
    run_agent_with_recording(
        description="wood_and_crafting_table",
        num_episodes=3,
        max_steps=3000,  # Increased from 1000 to allow more time
        update_interval=50,  # Agent replans every 50 steps
        cond_scale=5.0,
        verbose=True,  # Show detailed agent thinking
    )
    
    # Option 2: Simple single episode (good for debugging)
    # run_simple_loop_no_pipeline(
    #     description="simple_test",
    #     max_steps=500,
    #     update_interval=50,
    #     cond_scale=5.0,
    #     verbose=True,
    # )
    
    # ========================================================================
    # PARAMETERS EXPLAINED:
    # ========================================================================
    # description:      Label for the experiment (used in directory name)
    # num_episodes:     How many episodes to run (full pipeline only)
    # max_steps:        Maximum steps per episode
    # update_interval:  How often Alex agent replans (in steps)
    #                   - Lower = more responsive but more LLM calls
    #                   - Higher = more consistent but less adaptive
    # cond_scale:       STEVE-1 conditioning strength (2.0-8.0)
    #                   - Higher = stricter command following
    #                   - Lower = more exploration
    #
    # ========================================================================
    # NOTES:
    # ========================================================================
    # - The Alex agent has a built-in planner that currently follows a simple
    #   hardcoded strategy: collect wood -> craft table -> idle scan
    # - To customize agent behavior, modify alex/planner.py
    # - Agent uses its vision system to analyze scenes and make decisions
    # - STEVE-1 executes the low-level motor commands generated by the agent
    #
    # ========================================================================
    # TIPS:
    # ========================================================================
    # - Videos are saved as episode_0.mp4, episode_1.mp4, etc.
    # - Check experiment_info.txt for run details
    # - Use VLC or similar to watch recordings: vlc recordings/*/episode_0.mp4
    # - For debugging, use run_simple_loop_no_pipeline()
    # - If agent gets stuck, try lower update_interval (e.g., 30)
    # - If commands aren't followed, increase cond_scale (e.g., 6.0)
