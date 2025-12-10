from __future__ import annotations

from typing import Dict, Any, Optional

from ..agent import Agent, VerboseAgent
from ..core.extractor import extract_state

try:
    from minestudio.simulator.callbacks import MinecraftCallback
    _MINESTUDIO_AVAILABLE = True
except ImportError:
    class MinecraftCallback:
        pass
    _MINESTUDIO_AVAILABLE = False


class AlexAgentCallback(MinecraftCallback):
    def __init__(self, 
                 update_interval: int = 50,
                 cond_scale: float = 5.0,
                 verbose: bool = True):
        if _MINESTUDIO_AVAILABLE:
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
            
            if self.current_command == "explore around":
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
                        else:
                            print(f"\n[STEVE-1] Continuing: '{self.current_command}'")
                    else:
                        if action.status == "FAILED":
                            print(f"\n[STEVE-1] Execution failed, keeping: '{self.current_command}'")
                        else:
                            print(f"\n[STEVE-1] No new command, continuing: '{self.current_command}'")
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
