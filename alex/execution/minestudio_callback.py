from __future__ import annotations

from typing import Dict, Any, Optional
import os
import datetime
try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:
    _CV2_AVAILABLE = False

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
                 verbose: bool = True,
                 log_dir: Optional[str] = None):
        if _MINESTUDIO_AVAILABLE:
            super().__init__()
        self.agent = VerboseAgent() if verbose else Agent()
        self.update_interval = update_interval
        self.cond_scale = cond_scale
        self.verbose = verbose
        self.timestep = 0
        self.current_command = "explore around"
        # Prepare logging file (if verbose)
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        if verbose:
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception:
                pass
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = os.path.join(self.log_dir, f"agent_{ts}.log")
            try:
                self._log_fh = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                self._log_fh = None
        else:
            self._log_fh = None
    
    def after_reset(self, sim, obs, info):
        self.timestep = 0
        
        try:
            self._log("\n" + "="*80)
            self._log("AGENT RESET - Getting Initial Command")
            self._log("="*80)
            
            state = extract_state(info)
            
            self._log(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
            health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
            hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
            self._log(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
            self._log(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
            self._log(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
        
            self._log(f"\n[AGENT] Processing state through pipeline...")
            action = self.agent.step(obs)
            
            self._log(f"[AGENT] Action status: {action.status}")
            if hasattr(action, 'info') and action.info:
                self._log(f"[AGENT] Action info: {action.info}")
                if 'steve_prompt' in action.info:
                    self.current_command = action.info['steve_prompt']
                    self._log(f"\n[STEVE-1] Command from action: '{self.current_command}'")
            
            if self.current_command == "explore around":
                self._log(f"\n[STEVE-1] Default command: '{self.current_command}'")
            
            self._log("="*80 + "\n")
            
        except Exception as e:
                self._log(f"\n[ERROR] Failed to get initial command: {e}")
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
                self._log("\n" + "="*80)
                self._log(f"AGENT REPLANNING - Step {self.timestep}")
                self._log("="*80)
                
                state = extract_state(info)
                
                self._log(f"\n[STATE] Inventory: {state.inventory_agg if state.inventory_agg else {}}")
                health_str = f"{state.health:.1f}" if state.health is not None else "N/A"
                hunger_str = f"{state.hunger}" if state.hunger is not None else "N/A"
                self._log(f"[STATE] Health: {health_str}, Hunger: {hunger_str}")
                if state.player_pos and isinstance(state.player_pos, dict):
                    x = state.player_pos.get('x', 0)
                    y = state.player_pos.get('y', 0)
                    z = state.player_pos.get('z', 0)
                    self._log(f"[STATE] Position: ({x:.1f}, {y:.1f}, {z:.1f})")
                self._log(f"[STATE] Nearby mobs: {len(state.mobs) if state.mobs else 0}")
                self._log(f"[STATE] Nearby blocks: {len(state.blocks) if state.blocks else 0}")
                
                self._log(f"\n[AGENT] Running planner...")
                action = self.agent.step(obs)
                
                self._log(f"[AGENT] Action status: {action.status}")
                if hasattr(action, 'info') and action.info:
                    self._log(f"[AGENT] Action info: {action.info}")
                    if 'steve_prompt' in action.info:
                        old_command = self.current_command
                        self.current_command = action.info['steve_prompt']
                        if old_command != self.current_command:
                            self._log(f"\n[STEVE-1] COMMAND CHANGED (from info)")
                            self._log(f"[STEVE-1]   Old: '{old_command}'")
                            self._log(f"[STEVE-1]   New: '{self.current_command}'")
                        else:
                            self._log(f"\n[STEVE-1] Continuing: '{self.current_command}'")
                    else:
                        if action.status == "FAILED":
                            self._log(f"\n[STEVE-1] Execution failed, keeping: '{self.current_command}'")
                        else:
                            self._log(f"\n[STEVE-1] No new command, continuing: '{self.current_command}'")
                else:
                    if action.status == "FAILED":
                        self._log(f"\n[STEVE-1] Execution failed, keeping: '{self.current_command}'")
                    else:
                        self._log(f"\n[STEVE-1] No new command, continuing: '{self.current_command}'")
                
                self._log("="*80 + "\n")
                        
            except Exception as e:
                self._log(f"\n[ERROR t={self.timestep}] Failed to update agent: {e}")
                import traceback
                traceback.print_exc()
        
        if 'condition' not in obs:
            obs['condition'] = {}
        obs['condition'] = {
            "cond_scale": self.cond_scale,
            "text": self.current_command
        }
        
        return obs, reward, terminated, truncated, info

    def before_render(self, sim, image):
        """Overlay the current command/action on the frame before render."""
        if not self.verbose:
            return image
        try:
            if image is None:
                return image
            if _CV2_AVAILABLE:
                arr = image
                h, w = arr.shape[:2]
                overlay = arr.copy()
                cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, arr, 1 - alpha, 0, arr)
                text = f"Cmd: {self.current_command}"
                cv2.putText(arr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                return arr
            else:
                return image
        except Exception as e:
            self._log(f"[ERROR] before_render overlay failed: {e}")
            return image

    def _log(self, msg: str):
        # Print (if verbose) and also write to log file
        try:
            if self.verbose:
                print(msg)
            if getattr(self, "_log_fh", None):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._log_fh.write(f"[{timestamp}] {msg}\n")
                self._log_fh.flush()
        except Exception:
            try:
                print(f"[LOGGING ERROR] {msg}")
            except Exception:
                pass

    def __del__(self):
        try:
            if getattr(self, "_log_fh", None):
                self._log_fh.close()
        except Exception:
            pass
