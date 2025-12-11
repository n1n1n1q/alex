from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlexConfig:

    use_hf_planner: bool = False
    use_hf_reflex_manager: bool = False
    use_steve_executor: bool = False
    
    steve_model_path: str = "CraftJarvis/MineStudio_STEVE-1.official"
    mineclip_weights_path: Optional[str] = None
    
    device: Optional[str] = None
    
    steve_default_cond_scale: float = 5.0
    steve_default_max_steps: int = 100
    
    hf_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_temperature: float = 0.6
    hf_max_tokens: int = 1024
    hf_reflex_model_name: Optional[str] = None
    hf_reflex_temperature: float = 0.35
    hf_reflex_max_tokens: int = 96
    
    mcp_server_path: str = "mcp_server.py"
    
    verbose: bool = True
    
    @classmethod
    def from_env(cls) -> AlexConfig:

        use_hf = os.getenv("USE_HF_PLANNER", "false").lower() in ("true", "1", "yes")
        use_hf_reflex = os.getenv("USE_HF_REFLEX_MANAGER", "false").lower() in ("true", "1", "yes")
        
        use_steve = os.getenv("USE_STEVE_EXECUTOR", "false").lower() in ("true", "1", "yes")
        
        mineclip_weights = os.getenv("MINECLIP_WEIGHTS_PATH")
        if mineclip_weights is None:
            default_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../models/avg.pth'
            ))
            mineclip_weights = default_path if os.path.exists(default_path) else None
        
        return cls(
            use_hf_planner=use_hf,
            use_hf_reflex_manager=use_hf_reflex,
            use_steve_executor=use_steve,
            steve_model_path=os.getenv("STEVE_MODEL_PATH", cls.steve_model_path),
            mineclip_weights_path=mineclip_weights,
            device=os.getenv("DEVICE"),
            hf_model_name=os.getenv("HF_MODEL_NAME", cls.hf_model_name),
            hf_reflex_model_name=os.getenv("HF_REFLEX_MODEL_NAME"),
            hf_reflex_temperature=float(os.getenv("HF_REFLEX_TEMPERATURE", cls.hf_reflex_temperature)),
            hf_reflex_max_tokens=int(os.getenv("HF_REFLEX_MAX_TOKENS", cls.hf_reflex_max_tokens)),
            verbose=os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes"),
        )
    
    def validate(self) -> list[str]:

        issues = []
        
        if self.use_steve_executor:
            try:
                import minestudio
            except ImportError:
                issues.append("STEVE executor enabled but minestudio not installed")
        
        if self.mineclip_weights_path and not os.path.exists(self.mineclip_weights_path):
            issues.append(f"MineCLIP weights not found: {self.mineclip_weights_path}")
        
        return issues
    
    def print_summary(self) -> None:
        print("=== Alex Configuration ===")
        print(f"  HF Planner: {'enabled' if self.use_hf_planner else 'disabled'}")
        print(f"  HF Reflex Manager: {'enabled' if self.use_hf_reflex_manager else 'disabled'}")
        print(f"  STEVE Executor: {'enabled' if self.use_steve_executor else 'disabled'}")
        print(f"  Device: {self.device or 'auto-detect'}")
        print(f"  Verbose: {self.verbose}")
        
        if self.mineclip_weights_path:
            print(f"  MineCLIP Weights: {os.path.basename(self.mineclip_weights_path)}")
        
        issues = self.validate()
        if issues:
            print("\n  Configuration Issues:")
            for issue in issues:
                print(f"    - {issue}")
        print("=" * 27)


_config: Optional[AlexConfig] = None


def get_config() -> AlexConfig:

    global _config
    if _config is None:
        _config = AlexConfig.from_env()
    return _config


def set_config(config: AlexConfig) -> None:

    global _config
    _config = config


__all__ = ['AlexConfig', 'get_config', 'set_config']
