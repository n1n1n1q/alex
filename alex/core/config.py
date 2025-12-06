"""
Configuration management for Alex agent.

Centralizes all environment-based configuration and provides
a clean interface for accessing settings throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlexConfig:
    """Central configuration for Alex agent system."""
    
    # API Keys
    gemini_api_key: Optional[str] = None
    
    # Feature flags
    use_gemini_planner: bool = False
    use_steve_executor: bool = False
    
    # Model paths
    steve_model_path: str = "CraftJarvis/MineStudio_STEVE-1.official"
    mineclip_weights_path: Optional[str] = None
    
    # Device configuration
    device: Optional[str] = None  # None = auto-detect
    
    # Execution parameters
    steve_default_cond_scale: float = 5.0
    steve_default_max_steps: int = 100
    
    # Gemini parameters
    gemini_model_name: str = "gemini-2.0-flash-exp"
    gemini_temperature: float = 0.7
    gemini_max_tokens: int = 1024
    
    # MCP server
    mcp_server_path: str = "mcp_server.py"
    
    # Logging
    verbose: bool = True
    
    @classmethod
    def from_env(cls) -> AlexConfig:
        """
        Create configuration from environment variables.
        
        Environment variables:
            GEMINI_API_KEY: API key for Google Gemini
            USE_GEMINI_PLANNER: Enable Gemini-based planning (true/false)
            USE_STEVE_EXECUTOR: Enable STEVE-1 execution (true/false)
            STEVE_MODEL_PATH: HuggingFace path for STEVE-1 model
            MINECLIP_WEIGHTS_PATH: Path to MineCLIP weights
            DEVICE: Compute device (cpu/cuda/mps)
            VERBOSE: Enable verbose logging (true/false)
        """
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Auto-enable features based on API key availability
        use_gemini = gemini_api_key is not None
        if os.getenv("USE_GEMINI_PLANNER") is not None:
            use_gemini = os.getenv("USE_GEMINI_PLANNER", "").lower() in ("true", "1", "yes")
        
        use_steve = os.getenv("USE_STEVE_EXECUTOR", "false").lower() in ("true", "1", "yes")
        
        mineclip_weights = os.getenv("MINECLIP_WEIGHTS_PATH")
        if mineclip_weights is None:
            # Default to ../models/avg.pth relative to this file
            default_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../models/avg.pth'
            ))
            mineclip_weights = default_path if os.path.exists(default_path) else None
        
        return cls(
            gemini_api_key=gemini_api_key,
            use_gemini_planner=use_gemini,
            use_steve_executor=use_steve,
            steve_model_path=os.getenv("STEVE_MODEL_PATH", cls.steve_model_path),
            mineclip_weights_path=mineclip_weights,
            device=os.getenv("DEVICE"),
            verbose=os.getenv("VERBOSE", "true").lower() in ("true", "1", "yes"),
        )
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages (empty if valid)
        """
        issues = []
        
        if self.use_gemini_planner and not self.gemini_api_key:
            issues.append("Gemini planner enabled but GEMINI_API_KEY not set")
        
        if self.use_steve_executor:
            try:
                import minestudio  # noqa
            except ImportError:
                issues.append("STEVE executor enabled but minestudio not installed")
        
        if self.mineclip_weights_path and not os.path.exists(self.mineclip_weights_path):
            issues.append(f"MineCLIP weights not found: {self.mineclip_weights_path}")
        
        return issues
    
    def print_summary(self) -> None:
        """Print configuration summary."""
        print("=== Alex Configuration ===")
        print(f"  Gemini Planner: {'✓' if self.use_gemini_planner else '✗'}")
        print(f"  STEVE Executor: {'✓' if self.use_steve_executor else '✗'}")
        print(f"  Device: {self.device or 'auto-detect'}")
        print(f"  Verbose: {self.verbose}")
        
        if self.mineclip_weights_path:
            print(f"  MineCLIP Weights: {os.path.basename(self.mineclip_weights_path)}")
        
        issues = self.validate()
        if issues:
            print("\n  ⚠ Configuration Issues:")
            for issue in issues:
                print(f"    - {issue}")
        print("=" * 27)


# Global config instance (lazy loaded)
_config: Optional[AlexConfig] = None


def get_config() -> AlexConfig:
    """Get or create global configuration instance."""
    global _config
    if _config is None:
        _config = AlexConfig.from_env()
    return _config


def set_config(config: AlexConfig) -> None:
    """Set global configuration (useful for testing)."""
    global _config
    _config = config


__all__ = ['AlexConfig', 'get_config', 'set_config']
