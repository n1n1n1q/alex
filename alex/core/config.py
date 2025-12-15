from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


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

    hf_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_temperature: float = 0.6
    hf_max_tokens: int = 1024
    hf_reflex_model_name: Optional[str] = None
    hf_reflex_temperature: float = 0.35
    hf_reflex_max_tokens: int = 96

    mcp_server_path: str = "mcp_server.py"

    verbose: bool = True

    @classmethod
    def from_config_file(
        cls,
        config_dir: Optional[Path] = None,
        config_name: str = "config",
        overrides: Optional[Sequence[str]] = None,
    ) -> "AlexConfig":
        """
        Load configuration from Hydra YAML (no environment variable side-effects).
        """
        base_dir = config_dir or Path(__file__).resolve().parent.parent.parent / "conf"

        with initialize_config_dir(config_dir=str(base_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=list(overrides or []))

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mineclip_path = cfg_dict.get("mineclip_weights_path")
        if not mineclip_path:
            default_path = os.path.abspath(
                os.path.join(
                    Path(__file__).resolve().parent.parent, "models", "avg.pth"
                )
            )
            cfg_dict["mineclip_weights_path"] = (
                default_path if os.path.exists(default_path) else None
            )

        return cls(**cfg_dict)

    def validate(self) -> list[str]:
        issues = []

        if self.use_steve_executor:
            try:
                import minestudio
            except ImportError:
                issues.append("STEVE executor enabled but minestudio not installed")

        if self.mineclip_weights_path and not os.path.exists(
            self.mineclip_weights_path
        ):
            issues.append(f"MineCLIP weights not found: {self.mineclip_weights_path}")

        return issues

    def print_summary(self) -> None:
        print("=== Alex Configuration ===")
        print(f"  HF Planner: {'enabled' if self.use_hf_planner else 'disabled'}")
        print(
            f"  HF Reflex Manager: {'enabled' if self.use_hf_reflex_manager else 'disabled'}"
        )
        print(
            f"  STEVE Executor: {'enabled' if self.use_steve_executor else 'disabled'}"
        )
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


def get_config(
    config_dir: Optional[Path] = None,
    config_name: str = "config",
    overrides: Optional[Sequence[str]] = None,
) -> AlexConfig:
    """
    Returns a singleton AlexConfig loaded via Hydra.
    """
    global _config
    if _config is None:
        _config = AlexConfig.from_config_file(
            config_dir=config_dir, config_name=config_name, overrides=overrides
        )
    return _config


def set_config(config: AlexConfig) -> None:
    global _config
    _config = config


__all__ = ["AlexConfig", "get_config", "set_config"]
