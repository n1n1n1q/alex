"""
Benchmark configuration loader.
Loads configuration from YAML file and provides easy access to parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default benchmark_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "benchmark_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_path(self, model_name: str) -> str:
        """Get the path/ID for a model."""
        return self.config['models'][model_name]['path']
    
    def get_model_weights(self, model_name: str) -> Optional[str]:
        """Get the weights path for a model (if any)."""
        weights = self.config['models'][model_name].get('weights', '')
        return weights if weights else None
    
    def get_output_dir(self) -> str:
        """Get the output directory for results."""
        return self.config['general']['output_dir']
    
    def get_device(self) -> str:
        """Get the device to run on (cuda/cpu)."""
        return self.config['general']['device']
    
    def get_task_trials(self, task_name: str, model_name: Optional[str] = None) -> int:
        """
        Get number of trials for a task.
        
        Args:
            task_name: Name of the task (e.g., 'crafting_table')
            model_name: Optional model name to check for overrides
        
        Returns:
            Number of trials
        """
        # Check for model-specific override
        if model_name and 'overrides' in self.config:
            override = (self.config['overrides']
                       .get(model_name, {})
                       .get(task_name, {})
                       .get('trials'))
            if override:
                return override
        
        # Return default from task config
        return self.config['tasks'][task_name]['trials']
    
    def get_task_max_steps(self, task_name: str, model_name: Optional[str] = None) -> int:
        """
        Get max steps for a task.
        
        Args:
            task_name: Name of the task (e.g., 'crafting_table')
            model_name: Optional model name to check for overrides
        
        Returns:
            Maximum number of steps
        """
        # Check for model-specific override
        if model_name and 'overrides' in self.config:
            override = (self.config['overrides']
                       .get(model_name, {})
                       .get(task_name, {})
                       .get('max_steps'))
            if override:
                return override
        
        # Return default from task config
        return self.config['tasks'][task_name]['max_steps']
    
    def get_dirt_trials(self, model_name: Optional[str] = None) -> int:
        """Get number of trials for dirt mining benchmark."""
        # Check for model-specific override
        if model_name and 'overrides' in self.config:
            override = (self.config['overrides']
                       .get(model_name, {})
                       .get('dirt_mining', {})
                       .get('trials'))
            if override:
                return override
        
        return self.config['dirt_mining']['trials']
    
    def get_dirt_max_steps(self, model_name: Optional[str] = None) -> int:
        """Get max steps for dirt mining benchmark."""
        # Check for model-specific override
        if model_name and 'overrides' in self.config:
            override = (self.config['overrides']
                       .get(model_name, {})
                       .get('dirt_mining', {})
                       .get('max_steps'))
            if override:
                return override
        
        return self.config['dirt_mining']['max_steps']
    
    def get_all_tasks(self) -> list:
        """Get list of all task names."""
        return list(self.config['tasks'].keys())
    
    def get_all_models(self) -> list:
        """Get list of all model names."""
        return list(self.config['models'].keys())
    
    def print_config(self):
        """Print configuration summary."""
        print("="*80)
        print("BENCHMARK CONFIGURATION")
        print("="*80)
        print("\nModels:")
        for model_name, model_config in self.config['models'].items():
            print(f"  {model_name:10s}: {model_config['path']}")
        
        print(f"\nOutput Directory: {self.get_output_dir()}")
        print(f"Device: {self.get_device()}")
        
        print("\nTask Configurations:")
        for task_name, task_config in self.config['tasks'].items():
            print(f"  {task_name}:")
            print(f"    Trials: {task_config['trials']}, Max Steps: {task_config['max_steps']}")
        
        print("\nDirt Mining:")
        print(f"  Trials: {self.config['dirt_mining']['trials']}, "
              f"Max Steps: {self.config['dirt_mining']['max_steps']}")
        
        if 'overrides' in self.config and self.config['overrides']:
            print("\nModel-Specific Overrides:")
            for model_name, overrides in self.config['overrides'].items():
                if overrides:
                    print(f"  {model_name}:")
                    for task, params in overrides.items():
                        print(f"    {task}: {params}")
        
        print("="*80)


if __name__ == "__main__":
    # Test loading configuration
    config = BenchmarkConfig()
    config.print_config()
