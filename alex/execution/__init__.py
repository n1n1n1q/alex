"""
Execution components for the Alex agent.

This module contains policy executors and STEVE-1 integration for
low-level action generation.
"""

from .policy_executor import execute_policy_skill

# Conditional import for STEVE executor
try:
    from .steve_executor import SteveExecutor, CommandCallback, STEVE_AVAILABLE
    __all__ = [
        'execute_policy_skill',
        'SteveExecutor',
        'CommandCallback',
        'STEVE_AVAILABLE',
    ]
except ImportError:
    __all__ = [
        'execute_policy_skill',
    ]
