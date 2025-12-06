"""
Prompt generation and MCP server components.

This module contains prompt generators, few-shot examples,
and MCP server integration.
"""

# Conditional import for action prompt generator
try:
    from .action_prompt_generator import ActionPromptGenerator, GEMINI_AVAILABLE
    __all__ = [
        'ActionPromptGenerator',
        'GEMINI_AVAILABLE',
    ]
except ImportError:
    __all__ = []
