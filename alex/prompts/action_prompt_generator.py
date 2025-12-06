"""
Action Prompt Generator

Converts high-level SkillRequests into short, STEVE-1-compatible text prompts.

STEVE-1 requires specific prompt formats:
- 2-3 word commands
- Goal-oriented (not pure movement)
- Minecraft terminology
- Object-focused behavior

Examples:
- SkillRequest(name="gather_wood") -> "mine log"
- SkillRequest(name="hunt_mob", params={"mob": "cow"}) -> "kill cow"
- SkillRequest(name="craft_table") -> "craft table"

The LLM generates these prompts based on the skill name and parameters,
using few-shot examples and Minecraft knowledge.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import os

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Few-shot examples for prompt generation
PROMPT_GENERATION_EXAMPLES = """
# Examples of converting SkillRequests to STEVE-1 prompts:

SkillRequest: {"name": "gather_wood", "params": {}}
STEVE Prompt: mine log

SkillRequest: {"name": "collect_dirt", "params": {"count": 10}}
STEVE Prompt: mine dirt

SkillRequest: {"name": "hunt_mob", "params": {"mob": "cow"}}
STEVE Prompt: kill cow

SkillRequest: {"name": "hunt_mob", "params": {"mob": "pig"}}
STEVE Prompt: kill pig

SkillRequest: {"name": "fight_mob", "params": {"mob": "zombie"}}
STEVE Prompt: kill zombie

SkillRequest: {"name": "craft_table", "params": {}}
STEVE Prompt: craft table

SkillRequest: {"name": "build_shelter", "params": {}}
STEVE Prompt: place blocks

SkillRequest: {"name": "mine_stone", "params": {}}
STEVE Prompt: mine stone

SkillRequest: {"name": "gather_food", "params": {}}
STEVE Prompt: kill chicken

SkillRequest: {"name": "retreat", "params": {}}
STEVE Prompt: escape danger

SkillRequest: {"name": "collect_wood", "params": {}}
STEVE Prompt: obtain log

SkillRequest: {"name": "idle_scan", "params": {}}
STEVE Prompt: look around
"""

SYSTEM_PROMPT = """You are an expert at converting high-level Minecraft skills into short, actionable STEVE-1 prompts.

STEVE-1 is a trained Minecraft policy that responds to short 2-3 word goal-oriented commands.

CRITICAL RULES:
1. Use ONLY 2-3 words maximum
2. Use Minecraft terminology (log, dirt, zombie, craft, mine, kill)
3. Be GOAL-ORIENTED (object-focused), NOT movement-focused
4. Avoid pure movement commands ("move forward", "run", "walk")
5. Focus on tangible objects and actions

Good prompts:
- "mine log" (gathering wood)
- "kill cow" (hunting)
- "craft table" (crafting)
- "mine stone" (mining)

Bad prompts:
- "move forward and collect wood" (too long, movement-focused)
- "go to forest" (too vague, movement)
- "explore area" (not object-focused)

Convert the SkillRequest to a short STEVE-1 prompt. Return ONLY the prompt, nothing else."""


class ActionPromptGenerator:
    """Generates STEVE-1 compatible prompts from SkillRequests using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, verbose: bool = True):
        """
        Initialize prompt generator.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            verbose: Whether to print thought process
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not available. Install with: pip install google-generativeai")
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.verbose = verbose
        
    def generate_prompt(
        self,
        skill_name: str,
        skill_params: Dict[str, Any],
    ) -> str:
        """
        Generate STEVE-1 prompt from skill request.
        
        Args:
            skill_name: Name of the skill (e.g., "gather_wood", "hunt_mob")
            skill_params: Skill parameters (e.g., {"mob": "cow", "count": 5})
            
        Returns:
            Short STEVE-1 prompt (e.g., "mine log", "kill cow")
        """
        if self.verbose:
            print(f"\n[STEVE-1 Prompt Generator]")
            print(f"  Skill: {skill_name}")
            print(f"  Params: {skill_params}")
        
        # Build user prompt
        user_prompt = f"""Convert this SkillRequest to a STEVE-1 prompt:

SkillRequest: {{"name": "{skill_name}", "params": {skill_params}}}

Remember: 2-3 words maximum, goal-oriented, Minecraft terminology.

Here are some examples:
{PROMPT_GENERATION_EXAMPLES}

STEVE Prompt:"""
        
        # Generate with Gemini
        try:
            response = self.model.generate_content(
                [
                    {"role": "user", "parts": [SYSTEM_PROMPT]},
                    {"role": "model", "parts": ["I understand. I will generate short 2-3 word STEVE-1 prompts that are goal-oriented and use Minecraft terminology."]},
                    {"role": "user", "parts": [user_prompt]},
                ],
                generation_config={
                    "temperature": 0.3,  # Low temperature for consistency
                    "max_output_tokens": 20,  # Very short output
                }
            )
            
            prompt = response.text.strip()
            
            # Clean up common issues
            prompt = prompt.replace('"', '').replace("'", '')
            prompt = prompt.split('\n')[0]  # Take first line only
            
            # Validate length (should be 2-5 words)
            words = prompt.split()
            if len(words) > 5:
                # Truncate to first 3 words
                prompt = ' '.join(words[:3])
            
            if self.verbose:
                print(f"  → Generated STEVE Prompt: '{prompt}'")
            
            return prompt
            
        except Exception as e:
            if self.verbose:
                print(f"  ⚠ LLM failed ({e}), using fallback")
            # Fallback to simple mapping if LLM fails
            fallback = self._fallback_prompt(skill_name, skill_params)
            if self.verbose:
                print(f"  → Fallback STEVE Prompt: '{fallback}'")
            return fallback
    
    def _fallback_prompt(self, skill_name: str, skill_params: Dict[str, Any]) -> str:
        """Fallback to rule-based prompt generation if LLM fails."""
        # Simple mapping for common skills
        mapping = {
            "gather_wood": "mine log",
            "collect_wood": "mine log",
            "collect_dirt": "mine dirt",
            "mine_stone": "mine stone",
            "craft_table": "craft table",
            "hunt_mob": f"kill {skill_params.get('mob', 'cow')}",
            "fight_mob": f"kill {skill_params.get('mob', 'zombie')}",
            "gather_food": "kill chicken",
            "retreat": "escape danger",
            "build_shelter": "place blocks",
            "idle_scan": "look around",
        }
        
        return mapping.get(skill_name, "mine dirt")  # Default fallback


__all__ = [
    "ActionPromptGenerator",
    "GEMINI_AVAILABLE",
]
