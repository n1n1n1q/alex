from __future__ import annotations

from typing import Any, Dict, Optional
import os
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


PROMPT_GENERATION_EXAMPLES = """
Examples of converting SkillRequests to STEVE-1 prompts:

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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[str] = None, verbose: bool = True):
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers torch")
        
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            print(f"[ActionPromptGenerator] Loading model: {model_name} on {self.device}...")
        
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )
        
        self.generation_config = {
            "max_new_tokens": 20,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False,
        }
        
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def _generate_sync(self, messages):
        outputs = self.pipe(messages, **self.generation_config)
        return outputs[0]["generated_text"]
    
    def generate_prompt(
        self,
        skill_name: str,
        skill_params: Dict[str, Any],
    ) -> str:

        if self.verbose:
            print(f"\n[STEVE-1 Prompt Generator]")
            print(f"  Skill: {skill_name}")
            print(f"  Params: {skill_params}")
        
        user_prompt = f"""Convert this SkillRequest to a STEVE-1 prompt:

SkillRequest: {{"name": "{skill_name}", "params": {skill_params}}}

Remember: 2-3 words maximum, goal-oriented, Minecraft terminology.

Here are some examples:
{PROMPT_GENERATION_EXAMPLES}

STEVE Prompt:"""
        
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            loop = asyncio.get_event_loop()
            if loop.is_running():
                response_text = self._generate_sync(messages)
            else:
                response_text = asyncio.run(loop.run_in_executor(self._executor, self._generate_sync, messages))
            
            prompt = response_text.strip()
            
            prompt = prompt.replace('"', '').replace("'", '')
            prompt = prompt.split('\n')[0]
            
            words = prompt.split()
            if len(words) > 5:
                prompt = ' '.join(words[:3])
            
            if self.verbose:
                print(f"  Generated STEVE Prompt: '{prompt}'")
            
            return prompt
            
        except Exception as e:
            if self.verbose:
                print(f"  LLM failed ({e}), using fallback")
            fallback = self._fallback_prompt(skill_name, skill_params)
            if self.verbose:
                print(f"  Fallback STEVE Prompt: '{fallback}'")
            return fallback
    
    def _fallback_prompt(self, skill_name: str, skill_params: Dict[str, Any]) -> str:

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
        
        return mapping.get(skill_name, "mine dirt")


__all__ = [
    "ActionPromptGenerator",
    "HF_AVAILABLE",
]
