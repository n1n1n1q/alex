"""HuggingFace-based planner using local transformers models."""

import json
import asyncio
import torch
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from transformers import pipeline

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.types import GameState, Subgoal
from .base_planner import BasePlanner

class HuggingFacePlanner(BasePlanner):
    """
    Local Transformers-powered planner using FastMCP server for prompt engineering.
    Uses local Hugging Face models (e.g., Llama-3, Qwen, Mistral) for inference.
    """
    
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct", # Lightweight, good instruction following
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mcp_server_path: str = "mcp_server.py",
        verbose: bool = True
    ):
        
        self.verbose = verbose
        self.device = device
        
        if self.verbose:
            print(f"[{self.__class__.__name__}] Loading model: {model_name} on {device}...")

        # Initialize Transformers Pipeline
        # We use torch_dtype=torch.bfloat16 for modern GPU efficiency if available
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )

        # Generation configuration
        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.6, # Slightly lower for more stable JSON
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False, # Only return the new tokens
        }

        self.server_params = StdioServerParameters(
            command="python",
            args=[mcp_server_path],
            env=None
        )
        
        # Executor for running blocking HF inference
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _state_to_dict(self, state: GameState) -> Dict[str, Any]:
        """Convert GameState to JSON-serializable dict"""
        state_dict = {
            "inventory": state.inventory_agg if state.inventory_agg else {},
            "health": state.health,
            "hunger": state.hunger,
            "position": state.player_pos if state.player_pos else {},
            "biome": state.env_state.get("biome_id") if state.env_state else None,
            "mobs": state.mobs if state.mobs else [],
            "blocks": state.blocks if state.blocks else [],
        }
        
        if hasattr(state, 'extras') and state.extras and 'vision' in state.extras:
            state_dict['vision'] = state.extras['vision']
        
        return {k: v for k, v in state_dict.items() if v is not None}
    
    def _clean_json_output(self, text: str) -> str:
        """Strip Markdown code blocks and whitespace to isolate JSON."""
        text = text.strip()
        # Regex to capture content inside ```json ... ``` or just ``` ... ```
        pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    async def _generate_hf(self, system_prompt: str, user_prompt: str) -> str:
        """Run blocking transformers inference in an executor"""
        
        # Structure messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        loop = asyncio.get_running_loop()
        
        def run_inference():
            # Pipeline automatically handles tokenizer.apply_chat_template
            outputs = self.pipe(messages, **self.generation_config)
            return outputs[0]["generated_text"]
            
        return await loop.run_in_executor(self._executor, run_inference)

    async def plan_async(self, state: GameState) -> List[Subgoal]:
        """Generate subgoals using Local HF Model with MCP server"""
        state_dict = self._state_to_dict(state)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"HF TRANSFORMERS PLANNER - Thinking...")
            print(f"{'='*70}")
            print(f"[Input State]")
            print(f"  Inventory: {state_dict.get('inventory', {})}")
            
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # 1. Get prompt from MCP server
                    result = await session.call_tool(
                        "plan_actions",
                        arguments={"game_state": state_dict}
                    )
                    
                    # The MCP returns the specific context/instructions based on game state
                    mcp_prompt = result.content[0].text
                    
                    if self.verbose:
                        print(f"\n[MCP Prompt] (first 200 chars)")
                        print(f"{mcp_prompt[:200]}...")
                    
                    # 2. Generate Response using local HF model
                    # We inject a system prompt to enforce JSON formatting strictly
                    system_instruction = (
                        "You are an expert Minecraft agent planner. "
                        "You must output ONLY valid JSON. "
                        "Do not include markdown formatting or explanations."
                    )
                    
                    raw_response = await self._generate_hf(system_instruction, mcp_prompt)
                    cleaned_json_str = self._clean_json_output(raw_response)

                    if self.verbose:
                        print(f"\n[Model Response]")
                        print(f"{cleaned_json_str[:500]}..." if len(cleaned_json_str) > 500 else cleaned_json_str)
                    
                    # 3. Validate JSON structure internally first
                    try:
                        plan_json = json.loads(cleaned_json_str)
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"[Error] Invalid JSON generated: {e}")
                        raise e

                    # 4. Validate plan logic via MCP
                    validation = await session.call_tool(
                        "validate_action_plan",
                        arguments={"plan_json": cleaned_json_str}
                    )
                    
                    validation_result = json.loads(validation.content[0].text)
                    
                    if self.verbose:
                        print(f"\n[Validation]")
                        print(f"  Valid: {validation_result['valid']}")
                        if not validation_result["valid"]:
                            print(f"  Warnings: {validation_result['warnings']}")
                    
                    if not validation_result["valid"]:
                        print(f"⚠ Plan validation warnings: {validation_result['warnings']}")
                    
                    # 5. Convert to objects
                    subgoals = self._plan_to_subgoals(plan_json)
                    
                    if self.verbose:
                        print(f"\n[Generated Subgoals]")
                        for i, sg in enumerate(subgoals, 1):
                            print(f"  {i}. {sg.name} (priority={sg.priority}, params={sg.params})")
                        print(f"{'='*70}\n")
                    
                    return subgoals
                    
        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] HF planning failed: {e}")
                import traceback
                traceback.print_exc()
                print(f"[Fallback] Using simple rule-based planning")
                print(f"{'='*70}\n")
            else:
                print(f"HF planning failed: {e}")
            
            return self.fallback_plan(state)
    
    def plan(self, state: GameState) -> List[Subgoal]:
        """Synchronous wrapper for plan_async"""
        return asyncio.run(self.plan_async(state))
    
    def _plan_to_subgoals(self, plan: Dict[str, Any]) -> List[Subgoal]:
        """Convert the parsed plan to Subgoal objects"""
        subgoals = []
        
        # Handle cases where the model might wrap the list in a root key like "actions" or "plan"
        # Adapting to common JSON variances
        items = plan.get("subgoals", [])
        
        for sg_dict in items:
            subgoal = Subgoal(
                name=sg_dict["name"],
                params=sg_dict.get("params", {}),
                priority=sg_dict.get("priority", 50)
            )
            subgoals.append(subgoal)
        
        return subgoals
