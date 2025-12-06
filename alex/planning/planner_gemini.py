from __future__ import annotations
from typing import List, Dict, Any
import json
import os
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.types import GameState, Subgoal
from .base_planner import BasePlanner


class GeminiMCPPlanner(BasePlanner):
    """
    Gemini-powered planner using FastMCP server for prompt engineering.
    The MCP server provides structured prompts and validation.
    """
    
    def __init__(
        self, 
        gemini_api_key: str = None,
        model_name: str = "gemini-2.0-flash-exp",
        mcp_server_path: str = "mcp_server.py",
        verbose: bool = True
    ):
        # Get API key from env if not provided
        if gemini_api_key is None:
            gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or parameters")
            
        genai.configure(api_key=gemini_api_key)
        
        self.verbose = verbose
        
        # Set up generation config for JSON output
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",  # Force JSON output
        }
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        
        # MCP server parameters
        self.server_params = StdioServerParameters(
            command="python",
            args=[mcp_server_path],
            env=None
        )
        
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
        
        # Include vision data if available
        if hasattr(state, 'extras') and state.extras and 'vision' in state.extras:
            state_dict['vision'] = state.extras['vision']
        
        return {k: v for k, v in state_dict.items() if v is not None}
    
    async def plan_async(self, state: GameState) -> List[Subgoal]:
        """Generate subgoals using Gemini with MCP server (async)"""
        state_dict = self._state_to_dict(state)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"GEMINI PLANNER - Thinking...")
            print(f"{'='*70}")
            print(f"[Input State]")
            print(f"  Inventory: {state_dict.get('inventory', {})}")
            print(f"  Health: {state_dict.get('health', 'N/A')}")
            print(f"  Hunger: {state_dict.get('hunger', 'N/A')}")
            if 'vision' in state_dict:
                print(f"  Scene: {state_dict['vision'].get('scene_type', 'unknown')}")
        
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get prompt from MCP server
                    result = await session.call_tool(
                        "plan_actions",
                        arguments={"game_state": state_dict}
                    )
                    
                    prompt = result.content[0].text
                    
                    if self.verbose:
                        print(f"\n[MCP Prompt] (first 500 chars)")
                        print(f"{prompt[:500]}...")
                    
                    # Get Gemini response
                    response = self.model.generate_content(prompt)
                    
                    if self.verbose:
                        print(f"\n[Gemini Response]")
                        print(f"{response.text}")
                    
                    plan = json.loads(response.text)
                    
                    # Validate plan
                    validation = await session.call_tool(
                        "validate_action_plan",
                        arguments={"plan_json": response.text}
                    )
                    
                    validation_result = json.loads(validation.content[0].text)
                    
                    if self.verbose:
                        print(f"\n[Validation]")
                        print(f"  Valid: {validation_result['valid']}")
                        if not validation_result["valid"]:
                            print(f"  Warnings: {validation_result['warnings']}")
                    
                    if not validation_result["valid"]:
                        print(f"⚠ Plan validation warnings: {validation_result['warnings']}")
                    
                    subgoals = self._plan_to_subgoals(plan)
                    
                    if self.verbose:
                        print(f"\n[Generated Subgoals]")
                        for i, sg in enumerate(subgoals, 1):
                            print(f"  {i}. {sg.name} (priority={sg.priority}, params={sg.params})")
                        print(f"{'='*70}\n")
                    
                    return subgoals
                    
        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] Gemini planning failed: {e}")
                print(f"[Fallback] Using simple rule-based planning")
                print(f"{'='*70}\n")
            else:
                print(f"Gemini planning failed: {e}")
            # Use inherited fallback_plan method
            return self.fallback_plan(state)
    
    def plan(self, state: GameState) -> List[Subgoal]:
        """Synchronous wrapper for plan_async"""
        import asyncio
        return asyncio.run(self.plan_async(state))
    
    def _plan_to_subgoals(self, plan: Dict[str, Any]) -> List[Subgoal]:
        """Convert the parsed plan to Subgoal objects"""
        subgoals = []
        
        for sg_dict in plan.get("subgoals", []):
            subgoal = Subgoal(
                name=sg_dict["name"],
                params=sg_dict.get("params", {}),
                priority=sg_dict.get("priority", 50)
            )
            subgoals.append(subgoal)
        
        return subgoals


# Alias for backwards compatibility
GeminiPlanner = GeminiMCPPlanner


__all__ = ['GeminiMCPPlanner', 'GeminiPlanner']