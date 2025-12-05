from __future__ import annotations
from typing import List, Dict, Any
import json
import os
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .types import GameState, Subgoal


class GeminiMCPPlanner:
    """
    Gemini-powered planner using FastMCP server for prompt engineering.
    The MCP server provides structured prompts and validation.
    """
    
    def __init__(
        self, 
        gemini_api_key: str,
        model_name: str = "gemini-2.0-flash-exp",
        mcp_server_path: str = "mcp_server.py"
    ):
        genai.configure(api_key=gemini_api_key)
        
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
            "inventory": state.inventory,
            "health": state.health,
            "hunger": state.hunger,
            "position": state.position,
            "biome": state.biome,
            "time_of_day": state.time_of_day,
            "nearby_entities": state.nearby_entities,
        }
        return {k: v for k, v in state_dict.items() if v is not None}
    
    async def plan_async(self, state: GameState) -> List[Subgoal]:
        """Generate subgoals using Gemini with MCP server (async)"""
        state_dict = self._state_to_dict(state)
        
        try:

            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    result = await session.call_tool(
                        "plan_actions",
                        arguments={"game_state": state_dict}
                    )
                    
                    prompt = result.content[0].text
                    
                    response = self.model.generate_content(prompt)
                    
                    plan = json.loads(response.text)
                    
                    validation = await session.call_tool(
                        "validate_action_plan",
                        arguments={"plan_json": response.text}
                    )
                    
                    validation_result = json.loads(validation.content[0].text)
                    
                    if not validation_result["valid"]:
                        print(f"Plan validation warnings: {validation_result['warnings']}")
                    
                    return self._plan_to_subgoals(plan)
                    
        except Exception as e:
            print(f"Gemini planning failed: {e}")
            return self._fallback_plan(state)
    
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
    
    def _fallback_plan(self, state: GameState) -> List[Subgoal]:
        """Fallback to basic heuristics if Gemini/MCP fails"""
        subgoals: List[Subgoal] = []
        
        # Emergency situations
        if state.health is not None and state.health <= 4:
            return [Subgoal(name="emergency_retreat", params={}, priority=100)]
        
        # Night without light
        if state.time_of_day == "night" and state.inventory.get("torch", 0) == 0:
            subgoals.append(Subgoal(name="seek_shelter", params={}, priority=90))
        
        # Basic progression
        if state.inventory.get("planks", 0) < 4 and state.inventory.get("logs", 0) < 2:
            subgoals.append(Subgoal(name="collect_wood", params={"count": 4}, priority=10))
        elif state.inventory.get("crafting_table", 0) < 1:
            subgoals.append(Subgoal(name="craft_table", params={}, priority=8))
        else:
            subgoals.append(Subgoal(name="idle_scan", params={}, priority=0))
        
        return subgoals