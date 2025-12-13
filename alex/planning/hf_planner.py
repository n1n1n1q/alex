import re
import ray
import json
import time
import asyncio
import torch
import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from transformers import pipeline

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.types import GameState, Subgoal
from .base_planner import BasePlanner


class HuggingFacePlanner(BasePlanner):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mcp_server_path: str = "alex.prompts.mcp_server",
        verbose: bool = True,
    ):

        self.verbose = verbose
        self.device = device
        self.mcp_server_path = mcp_server_path

        if self.verbose:
            print(
                f"[{self.__class__.__name__}] Loading model: {model_name} on {device}..."
            )

        self.model_name = model_name
        self.pipe = None

        self._last_raw_response = None
        self._last_cleaned_response = None

        self.generation_config = {
            "max_new_tokens": 1024,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False,
        }

        self.server_params = None
        self._executor = None
        self._last_successful_plan = []

    def _ensure_resources(self):
        if self.pipe is None:
            device_index = 0 if self.device == "cuda" else -1
            dtype = (
                torch.bfloat16
                if self.device == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model_name,
                device=device_index,
                dtype=dtype,
                trust_remote_code=True,
            )
            time.sleep(10)

        if self.server_params is None:
            module_path = self.mcp_server_path.replace("/", ".")
            print(">>>>", module_path)
            self.server_params = StdioServerParameters(
                command=sys.executable, args=["-m", module_path], env=os.environ.copy()
            )

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)

    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable types to Python native types"""
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()  # Convert single-element array to scalar
            return obj.tolist()  # Convert multi-element array to list
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert numpy scalar to Python scalar
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def _state_to_dict(self, state: GameState) -> Dict[str, Any]:

        state_dict = {
            "inventory": state.inventory_agg if state.inventory_agg else {},
            "health": state.health,
            "hunger": state.hunger,
            "position": state.player_pos if state.player_pos else {},
            "biome": state.env_state.get("biome_id") if state.env_state else None,
            "mobs": state.mobs if state.mobs else [],
            "blocks": state.blocks if state.blocks else [],
        }

        if hasattr(state, "extras") and state.extras and "vision" in state.extras:
            state_dict["vision"] = state.extras["vision"]

        state_dict = self._convert_to_serializable(state_dict)

        return {k: v for k, v in state_dict.items() if v is not None}

    def _clean_json_output(self, text: str) -> str:

        text = text.strip()
        pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    async def _generate_hf(self, system_prompt: str, user_prompt: str) -> str:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        loop = asyncio.get_running_loop()

        def run_inference():
            outputs = self.pipe(messages, **self.generation_config)
            return outputs[0]["generated_text"]

        return await loop.run_in_executor(self._executor, run_inference)

    async def plan_async(self, state: GameState) -> List[Subgoal]:

        state_dict = self._state_to_dict(state)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"HF TRANSFORMERS PLANNER - Thinking...")
            print(f"{'='*70}")
            print(f"[Input State]")
            print(f"  Inventory: {state_dict.get('inventory', {})}")

        print(">>> [STATE DICT]", state_dict)

        try:
            if self.verbose:
                print(
                    f"\n[MCP] Starting server: {self.server_params.command} {' '.join(self.server_params.args)}"
                )

            async with stdio_client(self.server_params) as (read, write):
                if self.verbose:
                    print(f"[MCP] Server started, creating session...")

                async with ClientSession(read, write) as session:
                    if self.verbose:
                        print(f"[MCP] Session created, initializing...")
                    await session.initialize()

                    result = await session.call_tool(
                        "plan_actions", arguments={"game_state": state_dict}
                    )

                    mcp_prompt = result.content[0].text

                    if self.verbose:
                        print(f"\n[MCP Prompt] (first 200 chars)")
                        print(f"{mcp_prompt[:200]}...")

                    system_instruction = (
                        "You are an expert Minecraft short-term agent planner. "
                        "You must output ONLY valid JSON. "
                        "Do not include markdown formatting or explanations."
                    )

                    raw_response = await self._generate_hf(
                        system_instruction, mcp_prompt
                    )
                    cleaned_json_str = self._clean_json_output(raw_response)

                    self._last_raw_response = raw_response
                    self._last_cleaned_response = cleaned_json_str

                    if self.verbose:
                        print(f"\n[Model Response]")

                    try:
                        plan_json = json.loads(cleaned_json_str)
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"[Error] Invalid JSON generated: {e}")
                        raise e

                    validation = await session.call_tool(
                        "validate_action_plan",
                        arguments={"plan_json": cleaned_json_str},
                    )

                    validation_result = json.loads(validation.content[0].text)

                    if self.verbose:
                        print(f"\n[Validation]")
                        print(f"  Valid: {validation_result['valid']}")
                        if not validation_result["valid"]:
                            print(f"  Warnings: {validation_result['warnings']}")

                    if not validation_result["valid"]:
                        print(
                            f"Plan validation warnings: {validation_result['warnings']}"
                        )

                    subgoals = self._plan_to_subgoals(plan_json)

                    if self.verbose:
                        print(f"\n[Generated Subgoals]")
                        for i, sg in enumerate(subgoals, 1):
                            print(
                                f"  {i}. {sg.name} (priority={sg.priority}, params={sg.params})"
                            )
                        print(f"{'='*70}\n")

                    self._last_successful_plan = subgoals

                    return subgoals

        except Exception as e:
            if self.verbose:
                print(f"\n[ERROR] HF planning failed: {e}")
                import traceback

                traceback.print_exc()

                def print_exception_group(exc, level=0):
                    if hasattr(exc, "exceptions"):
                        for i, sub_exc in enumerate(exc.exceptions, 1):
                            indent = "  " * level
                            print(f"{indent}--- Sub-exception {i} (level {level}) ---")
                            traceback.print_exception(
                                type(sub_exc), sub_exc, sub_exc.__traceback__
                            )

                            err_str = str(sub_exc).lower()
                            if (
                                "modulenotfounderror" in str(type(sub_exc)).lower()
                                or "no module" in err_str
                            ):
                                print(f"\n{indent}Module import error detected.")
                                print(f"{indent}   Error: {sub_exc}")
                                if "mcp" in err_str:
                                    print(
                                        f"{indent}   Install: pip install mcp>=0.9.0 fastmcp>=0.2.0"
                                    )

                            print_exception_group(sub_exc, level + 1)

                print_exception_group(e)
                print(f"{'='*70}\n")
            else:
                print(f"HF planning failed: {e}")

            if self._last_successful_plan:
                if self.verbose:
                    print(f"[Fallback] Strategy: Repeating last successful plan")
                return self._last_successful_plan

            return self.fallback_plan(state)

    def plan(self, state: GameState) -> List[Subgoal]:
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.plan_async(state))
                    return future.result()
            except RuntimeError:
                return asyncio.run(self.plan_async(state))
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Failed to run async planning: {e}")
                import traceback

                traceback.print_exc()
            return self.fallback_plan(state)

    def _plan_to_subgoals(self, plan: Dict[str, Any]) -> List[Subgoal]:

        subgoals = []

        items = plan.get("subgoals", [])

        for sg_dict in items:
            subgoal = Subgoal(
                name=sg_dict["name"],
                params=sg_dict.get("params", {}),
                priority=sg_dict.get("priority", 50),
            )
            subgoals.append(subgoal)

        return subgoals
